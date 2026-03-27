import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# 基础配置
NUM_INPUT_CHANNELS = 11
BOARD_SIZE = 9


class RMSNorm(nn.Module):
    """
    改用 RMSNorm。
    """

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        dtype = x.dtype
        x_f32 = x.float()
        res = x_f32 * torch.rsqrt(x_f32.pow(2).mean(-1, keepdim=True) + self.eps)
        return res.to(dtype) * self.weight


class PoPEPositionalManager(nn.Module):
    """
    Polar Coordinate Position Embedding (PoPE)
    1. 2 路 (x, y)，最大化每一个正交方向的频率分辨率。
     (对角线信息本就可由 QKV 投影矩阵自己学到 x 和 y 的线性组合)
    2. 引入受限可学习的频率。
       通过 Sigmoid 将波长限制在 [1.8, 1024.0] 范围内，
       2.0 是奈奎斯特极限，1.8 为梯度传播提供容错，1024.0 代替 inf
    解耦 What (Content) 和 Where (Position)
    """

    def __init__(self, dim, board_size, min_wave=1.8, max_wave=1024.0):
        super().__init__()
        self.dim = dim
        self.sub_dim = dim // 2
        self.board_size = board_size
        self.min_wave = min_wave
        self.max_wave = max_wave

        # 对数空间均匀初始化波长
        init_lambdas = torch.exp(torch.linspace(math.log(min_wave), math.log(max_wave), self.sub_dim))

        # 逆推 Sigmoid 的输入 w，使得初始化符合 init_lambdas
        target_sigmoid = torch.log(init_lambdas / min_wave) / math.log(max_wave / min_wave)
        target_sigmoid = torch.clamp(target_sigmoid, 1e-4, 1.0 - 1e-4)
        init_w = torch.log(target_sigmoid / (1.0 - target_sigmoid))

        # 可学习参数
        self.w_x = nn.Parameter(init_w.clone())
        self.w_y = nn.Parameter(init_w.clone())

        # 基础坐标网格
        coords = torch.arange(board_size).float()
        y_c, x_c = torch.meshgrid(coords, coords, indexing='ij')
        self.register_buffer("c_x", x_c.reshape(-1, 1))
        self.register_buffer("c_y", y_c.reshape(-1, 1))

        # 推理加速用的静态 Buffer
        # 在 eval() 模式下直接读取，无需重复计算三角函数
        self.register_buffer("cos_phi_cached", torch.zeros(board_size * board_size, dim))
        self.register_buffer("sin_phi_cached", torch.zeros(board_size * board_size, dim))

    def _get_freqs(self, w):
        # 计算波长 -> 频率的连续映射
        ratio = torch.sigmoid(w)
        lambdas = self.min_wave * ((self.max_wave / self.min_wave) ** ratio)
        return 2 * math.pi / lambdas

    def update_working_buffers(self):
        """
        在推理/导出 TRT 前调用此函数。
        将当前学到的可学习波长固化到缓存 Buffer 中。
        """
        with torch.no_grad():
            freq_x = self._get_freqs(self.w_x)
            freq_y = self._get_freqs(self.w_y)
            phi_x = self.c_x * freq_x.unsqueeze(0)
            phi_y = self.c_y * freq_y.unsqueeze(0)
            phi = torch.cat([phi_x, phi_y], dim=-1)
            self.cos_phi_cached.copy_(torch.cos(phi))
            self.sin_phi_cached.copy_(torch.sin(phi))

    def get_sincos(self):
        if self.training:
            # 训练模式：保持动态计算，允许梯度回传
            freq_x = self._get_freqs(self.w_x)
            freq_y = self._get_freqs(self.w_y)
            phi_x = self.c_x * freq_x.unsqueeze(0)
            phi_y = self.c_y * freq_y.unsqueeze(0)
            phi = torch.cat([phi_x, phi_y], dim=-1)
            return torch.cos(phi), torch.sin(phi)
        else:
            # 推理模式：直接返回缓存，性能最优且 TRT 友好
            return self.cos_phi_cached, self.sin_phi_cached


class PoPE2DAttention(nn.Module):
    """
    Softplus 只是为了防止负数产生的 π 相位
    不如直接改用 TRT 高度优化的 ReLU
    """

    def __init__(self, dim, num_heads, board_size):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv_proj = nn.Linear(dim, dim * 3, bias=True)
        self.out_proj = nn.Linear(dim, dim, bias=True)
        self.pos_manager = PoPEPositionalManager(self.head_dim, board_size)

    def forward(self, x):
        b, n, d = x.shape
        h, hd = self.num_heads, self.head_dim

        qkv = self.qkv_proj(x).view(b, n, 3, h, hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # 极坐标旋转 (PoPE 核心)
        mu_q, mu_k = F.relu(q), F.relu(k)
        cos_p, sin_p = self.pos_manager.get_sincos()

        q_p = torch.cat([mu_q * cos_p, mu_q * sin_p], dim=-1)
        k_p = torch.cat([mu_k * cos_p, mu_k * sin_p], dim=-1)

        out = F.scaled_dot_product_attention(q_p, k_p, v, scale=self.scale)
        return self.out_proj(out.transpose(1, 2).reshape(b, n, d))


class AttnResBlock(nn.Module):
    """
    集成 Attention Residuals (AttnRes) 理念的改进块。
    核心逻辑：将固定权重的残差累加替换为基于输入的动态 Softmax 聚合，
    解决深层网络中的贡献稀释（Dilution）问题。
    """

    def __init__(self, dim, num_heads, board_size, n_streams, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_streams = n_streams

        self.norm1 = RMSNorm(dim)
        self.attn = PoPE2DAttention(dim, num_heads, board_size)
        self.norm2 = RMSNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 3 * dim),
            nn.ReLU(inplace=True),
            nn.Linear(3 * dim, dim)
        )
        self.mhc_mixer = nn.Linear(dim, n_streams * n_streams)
        self.gamma = nn.Parameter(torch.full((dim,), 1e-5))

    def forward(self, x_stream):
        B, L, N, D = x_stream.shape
        combined_input = x_stream.mean(dim=2)

        attn_out = self.attn(self.norm1(combined_input))
        ffn_out = self.mlp(self.norm2(combined_input + attn_out))
        delta = (attn_out + ffn_out) * self.gamma

        mixer_logits = self.mhc_mixer(combined_input).view(B, L, N, N)
        mixer_weights = F.softmax(mixer_logits, dim=-1)

        x_stream_reshaped = x_stream.view(-1, N, D)
        # 线性变换流状态：x_next = Softmax(W) * x_prev
        x_stream_next = torch.bmm(mixer_weights.view(-1, N, N), x_stream_reshaped)
        x_stream_next = x_stream_next.view(B, L, N, D)

        return x_stream_next + delta.unsqueeze(2)


class PomPomNN(nn.Module):
    """
    PoPE(相对坐标)与绝对坐标信息都是必要的
    注意力块本身就包含了良好的信息处理步骤
    任何其他经典的处理方法都只会影响收敛速度
    """

    def __init__(self, num_layers=8, num_filters=128, num_heads=4, n_streams=2):
        super().__init__()
        self.n_streams = n_streams
        self.num_layers = num_layers

        self.input_projection = nn.Sequential(
            nn.Linear(NUM_INPUT_CHANNELS + 2, num_filters),
            nn.ReLU(inplace=True),
            nn.Linear(num_filters, num_filters)
        )

        self.blocks = nn.ModuleList([
            AttnResBlock(num_filters, num_heads, BOARD_SIZE, n_streams, i)
            for i in range(num_layers)
        ])

        self.final_norm = RMSNorm(num_filters)
        self.policy_head = nn.Linear(num_filters, 2)
        self.value_head = nn.Sequential(
            nn.Linear(num_filters, num_filters),
            nn.ReLU(inplace=True),
            nn.Linear(num_filters, 1)
        )

        coords = torch.linspace(-1, 1, BOARD_SIZE)
        y_c, x_c = torch.meshgrid(coords, coords, indexing='ij')
        self.register_buffer("static_coords", torch.stack([x_c, y_c], dim=0))

    def update_all_buffers(self):
        """推理前统一更新所有层的位置编码缓存"""
        for block in self.blocks:
            block.attn.pos_manager.update_working_buffers()

    def forward(self, x):
        b = x.shape[0]
        c_feat = self.static_coords.unsqueeze(0).expand(b, -1, -1, -1)
        x = torch.cat([x, c_feat], dim=1)

        x = x.view(b, x.size(1), -1).transpose(1, 2)
        x = self.input_projection(x)

        x_stream = x.unsqueeze(2).repeat(1, 1, self.n_streams, 1)

        for block in self.blocks:
            x_stream = block(x_stream)

        x_tokens = self.final_norm(x_stream.mean(dim=2))

        p_out = self.policy_head(x_tokens)
        pol = p_out[..., 0]
        soft_pol = p_out[..., 1]

        v_aggregated = x_tokens.mean(dim=1)
        val = torch.tanh(self.value_head(v_aggregated))

        return pol, val, soft_pol


if __name__ == "__main__":
    model = PomPomNN()

    # 模拟训练结束后的准备工作
    model.eval()
    model.update_all_buffers()  # 更新推理缓存

    dummy_input = torch.randn(1, 11, 9, 9)
    with torch.no_grad():
        p, v, sp = model(dummy_input)
    print("--- 模型初始化成功 ---")