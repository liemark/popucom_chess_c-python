import torch
import torch.nn as nn
import torch.nn.functional as F
import math

#
NUM_INPUT_CHANNELS = 11
BOARD_SIZE = 9
THETA_BASE = 10


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
    解耦 What (Content) 和 Where (Position)
    对于明确已知边界的任务，不需要原论文中的可变参数 delta
    方便预计算缓存位置嵌入相位
    """

    def __init__(self, dim, board_size, theta_base=THETA_BASE):
        super().__init__()
        half_hd = dim // 2
        inv_freq = 1.0 / (theta_base ** (torch.arange(0, half_hd).float() / (half_hd - 1 if half_hd > 1 else 1)))
        coords = torch.arange(board_size).float()
        y_c, x_c = torch.meshgrid(coords, coords, indexing='ij')

        phi_x = x_c.reshape(-1, 1) * inv_freq.unsqueeze(0)
        phi_y = y_c.reshape(-1, 1) * inv_freq.unsqueeze(0)
        phi = torch.cat([phi_x, phi_y], dim=-1)  # (81, HD)

        self.register_buffer("cos_phi", torch.cos(phi))
        self.register_buffer("sin_phi", torch.sin(phi))

    def get_sincos(self):
        return self.cos_phi, self.sin_phi


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

        # 核心特征变换路径
        self.norm1 = RMSNorm(dim)
        self.attn = PoPE2DAttention(dim, num_heads, board_size)
        self.norm2 = RMSNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 3 * dim),
            nn.ReLU(inplace=True),
            nn.Linear(3 * dim, dim)
        )

        # 多流转换系数 (近似 Sinkhorn / Attention Residuals 中的跨层权重)
        # 允许每一层根据当前输入 Content 动态决定从哪些流读取/写入信息
        self.mhc_mixer = nn.Linear(dim, n_streams * n_streams)

        # LayerScale: 用于稳定极深网络的训练
        self.gamma = nn.Parameter(torch.full((dim,), 1e-5))

    def forward(self, x_stream):
        """
        x_stream: (B, L, N, D) -> Batch, SeqLen, NumStreams, Dim
        """
        B, L, N, D = x_stream.shape

        # 1. 跨流聚合 (Cross-stream Aggregation)
        # 将多个流的信息汇聚，作为当前层处理的 Query 背景
        combined_input = x_stream.mean(dim=2)

        # 2. 变换层计算
        attn_out = self.attn(self.norm1(combined_input))
        ffn_out = self.mlp(self.norm2(combined_input + attn_out))
        # 增量信息 (New Content)
        delta = (attn_out + ffn_out) * self.gamma

        # 3. 动态残差连接 (AttnRes 核心逻辑)
        # 计算输入相关的混合权重矩阵，实现非等权的层间/流间信息整合
        mixer_logits = self.mhc_mixer(combined_input).view(B, L, N, N)
        mixer_weights = F.softmax(mixer_logits, dim=-1)

        x_stream_reshaped = x_stream.view(-1, N, D)
        # 线性变换流状态：x_next = Softmax(W) * x_prev
        x_stream_next = torch.bmm(mixer_weights.view(-1, N, N), x_stream_reshaped)
        x_stream_next = x_stream_next.view(B, L, N, D)

        # 将新特征 delta 以残差形式注入所有流
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

        # 输入投影：第一层投影功能明确，不使用残差
        self.input_projection = nn.Sequential(
            nn.Linear(NUM_INPUT_CHANNELS + 2, num_filters),
            nn.ReLU(inplace=True),
            nn.Linear(num_filters, num_filters)
        )

        # 使用集成了 AttnRes 理念的 Block
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

        # 预计算坐标并注册为 buffer
        coords = torch.linspace(-1, 1, BOARD_SIZE)
        y_c, x_c = torch.meshgrid(coords, coords, indexing='ij')
        c_feat = torch.stack([x_c, y_c], dim=0)
        self.register_buffer("static_coords", c_feat)

    def forward(self, x):
        b, c, h, w = x.shape

        # 1. 注入坐标x,y (CoordConv)
        c_feat = self.static_coords.unsqueeze(0).expand(b, -1, -1, -1)
        x = torch.cat([x, c_feat], dim=1)

        # 2. 初始 MLP 投影 (B, 81, Channels) -> (B, 81, Filters)
        x = x.view(b, x.size(1), -1).transpose(1, 2)
        x = self.input_projection(x)

        # 3. 初始化多流 (Multi-stream)
        # 通过多流维护隐状态，配合 AttnRes 动态权重实现更深度的特征传递
        x_stream = x.unsqueeze(2).repeat(1, 1, self.n_streams, 1)

        for block in self.blocks:
            x_stream = block(x_stream)

        # 4. 平均池化恢复
        x_tokens = self.final_norm(x_stream.mean(dim=2))

        # 5. 策略输出
        p_out = self.policy_head(x_tokens)
        pol = p_out[..., 0]
        soft_pol = p_out[..., 1]

        # 6. 价值输出
        v_aggregated = x_tokens.mean(dim=1)
        val = torch.tanh(self.value_head(v_aggregated))

        return pol, val, soft_pol


if __name__ == "__main__":
    model = PomPomNN(num_layers=12, num_filters=128, num_heads=4, n_streams=4)
    model.eval()
    dummy_input = torch.randn(1, 11, 9, 9)
    with torch.no_grad():
        p, v, sp = model(dummy_input)
    print("--- 改进版模型初始化成功 ---")