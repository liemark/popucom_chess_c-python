import torch
import torch.nn as nn
import torch.nn.functional as F

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


class mHCTurboBlock(nn.Module):
    """
    Multi-Head Collaborative (mHC) Block。
    直接用 softmax 近似 Sinkhorn
    """

    def __init__(self, dim, num_heads, board_size, n_streams=4):
        super().__init__()
        self.n = n_streams
        self.mhc_proj = nn.Linear(dim, n_streams * 2 + n_streams * n_streams)
        self.alpha = nn.Parameter(torch.full((3,), 0.01))
        self.layer_scale = nn.Parameter(torch.full((dim,), 1e-5))

        self.norm1 = RMSNorm(dim)
        self.attn = PoPE2DAttention(dim, num_heads, board_size)
        self.norm2 = RMSNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.ReLU(inplace=True),
            nn.Linear(4 * dim, dim)
        )

    def forward(self, x_stream):
        B, L, N, D = x_stream.shape
        x_avg = x_stream.mean(dim=2)

        coeffs = self.mhc_proj(x_avg)
        h_pre = torch.sigmoid(coeffs[:, :, :N] * self.alpha[0]).unsqueeze(-1)
        h_post = (torch.sigmoid(coeffs[:, :, N:2 * N] * self.alpha[1]) * 2.0).unsqueeze(-1)

        h_res_raw = coeffs[:, :, 2 * N:].view(B, L, N, N) * self.alpha[2]
        h_res = F.softmax(h_res_raw, dim=-1)

        layer_input = (x_stream * h_pre).sum(dim=2)

        res = self.attn(self.norm1(layer_input))
        res = res + self.mlp(self.norm2(layer_input + res))

        x_stream_next = torch.bmm(h_res.reshape(-1, N, N), x_stream.reshape(-1, N, D)).reshape(B, L, N, D)
        return x_stream_next + (res.unsqueeze(2) * (h_post * self.layer_scale))


class PomPomNN(nn.Module):
    """
    PoPE(相对坐标)与绝对坐标信息都是必要的
    注意力块本身就包含了良好的信息处理步骤
    任何其他经典的处理方法都只会影响收敛速度
    """

    def __init__(self, num_layers=6, num_filters=320, num_heads=4, n_streams=2):
        super().__init__()
        self.n_streams = n_streams

        self.input_projection = nn.Sequential(
            nn.Linear(NUM_INPUT_CHANNELS + 2, num_filters),
            nn.ReLU(inplace=True),
            nn.Linear(num_filters, num_filters)
        )

        self.blocks = nn.ModuleList([
            mHCTurboBlock(num_filters, num_heads, BOARD_SIZE, n_streams)
            for _ in range(num_layers)
        ])

        self.final_norm = RMSNorm(num_filters)
        self.policy_head = nn.Linear(num_filters, 2)

        self.value_head = nn.Sequential(
            nn.Linear(num_filters, num_filters),
            nn.ReLU(inplace=True),
            nn.Linear(num_filters, 1)
        )

    def forward(self, x):
        b, c, h, w = x.shape

        # 1. 注入坐标x,y (CoordConv)
        coords = torch.linspace(-1, 1, BOARD_SIZE, device=x.device)
        y_c, x_c = torch.meshgrid(coords, coords, indexing='ij')
        c_feat = torch.stack([x_c, y_c], dim=0).unsqueeze(0).repeat(b, 1, 1, 1)
        x = torch.cat([x, c_feat], dim=1)

        # 2. MLP (B, 81, 13) -> (B, 81, 128)
        x = x.view(b, x.size(1), -1).transpose(1, 2)
        # 这里不要用残差连接，第一层的投影功能本来就很明确
        x = self.input_projection(x)

        # 3. mHC
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
        # 直接平均池化即可，注意力层会均衡好策略头与价值头的信息的
        v_aggregated = x_tokens.mean(dim=1)
        val = torch.tanh(self.value_head(v_aggregated))

        return pol, val, soft_pol


if __name__ == "__main__":
    model = PomPomNN(num_layers=8, num_filters=128, num_heads=4, n_streams=2)
    model.eval()
    dummy_input = torch.randn(1, 11, 9, 9)
    with torch.no_grad():
        p, v, sp = model(dummy_input)
    print("--- 模型初始化成功 ---")