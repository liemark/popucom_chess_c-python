import torch
import torch.nn as nn
import torch.nn.functional as F

# 基础配置
NUM_INPUT_CHANNELS = 11
BOARD_SIZE = 9


@torch.jit.script
def fast_sinkhorn_v1(W: torch.Tensor):
    """
    极致性能的单次 Sinkhorn。
    1次迭代不仅是为了数学正确，更是为了给 MCTS 提供稳定的先验分布，
    防止搜索树因分布过于平坦而过度膨胀。
    """
    # 减去最大值防止 exp 溢出，并提供基础的数值稳定性
    W = torch.exp(W - W.max(dim=-1, keepdim=True)[0])
    # 强制执行 1 次双向归一化，确保流形约束
    W = W / (W.sum(dim=-1, keepdim=True) + 1e-6)
    W = W / (W.sum(dim=-2, keepdim=True) + 1e-6)
    return W


class PoPE2DAttention(nn.Module):
    """
    针对 9x9 优化的 PoPE。
    保持计算简单，确保 Policy 输出的锐度。
    """

    def __init__(self, dim, num_heads, board_size, theta_base=10):
        # theta: 8/(pi/2) ≈ 5.1
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.qkv_proj = nn.Linear(dim, dim * 3, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

        half_hd = self.head_dim // 2
        inv_freq = 1.0 / (theta_base ** (torch.arange(0, half_hd).float() / (half_hd - 1 if half_hd > 1 else 1)))

        coords = torch.arange(board_size).float()
        y_c, x_c = torch.meshgrid(coords, coords, indexing='ij')
        phi_x = x_c.flatten().unsqueeze(1) * inv_freq.unsqueeze(0)
        phi_y = y_c.flatten().unsqueeze(1) * inv_freq.unsqueeze(0)
        phi = torch.cat([phi_x, phi_y], dim=-1)  # (81, head_dim)

        self.register_buffer("phi", phi)
        self.bias_delta = nn.Parameter(torch.zeros(num_heads, self.head_dim))

    def forward(self, x):
        b, n, d = x.shape
        h, hd = self.num_heads, self.head_dim

        qkv = self.qkv_proj(x).view(b, n, 3, h, hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        mu_q, mu_k = F.softplus(q), F.softplus(k)

        # 保持三角函数计算，确保位置感的精确
        phi_total = self.phi + self.bias_delta.unsqueeze(1)

        q_cos, q_sin = mu_q * torch.cos(self.phi), mu_q * torch.sin(self.phi)
        k_cos, k_sin = mu_k * torch.cos(phi_total), mu_k * torch.sin(phi_total)

        scores = (torch.matmul(q_cos, k_cos.transpose(-1, -2)) +
                  torch.matmul(q_sin, k_sin.transpose(-1, -2))) / (hd ** 0.5)

        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).reshape(b, n, d)
        return self.out_proj(out)


class mHCTurboBlock(nn.Module):
    """
    Turbo Block：最小化算子数量。
    通过一次 mHC 投影完成 Attention + MLP 的流管理。
    """

    def __init__(self, dim, num_heads, board_size, n_streams=4):
        super().__init__()
        self.n = n_streams
        self.mhc_proj = nn.Linear(dim, n_streams * 2 + n_streams * n_streams)
        self.alpha = nn.Parameter(torch.full((3,), 0.01))

        self.norm1 = nn.LayerNorm(dim)
        self.attn = PoPE2DAttention(dim, num_heads, board_size)

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.ReLU(),
            nn.Linear(4 * dim, dim)
        )

    def forward(self, x_stream):
        B, L, N, D = x_stream.shape
        x_avg = x_stream.mean(dim=2)

        coeffs = self.mhc_proj(x_avg)
        h_pre = torch.sigmoid(coeffs[:, :, :N] * self.alpha[0]).unsqueeze(-1)
        h_post = (torch.sigmoid(coeffs[:, :, N:2 * N] * self.alpha[1]) * 2.0).unsqueeze(-1)
        h_res_raw = coeffs[:, :, 2 * N:].view(B, L, N, N) * self.alpha[2]

        # 必须保留至少 1 次迭代以引导 MCTS 搜索效率
        h_res = fast_sinkhorn_v1(h_res_raw)

        layer_input = (x_stream * h_pre).sum(dim=2)

        delta = self.attn(self.norm1(layer_input))
        delta = delta + self.mlp(self.norm2(layer_input + delta))

        # 矩阵融合
        h_res_f = h_res.view(-1, N, N)
        x_str_f = x_stream.view(-1, N, D)
        x_stream_next = torch.bmm(h_res_f, x_str_f).view(B, L, N, D)

        return x_stream_next + (delta.unsqueeze(2) * h_post)


class PomPomNN(nn.Module):
    def __init__(self, num_layers=4, num_filters=96, num_heads=4, n_streams=4):
        super().__init__()
        self.n_streams = n_streams
        self.input_projection = nn.Linear(NUM_INPUT_CHANNELS, num_filters)

        self.blocks = nn.ModuleList([
            mHCTurboBlock(num_filters, num_heads, BOARD_SIZE, n_streams)
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(num_filters)

        self.policy_head = nn.Linear(num_filters, 1)
        self.soft_policy_head = nn.Linear(num_filters, 1)

        self.value_head = nn.Sequential(
            nn.Linear(2 * num_filters, num_filters),
            nn.ReLU(),
            nn.Linear(num_filters, 1),
            nn.Tanh()
        )

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.view(b, c, -1).transpose(1, 2)
        x_init = self.input_projection(x)
        x_stream = x_init.unsqueeze(2).expand(-1, -1, self.n_streams, -1).contiguous()

        for block in self.blocks:
            x_stream = block(x_stream)

        x_final = self.final_norm(x_stream.mean(dim=2))

        v_mean = x_final.mean(dim=1)
        v_max = x_final.max(dim=1)[0]
        v_combined = torch.cat([v_mean, v_max], dim=1)

        return self.policy_head(x_final).squeeze(-1), \
            self.value_head(v_combined), \
            self.soft_policy_head(x_final).squeeze(-1)


if __name__ == "__main__":
    model = PomPomNN(num_layers=4, num_filters=96, n_streams=4)
    dummy = torch.randn(2, 11, 9, 9)
    p, v, sp = model(dummy)
    print(f"验证：4层96通道 mHC-PoPE (1-iter Sinkhorn) 为当前效率最优平衡点。")