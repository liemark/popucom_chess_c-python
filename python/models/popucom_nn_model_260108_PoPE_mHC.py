import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# 棋盘基础配置
NUM_INPUT_CHANNELS = 11
BOARD_SIZE = 9
# 位置编码基数：8 / (pi/2) ≈ 5.093
# 取 10 效果较好
THETA_BASE = 10


@torch.jit.script
def fast_sinkhorn_v1(W: torch.Tensor):
    """
    极致性能的单次 Sinkhorn 变换。
    用于流形约束，为 MCTS 提供稳定的先验分布。
    """
    # 减去最大值防止溢出
    W = torch.exp(W - W.max(dim=-1, keepdim=True)[0])
    # 行列归一化即可达到较好的投影效果，直接写死三次
    W = W / (W.sum(dim=-1, keepdim=True) + 1e-6)
    W = W / (W.sum(dim=-2, keepdim=True) + 1e-6)
    W = W / (W.sum(dim=-1, keepdim=True) + 1e-6)
    W = W / (W.sum(dim=-2, keepdim=True) + 1e-6)
    W = W / (W.sum(dim=-1, keepdim=True) + 1e-6)
    W = W / (W.sum(dim=-2, keepdim=True) + 1e-6)
    return W

class PoPEPositionalManager(nn.Module):
    """
    统一管理 2D 棋盘的 PoPE 位置相位。
    """

    def __init__(self, dim, board_size, theta_base=THETA_BASE):
        super().__init__()
        half_hd = dim // 2
        # 生成频率序列
        inv_freq = 1.0 / (theta_base ** (torch.arange(0, half_hd).float() / (half_hd - 1 if half_hd > 1 else 1)))

        # 生成 2D 坐标网格 (9, 9)
        coords = torch.arange(board_size).float()
        y_c, x_c = torch.meshgrid(coords, coords, indexing='ij')

        # 计算 X 和 Y 方向的相位 (81, half_hd)
        phi_x = x_c.reshape(-1, 1) * inv_freq.unsqueeze(0)
        phi_y = y_c.reshape(-1, 1) * inv_freq.unsqueeze(0)

        # 拼接为完整的 2D 相位 (81, dim)
        phi = torch.cat([phi_x, phi_y], dim=-1)
        self.register_buffer("phi", phi)

    def get_phi(self):
        return self.phi


class ValueAttentionPooling(nn.Module):
    """
    PoPE 全局价值池化：利用可学习的 Query 和位置解耦的相位聚合棋盘特征。
    """

    def __init__(self, dim, num_heads, board_size=BOARD_SIZE):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # 全局查询向量 (Query) - 注意：PoPE 中 Q 被视为位置 0
        self.query = nn.Parameter(torch.randn(1, 1, dim) * 0.02)

        # 映射层
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

        # 位置管理
        self.pos_manager = PoPEPositionalManager(self.head_dim, board_size)

        # 关键：PoPE 的可学习相位偏置 delta
        self.bias_delta = nn.Parameter(torch.zeros(num_heads, self.head_dim))

    def forward(self, x):
        # x: (B, 81, D)
        B, L, D = x.shape
        h, hd = self.num_heads, self.head_dim

        # 1. 准备 Q (位置 0), K (带位置), V
        q = self.query.expand(B, -1, -1).view(B, 1, h, hd).transpose(1, 2)
        k = self.k_proj(x).view(B, L, h, hd).transpose(1, 2)
        v = self.v_proj(x).view(B, L, h, hd).transpose(1, 2)

        # 2. 计算 PoPE 模长 (What)
        mu_q = F.softplus(q)  # (B, H, 1, HD)
        mu_k = F.softplus(k)  # (B, H, 81, HD)

        # 3. 计算 PoPE 相位 (Where)
        # K 的相位 = 原始位置 phi + 学习偏置 delta
        # Q 的相位固定为 0，简化 cos(0 - phi_k) -> cos(phi_k)
        phi_k = self.pos_manager.get_phi().unsqueeze(0) + self.bias_delta.unsqueeze(1)  # (H, 81, HD)

        # 4. 计算得分: sum(mu_q * mu_k * cos(phi_k))
        # 因为 Q 相位为 0，其 sin 分量为 0，计算简化
        k_cos = torch.cos(phi_k)
        # 这里的相乘在 HD 维度求和
        scores = torch.matmul(mu_q * 1.0, (mu_k * k_cos).transpose(-1, -2)) * self.scale

        attn = F.softmax(scores, dim=-1)

        # 5. 加权聚合
        out = (attn @ v).transpose(1, 2).reshape(B, 1, D)
        return self.out_proj(out).squeeze(1)


class PoPE2DAttention(nn.Module):
    """
    针对棋盘自注意力的 PoPE 实现，支持完整的相对位置感应。
    """

    def __init__(self, dim, num_heads, board_size, theta_base=THETA_BASE):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv_proj = nn.Linear(dim, dim * 3, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

        # 位置相位
        self.pos_manager = PoPEPositionalManager(self.head_dim, board_size, theta_base)

        # 每一个通道独立的偏置
        self.bias_delta = nn.Parameter(torch.zeros(num_heads, self.head_dim))

    def forward(self, x):
        b, n, d = x.shape
        h, hd = self.num_heads, self.head_dim

        qkv = self.qkv_proj(x).view(b, n, 3, h, hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # 1. 模长 (What)
        mu_q, mu_k = F.softplus(q), F.softplus(k)

        # 2. 相位 (Where)
        phi = self.pos_manager.get_phi()  # (81, HD)
        # 增加偏置项
        phi_k = phi.unsqueeze(0) + self.bias_delta.unsqueeze(1)  # (H, 81, HD)

        # 3. 旋转映射
        q_cos, q_sin = mu_q * torch.cos(phi), mu_q * torch.sin(phi)
        k_cos, k_sin = mu_k * torch.cos(phi_k), mu_k * torch.sin(phi_k)

        # 4. 这里的 Dot Product 等效于 cos(phi_q - (phi_k + delta))
        scores = (torch.matmul(q_cos, k_cos.transpose(-1, -2)) +
                  torch.matmul(q_sin, k_sin.transpose(-1, -2))) * self.scale

        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).reshape(b, n, d)
        return self.out_proj(out)


class mHCTurboBlock(nn.Module):
    """
    极致优化的 Turbo Block，集成流式处理和 PoPE 注意力。
    """

    def __init__(self, dim, num_heads, board_size, n_streams=4):
        super().__init__()
        self.n = n_streams
        self.mhc_proj = nn.Linear(dim, n_streams * 2 + n_streams * n_streams)
        self.alpha = nn.Parameter(torch.full((3,), 0.01))
        self.layer_scale = nn.Parameter(torch.full((dim,), 1e-5))

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

        # 生成流控制系数
        coeffs = self.mhc_proj(x_avg)
        h_pre = torch.sigmoid(coeffs[:, :, :N] * self.alpha[0]).unsqueeze(-1)
        h_post = (torch.sigmoid(coeffs[:, :, N:2 * N] * self.alpha[1]) * 2.0).unsqueeze(-1)
        h_res_raw = coeffs[:, :, 2 * N:].view(B, L, N, N) * self.alpha[2]

        # 使用 Sinkhorn 保持流的稳定性
        h_res = fast_sinkhorn_v1(h_res_raw)

        # 聚合输入进行主路径计算
        layer_input = (x_stream * h_pre).sum(dim=2)

        # 注意力 + MLP
        delta = self.attn(self.norm1(layer_input))
        delta = delta + self.mlp(self.norm2(layer_input + delta))
        delta = delta * self.layer_scale

        # 流分发
        h_res_f = h_res.view(-1, N, N)
        x_str_f = x_stream.view(-1, N, D)
        x_stream_next = torch.bmm(h_res_f, x_str_f).view(B, L, N, D)

        return x_stream_next + (delta.unsqueeze(2) * h_post)


class PomPomNN(nn.Module):
    """
    集成 PoPE 与 ValueAttentionPooling 的高性能棋类神经网络。
    """

    def __init__(self, num_layers=4, num_filters=96, num_heads=4, n_streams=4):
        super().__init__()
        self.n_streams = n_streams
        self.input_projection = nn.Linear(NUM_INPUT_CHANNELS, num_filters)

        self.blocks = nn.ModuleList([
            mHCTurboBlock(num_filters, num_heads, BOARD_SIZE, n_streams)
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(num_filters)

        # 策略头
        self.policy_head = nn.Linear(num_filters, 1)
        self.soft_policy_head = nn.Linear(num_filters, 1)

        # 价值聚合：彻底 PoPE 化，使用针对 9x9 优化的 theta_base
        self.value_pooling = ValueAttentionPooling(num_filters, num_heads=num_heads, board_size=BOARD_SIZE)
        self.value_head = nn.Linear(num_filters, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        # 1. 输入投影与流初始化
        x = x.view(b, c, -1).transpose(1, 2)
        x_init = self.input_projection(x)
        x_stream = x_init.unsqueeze(2).expand(-1, -1, self.n_streams, -1).contiguous()

        # 2. 骨干 Turbo Blocks
        for block in self.blocks:
            x_stream = block(x_stream)

        # 3. 最终特征聚合 (B, 81, D)
        x_tokens = self.final_norm(x_stream.mean(dim=2))

        # 4. 策略输出 (B, 81)
        pol = self.policy_head(x_tokens).squeeze(-1)
        soft_pol = self.soft_policy_head(x_tokens).squeeze(-1)

        # 5. 价值输出 (B, 1)
        # 通过 PoPE 注意力池化捕获全局盘面语义
        v_aggregated = self.value_pooling(x_tokens)
        val = torch.tanh(self.value_head(v_aggregated))

        return pol, val, soft_pol


if __name__ == "__main__":
    # 实例化测试
    model = PomPomNN(num_layers=4, num_filters=96, n_streams=4)
    dummy = torch.randn(2, 11, 9, 9)
    p, v, sp = model(dummy)
    print(f"--- PomPomNN PoPE Edition ---")
    print(f"棋盘尺寸: {BOARD_SIZE}x{BOARD_SIZE}")
    print(f"PoPE 基数 (Theta Base): {THETA_BASE}")
    print(f"价值头聚合方式: PoPE-Pooling (Global Q + Learnable Delta)")
    print(f"输出形状: Policy={p.shape}, Value={v.shape}")