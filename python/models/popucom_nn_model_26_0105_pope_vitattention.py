import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# 尝试导入接口常量
try:
    from popucom_nn_interface import NUM_INPUT_CHANNELS, BOARD_SIZE
except ImportError:
    NUM_INPUT_CHANNELS = 11
    BOARD_SIZE = 9


class PoPE2DAttention(nn.Module):
    """
    2D 极坐标位置嵌入注意力 (2D-PoPE)
    实现了论文中提到的 "What" (幅度) 与 "Where" (相位) 的完全解耦。
    """

    def __init__(self, dim, num_heads, board_size, theta_base=10.0, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.board_size = board_size

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

        # 分配 X 和 Y 的频率
        half_hd = self.head_dim // 2
        inv_freq = 1.0 / (theta_base ** (torch.arange(0, half_hd).float() / half_hd))
        self.register_buffer("inv_freq", inv_freq)

        # 可学习的偏置 delta
        self.bias_delta = nn.Parameter(torch.zeros(num_heads, self.head_dim))

    def forward(self, x):
        b, n, d = x.shape
        h = self.num_heads
        hd = self.head_dim
        L = self.board_size

        q = self.q_proj(x).view(b, n, h, hd).transpose(1, 2)
        k = self.k_proj(x).view(b, n, h, hd).transpose(1, 2)
        v = self.v_proj(x).view(b, n, h, hd).transpose(1, 2)

        # 1. 提取幅度 (What): 使用 Softplus 确保内容特征为正
        mu_q = F.softplus(q)
        mu_k = F.softplus(k)

        # 2. 计算位置相位 (Where)
        y_coords, x_coords = torch.meshgrid(
            torch.arange(L, device=x.device),
            torch.arange(L, device=x.device),
            indexing='ij'
        )
        x_flat = x_coords.flatten().float()  # (seq_len,)
        y_flat = y_coords.flatten().float()  # (seq_len,)

        # 构造 2D 相位: 前半部分 X，后半部分 Y
        phi_x = x_flat.unsqueeze(1) * self.inv_freq.unsqueeze(0)
        phi_y = y_flat.unsqueeze(1) * self.inv_freq.unsqueeze(0)
        phi = torch.cat([phi_x, phi_y], dim=-1)  # (seq_len, head_dim)

        # 3. 计算 PoPE 注意力分数
        # 核心逻辑: sum(mu_q * mu_k * cos(phi_k - phi_q + delta))
        q_cos = mu_q * torch.cos(phi.unsqueeze(0).unsqueeze(0))
        q_sin = mu_q * torch.sin(phi.unsqueeze(0).unsqueeze(0))

        k_phi_delta = phi.unsqueeze(0).unsqueeze(0) + self.bias_delta.unsqueeze(1)
        k_cos = mu_k * torch.cos(k_phi_delta)
        k_sin = mu_k * torch.sin(k_phi_delta)

        # 利用复数乘法性质计算相对位置分数
        scores = (torch.matmul(q_cos, k_cos.transpose(-1, -2)) +
                  torch.matmul(q_sin, k_sin.transpose(-1, -2))) / (hd ** 0.5)

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v).transpose(1, 2).reshape(b, n, d)
        return self.out_proj(out)


class TransformerBlock(nn.Module):
    """标准的 Transformer 块 (Pre-Norm 结构)"""

    def __init__(self, dim, num_heads, board_size, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = PoPE2DAttention(dim, num_heads, board_size, dropout=dropout)

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class PomPomNN(nn.Module):
    """
    基于 Vision Transformer (ViT) 架构的棋盘模型
    完全去除了卷积层，使用 2D-PoPE 捕捉棋盘几何关系。
    """

    def __init__(self, num_layers=6, num_filters=96, num_heads=4, dropout=0.1):
        super().__init__()
        self.board_size = BOARD_SIZE
        self.embed_dim = num_filters

        # 初始 Embedding 层 (线性投影将棋盘通道映射到 Embedding 维度)
        # 将 (B, Channels, H, W) 视为长度为 H*W 的序列
        self.input_projection = nn.Linear(NUM_INPUT_CHANNELS, num_filters)

        # Transformer 骨架
        self.blocks = nn.ModuleList([
            TransformerBlock(num_filters, num_heads, BOARD_SIZE, dropout=dropout)
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(num_filters)

        # --- Policy Head ---
        self.policy_head = nn.Sequential(
            nn.Linear(num_filters, num_filters),
            nn.ReLU(),
            nn.Linear(num_filters, 1)  # 每个位置输出一个概率分值
        )

        # --- Soft Policy Head ---
        self.soft_policy_head = nn.Sequential(
            nn.Linear(num_filters, num_filters),
            nn.ReLU(),
            nn.Linear(num_filters, 1)
        )

        # --- Value Head ---
        self.value_head = nn.Sequential(
            nn.Linear(num_filters * BOARD_SIZE * BOARD_SIZE, num_filters),
            nn.ReLU(),
            nn.Linear(num_filters, 1),
            nn.Tanh()
        )

    def forward(self, x):
        # x: (batch, channels, H, W)
        b, c, h, w = x.shape

        # 1. 扁平化并投影 (B, seq_len, Channels)
        x = x.view(b, c, -1).transpose(1, 2)  # (B, 81, Channels)
        x = self.input_projection(x)

        # 2. 经过纯注意力层 (PoPE 自动处理位置)
        for block in self.blocks:
            x = block(x)

        x = self.final_norm(x)  # (B, 81, num_filters)

        # 3. Policy Head: 对每个 Token 输出一个分值
        policy_logits = self.policy_head(x).squeeze(-1)  # (B, 81)

        # 4. Soft Policy Head
        soft_policy_logits = self.soft_policy_head(x).squeeze(-1)  # (B, 81)

        # 5. Value Head: 使用全局特征 (平铺所有 Token)
        v_flat = x.view(b, -1)
        value_output = self.value_head(v_flat)  # (B, 1)

        return policy_logits, value_output, soft_policy_logits


if __name__ == "__main__":
    # 测试输入
    dummy_input = torch.randn(2, NUM_INPUT_CHANNELS, BOARD_SIZE, BOARD_SIZE)
    model = PomPomNN(num_layers=6, num_filters=96, num_heads=4)

    p, v, sp = model(dummy_input)

    print(f"输入形状: {dummy_input.shape}")
    print(f"Policy Logits: {p.shape}")
    print(f"Value Output: {v.shape}")
    print(f"Soft Policy Logits: {sp.shape}")

    # 验证是否能够运行成功
    print("\n[成功] 纯注意力模型初始化并完成前向传播。")
    print(f"模型使用了 {BOARD_SIZE * BOARD_SIZE} 个 Token，每个 Token 维度为 96。")