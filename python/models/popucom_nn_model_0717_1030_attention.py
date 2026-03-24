import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# 导入神经网络输入接口相关的常量
try:
    # 尝试从接口文件导入，这是在实际训练环境中的做法
    from popucom_nn_interface import NUM_INPUT_CHANNELS, BOARD_SIZE
except ImportError:
    # 如果导入失败（例如，在独立运行此脚本时），则使用默认值
    print("警告: 无法导入 popucom_nn_interface.py。将使用默认值。")
    NUM_INPUT_CHANNELS = 11  # 棋盘内容特征通道数
    BOARD_SIZE = 9  # 棋盘尺寸


class GlobalPoolingBias(nn.Module):
    """
    实现全局池化偏置结构，用于向空间特征图添加通道维度的偏置。
    该偏置从特征图自身的全局信息（均值和最大值）中学习，类似于Squeeze-and-Excitation机制。
    """

    def __init__(self, num_filters):
        super(GlobalPoolingBias, self).__init__()
        self.bn = nn.BatchNorm2d(num_filters)
        self.relu = nn.ReLU(inplace=True)
        # 拼接均值和最大值池化结果，因此输入维度是 2 * num_filters
        self.fc = nn.Linear(2 * num_filters, num_filters)

    def forward(self, x):
        # 应用批归一化和ReLU激活
        g_features = self.relu(self.bn(x))

        # 计算全局平均池化和全局最大池化
        mean_pooled = F.adaptive_avg_pool2d(g_features, (1, 1)).view(g_features.size(0), -1)
        max_pooled = F.adaptive_max_pool2d(g_features, (1, 1)).view(g_features.size(0), -1)

        # 拼接池化特征
        pooled_features = torch.cat([mean_pooled, max_pooled], dim=1)

        # 通过全连接层获取偏置值，并调整形状以便广播相加
        bias_values = self.fc(pooled_features).unsqueeze(2).unsqueeze(3)

        # 将学习到的偏置加到原始输入上
        return x + bias_values


class MultiHeadSelfAttentionWithRelativeBias(nn.Module):
    """
    一个自定义的多头自注意力模块，它将可学习的相对位置偏置添加到注意力分数中。
    这使得模型能够直接捕捉不同位置之间的相对空间关系，对于棋盘类任务尤其有效。
    """

    def __init__(self, embed_dim, num_heads, board_size, dropout=0.1):
        super(MultiHeadSelfAttentionWithRelativeBias, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.board_size = board_size
        self.dropout = nn.Dropout(dropout)

        if self.head_dim * num_heads != self.embed_dim:
            raise ValueError(f"embed_dim ({embed_dim}) 必须能被 num_heads ({num_heads}) 整除")

        # Query, Key, Value 的投影层
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        # 可学习的相对位置偏置
        # 相对距离范围是 -(BOARD_SIZE - 1) 到 (BOARD_SIZE - 1)，共 2 * BOARD_SIZE - 1 种可能
        self.relative_x_bias_embedding = nn.Parameter(torch.randn(num_heads, 2 * board_size - 1))
        self.relative_y_bias_embedding = nn.Parameter(torch.randn(num_heads, 2 * board_size - 1))

    def forward(self, x):
        # x shape: (batch_size, seq_len, embed_dim)
        batch_size, seq_len, _ = x.shape
        H = W = self.board_size

        # 1. 投影 Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # 2. 分割成多头
        # -> (batch_size, num_heads, seq_len, head_dim)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 3. 计算注意力分数
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # 4. 计算并添加相对位置偏置
        # 生成所有 query-key 对的相对坐标索引
        coords = torch.arange(seq_len, device=x.device)
        coords_x = coords % W
        coords_y = coords // W

        # 广播计算相对距离 -> (seq_len, seq_len)
        relative_x_indices = coords_x.unsqueeze(0) - coords_x.unsqueeze(1)
        relative_y_indices = coords_y.unsqueeze(0) - coords_y.unsqueeze(1)

        # 映射到偏置嵌入的索引 (0 到 2*BOARD_SIZE - 2)
        offset = self.board_size - 1
        relative_x_indices = (relative_x_indices + offset).long()
        relative_y_indices = (relative_y_indices + offset).long()

        # 查找偏置 -> (num_heads, seq_len, seq_len)
        relative_x_bias = self.relative_x_bias_embedding[:, relative_x_indices]
        relative_y_bias = self.relative_y_bias_embedding[:, relative_y_indices]

        # 添加偏置
        attention_scores = attention_scores + (relative_x_bias + relative_y_bias).unsqueeze(0)

        # 5. 应用 softmax 和 dropout
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # 6. 加权求和 Value
        output = torch.matmul(attention_weights, v)

        # 7. 拼接头并投影回原始维度
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        output = self.out_proj(output)

        return output


class SpatialSelfAttentionBlock(nn.Module):
    """
    一个完整的空间自注意力块，类似于Transformer编码器层。
    结构: LayerNorm -> Attention -> Residual -> LayerNorm -> FFN -> Residual
    """

    def __init__(self, num_filters, board_size, num_heads, dropout=0.1):
        super(SpatialSelfAttentionBlock, self).__init__()
        self.norm1 = nn.LayerNorm(num_filters)
        self.attention = MultiHeadSelfAttentionWithRelativeBias(
            embed_dim=num_filters,
            num_heads=num_heads,
            board_size=board_size,
            dropout=dropout
        )
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(num_filters)
        self.ffn = nn.Sequential(
            nn.Linear(num_filters, 4 * num_filters),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(4 * num_filters, num_filters)
        )
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: (batch_size, num_filters, H, W)
        batch_size, channels, H, W = x.shape

        # 存下残差连接的输入
        residual = x

        # 展平为序列: (batch, H*W, channels)
        x_flat = x.view(batch_size, channels, -1).permute(0, 2, 1)

        # 第一个子层: 注意力
        attn_output = self.attention(self.norm1(x_flat))
        # 残差连接 1
        x_flat = x_flat + self.dropout1(attn_output)

        # 第二个子层: FFN
        ffn_output = self.ffn(self.norm2(x_flat))
        # 残差连接 2
        x_flat = x_flat + self.dropout2(ffn_output)

        # 转换回图像形状: (batch, channels, H, W)
        out = x_flat.permute(0, 2, 1).view(batch_size, channels, H, W)

        return out


class PomPomNN(nn.Module):
    """
    泡姆棋的神经网络模型（重构版）。
    - 主干网络: 完全由自注意力块堆叠而成。
    - 输出头: (策略Logits, 价值, 软策略Logits)。
    """
    ACTUAL_INPUT_CHANNELS = NUM_INPUT_CHANNELS + 2

    def __init__(self, num_attention_blocks=4, num_filters=96, num_attention_heads=6):
        super(PomPomNN, self).__init__()
        self.num_attention_blocks = num_attention_blocks
        self.num_filters = num_filters
        self.num_attention_heads = num_attention_heads

        # 初始卷积层，用于将输入转换为合适的特征维度
        self.initial_conv = nn.Conv2d(self.ACTUAL_INPUT_CHANNELS, num_filters, kernel_size=3, padding=1, bias=False)
        self.initial_bn = nn.BatchNorm2d(num_filters)
        self.initial_relu = nn.ReLU(inplace=True)
        self.global_pool_bias_initial = GlobalPoolingBias(num_filters)

        # 主干网络: 堆叠的自注意力块
        self.attention_blocks = nn.ModuleList([
            SpatialSelfAttentionBlock(num_filters, BOARD_SIZE, num_attention_heads)
            for _ in range(num_attention_blocks)
        ])

        # --- 三个输出头 ---

        # 1. 主策略头 (Policy Head)
        self.policy_conv = nn.Conv2d(num_filters, 2, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_relu = nn.ReLU(inplace=True)
        self.global_pool_bias_policy = GlobalPoolingBias(2)
        self.policy_fc = nn.Linear(2 * BOARD_SIZE * BOARD_SIZE, BOARD_SIZE * BOARD_SIZE)

        # 2. 价值头 (Value Head)
        self.value_conv = nn.Conv2d(num_filters, 1, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_relu = nn.ReLU(inplace=True)
        self.global_pool_bias_value = GlobalPoolingBias(1)
        self.value_fc1 = nn.Linear(BOARD_SIZE * BOARD_SIZE, num_filters)
        self.value_fc2 = nn.Linear(num_filters, 1)

        # 3. 软策略头 (Soft Policy Head)
        self.soft_policy_conv = nn.Conv2d(num_filters, 2, kernel_size=1, bias=False)
        self.soft_policy_bn = nn.BatchNorm2d(2)
        self.soft_policy_relu = nn.ReLU(inplace=True)
        self.soft_global_pool_bias_policy = GlobalPoolingBias(2)
        self.soft_policy_fc = nn.Linear(2 * BOARD_SIZE * BOARD_SIZE, BOARD_SIZE * BOARD_SIZE)

    def forward(self, x):
        batch_size, _, H, W = x.shape

        # 动态生成并拼接坐标通道
        x_coords = torch.linspace(-1, 1, W, device=x.device).view(1, 1, 1, W).expand(batch_size, 1, H, W)
        y_coords = torch.linspace(-1, 1, H, device=x.device).view(1, 1, H, 1).expand(batch_size, 1, H, W)
        x_with_coords = torch.cat([x, x_coords, y_coords], dim=1)

        # --- 主干网络前向传播 ---
        # 初始卷积
        trunk = self.initial_relu(self.initial_bn(self.initial_conv(x_with_coords)))
        trunk = self.global_pool_bias_initial(trunk)

        # 堆叠的自注意力块
        for block in self.attention_blocks:
            trunk = block(trunk)

        # --- 各个头的前向传播 ---

        # 1. 主策略头 -> 输出 Logits
        policy = self.policy_relu(self.policy_bn(self.policy_conv(trunk)))
        policy = self.global_pool_bias_policy(policy)
        policy = policy.view(policy.size(0), -1)
        policy_logits = self.policy_fc(policy)

        # 2. 价值头 -> 输出 tanh 激活后的值 [-1, 1]
        value = self.value_relu(self.value_bn(self.value_conv(trunk)))
        value = self.global_pool_bias_value(value)
        value = value.view(value.size(0), -1)
        value = F.relu(self.value_fc1(value))
        value_output = torch.tanh(self.value_fc2(value))

        # 3. 软策略头 -> 输出 Logits
        soft_policy = self.soft_policy_relu(self.soft_policy_bn(self.soft_policy_conv(trunk)))
        soft_policy = self.soft_global_pool_bias_policy(soft_policy)
        soft_policy = soft_policy.view(soft_policy.size(0), -1)
        soft_policy_logits = self.soft_policy_fc(soft_policy)

        return policy_logits, value_output, soft_policy_logits


# --- 示例用法和验证 ---
if __name__ == "__main__":
    # 创建一个虚拟输入张量 (batch_size=2)
    dummy_input = torch.randn(2, NUM_INPUT_CHANNELS, BOARD_SIZE, BOARD_SIZE)

    # 实例化模型
    # 您可以调整注意力块的数量、滤波器数量和注意力头数
    model = PomPomNN(
        num_attention_blocks=4,  # 例如，使用6个注意力块
        num_filters=96,  # 例如，使用96个嵌入维度
        num_attention_heads=6  # 例如，使用8个头
    )

    # 将模型设置为评估模式（影响BatchNorm和Dropout的行为）
    model.eval()

    # 执行前向传播
    with torch.no_grad():
        policy_logits, value, soft_policy_logits = model(dummy_input)

    print("--- 神经网络模型重构完成 ---")
    print(f"模型主干: {model.num_attention_blocks}个 SpatialSelfAttentionBlock 堆叠")
    print(f"模型滤波器/嵌入维度: {model.num_filters}")
    print(f"模型注意力头数: {model.num_attention_heads}")
    print("-" * 30)
    print(f"输入张量形状 (仅内容): {dummy_input.shape}")
    print(f"策略头 Logits 输出形状: {policy_logits.shape}")
    print(f"价值头输出形状: {value.shape}")
    print(f"软策略头 Logits 输出形状: {soft_policy_logits.shape}")
    print("-" * 30)
    print(f"价值头输出示例: {value.squeeze().tolist()}")
    print(f"价值头输出范围: [-1, 1] (通过 Tanh 激活)")

    # 验证价值输出范围
    assert torch.all(value >= -1) and torch.all(value <= 1), "价值输出不在 [-1, 1] 范围内"
    print("\n模型定义、前向传播和输出验证成功。")