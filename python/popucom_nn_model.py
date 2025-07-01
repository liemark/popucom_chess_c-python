import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# 导入神经网络输入接口相关的常量
try:
    from popucom_nn_interface import NUM_INPUT_CHANNELS, BOARD_SIZE
except ImportError:
    print("错误: 无法导入 popucom_nn_interface.py。请确保它在同一目录下。")
    NUM_INPUT_CHANNELS = 11  # 默认值，这里的 NUM_INPUT_CHANNELS 指的是棋盘内容通道数
    BOARD_SIZE = 9  # 默认值


class ResidualBlock(nn.Module):
    """
    一个标准的残差块，采用 KataGo 论文中描述的前激活 (Pre-activation) 结构。
    结构为：BN -> ReLU -> Conv -> BN -> ReLU -> Conv，然后是跳跃连接。
    这种前激活设计对于优化深度网络中的梯度传播至关重要，因为恒等映射绕过了非线性，允许梯度直接流动。
    """

    def __init__(self, num_filters):
        super(ResidualBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.relu1 = nn.ReLU(inplace=True)  # inplace=True 节省内存
        self.conv1 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_filters)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        residual = x  # 保存输入用于跳跃连接

        # BN -> ReLU -> Conv
        out = self.relu1(self.bn1(x))
        out = self.conv1(out)

        # BN -> ReLU -> Conv
        out = self.relu2(self.bn2(out))
        out = self.conv2(out)

        out += residual  # 残差连接：将原始输入加到卷积结果上
        return out


class GlobalPoolingBias(nn.Module):
    """
    实现全局池化偏置结构，用于向空间特征图添加通道维度的偏置。
    该偏置从特征图自身的全局信息（均值和最大值）中学习。
    这类似于 Squeeze-and-Excitation 模块中的通道注意力机制。
    此模块内部也包含一个加性跳跃连接 (x + bias_values)，有助于梯度传播。
    """

    def __init__(self, num_filters):
        super(GlobalPoolingBias, self).__init__()
        self.bn = nn.BatchNorm2d(num_filters)
        self.relu = nn.ReLU(inplace=True)
        # 全局池化后，我们将均值和最大值池化结果拼接，因此输入维度是 2 * num_filters
        self.fc = nn.Linear(2 * num_filters, num_filters)  # 全连接层将池化特征映射回原始通道数

    def forward(self, x):
        # x 是输入空间张量，同时作为 KataGo 描述中的 X 和 G（此处简化 G=X）

        # 应用批归一化和 ReLU 激活 (对应 KataGo 中的 G 路径)
        g_features = self.relu(self.bn(x))

        # 计算每个通道的全局平均池化
        mean_pooled = F.adaptive_avg_pool2d(g_features, (1, 1)).view(g_features.size(0), -1)
        # 计算每个通道的全局最大池化
        max_pooled = F.adaptive_max_pool2d(g_features, (1, 1)).view(g_features.size(0), -1)

        # 拼接均值和最大值池化特征
        pooled_features = torch.cat([mean_pooled, max_pooled], dim=1)

        # 通过全连接层获取通道维度的偏置，并调整形状使其能与空间特征图相加
        bias_values = self.fc(pooled_features).unsqueeze(2).unsqueeze(3)  # 形状变为 (batch, channels, 1, 1)

        # 将学习到的偏置加到原始输入空间特征上
        return x + bias_values


class MultiHeadSelfAttentionWithRelativeBias(nn.Module):
    """
    一个自定义的多头自注意力模块，它将可学习的相对位置偏置添加到注意力分数中。
    这使得模型能够直接捕捉不同位置之间的相对空间关系。
    """

    def __init__(self, embed_dim, num_heads, board_size, dropout=0.1):
        super(MultiHeadSelfAttentionWithRelativeBias, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.board_size = board_size
        self.dropout = nn.Dropout(dropout)

        if self.head_dim * num_heads != self.embed_dim:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")

        # Query, Key, Value 投影层
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        # 输出投影层
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        # 可学习的相对位置偏置
        # 相对距离的范围是 -(BOARD_SIZE - 1) 到 (BOARD_SIZE - 1)
        # 共有 2 * BOARD_SIZE - 1 种可能的相对距离
        # 每个头为一个相对距离学习一个偏置
        self.relative_coords_bias_embedding = nn.Parameter(torch.randn(num_heads, 2 * board_size - 1))

    def forward(self, x):
        # x shape: (batch_size, seq_len, embed_dim)
        batch_size, seq_len, embed_dim = x.shape
        H = W = self.board_size  # For board-like inputs, seq_len = H*W

        # 1. 投影 Q, K, V
        # 结果形状: (batch_size, seq_len, embed_dim)
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # 2. 分割头并转置
        # 结果形状: (batch_size, num_heads, seq_len, head_dim)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 3. 计算注意力分数 (Query @ Key.transpose)
        # 结果形状: (batch_size, num_heads, seq_len, seq_len)
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # 4. 计算相对位置偏置并添加到注意力分数
        # 生成所有 query-key 对的相对 X 和 Y 坐标索引
        # query_coords 和 key_coords 形状都是 (seq_len, 2) (row_idx, col_idx)
        query_coords_flat = torch.arange(seq_len, device=x.device).view(H, W)
        query_rows = query_coords_flat // W  # (H, W)
        query_cols = query_coords_flat % W  # (H, W)

        key_rows = query_rows.clone()  # (H, W)
        key_cols = query_cols.clone()  # (H, W)

        # 广播计算所有 query-key 对的相对距离
        # 形状 (seq_len, seq_len)
        relative_x_indices = key_cols.flatten().unsqueeze(0) - query_cols.flatten().unsqueeze(1)
        relative_y_indices = key_rows.flatten().unsqueeze(0) - query_rows.flatten().unsqueeze(1)

        # 将相对距离映射到偏置嵌入的索引 (范围 0 到 2*BOARD_SIZE - 2)
        # -(BOARD_SIZE - 1) 对应索引 0
        # (BOARD_SIZE - 1) 对应索引 2*BOARD_SIZE - 2
        offset = self.board_size - 1
        relative_x_indices = (relative_x_indices + offset).long()  # (seq_len, seq_len)
        relative_y_indices = (relative_y_indices + offset).long()  # (seq_len, seq_len)

        # 查找可学习的相对偏置
        # bias_x_per_head_per_pair 形状: (num_heads, seq_len, seq_len)
        relative_x_bias = self.relative_coords_bias_embedding[:, relative_x_indices]  # (num_heads, seq_len, seq_len)
        relative_y_bias = self.relative_coords_bias_embedding[:, relative_y_indices]  # (num_heads, seq_len, seq_len)

        # 将偏置添加到注意力分数上
        # (batch_size, num_heads, seq_len, seq_len) + (num_heads, seq_len, seq_len) -> 广播到 batch_size
        attention_scores = attention_scores + relative_x_bias.unsqueeze(0) + relative_y_bias.unsqueeze(0)

        # 5. 应用 softmax 和 dropout
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # 6. 加权求和 Value
        # 结果形状: (batch_size, num_heads, seq_len, head_dim)
        output = torch.matmul(attention_weights, v)

        # 7. 拼接头并投影回原始维度
        # 结果形状: (batch_size, seq_len, embed_dim)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        output = self.out_proj(output)

        return output


class SpatialSelfAttentionBlock(nn.Module):
    """
    A spatial self-attention block to capture long-range dependencies within feature maps.
    It transforms the 2D feature map into a sequence, applies multi-head self-attention,
    and then transforms it back to 2D. This full (global) self-attention allows for
    direct interaction between any two positions on the board, which can implicitly
    capture diagonal relationships.
    Includes Layer Normalization and residual connections for stability.
    This version uses a custom MultiHeadSelfAttentionWithRelativeBias to incorporate
    learnable relative positional biases instead of absolute positional embeddings.
    """

    def __init__(self, num_filters, board_size, num_heads=8, dropout=0.1):
        super(SpatialSelfAttentionBlock, self).__init__()
        self.num_filters = num_filters
        self.board_size = board_size
        self.num_heads = num_heads

        # LayerNorm is typically used in Transformer architectures
        self.norm = nn.LayerNorm(num_filters)

        # 使用自定义的 MultiHeadSelfAttentionWithRelativeBias 模块
        self.attention = MultiHeadSelfAttentionWithRelativeBias(
            embed_dim=num_filters,
            num_heads=num_heads,
            board_size=board_size,
            dropout=dropout
        )

        # Feed-forward network (FFN)
        self.fc1 = nn.Linear(num_filters, 4 * num_filters)  # Expand dimension
        self.fc2 = nn.Linear(4 * num_filters, num_filters)  # Reduce dimension
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: (batch_size, num_filters, board_size, board_size)
        batch_size, channels, H, W = x.shape

        # Flatten spatial dimensions and transpose, making it (batch_size, H*W, channels)
        # H*W is sequence_length, channels is embedding_dimension
        x_flat = x.view(batch_size, channels, -1).permute(0, 2, 1)  # (batch_size, H*W, num_filters)

        # Save input for residual connection for the entire block
        residual_block_input = x_flat

        # Apply Layer Normalization before attention
        norm_x_flat = self.norm(x_flat)

        # Self-attention with relative bias
        attn_output = self.attention(norm_x_flat)

        # Residual connection 1: Attention output + original flattened input
        attn_output = self.dropout(attn_output)
        out = residual_block_input + attn_output

        # Feed-forward network
        ffn_input = self.norm(out)  # Apply LayerNorm again
        ffn_output = self.fc2(self.dropout(F.relu(self.fc1(ffn_input))))
        ffn_output = self.dropout(ffn_output)

        # Residual connection 2: FFN output + attention residual output
        out = out + ffn_output

        # Transform back to original 2D shape: (batch_size, num_filters, H, W)
        out = out.permute(0, 2, 1).view(batch_size, channels, H, W)
        return out


class PomPomNN(nn.Module):
    """
    Neural network model for PomPom chess, using convolutional neural networks and residual connections,
    and incorporating global pooling bias.
    Includes a policy head for predicting move probabilities, a value head for predicting win/loss or score,
    and an ownership head for predicting final ownership of board squares.
    """

    # Define the number of content channels (from popucom_nn_interface) and 2 additional coordinate channels
    # The actual input channel count will be content_channels + coord_channels
    ACTUAL_INPUT_CHANNELS = NUM_INPUT_CHANNELS + 2

    def __init__(self, num_res_blocks=6, num_filters=96, num_attention_heads=8):
        """
        Initializes the neural network model.

        Args:
            num_res_blocks (int): Number of stacked residual blocks.
            num_filters (int): Number of filters (channels) used in convolutional layers.
            num_attention_heads (int): Number of heads in the self-attention layer.
        """
        super(PomPomNN, self).__init__()
        self.num_res_blocks = num_res_blocks
        self.num_filters = num_filters
        self.num_attention_heads = num_attention_heads

        # Initial convolutional layer
        # Input channels are now content channels (NUM_INPUT_CHANNELS) + 2 coordinate channels
        self.initial_conv = nn.Conv2d(self.ACTUAL_INPUT_CHANNELS, num_filters, kernel_size=3, padding=1, bias=False)
        self.initial_bn = nn.BatchNorm2d(num_filters)
        self.initial_relu = nn.ReLU(inplace=True)
        # Add Global Pooling Bias after the initial convolutional layer
        self.global_pool_bias_initial = GlobalPoolingBias(num_filters)

        # Stacked Residual Blocks
        self.res_blocks = nn.ModuleList([ResidualBlock(num_filters) for _ in range(num_res_blocks)])

        # New: Spatial Self-Attention Layer (global attention)
        # Now uses relative positional biases to better capture coordinate differences.
        # Placed after some residual blocks. Its placement and number can be adjusted based on empirical results.
        self.spatial_attention = SpatialSelfAttentionBlock(num_filters, BOARD_SIZE, num_heads=self.num_attention_heads)

        # Policy Head
        # Input: Main network output (batch_size, num_filters, BOARD_SIZE, BOARD_SIZE)
        # Output: Probability distribution over possible moves on a 9x9 board (after softmax normalization)
        self.policy_conv = nn.Conv2d(num_filters, 2, kernel_size=1,
                                     bias=False)  # 1x1 convolution, outputs 2 feature maps
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_relu = nn.ReLU(inplace=True)
        # Add Global Pooling Bias after the policy head's convolution output
        self.global_pool_bias_policy = GlobalPoolingBias(2)  # Policy head output channel is 2

        # Flatten to (batch_size, 2 * BOARD_SIZE * BOARD_SIZE)
        # Linear layer maps features to 9x9 = 81 possible move positions
        self.policy_fc = nn.Linear(2 * BOARD_SIZE * BOARD_SIZE, BOARD_SIZE * BOARD_SIZE)

        # soft_policy
        self.soft_policy_conv = nn.Conv2d(num_filters, 2, kernel_size=1,bias=False)
        self.soft_policy_bn = nn.BatchNorm2d(2)
        self.soft_policy_relu = nn.ReLU(inplace=True)
        self.soft_global_pool_bias_policy = GlobalPoolingBias(2)
        self.soft_policy_fc = nn.Linear(2 * BOARD_SIZE * BOARD_SIZE, BOARD_SIZE * BOARD_SIZE)

        # Value Head (now consolidates value and score prediction)
        # Input: Main network output (batch_size, num_filters, BOARD_SIZE, BOARD_SIZE)
        # Output: A scalar value representing the current player's win/loss evaluation or score evaluation (-1 to 1)
        self.value_conv = nn.Conv2d(num_filters, 1, kernel_size=1, bias=False)  # 1x1 convolution, outputs 1 feature map
        self.value_bn = nn.BatchNorm2d(1)
        self.value_relu = nn.ReLU(inplace=True)
        # Add Global Pooling Bias after the value head's convolution output
        self.global_pool_bias_value = GlobalPoolingBias(1)  # Value head output channel is 1

        # Flatten to (batch_size, BOARD_SIZE * BOARD_SIZE)
        self.value_fc1 = nn.Linear(BOARD_SIZE * BOARD_SIZE, num_filters)  # Through a hidden layer
        self.value_fc2 = nn.Linear(num_filters, 1)  # Final output is a scalar

        # Ownership Head
        # Input: Main network output (batch_size, num_filters, BOARD_SIZE, BOARD_SIZE)
        # Output: Tensor of shape (BOARD_SIZE, BOARD_SIZE) representing ownership of each square (-1 to 1)
        self.ownership_conv = nn.Conv2d(num_filters, 1, kernel_size=1,
                                        bias=False)  # 1x1 convolution, outputs 1 feature map
        self.ownership_bn = nn.BatchNorm2d(1)
        # Note: Ownership head typically uses tanh activation directly, or sigmoid with binary cross-entropy.
        # KataGo uses tanh, outputting -1 to 1, so no ReLU here, directly tanh.

    def forward(self, x):
        """
        Forward pass of the neural network.

        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, NUM_INPUT_CHANNELS, BOARD_SIZE, BOARD_SIZE).
                              Here NUM_INPUT_CHANNELS refers to content channels, not including coordinate channels.

        Returns:
            tuple: (policy_output, value_output, ownership_output)
                - policy_output (torch.Tensor): Policy output, shape (batch_size, BOARD_SIZE * BOARD_SIZE),
                                                softmax normalized, range [0, 1].
                - value_output (torch.Tensor): Value output, shape (batch_size, 1)，
                                               Tanh activated, range [-1, 1].
                - ownership_output (torch.Tensor): Ownership output, shape (batch_size, BOARD_SIZE, BOARD_SIZE)，
                                                  Tanh activated, range [-1, 1].
        """
        batch_size, _, H, W = x.shape

        # Generate and normalize X, Y coordinate channels
        # X coordinate channel (row index)
        x_coords = torch.linspace(-1, 1, W, device=x.device).view(1, 1, 1, W).expand(batch_size, 1, H, W)
        # Y coordinate channel (column index)
        y_coords = torch.linspace(-1, 1, H, device=x.device).view(1, 1, H, 1).expand(batch_size, 1, H, W)

        # Concatenate coordinate channels to the original input features
        # The tensor shape before input to the initial convolutional layer will be (batch_size, NUM_INPUT_CHANNELS + 2, BOARD_SIZE, BOARD_SIZE)
        x_with_coords = torch.cat([x, x_coords, y_coords], dim=1)

        # Main network path
        # Initial convolutional layer and its pre-activation
        x = self.initial_relu(self.initial_bn(self.initial_conv(x_with_coords)))
        # Apply initial global pooling bias
        x = self.global_pool_bias_initial(x)

        # Stacked Residual Blocks
        # Place attention block in the middle of residual blocks to balance local and global feature learning
        attention_insertion_point = self.num_res_blocks // 2
        for i, block in enumerate(self.res_blocks):
            x = block(x)
            if i == attention_insertion_point:
                x = self.spatial_attention(x)

        # Policy Head
        # Policy head's convolutional layer and its pre-activation
        policy = self.policy_relu(self.policy_bn(self.policy_conv(x)))
        # Apply policy head global pooling bias
        policy = self.global_pool_bias_policy(policy)
        policy = policy.view(policy.size(0), -1)  # Flatten (batch_size, 2 * 9 * 9)
        policy_logits = self.policy_fc(policy)  # Linear layer output (batch_size, 81)

        # Value Head (now consolidates value and score prediction)
        # Value head's convolutional layer and its pre-activation
        value = self.value_relu(self.value_bn(self.value_conv(x)))
        # Apply value head global pooling bias
        value = self.global_pool_bias_value(value)
        value = value.view(value.size(0), -1)  # Flatten (batch_size, 9 * 9)
        value = F.relu(self.value_fc1(value))  # Hidden layer
        value_output = torch.tanh(self.value_fc2(value))  # Tanh activation, output range [-1, 1]

        # Ownership Head
        ownership = self.ownership_conv(x)  # 1x1 convolution
        ownership = self.ownership_bn(ownership)  # BatchNorm
        ownership_output = torch.tanh(ownership).squeeze(1)
        # Tanh activation, range [-1, 1], and remove singleton channel dimension, shape becomes (batch, BOARD_SIZE, BOARD_SIZE)

        soft_policy = self.soft_policy_relu(self.soft_policy_bn(self.soft_policy_conv(x)))
        soft_policy = self.soft_global_pool_bias_policy(soft_policy)
        soft_policy = soft_policy.view(soft_policy.size(0), -1)  # Flatten (batch_size, 2 * 9 * 9)
        soft_policy_logits = self.soft_policy_fc(soft_policy)  # Linear layer output (batch_size, 81)

        return policy_logits, value_output, ownership_output, soft_policy_logits


# Example usage (in actual training, DataLoader and optimizer are needed)
if __name__ == "__main__":
    # Create a dummy input tensor
    # Assuming batch_size = 1
    # Here dummy_input shape only contains content channels, coordinate channels will be dynamically generated in forward
    dummy_input = torch.randn(1, NUM_INPUT_CHANNELS, BOARD_SIZE, BOARD_SIZE)

    # Instantiate the model
    # You can adjust num_res_blocks and num_filters as needed
    model = PomPomNN(num_res_blocks=6, num_filters=96, num_attention_heads=8)

    # Set the model to evaluation mode (affects BatchNorm and Dropout behavior)
    model.eval()

    # Perform forward pass
    with torch.no_grad():  # Disable gradient calculation during inference
        policy, value, ownership = model(dummy_input)  # Receive three outputs

    print(f"Simulated input tensor shape (content only): {dummy_input.shape}")
    print(f"Policy output shape: {policy.shape}")
    print(f"Policy output example (first 5 values): {policy[0, :5]}")  # Print first 5 values
    print(f"Policy output sum: {torch.sum(policy)}")  # Sum of policy output should be close to 1

    print(f"\nValue output shape: {value.shape}")
    print(f"Value output example: {value.item()}")

    print(f"\nOwnership output shape: {ownership.shape}")
    print(f"Ownership output example (top-left portion):\n{ownership[0, :3, :3]}")

    # Validate output ranges
    assert torch.all(policy >= 0) and torch.all(policy <= 1), "Policy output not in [0, 1] range"
    assert torch.all(value >= -1) and torch.all(value <= 1), "Value output not in [-1, 1] range"
    assert torch.all(ownership >= -1) and torch.all(ownership <= 1), "Ownership output not in [-1, 1] range"

    print("\n神经网络模型已定义，并验证了输入输出形状和范围。")
    print("已成功将全局自注意力层放置在残差块堆叠的中间位置，并保留坐标通道作为输入。")
    print("SpatialSelfAttentionBlock 现在使用自定义实现，包含了可学习的相对位置偏置。")
