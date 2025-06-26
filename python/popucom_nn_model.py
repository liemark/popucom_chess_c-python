# popucom_nn_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# 导入神经网络输入接口相关的常量
try:
    from popucom_nn_interface import NUM_INPUT_CHANNELS, BOARD_SIZE
except ImportError:
    print("错误: 无法导入 popucom_nn_interface.py。请确保它在同一目录下。")
    # Provide default values as a fallback for standalone execution
    NUM_INPUT_CHANNELS = 11
    BOARD_SIZE = 9


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
        for i, block in enumerate(self.res_blocks):
            x = block(x)

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

        return policy_logits, value_output, ownership_output


# --- 示例用法 ---
if __name__ == "__main__":
    dummy_input = torch.randn(2, NUM_INPUT_CHANNELS, BOARD_SIZE, BOARD_SIZE)  # batch_size=2

    model = PomPomNN(num_res_blocks=6, num_filters=96, num_attention_heads=8)
    model.eval()

    with torch.no_grad():
        policy_logits, value, ownership = model(dummy_input)

    print("高级版神经网络模型已定义。")
    print(f"输入张量形状 (仅内容通道): {dummy_input.shape}")
    print(f"策略头输出 (Logits) 形状: {policy_logits.shape}")
    print(f"价值头输出形状: {value.shape}")
    print(f"所有权头输出形状: {ownership.shape}")
    print(f"价值头输出示例: {value.squeeze().tolist()}")
    print(f"所有权头输出范围: [-1, 1]")
