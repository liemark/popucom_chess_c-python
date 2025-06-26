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
    """一个标准的残差块，采用前激活结构"""

    def __init__(self, num_filters):
        super(ResidualBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1, bias=False)

        self.bn2 = nn.BatchNorm2d(num_filters)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        residual = x
        out = self.relu1(self.bn1(x))
        out = self.conv1(out)
        out = self.relu2(self.bn2(out))
        out = self.conv2(out)
        out += residual
        return out


class SpatialSelfAttentionBlock(nn.Module):
    """空间自注意力模块，用于捕捉特征图内的长程依赖"""

    def __init__(self, num_filters, board_size, num_heads=8):
        super(SpatialSelfAttentionBlock, self).__init__()
        self.norm = nn.LayerNorm(num_filters)
        self.attention = nn.MultiheadAttention(embed_dim=num_filters, num_heads=num_heads, batch_first=True)
        # 简单的绝对位置编码
        self.pos_embedding = nn.Parameter(torch.randn(1, board_size * board_size, num_filters))

    def forward(self, x):
        B, C, H, W = x.shape
        x_flat = x.flatten(2).transpose(1, 2)  # Shape: (B, H*W, C)

        # 添加位置编码并进行自注意力计算
        x_pos = x_flat + self.pos_embedding
        attn_output, _ = self.attention(self.norm(x_pos), self.norm(x_pos), self.norm(x_pos))

        # 残差连接
        out_flat = x_flat + attn_output

        # 恢复形状
        return out_flat.transpose(1, 2).view(B, C, H, W)


class PomPomNN(nn.Module):
    """泡姆棋的神经网络模型 (高级版)"""

    def __init__(self, num_res_blocks=6, num_filters=96, num_attention_heads=8):
        super(PomPomNN, self).__init__()
        self.num_res_blocks = num_res_blocks

        # 定义实际输入通道数 = 内容通道 + 2个坐标通道
        actual_input_channels = NUM_INPUT_CHANNELS + 2

        # 初始卷积层
        self.initial_conv = nn.Conv2d(actual_input_channels, num_filters, kernel_size=3, padding=1, bias=False)
        self.initial_bn = nn.BatchNorm2d(num_filters)
        self.initial_relu = nn.ReLU(inplace=True)

        # 残差块和自注意力块堆叠
        self.blocks = nn.ModuleList()
        attention_insertion_point = num_res_blocks // 2
        for i in range(num_res_blocks):
            self.blocks.append(ResidualBlock(num_filters))
            if i == attention_insertion_point:
                self.blocks.append(SpatialSelfAttentionBlock(num_filters, BOARD_SIZE, num_attention_heads))

        # 策略头 (Policy Head)
        self.policy_conv = nn.Conv2d(num_filters, 2, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_relu = nn.ReLU(inplace=True)
        self.policy_fc = nn.Linear(2 * BOARD_SIZE * BOARD_SIZE, BOARD_SIZE * BOARD_SIZE)

        # 价值头 (Value Head)
        self.value_conv = nn.Conv2d(num_filters, 1, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_relu = nn.ReLU(inplace=True)
        self.value_fc1 = nn.Linear(BOARD_SIZE * BOARD_SIZE, num_filters)
        self.value_fc2 = nn.Linear(num_filters, 1)

        # 所有权头 (Ownership Head)
        self.ownership_conv = nn.Conv2d(num_filters, 1, kernel_size=1, bias=False)
        self.ownership_bn = nn.BatchNorm2d(1)

    def forward(self, x_content):
        # x_content 的形状: (batch, NUM_INPUT_CHANNELS, H, W)
        B, _, H, W = x_content.shape

        # 动态生成坐标通道并拼接
        x_coords = torch.linspace(-1, 1, W, device=x_content.device).view(1, 1, 1, W).expand(B, 1, H, W)
        y_coords = torch.linspace(-1, 1, H, device=x_content.device).view(1, 1, H, 1).expand(B, 1, H, W)
        x = torch.cat([x_content, x_coords, y_coords], dim=1)

        # 主干网络
        x = self.initial_relu(self.initial_bn(self.initial_conv(x)))
        for block in self.blocks:
            x = block(x)

        # 策略头
        policy = self.policy_relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.view(B, -1)
        policy_logits = self.policy_fc(policy)

        # 价值头
        value = self.value_relu(self.value_bn(self.value_conv(x)))
        value = value.view(B, -1)
        value = F.relu(self.value_fc1(value))
        value_output = torch.tanh(self.value_fc2(value))

        # 所有权头
        ownership = self.ownership_bn(self.ownership_conv(x))
        ownership_output = torch.tanh(ownership).squeeze(1)  # Tanh输出[-1,1], 并移除通道维度

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
