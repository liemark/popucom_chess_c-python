import torch
import torch.nn as nn
import torch.nn.functional as F

# 导入神经网络输入接口相关的常量
from popucom_nn_interface import NUM_INPUT_CHANNELS, BOARD_SIZE


# --- U-Net 核心组件 ---

class DoubleConv(nn.Module):
    """(卷积 => BN => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """下采样模块：最大池化 + DoubleConv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """上采样模块：转置卷积 + 跳跃连接融合 + DoubleConv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 使用转置卷积进行上采样
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        # x1 是来自解码器上一步的特征图
        # x2 是来自编码器对应层的跳跃连接特征图
        x1 = self.up(x1)

        # 对齐x1和x2的空间维度
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        # 沿通道维度拼接
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


# --- 完整的U-Net模型 ---

class PomPomNN(nn.Module):
    """
    使用U-Net作为主干网络的泡姆棋模型。
    """
    # 额外增加2个坐标通道
    ACTUAL_INPUT_CHANNELS = NUM_INPUT_CHANNELS + 2

    def __init__(self, num_filters_base=64):
        super(PomPomNN, self).__init__()

        # --- U-Net 主干网络 ---
        # 编码器 (下采样路径)
        self.inc = DoubleConv(self.ACTUAL_INPUT_CHANNELS, num_filters_base)
        self.down1 = Down(num_filters_base, num_filters_base * 2)
        # 经过 down1 后，9x9 -> 4x4 (近似)
        self.down2 = Down(num_filters_base * 2, num_filters_base * 4)
        # 经过 down2 后，4x4 -> 2x2 (近似)

        # 解码器 (上采样路径)
        self.up1 = Up(num_filters_base * 4, num_filters_base * 2)
        self.up2 = Up(num_filters_base * 2, num_filters_base)

        # --- 输出头 (与你之前的模型类似) ---
        # 策略头
        self.policy_head = nn.Sequential(
            nn.Conv2d(num_filters_base, 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True)
        )
        self.policy_fc = nn.Linear(2 * BOARD_SIZE * BOARD_SIZE, BOARD_SIZE * BOARD_SIZE)

        # 价值头
        self.value_head_conv = nn.Sequential(
            nn.Conv2d(num_filters_base, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True)
        )
        self.value_fc = nn.Sequential(
            nn.Linear(1 * BOARD_SIZE * BOARD_SIZE, num_filters_base),
            nn.ReLU(inplace=True),
            nn.Linear(num_filters_base, 1),
            nn.Tanh()
        )

        # 软策略头 (结构与主策略头相同)
        self.soft_policy_head = nn.Sequential(
            nn.Conv2d(num_filters_base, 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True)
        )
        self.soft_policy_fc = nn.Linear(2 * BOARD_SIZE * BOARD_SIZE, BOARD_SIZE * BOARD_SIZE)

    def forward(self, x):
        # 添加坐标通道
        batch_size, _, H, W = x.shape
        x_coords = torch.linspace(-1, 1, W, device=x.device).view(1, 1, 1, W).expand(batch_size, 1, H, W)
        y_coords = torch.linspace(-1, 1, H, device=x.device).view(1, 1, H, 1).expand(batch_size, 1, H, W)
        x_with_coords = torch.cat([x, x_coords, y_coords], dim=1)

        # --- U-Net 前向传播 ---
        # 编码器
        x1 = self.inc(x_with_coords)  # 9x9
        x2 = self.down1(x1)  # 4x4
        x3 = self.down2(x2)  # 2x2

        # 解码器 (使用跳跃连接)
        x = self.up1(x3, x2)  # 4x4
        x = self.up2(x, x1)  # 9x9

        # --- 输出头计算 ---
        # 主策略头
        policy = self.policy_head(x)
        policy = policy.view(policy.size(0), -1)
        policy_logits = self.policy_fc(policy)

        # 价值头
        value = self.value_head_conv(x)
        value = value.view(value.size(0), -1)
        value_output = self.value_fc(value)

        # 软策略头
        soft_policy = self.soft_policy_head(x)
        soft_policy = soft_policy.view(soft_policy.size(0), -1)
        soft_policy_logits = self.soft_policy_fc(soft_policy)

        return policy_logits, value_output, soft_policy_logits


# --- 示例用法 ---
if __name__ == "__main__":
    # 创建一个虚拟输入
    dummy_input = torch.randn(2, NUM_INPUT_CHANNELS, BOARD_SIZE, BOARD_SIZE)

    # 实例化U-Net模型
    model = PomPomNN()
    model.eval()

    print("U-Net 架构的神经网络模型已定义。")
    print(f"输入张量形状: {dummy_input.shape}")

    with torch.no_grad():
        policy_logits, value, soft_policy_logits = model(dummy_input)

    print(f"策略头输出 (Logits) 形状: {policy_logits.shape}")
    print(f"价值头输出形状: {value.shape}")
    print(f"软策略头输出 (Logits) 形状: {soft_policy_logits.shape}")
