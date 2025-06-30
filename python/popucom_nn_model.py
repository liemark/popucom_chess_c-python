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
    """

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


class GlobalPoolingBias(nn.Module):
    """
    实现全局池化偏置结构。
    """

    def __init__(self, num_filters):
        super(GlobalPoolingBias, self).__init__()
        self.bn = nn.BatchNorm2d(num_filters)
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(2 * num_filters, num_filters)

    def forward(self, x):
        g_features = self.relu(self.bn(x))
        mean_pooled = F.adaptive_avg_pool2d(g_features, (1, 1)).view(g_features.size(0), -1)
        max_pooled = F.adaptive_max_pool2d(g_features, (1, 1)).view(g_features.size(0), -1)
        pooled_features = torch.cat([mean_pooled, max_pooled], dim=1)
        bias_values = self.fc(pooled_features).unsqueeze(2).unsqueeze(3)
        return x + bias_values


class PomPomNN(nn.Module):
    """
    泡姆棋的神经网络模型，现在包含了四个头：策略、价值、所有权和辅助软策略。
    """
    ACTUAL_INPUT_CHANNELS = NUM_INPUT_CHANNELS + 2

    def __init__(self, num_res_blocks=6, num_filters=96):
        super(PomPomNN, self).__init__()
        self.num_res_blocks = num_res_blocks
        self.num_filters = num_filters

        # 主干网络
        self.initial_conv = nn.Conv2d(self.ACTUAL_INPUT_CHANNELS, num_filters, kernel_size=3, padding=1, bias=False)
        self.initial_bn = nn.BatchNorm2d(num_filters)
        self.initial_relu = nn.ReLU(inplace=True)
        self.global_pool_bias_initial = GlobalPoolingBias(num_filters)
        self.res_blocks = nn.ModuleList([ResidualBlock(num_filters) for _ in range(num_res_blocks)])

        # 主策略头
        self.policy_conv = nn.Conv2d(num_filters, 2, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_relu = nn.ReLU(inplace=True)
        self.global_pool_bias_policy = GlobalPoolingBias(2)
        self.policy_fc = nn.Linear(2 * BOARD_SIZE * BOARD_SIZE, BOARD_SIZE * BOARD_SIZE)

        # 价值头
        self.value_conv = nn.Conv2d(num_filters, 1, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_relu = nn.ReLU(inplace=True)
        self.global_pool_bias_value = GlobalPoolingBias(1)
        self.value_fc1 = nn.Linear(BOARD_SIZE * BOARD_SIZE, num_filters)
        self.value_fc2 = nn.Linear(num_filters, 1)

        # 所有权头
        self.ownership_conv = nn.Conv2d(num_filters, 1, kernel_size=1, bias=False)
        self.ownership_bn = nn.BatchNorm2d(1)

        # 软策略头
        # 它的结构与主策略头完全相同，但有自己独立的权重。
        self.soft_policy_conv = nn.Conv2d(num_filters, 2, kernel_size=1, bias=False)
        self.soft_policy_bn = nn.BatchNorm2d(2)
        self.soft_policy_relu = nn.ReLU(inplace=True)
        self.soft_global_pool_bias_policy = GlobalPoolingBias(2)
        self.soft_policy_fc = nn.Linear(2 * BOARD_SIZE * BOARD_SIZE, BOARD_SIZE * BOARD_SIZE)

    def forward(self, x):
        batch_size, _, H, W = x.shape
        x_coords = torch.linspace(-1, 1, W, device=x.device).view(1, 1, 1, W).expand(batch_size, 1, H, W)
        y_coords = torch.linspace(-1, 1, H, device=x.device).view(1, 1, H, 1).expand(batch_size, 1, H, W)
        x_with_coords = torch.cat([x, x_coords, y_coords], dim=1)

        # 主干网络
        x = self.initial_relu(self.initial_bn(self.initial_conv(x_with_coords)))
        x = self.global_pool_bias_initial(x)
        for block in self.res_blocks:
            x = block(x)

        # --- 各个头的计算 ---

        # 主策略头
        policy = self.policy_relu(self.policy_bn(self.policy_conv(x)))
        policy = self.global_pool_bias_policy(policy)
        policy = policy.view(policy.size(0), -1)
        policy_logits = self.policy_fc(policy)

        # 价值头
        value = self.value_relu(self.value_bn(self.value_conv(x)))
        value = self.global_pool_bias_value(value)
        value = value.view(value.size(0), -1)
        value = F.relu(self.value_fc1(value))
        value_output = torch.tanh(self.value_fc2(value))

        # 所有权头
        ownership = self.ownership_conv(x)
        ownership = self.ownership_bn(ownership)
        ownership_output = torch.tanh(ownership).squeeze(1)

        # 软策略头的前向传播
        soft_policy = self.soft_policy_relu(self.soft_policy_bn(self.soft_policy_conv(x)))
        soft_policy = self.soft_global_pool_bias_policy(soft_policy)
        soft_policy = soft_policy.view(soft_policy.size(0), -1)
        soft_policy_logits = self.soft_policy_fc(soft_policy)

        # 返回所有四个头的输出
        return policy_logits, value_output, ownership_output, soft_policy_logits


# --- 示例用法 ---
if __name__ == "__main__":
    dummy_input = torch.randn(2, NUM_INPUT_CHANNELS, BOARD_SIZE, BOARD_SIZE)

    model = PomPomNN()
    model.eval()

    with torch.no_grad():
        policy_logits, value, ownership, soft_policy_logits = model(dummy_input)

    print("神经网络模型已定义。")
    print(f"输入张量形状 (仅内容通道): {dummy_input.shape}")
    print(f"策略头输出 (Logits) 形状: {policy_logits.shape}")
    print(f"价值头输出形状: {value.shape}")
    print(f"所有权头输出形状: {ownership.shape}")
    print(f"软策略头输出 (Logits) 形状: {soft_policy_logits.shape}")
