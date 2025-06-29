import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from popucom_nn_interface import NUM_INPUT_CHANNELS, BOARD_SIZE
except ImportError:
    NUM_INPUT_CHANNELS = 11
    BOARD_SIZE = 9


class ResidualBlock(nn.Module):
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
    # Model now has FIVE heads: policy, value, ownership, soft_policy, and value_uncertainty
    ACTUAL_INPUT_CHANNELS = NUM_INPUT_CHANNELS + 2

    def __init__(self, num_res_blocks=6, num_filters=96):
        super(PomPomNN, self).__init__()
        self.num_res_blocks, self.num_filters = num_res_blocks, num_filters

        # Main Body
        self.initial_conv = nn.Conv2d(self.ACTUAL_INPUT_CHANNELS, num_filters, kernel_size=3, padding=1, bias=False)
        self.initial_bn = nn.BatchNorm2d(num_filters)
        self.initial_relu = nn.ReLU(inplace=True)
        self.global_pool_bias_initial = GlobalPoolingBias(num_filters)
        self.res_blocks = nn.ModuleList([ResidualBlock(num_filters) for _ in range(num_res_blocks)])

        # Policy Head
        self.policy_conv = nn.Conv2d(num_filters, 2, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_relu = nn.ReLU(inplace=True)
        self.global_pool_bias_policy = GlobalPoolingBias(2)
        self.policy_fc = nn.Linear(2 * BOARD_SIZE * BOARD_SIZE, BOARD_SIZE * BOARD_SIZE)

        # Value Head
        self.value_conv = nn.Conv2d(num_filters, 1, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_relu = nn.ReLU(inplace=True)
        self.global_pool_bias_value = GlobalPoolingBias(1)
        self.value_fc1 = nn.Linear(BOARD_SIZE * BOARD_SIZE, num_filters)
        self.value_fc2 = nn.Linear(num_filters, 1)

        # Ownership Head
        self.ownership_conv = nn.Conv2d(num_filters, 1, kernel_size=1, bias=False)
        self.ownership_bn = nn.BatchNorm2d(1)

        # Soft Policy Head
        self.soft_policy_conv = nn.Conv2d(num_filters, 2, kernel_size=1, bias=False)
        self.soft_policy_bn = nn.BatchNorm2d(2)
        self.soft_policy_relu = nn.ReLU(inplace=True)
        self.soft_global_pool_bias_policy = GlobalPoolingBias(2)
        self.soft_policy_fc = nn.Linear(2 * BOARD_SIZE * BOARD_SIZE, BOARD_SIZE * BOARD_SIZE)

        # --- NEW: Value Uncertainty Head ---
        # It has a similar structure to the value head, but predicts a non-negative value.
        self.uncertainty_conv = nn.Conv2d(num_filters, 1, kernel_size=1, bias=False)
        self.uncertainty_bn = nn.BatchNorm2d(1)
        self.uncertainty_relu = nn.ReLU(inplace=True)
        self.uncertainty_fc1 = nn.Linear(BOARD_SIZE * BOARD_SIZE, num_filters // 2)
        self.uncertainty_fc2 = nn.Linear(num_filters // 2, 1)

    def forward(self, x):
        batch_size, _, H, W = x.shape
        x_coords = torch.linspace(-1, 1, W, device=x.device).view(1, 1, 1, W).expand(batch_size, 1, H, W)
        y_coords = torch.linspace(-1, 1, H, device=x.device).view(1, 1, H, 1).expand(batch_size, 1, H, W)
        x_with_coords = torch.cat([x, x_coords, y_coords], dim=1)

        # Main Body
        x = self.initial_relu(self.initial_bn(self.initial_conv(x_with_coords)))
        x = self.global_pool_bias_initial(x)
        for block in self.res_blocks:
            x = block(x)

        # Policy Head
        policy = self.policy_relu(self.policy_bn(self.policy_conv(x)))
        policy = self.global_pool_bias_policy(policy)
        policy_logits = self.policy_fc(policy.view(batch_size, -1))

        # Value Head
        value_features = self.value_relu(self.value_bn(self.value_conv(x)))
        value_features = self.global_pool_bias_value(value_features)
        value = F.relu(self.value_fc1(value_features.view(batch_size, -1)))
        value_output = torch.tanh(self.value_fc2(value))

        # Ownership Head
        ownership = self.ownership_bn(self.ownership_conv(x))
        ownership_output = torch.tanh(ownership).squeeze(1)

        # Soft Policy Head
        soft_policy = self.soft_policy_relu(self.soft_policy_bn(self.soft_policy_conv(x)))
        soft_policy = self.soft_global_pool_bias_policy(soft_policy)
        soft_policy_logits = self.soft_policy_fc(soft_policy.view(batch_size, -1))

        # --- NEW: Uncertainty Head Forward Pass ---
        uncertainty_features = self.uncertainty_relu(self.uncertainty_bn(self.uncertainty_conv(x)))
        uncertainty = F.relu(self.uncertainty_fc1(uncertainty_features.view(batch_size, -1)))
        # Use softplus to ensure the output is always non-negative, representing squared error.
        uncertainty_output = F.softplus(self.uncertainty_fc2(uncertainty))

        return policy_logits, value_output, ownership_output, soft_policy_logits, uncertainty_output
