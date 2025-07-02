# popucom_nn_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# 导入神经网络输入接口相关的常量
try:
    from popucom_nn_interface import NUM_INPUT_CHANNELS, BOARD_SIZE
except ImportError:
    print("Warning: Could not import from popucom_nn_interface.py. Using default values.")
    NUM_INPUT_CHANNELS = 11  # Default value for content channels
    BOARD_SIZE = 9          # Default board size


class ResidualBlock(nn.Module):
    """
    A standard residual block using the pre-activation design described in the KataGo paper.
    Structure: BN -> ReLU -> Conv -> BN -> ReLU -> Conv, followed by a skip connection.
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
    Implements the global pooling bias structure to add channel-wise bias to spatial feature maps,
    learned from the global information (mean and max) of the feature maps themselves.
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


# Note: The self-attention block definitions are omitted for brevity as they are unchanged.
# You can copy them from your original file.
class MultiHeadSelfAttentionWithRelativeBias(nn.Module):
    """
    (This class is unchanged from your provided code)
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
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.relative_coords_bias_embedding = nn.Parameter(torch.randn(num_heads, 2 * board_size - 1))

    def forward(self, x):
        batch_size, seq_len, embed_dim = x.shape
        H = W = self.board_size
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        query_coords_flat = torch.arange(seq_len, device=x.device).view(H, W)
        query_rows, query_cols = query_coords_flat // W, query_coords_flat % W
        key_rows, key_cols = query_rows.clone(), query_cols.clone()
        relative_x_indices = key_cols.flatten().unsqueeze(0) - query_cols.flatten().unsqueeze(1)
        relative_y_indices = key_rows.flatten().unsqueeze(0) - query_rows.flatten().unsqueeze(1)
        offset = self.board_size - 1
        relative_x_indices = (relative_x_indices + offset).long()
        relative_y_indices = (relative_y_indices + offset).long()
        relative_x_bias = self.relative_coords_bias_embedding[:, relative_x_indices]
        relative_y_bias = self.relative_coords_bias_embedding[:, relative_y_indices]
        attention_scores = attention_scores + relative_x_bias.unsqueeze(0) + relative_y_bias.unsqueeze(0)
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        output = torch.matmul(attention_weights, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        output = self.out_proj(output)
        return output

class SpatialSelfAttentionBlock(nn.Module):
    """
    (This class is unchanged from your provided code)
    """
    def __init__(self, num_filters, board_size, num_heads=8, dropout=0.1):
        super(SpatialSelfAttentionBlock, self).__init__()
        self.num_filters = num_filters
        self.board_size = board_size
        self.num_heads = num_heads
        self.norm = nn.LayerNorm(num_filters)
        self.attention = MultiHeadSelfAttentionWithRelativeBias(
            embed_dim=num_filters, num_heads=num_heads, board_size=board_size, dropout=dropout
        )
        self.fc1 = nn.Linear(num_filters, 4 * num_filters)
        self.fc2 = nn.Linear(4 * num_filters, num_filters)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size, channels, H, W = x.shape
        x_flat = x.view(batch_size, channels, -1).permute(0, 2, 1)
        residual_block_input = x_flat
        norm_x_flat = self.norm(x_flat)
        attn_output = self.attention(norm_x_flat)
        attn_output = self.dropout(attn_output)
        out = residual_block_input + attn_output
        ffn_input = self.norm(out)
        ffn_output = self.fc2(self.dropout(F.relu(self.fc1(ffn_input))))
        ffn_output = self.dropout(ffn_output)
        out = out + ffn_output
        out = out.permute(0, 2, 1).view(batch_size, channels, H, W)
        return out


class PomPomNN(nn.Module):
    """
    Neural network model for PomPom chess.
    *** UPDATED: Replaced the Ownership Head with a Legal Moves Prediction Head. ***
    """
    ACTUAL_INPUT_CHANNELS = NUM_INPUT_CHANNELS + 2

    def __init__(self, num_res_blocks=4, num_filters=96, num_attention_heads=8):
        super(PomPomNN, self).__init__()
        self.num_res_blocks = num_res_blocks
        self.num_filters = num_filters
        self.num_attention_heads = num_attention_heads

        # --- Body ---
        self.initial_conv = nn.Conv2d(self.ACTUAL_INPUT_CHANNELS, num_filters, kernel_size=3, padding=1, bias=False)
        self.initial_bn = nn.BatchNorm2d(num_filters)
        self.initial_relu = nn.ReLU(inplace=True)
        self.global_pool_bias_initial = GlobalPoolingBias(num_filters)
        self.res_blocks = nn.ModuleList([ResidualBlock(num_filters) for _ in range(num_res_blocks)])
        self.spatial_attention = SpatialSelfAttentionBlock(num_filters, BOARD_SIZE, num_heads=self.num_attention_heads)
        self.spatial_attention2 = SpatialSelfAttentionBlock(num_filters, BOARD_SIZE, num_heads=self.num_attention_heads)

        # --- Policy Head ---
        self.policy_conv = nn.Conv2d(num_filters, 2, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_relu = nn.ReLU(inplace=True)
        self.global_pool_bias_policy = GlobalPoolingBias(2)
        self.policy_fc = nn.Linear(2 * BOARD_SIZE * BOARD_SIZE, BOARD_SIZE * BOARD_SIZE)

        # --- Soft Policy Head ---
        self.soft_policy_conv = nn.Conv2d(num_filters, 2, kernel_size=1, bias=False)
        self.soft_policy_bn = nn.BatchNorm2d(2)
        self.soft_policy_relu = nn.ReLU(inplace=True)
        self.soft_global_pool_bias_policy = GlobalPoolingBias(2)
        self.soft_policy_fc = nn.Linear(2 * BOARD_SIZE * BOARD_SIZE, BOARD_SIZE * BOARD_SIZE)

        # --- Value Head ---
        self.value_conv = nn.Conv2d(num_filters, 1, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_relu = nn.ReLU(inplace=True)
        self.global_pool_bias_value = GlobalPoolingBias(1)
        self.value_fc1 = nn.Linear(BOARD_SIZE * BOARD_SIZE, num_filters)
        self.value_fc2 = nn.Linear(num_filters, 1)

        # --- Legal Moves Head (Replaces Ownership Head) ---
        # This head predicts which moves are legal on the board.
        self.legal_moves_conv = nn.Conv2d(num_filters, 1, kernel_size=1, bias=False)
        self.legal_moves_bn = nn.BatchNorm2d(1)
        self.legal_moves_relu = nn.ReLU(inplace=True)
        self.legal_moves_fc = nn.Linear(BOARD_SIZE * BOARD_SIZE, BOARD_SIZE * BOARD_SIZE)

    def forward(self, x):
        """
        Forward pass of the neural network.
        *** UPDATED: Returns legal_moves_logits instead of ownership_output. ***
        """
        batch_size, _, H, W = x.shape

        # --- Coordinate Channels ---
        x_coords = torch.linspace(-1, 1, W, device=x.device).view(1, 1, 1, W).expand(batch_size, 1, H, W)
        y_coords = torch.linspace(-1, 1, H, device=x.device).view(1, 1, H, 1).expand(batch_size, 1, H, W)
        x_with_coords = torch.cat([x, x_coords, y_coords], dim=1)

        # --- Main Body Path ---
        x = self.initial_relu(self.initial_bn(self.initial_conv(x_with_coords)))
        x = self.global_pool_bias_initial(x)
        attention_insertion_point = self.num_res_blocks // 2
        for i, block in enumerate(self.res_blocks):
            x = block(x)
            if i == attention_insertion_point:
                x = self.spatial_attention(x)
                x = self.spatial_attention2(x)

        # --- Policy Head ---
        policy = self.policy_relu(self.policy_bn(self.policy_conv(x)))
        policy = self.global_pool_bias_policy(policy)
        policy = policy.view(policy.size(0), -1)
        policy_logits = self.policy_fc(policy)

        # --- Value Head ---
        value = self.value_relu(self.value_bn(self.value_conv(x)))
        value = self.global_pool_bias_value(value)
        value = value.view(value.size(0), -1)
        value = F.relu(self.value_fc1(value))
        value_output = torch.tanh(self.value_fc2(value))

        # --- Legal Moves Head ---
        legal_moves = self.legal_moves_relu(self.legal_moves_bn(self.legal_moves_conv(x)))
        legal_moves = legal_moves.view(legal_moves.size(0), -1)
        # Output logits directly. The BCEWithLogitsLoss in the training script will handle the sigmoid.
        legal_moves_logits = self.legal_moves_fc(legal_moves)

        # --- Soft Policy Head ---
        soft_policy = self.soft_policy_relu(self.soft_policy_bn(self.soft_policy_conv(x)))
        soft_policy = self.soft_global_pool_bias_policy(soft_policy)
        soft_policy = soft_policy.view(soft_policy.size(0), -1)
        soft_policy_logits = self.soft_policy_fc(soft_policy)

        return policy_logits, value_output, legal_moves_logits, soft_policy_logits

# Example usage
if __name__ == "__main__":
    dummy_input = torch.randn(4, NUM_INPUT_CHANNELS, BOARD_SIZE, BOARD_SIZE)
    model = PomPomNN()
    model.eval()
    with torch.no_grad():
        p_logits, v_out, l_logits, sp_logits = model(dummy_input)

    print("--- Model with Legal Moves Head ---")
    print(f"Input shape: {dummy_input.shape}")
    print(f"Policy logits output shape: {p_logits.shape}")
    print(f"Value output shape: {v_out.shape}")
    print(f"Legal Moves logits output shape: {l_logits.shape}")
    print(f"Soft Policy logits output shape: {sp_logits.shape}")
    print("\nModel definition updated successfully.")

