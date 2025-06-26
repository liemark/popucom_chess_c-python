import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import glob
import pickle
import time
import argparse

# --- 导入我们自己的模块 ---
from popucom_nn_model import PomPomNN


# --- 新增: 数据增强函数 ---
def augment_data(state, policy, ownership):
    """
    对棋盘状态、策略和所有权图进行随机的旋转和翻转。
    这能有效扩充数据集，让模型学习到棋盘的对称性。
    """
    # 策略从一维(81,)转换为二维(9,9)以进行空间变换
    policy_2d = policy.reshape(9, 9)

    # 对一个正方形有8种对称变换 (4种旋转 x 2种翻转)
    # 随机选择其中一种
    transform_type = np.random.randint(0, 8)

    # 应用旋转 (0, 90, 180, 270 度)
    # k代表逆时针旋转90度的次数
    k = transform_type % 4
    # 对 state 张量 (C, H, W) 沿着 H, W 轴进行旋转
    state = np.rot90(state, k, axes=(1, 2))
    # 对 policy 和 ownership 二维图进行同样的旋转
    policy_2d = np.rot90(policy_2d, k)
    ownership = np.rot90(ownership, k)

    # 应用翻转 (在旋转之后)
    if transform_type >= 4:
        # 对 state 张量沿着宽度轴 (左右) 进行翻转
        state = np.flip(state, axis=2)
        # 对 policy 和 ownership 进行同样的翻转
        policy_2d = np.flip(policy_2d, axis=1)
        ownership = np.flip(ownership, axis=1)

    # 将变换后的策略图重新展平为一维
    policy_flat = policy_2d.flatten()

    # 返回副本以避免修改原始数据
    return state.copy(), policy_flat.copy(), ownership.copy()


class PopucomDataset(Dataset):
    """
    自定义数据集，用于加载自对弈数据。
    新增了在获取数据时进行数据增强的选项。
    """

    def __init__(self, data, augment=True):
        self.data = data
        self.augment = augment

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        state, policy, value, ownership = self.data[idx]

        if self.augment:
            # 如果启用增强，则应用随机变换
            state, policy, ownership = augment_data(state, policy, ownership)

        return state, policy, value, ownership


def load_data(data_dir, max_files=100):
    """从目录加载多个 .pkl 数据文件"""
    all_data = []
    # glob.escape 可以在路径包含特殊字符时正常工作
    file_paths = sorted(glob.glob(os.path.join(glob.escape(data_dir), "*.pkl")), key=os.path.getmtime, reverse=True)

    print(f"找到 {len(file_paths)} 个数据文件。将加载最新的 {min(len(file_paths), max_files)} 个。")
    for file_path in file_paths[:max_files]:
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
                all_data.extend(data)
        except Exception as e:
            print(f"警告: 无法加载或解析文件 {file_path}: {e}")

    return all_data


def get_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="训练 PomPomNN 模型")
    parser.add_argument('--data-dir', type=str, default='self_play_data', help='自对弈数据所在的目录')
    parser.add_argument('--model-path', type=str, default='model.pth', help='模型加载和保存的路径')
    parser.add_argument('--epochs', type=int, default=10, help='训练的总轮数')
    parser.add_argument('--batch-size', type=int, default=256, help='训练批次大小')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='AdamW 优化器的权重衰减')

    parser.add_argument('--policy-weight', type=float, default=1.0, help='策略损失的权重')
    parser.add_argument('--value-weight', type=float, default=1.0, help='价值损失的权重')
    parser.add_argument('--ownership-weight', type=float, default=1.0, help='所有权损失的权重')

    # 新增: 控制数据增强的开关
    parser.add_argument('--no-augment', action='store_true', help='如果设置此项，则禁用数据增强')

    return parser.parse_args()


def train_model(args):
    """主训练函数"""
    print("--- 开始模型训练 ---")
    print("当前配置:")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")
    print("--------------------")

    training_data = load_data(args.data_dir)
    if not training_data:
        print(f"错误: 在 '{args.data_dir}' 目录中未找到训练数据。请先运行自对弈。")
        return

    # 根据命令行参数决定是否启用数据增强
    use_augmentation = not args.no_augment
    dataset = PopucomDataset(training_data, augment=use_augmentation)

    # 在Windows上，num_workers通常设为0以避免多进程问题
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    print(
        f"成功加载 {len(training_data)} 条训练样本，共 {len(dataloader)} 个批次。数据增强已{'启用' if use_augmentation else '禁用'}。")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    model = PomPomNN()
    try:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print(f"成功从 '{args.model_path}' 加载现有模型。")
    except FileNotFoundError:
        print(f"未找到模型 '{args.model_path}'。将从头开始训练一个新模型。")

    model.to(device)
    model.train()

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    policy_loss_fn = nn.CrossEntropyLoss()
    value_loss_fn = nn.MSELoss()
    ownership_loss_fn = nn.MSELoss()

    start_time = time.time()
    for epoch in range(args.epochs):
        total_loss = 0.0
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_ownership_loss = 0.0

        for batch_idx, (states, target_policies, target_values, target_ownerships) in enumerate(dataloader):
            states = states.to(device)
            target_policies = target_policies.to(device)
            target_values = target_values.to(device, dtype=torch.float32).unsqueeze(1)
            target_ownerships = target_ownerships.to(device, dtype=torch.float32)

            optimizer.zero_grad()

            pred_policy_logits, pred_values, pred_ownerships = model(states)

            loss_policy = policy_loss_fn(pred_policy_logits, target_policies)
            loss_value = value_loss_fn(pred_values, target_values)
            loss_ownership = ownership_loss_fn(pred_ownerships, target_ownerships)

            loss = (args.policy_weight * loss_policy +
                    args.value_weight * loss_value +
                    args.ownership_weight * loss_ownership)

            total_loss += loss.item()
            total_policy_loss += loss_policy.item()
            total_value_loss += loss_value.item()
            total_ownership_loss += loss_ownership.item()

            loss.backward()
            optimizer.step()

        avg_loss = total_loss / len(dataloader)
        avg_policy_loss = total_policy_loss / len(dataloader)
        avg_value_loss = total_value_loss / len(dataloader)
        avg_ownership_loss = total_ownership_loss / len(dataloader)

        print(f"Epoch {epoch + 1}/{args.epochs} | "
              f"总损失: {avg_loss:.4f} | "
              f"策略损失: {avg_policy_loss:.4f} | "
              f"价值损失: {avg_value_loss:.4f} | "
              f"所有权损失: {avg_ownership_loss:.4f}")

    end_time = time.time()
    print(f"\n训练完成，用时: {end_time - start_time:.2f} 秒。")

    torch.save(model.state_dict(), args.model_path)
    print(f"模型已保存至 '{args.model_path}'。")


if __name__ == "__main__":
    args = get_args()
    if not os.path.exists(args.data_dir):
        os.makedirs(args.data_dir)
        print(f"创建数据目录: '{args.data_dir}'")

    train_model(args)
