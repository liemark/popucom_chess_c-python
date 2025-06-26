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


class PopucomDataset(Dataset):
    """自定义数据集，用于加载自对弈数据"""

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 数据元组包含4个部分: state, policy, value, ownership
        return self.data[idx]


def load_data(data_dir, max_files=1000):
    """从目录加载多个 .pkl 数据文件"""
    all_data = []
    # glob.escape(data_dir) is important if the path contains special characters
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

    # 为不同的损失函数添加权重参数
    parser.add_argument('--policy-weight', type=float, default=1.0, help='策略损失的权重')
    parser.add_argument('--value-weight', type=float, default=1.0, help='价值损失的权重')
    parser.add_argument('--ownership-weight', type=float, default=1.0, help='所有权损失的权重')

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

    dataset = PopucomDataset(training_data)
    # 在Windows上，num_workers通常设为0以避免问题
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    print(f"成功加载 {len(training_data)} 条训练样本，共 {len(dataloader)} 个批次。")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    model = PomPomNN()
    try:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print(f"成功从 '{args.model_path}' 加载现有模型。")
    except FileNotFoundError:
        print(f"未找到模型 '{args.model_path}'。将从头开始训练一个新模型。")

    model.to(device)
    model.train()  # 设置为训练模式

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # 定义损失函数
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
            # 将数据移动到指定设备
            states = states.to(device)
            target_policies = target_policies.to(device)
            target_values = target_values.to(device, dtype=torch.float32).unsqueeze(1)
            target_ownerships = target_ownerships.to(device, dtype=torch.float32)

            optimizer.zero_grad()

            # 前向传播
            pred_policy_logits, pred_values, pred_ownerships = model(states)

            # 计算各个部分的损失
            loss_policy = policy_loss_fn(pred_policy_logits, target_policies)
            loss_value = value_loss_fn(pred_values, target_values)
            loss_ownership = ownership_loss_fn(pred_ownerships, target_ownerships)

            # 根据权重计算总损失
            loss = (args.policy_weight * loss_policy +
                    args.value_weight * loss_value +
                    args.ownership_weight * loss_ownership)

            # 记录损失
            total_loss += loss.item()
            total_policy_loss += loss_policy.item()
            total_value_loss += loss_value.item()
            total_ownership_loss += loss_ownership.item()

            # 反向传播和优化
            loss.backward()
            optimizer.step()

        # 计算并打印每个 epoch 的平均损失
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

    # 保存最终模型
    torch.save(model.state_dict(), args.model_path)
    print(f"模型已保存至 '{args.model_path}'。")


if __name__ == "__main__":
    args = get_args()
    if not os.path.exists(args.data_dir):
        os.makedirs(args.data_dir)
        print(f"创建数据目录: '{args.data_dir}'")

    train_model(args)
