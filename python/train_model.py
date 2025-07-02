# train_model.py

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
import gzip

from popucom_nn_model import PomPomNN


def augment_data(state, policy, ownership):
    """对棋盘状态、策略和所有权图进行随机的旋转和翻转。"""
    policy_2d = policy.reshape(9, 9)
    transform_type = np.random.randint(0, 8)
    k = transform_type % 4
    state = np.rot90(state, k, axes=(1, 2))
    policy_2d = np.rot90(policy_2d, k)
    ownership = np.rot90(ownership, k)
    if transform_type >= 4:
        state = np.flip(state, axis=2)
        policy_2d = np.flip(policy_2d, axis=1)
        ownership = np.flip(ownership, axis=1)
    return state.copy(), policy_2d.flatten().copy(), ownership.copy()


class PopucomDataset(Dataset):
    """自定义数据集，加载自对弈数据并应用数据增强。"""
    def __init__(self, data, augment=True):
        self.data = data
        self.augment = augment

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        state, policy, value, ownership = self.data[idx]
        if self.augment:
            state, policy, ownership = augment_data(state, policy, ownership)
        return state, policy, value, ownership


def load_data(data_dir, max_files=25):
    """从目录加载多个压缩的 .pkl.gz 数据文件"""
    all_data = []
    file_paths = sorted(glob.glob(os.path.join(glob.escape(data_dir), "*.pkl.gz")), key=os.path.getmtime, reverse=True)
    print(f"找到 {len(file_paths)} 个压缩数据文件。将加载最新的 {min(len(file_paths), max_files)} 个。")
    for file_path in file_paths[:max_files]:
        try:
            with gzip.open(file_path, 'rb') as f:
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
    parser.add_argument('--epochs', type=int, default=1, help='训练的总轮数 (推荐值为1)')
    parser.add_argument('--batch-size', type=int, default=256, help='训练批次大小')
    parser.add_argument('--lr', type=float, default=2e-5, help='学习率')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='AdamW 优化器的权重衰减')
    parser.add_argument('--policy-weight', type=float, default=1.0, help='策略损失的权重')
    parser.add_argument('--value-weight', type=float, default=1.0, help='价值损失的权重')
    parser.add_argument('--ownership-weight', type=float, default=0.1, help='所有权损失的权重')
    parser.add_argument('--soft-policy-weight', type=float, default=8.0, help='辅助软策略损失的权重')
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

    use_augmentation = not args.no_augment
    dataset = PopucomDataset(training_data, augment=use_augmentation)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    print(f"成功加载 {len(training_data)} 条训练样本。数据增强已{'启用' if use_augmentation else '禁用'}。")

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

    scaler = torch.amp.GradScaler(device=device.type,enabled=(device.type == 'cuda'))

    start_time = time.time()
    for epoch in range(args.epochs):
        losses = {'total': 0.0, 'policy': 0.0, 'value': 0.0, 'ownership': 0.0, 'soft_policy': 0.0}

        for states, target_policies, target_values, target_ownerships in dataloader:
            states = states.to(device)
            target_policies = target_policies.to(device)
            target_values = target_values.to(device, dtype=torch.float32).unsqueeze(1)
            target_ownerships = target_ownerships.to(device, dtype=torch.float32)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=device.type,enabled=(device.type == 'cuda')):
                pred_policy_logits, pred_values, pred_ownerships, pred_soft_policy_logits = model(states)

                soft_policy_temp = 4.0
                target_policies_soft = target_policies + 1e-8
                target_policies_soft = torch.pow(target_policies_soft, 1.0 / soft_policy_temp)
                target_policies_soft /= torch.sum(target_policies_soft, dim=1, keepdim=True)

                loss_policy = policy_loss_fn(pred_policy_logits, target_policies)
                loss_value = value_loss_fn(pred_values, target_values)
                loss_ownership = ownership_loss_fn(pred_ownerships, target_ownerships)
                loss_soft_policy = policy_loss_fn(pred_soft_policy_logits, target_policies_soft)

                loss = (args.policy_weight * loss_policy +
                        args.value_weight * loss_value +
                        args.ownership_weight * loss_ownership +
                        args.soft_policy_weight * loss_soft_policy)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            losses['total'] += loss.item()
            losses['policy'] += loss_policy.item()
            losses['value'] += loss_value.item()
            losses['ownership'] += loss_ownership.item()
            losses['soft_policy'] += loss_soft_policy.item()

        num_batches = len(dataloader)
        if num_batches > 0:
            print(f"Epoch {epoch + 1}/{args.epochs} | "
                  f"总损失: {losses['total'] / num_batches:.4f} | "
                  f"策略: {losses['policy'] / num_batches:.4f} | "
                  f"价值: {losses['value'] / num_batches:.4f} | "
                  f"所有权: {losses['ownership'] / num_batches:.4f} | "
                  f"软策略: {losses['soft_policy'] / num_batches:.4f}")

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
