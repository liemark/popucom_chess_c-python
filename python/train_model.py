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
import bisect  # 用于快速查找文件索引

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


class PopucomLazyDataset(Dataset):
    """
    自定义数据集，实现Lazyload以优化内存占用。
    它只在需要时才从磁盘加载单个文件。
    """

    def __init__(self, file_paths, augment=True):
        self.file_paths = file_paths
        self.augment = augment

        # --- 建立索引 ---
        # 需要预先知道每个文件里有多少样本，以便计算总长度
        self.cumulative_lengths = [0]
        print("正在建立数据集索引...")
        for file_path in self.file_paths:
            # 这里仍然需要打开每个文件一次来获取其长度，
            # 但数据不会被一直保留在内存中。
            try:
                with gzip.open(file_path, 'rb') as f:
                    data = pickle.load(f)
                    length = len(data)
                    self.cumulative_lengths.append(self.cumulative_lengths[-1] + length)
            except Exception as e:
                print(f"警告: 建立索引时无法读取文件 {file_path}: {e}")

        self.total_length = self.cumulative_lengths[-1]

        # --- 缓存机制 ---
        # 为了避免在连续访问同一个文件时反复进行I/O操作，仅缓存最后一个打开的文件。
        self.last_opened_file_path = None
        self.last_opened_file_data = None

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.total_length:
            raise IndexError("Index out of range")

        # 1. 使用二分查找快速定位索引idx属于哪个文件
        file_index = bisect.bisect_right(self.cumulative_lengths, idx) - 1

        # 2. 计算在该文件内的局部索引
        local_idx = idx - self.cumulative_lengths[file_index]
        target_file_path = self.file_paths[file_index]

        # 3. 检查缓存，如果需要的文件不是上一个文件，则加载新文件
        if target_file_path != self.last_opened_file_path:
            try:
                with gzip.open(target_file_path, 'rb') as f:
                    self.last_opened_file_data = pickle.load(f)
                self.last_opened_file_path = target_file_path
            except Exception as e:
                print(f"错误: __getitem__ 无法加载文件 {target_file_path}: {e}")
                # 返回一个虚拟数据或抛出异常
                return torch.zeros(11, 9, 9), torch.zeros(81), 0.0, torch.zeros(9, 9)

        # 4. 从缓存的数据中获取样本
        state, policy, value, ownership = self.last_opened_file_data[local_idx]

        # 5. 应用数据增强
        if self.augment:
            state, policy, ownership = augment_data(state, policy, ownership)

        return state, policy, value, ownership


def get_data_files(data_dir, max_files=25):
    """只获取数据文件的路径列表，而不加载它们。"""
    file_paths = sorted(glob.glob(os.path.join(glob.escape(data_dir), "*.pkl.gz")), key=os.path.getmtime, reverse=True)
    print(f"找到 {len(file_paths)} 个压缩数据文件。将使用最新的 {min(len(file_paths), max_files)} 个。")
    return file_paths[:max_files]


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

    # 现在只获取文件路径列表
    training_files = get_data_files(args.data_dir)
    if not training_files:
        print(f"错误: 在 '{args.data_dir}' 目录中未找到训练数据。请先运行自对弈。")
        return

    use_augmentation = not args.no_augment
    # 使用新的懒加载数据集
    dataset = PopucomLazyDataset(training_files, augment=use_augmentation)

    if len(dataset) == 0:
        print("错误: 数据集为空，无法开始训练。")
        return

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    print(f"成功创建数据集，总样本数: {len(dataset)}。数据增强已{'启用' if use_augmentation else '禁用'}。")

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

    scaler = torch.amp.GradScaler(enabled=(device.type == 'cuda'))

    start_time = time.time()
    for epoch in range(args.epochs):
        losses = {'total': 0.0, 'policy': 0.0, 'value': 0.0, 'ownership': 0.0, 'soft_policy': 0.0}

        for states, target_policies, target_values, target_ownerships in dataloader:
            states = states.to(device)
            target_policies = target_policies.to(device)
            target_values = target_values.to(device, dtype=torch.float32).unsqueeze(1)
            target_ownerships = target_ownerships.to(device, dtype=torch.float32)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
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
