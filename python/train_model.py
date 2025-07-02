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


def augment_data(state, policy, legal_moves):
    """
    对棋盘状态、策略和合法走法掩码进行随机的旋转和翻转。
    *** 新增：现在也对 legal_moves 进行增强 ***
    """
    policy_2d = policy.reshape(9, 9)
    # 将合法走法掩码也转换为2D形式
    legal_moves_2d = legal_moves.reshape(9, 9)

    transform_type = np.random.randint(0, 8)
    k = transform_type % 4
    state = np.rot90(state, k, axes=(1, 2))
    policy_2d = np.rot90(policy_2d, k)
    legal_moves_2d = np.rot90(legal_moves_2d, k) # 应用相同的旋转

    if transform_type >= 4:
        state = np.flip(state, axis=2)
        policy_2d = np.flip(policy_2d, axis=1)
        legal_moves_2d = np.flip(legal_moves_2d, axis=1) # 应用相同的翻转

    return state.copy(), policy_2d.flatten().copy(), legal_moves_2d.flatten().copy()


class PopucomDataset(Dataset):
    """
    自定义数据集，加载自对弈数据并应用数据增强。
    *** 已更新，现在加载合法走法掩码而不是所有权图 ***
    """
    def __init__(self, data, augment=True):
        self.data = data
        self.augment = augment

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 数据格式现在应为: (state, policy, value, legal_moves_mask)
        # 您需要确保您的自对弈数据生成脚本保存的是这个格式。
        state, policy, value, legal_moves = self.data[idx]
        if self.augment:
            state, policy, legal_moves = augment_data(state, policy, legal_moves)
        return state, policy, value, legal_moves


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
    """
    解析命令行参数
    *** 已更新：将 ownership-weight 替换为 legal-moves-weight ***
    """
    parser = argparse.ArgumentParser(description="训练 PomPomNN 模型")
    parser.add_argument('--data-dir', type=str, default='self_play_data', help='自对弈数据所在的目录')
    parser.add_argument('--model-path', type=str, default='model.pth', help='模型加载和保存的路径')
    parser.add_argument('--epochs', type=int, default=1, help='训练的总轮数 (推荐值为1)')
    parser.add_argument('--batch-size', type=int, default=256, help='训练批次大小')
    parser.add_argument('--lr', type=float, default=1e-5, help='学习率')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='AdamW 优化器的权重衰减')
    parser.add_argument('--policy-weight', type=float, default=1.0, help='策略损失的权重')
    parser.add_argument('--value-weight', type=float, default=1.0, help='价值损失的权重')
    # 替换了 ownership-weight
    parser.add_argument('--legal-moves-weight', type=float, default=0.25, help='合法走法预测损失的权重')
    parser.add_argument('--soft-policy-weight', type=float, default=8.0, help='辅助软策略损失的权重')
    parser.add_argument('--no-augment', action='store_true', help='如果设置此项，则禁用数据增强')
    return parser.parse_args()


def train_model(args):
    """
    主训练函数
    *** 已更新，使用新的合法走法预测任务 ***
    """
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
    # 为合法走法预测任务选择更合适的损失函数
    legal_moves_loss_fn = nn.BCEWithLogitsLoss()

    scaler = torch.amp.GradScaler(device=device.type,enabled=(device.type == 'cuda'))

    start_time = time.time()
    for epoch in range(args.epochs):
        losses = {'total': 0.0, 'policy': 0.0, 'value': 0.0, 'legal_moves': 0.0, 'soft_policy': 0.0}

        for states, target_policies, target_values, target_legal_moves in dataloader:
            states = states.to(device)
            target_policies = target_policies.to(device)
            target_values = target_values.to(device, dtype=torch.float32).unsqueeze(1)
            # 合法走法掩码的目标张量
            target_legal_moves = target_legal_moves.to(device, dtype=torch.float32)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=device.type,enabled=(device.type == 'cuda')):
                # 假设模型现在返回: pred_policy_logits, pred_values, pred_legal_logits, pred_soft_policy_logits
                # 您需要修改 popucom_nn_model.py 来实现这一点。
                pred_policy_logits, pred_values, pred_legal_logits, pred_soft_policy_logits = model(states)

                soft_policy_temp = 4.0
                target_policies_soft = target_policies + 1e-8
                target_policies_soft = torch.pow(target_policies_soft, 1.0 / soft_policy_temp)
                target_policies_soft /= torch.sum(target_policies_soft, dim=1, keepdim=True)

                loss_policy = policy_loss_fn(pred_policy_logits, target_policies)
                loss_value = value_loss_fn(pred_values, target_values)
                # 计算新的合法走法预测损失
                loss_legal_moves = legal_moves_loss_fn(pred_legal_logits, target_legal_moves)
                loss_soft_policy = policy_loss_fn(pred_soft_policy_logits, target_policies_soft)

                # 在总损失中包含新的损失项
                loss = (args.policy_weight * loss_policy +
                        args.value_weight * loss_value +
                        args.legal_moves_weight * loss_legal_moves +
                        args.soft_policy_weight * loss_soft_policy)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            losses['total'] += loss.item()
            losses['policy'] += loss_policy.item()
            losses['value'] += loss_value.item()
            losses['legal_moves'] += loss_legal_moves.item()
            losses['soft_policy'] += loss_soft_policy.item()

        num_batches = len(dataloader)
        if num_batches > 0:
            # 更新日志输出
            print(f"Epoch {epoch + 1}/{args.epochs} | "
                  f"总损失: {losses['total'] / num_batches:.4f} | "
                  f"策略: {losses['policy'] / num_batches:.4f} | "
                  f"价值: {losses['value'] / num_batches:.4f} | "
                  f"合法走法: {losses['legal_moves'] / num_batches:.4f} | "
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