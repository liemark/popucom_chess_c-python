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

# 导入模型
from popucom_nn_model import PomPomNN


# --- Muon 优化器核心实现 (已针对计算速度优化) ---
class Muon(torch.optim.Optimizer):
    """
    Muon 优化器实现 (基于 Moonlight 论文)
    通过 Newton-Schulz 迭代实现矩阵更新的正交化。
    优化点：根据矩阵形状自动选择最小维度的方阵进行迭代，减少计算量。
    """

    def __init__(self, params, lr=1e-3, momentum=0.95, n_steps=5, weight_decay=0.01):
        defaults = dict(lr=lr, momentum=momentum, n_steps=n_steps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            n_steps = group['n_steps']
            wd = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.state[p]
                if len(state) == 0:
                    state['momentum'] = torch.zeros_like(p.data)

                # 1. 更新动量 (Momentum update)
                buf = state['momentum']
                buf.mul_(momentum).add_(p.grad)

                # 2. 准备正交化
                g = buf
                shape = g.shape
                # 只处理 2D 矩阵
                if len(shape) == 2:
                    # 获取更新缩放因子 (Moonlight 论文建议: 0.2 * sqrt(max(A, B)))
                    scale = 0.2 * (max(shape[0], shape[1]) ** 0.5)

                    # 使用 bfloat16 或 float32 进行计算以加速
                    X = g.to(torch.bfloat16) if p.device.type == 'cuda' else g.float()
                    # 归一化初始值 (Frobenius norm)
                    X /= (X.norm() + 1e-7)

                    # Newton-Schulz 迭代
                    # 优化技巧：选择较小的维度进行乘法
                    if shape[0] < shape[1]:
                        for _ in range(n_steps):
                            XXT = X @ X.t()
                            # 减少冗余计算：X = 3.4445*X - 4.775*(XXT@X) + 2.0315*(XXT@(XXT@X))
                            tmp = XXT @ X
                            X = 3.4445 * X - 4.775 * tmp + 2.0315 * (XXT @ tmp)
                    else:
                        for _ in range(n_steps):
                            XTX = X.t() @ X
                            tmp = X @ XTX
                            X = 3.4445 * X - 4.775 * tmp + 2.0315 * (tmp @ XTX)

                    update = X.to(p.dtype) * scale
                else:
                    update = g  # 非矩阵参数回退到普通动量更新

                # 3. 应用权重衰减和更新
                if wd != 0:
                    p.data.mul_(1 - lr * wd)

                p.data.add_(update, alpha=-lr)


def augment_data(state, policy):
    """ 数据增强：随机旋转和翻转 """
    policy_2d = policy.reshape(9, 9)
    transform_type = np.random.randint(0, 8)
    k = transform_type % 4
    state = np.rot90(state, k, axes=(1, 2))
    policy_2d = np.rot90(policy_2d, k)
    if transform_type >= 4:
        state = np.flip(state, axis=2)
        policy_2d = np.flip(policy_2d, axis=1)
    return state.copy(), policy_2d.flatten().copy()


class PopucomDataset(Dataset):
    def __init__(self, data, augment=True):
        self.data = data
        self.augment = augment

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        state, policy, value = self.data[idx]
        if self.augment:
            state, policy = augment_data(state, policy)
        return state, policy, value


def load_data(data_dir, max_files=3):
    all_data = []
    file_paths = sorted(glob.glob(os.path.join(glob.escape(data_dir), "*.pkl.gz")), key=os.path.getmtime, reverse=True)
    for file_path in file_paths[:max_files]:
        try:
            with gzip.open(file_path, 'rb') as f:
                data = pickle.load(f)
                all_data.extend(data)
        except Exception as e:
            print(f"警告: 无法加载文件 {file_path}: {e}")
    return all_data


def get_args():
    parser = argparse.ArgumentParser(description="训练 PomPomNN (Muon 版)")
    parser.add_argument('--data-dir', type=str, default='self_play_data')
    parser.add_argument('--model-path', type=str, default='model.pth')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--policy-weight', type=float, default=1.0)
    parser.add_argument('--value-weight', type=float, default=1.0)
    parser.add_argument('--soft-policy-weight', type=float, default=8.0)
    return parser.parse_args()


def train_model(args):
    print("--- 开始模型训练 (已启用 Muon 优化器) ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    model = PomPomNN().to(device)

    if os.path.exists(args.model_path):
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print("已加载现有模型权重。")
    else:
        print(f"未找到模型 '{args.model_path}'。将初始化新模型。")

    # --- 参数分组 ---
    # 按照论文，矩阵参数用 Muon，向量/标量参数用 AdamW
    muon_params = [p for p in model.parameters() if p.ndim == 2 and p.requires_grad]
    adamw_params = [p for p in model.parameters() if p.ndim < 2 and p.requires_grad]

    optimizer_muon = Muon(muon_params, lr=args.lr, weight_decay=args.weight_decay)
    optimizer_adamw = optim.AdamW(adamw_params, lr=args.lr, weight_decay=args.weight_decay)

    # 数据准备
    training_data = load_data(args.data_dir)
    if not training_data:
        print(f"错误: 未找到训练数据。")
        return
    dataset = PopucomDataset(training_data)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    print(f"成功加载 {len(training_data)} 条训练样本。")

    ce_loss = nn.CrossEntropyLoss()
    mse_loss = nn.MSELoss()
    scaler = torch.amp.GradScaler(device=device.type, enabled=(device.type == 'cuda'))

    model.train()
    start_time = time.time()

    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        losses = {'total': 0.0, 'policy': 0.0, 'value': 0.0, 'soft_policy': 0.0}
        for states, target_policies, target_values in dataloader:
            states, target_policies = states.to(device), target_policies.to(device)
            target_values = target_values.to(device, dtype=torch.float32).unsqueeze(1)

            optimizer_muon.zero_grad(set_to_none=True)
            optimizer_adamw.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
                pred_policy_logits, pred_values, pred_soft_policy_logits = model(states)

                # 软策略目标计算
                soft_temp = 4.0
                target_policies_soft = torch.pow(target_policies + 1e-8, 1.0 / soft_temp)
                target_policies_soft /= target_policies_soft.sum(dim=1, keepdim=True)

                loss_policy = ce_loss(pred_policy_logits, target_policies)
                loss_value = mse_loss(pred_values, target_values)
                loss_soft_policy = ce_loss(pred_soft_policy_logits, target_policies_soft)

                total_loss = (args.policy_weight * loss_policy +
                              args.value_weight * loss_value +
                              args.soft_policy_weight * loss_soft_policy)

            scaler.scale(total_loss).backward()

            # 分组更新
            scaler.step(optimizer_muon)
            scaler.step(optimizer_adamw)
            scaler.update()

            losses['total'] += total_loss.item()
            losses['policy'] += loss_policy.item()
            losses['value'] += loss_value.item()
            losses['soft_policy'] += loss_soft_policy.item()

        num_batches = len(dataloader)
        if num_batches > 0:
            print(f"Epoch {epoch + 1}/{args.epochs} | "
                  f"总损失: {losses['total'] / num_batches:.4f} | "
                  f"策略: {losses['policy'] / num_batches:.4f} | "
                  f"价值: {losses['value'] / num_batches:.4f} | "
                  f"软策略: {losses['soft_policy'] / num_batches:.4f} | "
                  f"耗时: {time.time() - epoch_start_time:.2f}s")

    print(f"\n训练完成，用时: {time.time() - start_time:.2f} 秒。")

    torch.save(model.state_dict(), args.model_path)
    print(f"模型已保存至 {args.model_path}")


if __name__ == "__main__":
    train_model(get_args())