# arena_vs_old.py
# 新模型 vs 备份旧模型（backup_before_uncertainty，无不确定度头、无贴目通道）对练脚本。
# 【私人测试专用，不上传 github】
#
# 旧模型签名: 11 通道输入（无贴目平面）、3 输出（policy/value/soft_policy）
# 新模型签名: 12 通道输入（通道 11 = komi/20）、5 输出（+uncertainty/score）
#
# 依赖 compat DLL（popucom_core_compat.dll，由 c/build_compat.ps1 编译，
# 定义 UNCERTAINTY_SENTINEL 宏）：新旧模型混合对打时各自保留原生行为——
#   1. 新模型走 uncertainty 加权反传，旧模型向 C++ 传负值哨兵，
#      C++ 端将其转为中性反传（weight=1、σ=0，与旧接口完全等价），
#      避免旧模型 u=0 被 1/(u+0.05) 误当成 20 倍"高置信"
#   2. 任一方为旧 11 通道模型 → 双方输入都不填贴目平面（通道 11 = 0），
#      等价于贴目 0，新旧模型同视角对打
#   3. 对局贴目锁死 0（旧模型不懂贴目；结算与搜索终局估值均按无贴目）
# 主 DLL（popucom_core.dll）不含哨兵分支，保持热路径零额外判断。
#
# 用法（在 python 目录下运行）:
#   python arena_vs_old.py                       # 新 model.onnx vs 备份旧模型
#   python arena_vs_old.py --num_games 40 --simulations 50
#   python arena_vs_old.py --old_model <路径>    # 指定其他旧模型

import argparse
import os
import sys
from collections import defaultdict

import arena_onnx
from arena_onnx import load_onnx_session, print_results, BLACK_WIN, WHITE_WIN

# compat DLL：哨兵分支专用（主 DLL 不编译该分支）
COMPAT_LIB = "popucom_core_compat.dll" if sys.platform == "win32" else "popucom_core_compat.so"
if not os.path.exists(COMPAT_LIB):
    print(f"错误: 未找到 compat 库 {COMPAT_LIB}。请在 c/ 目录运行 build_compat.ps1 编译。")
    sys.exit(1)
# 覆盖 arena_onnx 的模块级 c_lib：ArenaRunner 通过模块全局引用 c_lib，
# 替换后所有对局均走 compat DLL（与主 DLL 各自独立实例，互不干扰）
arena_onnx.c_lib = arena_onnx.setup_c_library(COMPAT_LIB)
ArenaRunner = arena_onnx.ArenaRunner

# 备份旧模型的候选路径（按序探测）
OLD_MODEL_CANDIDATES = [
    os.path.join("..", "backup_before_uncertainty", "python", "model.onnx"),
    os.path.join("backup_before_uncertainty", "python", "model.onnx"),
]


def find_old_model(explicit=None):
    if explicit:
        if not os.path.exists(explicit):
            print(f"错误: 找不到旧模型 {explicit}")
            sys.exit(1)
        return explicit
    for p in OLD_MODEL_CANDIDATES:
        if os.path.exists(p):
            return p
    print("错误: 未找到备份旧模型，请用 --old_model 指定路径。"
          f"已尝试: {OLD_MODEL_CANDIDATES}")
    sys.exit(1)


def run_round(new_session, old_session, num_games, simulations, opening_moves_n,
              new_name, old_name):
    """一轮对打：new_session 执黑、old_session 执白。返回 {名称: 胜局}。"""
    # komi 锁 0：旧模型无贴目概念，双方输入贴目平面与终局结算均按无贴目；
    # per_side_uncertainty=True：compat DLL 哨兵模式，新模型加权、旧模型中性反传
    runner = ArenaRunner(new_session, old_session, num_games, simulations,
                         opening_moves_n=opening_moves_n, komi=0,
                         per_side_uncertainty=True)
    scores = defaultdict(int)
    for res in runner.run_matches():
        if res == BLACK_WIN:
            scores[new_name] += 1
        elif res == WHITE_WIN:
            scores[old_name] += 1
        else:
            scores['draw'] += 1
    return scores


def main(args):
    if args.num_games % 2 != 0:
        print("总对局数不是偶数。为保证公平，将自动减 1。")
        args.num_games -= 1

    old_path = find_old_model(args.old_model)
    session_new = load_onnx_session(args.new_model)
    session_old = load_onnx_session(old_path)
    new_name, old_name = args.new_model, old_path
    games_per_matchup = args.num_games // 2
    total_scores = defaultdict(int)

    if args.opening_moves > 0:
        print(f"将使用模型驱动的随机 {args.opening_moves} 步开局。")
    print("对局贴目: 0（旧模型无贴目，双方按无贴目对打）")

    print(f"\n--- 第一轮: 新模型 (黑) vs 旧模型 (白) ---")
    for k, v in run_round(session_new, session_old, games_per_matchup,
                          args.simulations, args.opening_moves, new_name, old_name).items():
        total_scores[k] += v
    print_results(total_scores, new_name, old_name)

    print(f"\n--- 第二轮: 旧模型 (黑) vs 新模型 (白) ---")
    for k, v in run_round(session_old, session_new, games_per_matchup,
                          args.simulations, args.opening_moves, old_name, new_name).items():
        total_scores[k] += v
    print_results(total_scores, new_name, old_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="新模型 vs 备份旧模型（无不确定度头）对练，贴目固定 0。")
    parser.add_argument("--new_model", type=str, default="model.onnx", help="新模型路径 (.onnx)")
    parser.add_argument("--old_model", type=str, default="model_old.onnx",
                        help="旧模型路径 (.onnx)，默认自动探测 backup_before_uncertainty")
    parser.add_argument("--num_games", type=int, default=50, help="总对局数 (必须是偶数)")
    parser.add_argument("--simulations", type=int, default=20, help="每一步的 MCTS 模拟次数")
    parser.add_argument("--opening_moves", type=int, default=6,
                        help="模型驱动的随机开局步数，设为 0 则纯确定性对弈")
    args = parser.parse_args()
    main(args)
