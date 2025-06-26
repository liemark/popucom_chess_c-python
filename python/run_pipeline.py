import subprocess
import sys
import time
import os

# --- 配置 ---
# 您可以在这里调整每次循环中要运行的自对弈脚本的次数
# 例如，设置为 5 意味着每训练一次模型，会先生成 5 * NUM_PARALLEL_GAMES 局游戏数据
SELF_PLAY_RUNS_PER_TRAINING = 1


def run_script(script_name):
    """一个辅助函数，用于调用另一个Python脚本并等待其完成。"""
    print(f"\n{'=' * 20}")
    print(f"  正在运行: {script_name}")
    print(f"{'=' * 20}\n")
    try:
        # 使用 sys.executable 来确保我们用的是同一个Python解释器环境
        process = subprocess.Popen([sys.executable, script_name])
        process.wait()  # 等待子进程完成
        if process.returncode != 0:
            print(f"\n警告: {script_name} 运行出错，返回代码 {process.returncode}。流水线将继续...")
            return False
    except FileNotFoundError:
        print(f"\n错误: 找不到脚本 '{script_name}'。请确保所有脚本都在同一目录下。")
        return False
    except Exception as e:
        print(f"\n运行 {script_name} 时发生未知错误: {e}")
        return False

    print(f"\n--- {script_name} 运行完成 ---")
    return True


def main_pipeline():
    """主流水线函数，无限循环执行自对弈和训练。"""
    iteration = 0
    while True:
        iteration += 1
        print(f"\n\n{'#' * 50}")
        print(f"  开始强化学习第 {iteration} 轮迭代")
        print(f"{'#' * 50}")

        # --- 步骤 1: 自对弈数据生成 ---
        print("\n>>> 阶段 1: 生成自对弈数据...")
        for i in range(SELF_PLAY_RUNS_PER_TRAINING):
            print(f"\n  -- 自对弈运行: 第 {i + 1}/{SELF_PLAY_RUNS_PER_TRAINING} 批 --")
            if not run_script("self_play_worker.py"):
                print("自对弈脚本执行失败，终止流水线。")
                break

        # --- 步骤 2: 模型训练 ---
        print("\n>>> 阶段 2: 使用新数据训练模型...")
        if not run_script("train_model.py"):
            print("训练脚本执行失败，终止流水线。")
            break

        print(f"\n第 {iteration} 轮迭代完成。开始下一轮...")


if __name__ == "__main__":
    # 确保所有必需的文件都存在
    required_files = ["self_play_worker.py", "train_model.py", "popucom_nn_model.py", "popucom_nn_interface.py"]
    for f in required_files:
        if not os.path.exists(f):
            print(f"错误: 必需文件 '{f}' 不存在。无法启动流水线。")
            exit()

    main_pipeline()
