import subprocess
import sys
import os

# --- 脚本文件名配置 ---
SELF_PLAY_SCRIPT = "self_play_worker_trt.py"
TRAIN_SCRIPT = "train_model.py"
BUILD_ENGINE_SCRIPT = "build_tensorrt_engine.py"

# --- 模型文件配置 ---
PYTORCH_MODEL_PATH = "model.pth"
TENSORRT_ENGINE_PATH = "model.plan"


def run_script(script_name):
    """一个辅助函数，用于调用另一个Python脚本并等待其完成。"""
    print(f"\n{'=' * 25}")
    print(f"  正在运行: {script_name}")
    print(f"{'=' * 25}\n")
    try:
        # 使用 sys.executable 确保用的是同一个Python解释器环境
        process = subprocess.Popen([sys.executable, script_name])
        process.wait()  # 等待子进程完成
        if process.returncode != 0:
            print(f"\n错误: {script_name} 运行失败，返回代码 {process.returncode}。")
            return False
        print(f"\n--- {script_name} 成功运行 ---")
        return True
    except FileNotFoundError:
        print(f"\n错误: 找不到脚本 '{script_name}'。请确保所有脚本都在同一目录下。")
        return False
    except Exception as e:
        print(f"\n运行 {script_name} 时发生未知错误: {e}")
        return False


def ensure_initial_engine_exists():
    """检查初始的TensorRT引擎是否存在，如果不存在，则创建它。"""
    if os.path.exists(TENSORRT_ENGINE_PATH):
        print(f"检测到已存在的 TensorRT 引擎 ({TENSORRT_ENGINE_PATH})，流水线将直接使用它。")
        return True

    print(f"未检测到 TensorRT 引擎。开始首次构建流程...")

    # 1. 检查PyTorch模型是否存在，如果不存在，则需要先运行训练脚本来创建一个初始模型
    if not os.path.exists(PYTORCH_MODEL_PATH):
        print(f"也未检测到 PyTorch 模型 ({PYTORCH_MODEL_PATH})。")
        print("将运行一次训练脚本以创建一个随机初始化的模型...")
        if not run_script(TRAIN_SCRIPT):
            print("创建初始PyTorch模型失败，无法继续。")
            return False

    # 2. 现在我们应该有了一个PyTorch模型，用它来构建TensorRT引擎
    print("开始构建初始 TensorRT 引擎...")
    if not run_script(BUILD_ENGINE_SCRIPT):
        print("构建初始TensorRT引擎失败，无法继续。")
        return False

    print("初始 TensorRT 引擎构建成功！")
    return True


def main_pipeline():
    """
    基于TensorRT的强化学习主流水线。
    循环执行: 自对弈 -> 训练 -> 构建新引擎。
    """
    # 首先，确保我们有一个可用的引擎
    if not ensure_initial_engine_exists():
        print("初始化失败，终止流水线。")
        return

    iteration = 0
    while True:
        iteration += 1
        print(f"\n\n{'#' * 60}")
        print(f"  开始强化学习第 {iteration} 轮迭代 (使用 TensorRT 引擎)")
        print(f"{'#' * 60}")

        # --- 步骤 1: 使用TensorRT引擎进行自对弈 ---
        print("\n>>> 阶段 1: 使用 TensorRT 引擎生成自对弈数据...")
        if not run_script(SELF_PLAY_SCRIPT):
            print("自对弈脚本执行失败，终止流水线。")
            break

        # --- 步骤 2: 使用新数据训练模型，生成新的 PyTorch 模型 ---
        print("\n>>> 阶段 2: 使用新数据训练，生成新版 PyTorch 模型...")
        if not run_script(TRAIN_SCRIPT):
            print("训练脚本执行失败，终止流水线。")
            break

        # --- 步骤 3: 将新训练的 PyTorch 模型转换为 TensorRT 引擎，供下一轮使用 ---
        print("\n>>> 阶段 3: 将新版模型转换为下一轮使用的 TensorRT 引擎...")
        if not run_script(BUILD_ENGINE_SCRIPT):
            print("构建新版 TensorRT 引擎失败，终止流水线。")
            break

        print(f"\n第 {iteration} 轮迭代完成。下一轮将使用刚刚生成的新引擎。")


if __name__ == "__main__":
    main_pipeline()
