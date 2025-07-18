import torch
from popucom_nn_model import PomPomNN
from popucom_nn_interface import NUM_INPUT_CHANNELS, BOARD_SIZE

# --- 配置 ---
MODEL_PATH_PTH = "model.pth"  # 你的PyTorch模型输入路径
MODEL_PATH_ONNX = "model.onnx"  # ONNX模型的输出路径
BATCH_SIZE = 1  # ONNX模型将使用动态批处理大小，这里设为1作为示例


def main():
    """
    加载PyTorch模型并将其转换为ONNX格式。
    """
    print(f"正在从 {MODEL_PATH_PTH} 加载PyTorch模型...")

    # 加载你的模型结构
    model = PomPomNN()

    # 加载训练好的权重
    try:
        model.load_state_dict(torch.load(MODEL_PATH_PTH, map_location='cpu'))
    except FileNotFoundError:
        print(f"错误: 未找到模型文件 '{MODEL_PATH_PTH}'。请确保模型文件存在。")
        return

    # 设置为评估模式，这很重要
    model.eval()

    print("模型加载成功。开始转换为ONNX格式...")

    # 创建一个符合模型输入的虚拟张量
    # 尺寸: (batch_size, channels, height, width)
    dummy_input = torch.randn(BATCH_SIZE, NUM_INPUT_CHANNELS, BOARD_SIZE, BOARD_SIZE)

    # 定义动态轴，这允许ONNX模型处理不同大小的批次
    dynamic_axes = {'input': {0: 'batch_size'},  # 输入张量的第0维是动态的
                    'policy': {0: 'batch_size'},  # 输出策略的第0维也是动态的
                    'value': {0: 'batch_size'},  # 输出价值的第0维也是动态的
                    'soft_policy': {0: 'batch_size'}}  # 软策略的第0维也是动态的

    # 导出模型
    torch.onnx.export(model,  # 要转换的模型
                      dummy_input,  # 虚拟输入
                      MODEL_PATH_ONNX,  # 输出文件名
                      export_params=True,  # 导出训练好的权重
                      opset_version=14,  # ONNX版本
                      do_constant_folding=True,  # 执行常量折叠优化
                      input_names=['input'],  # 输入张量的名字
                      output_names=['policy', 'value', 'soft_policy'],  # 输出张量的名字
                      dynamic_axes=dynamic_axes)  # 指定动态轴

    print("-" * 50)
    print(f"模型成功转换为ONNX格式，并保存为 '{MODEL_PATH_ONNX}'")
    print("-" * 50)


if __name__ == "__main__":
    main()
