import torch
import tensorrt as trt
import os
import sys

# 导入您的模型定义
from popucom_nn_model import PomPomNN

# --- 配置 ---
# 这个值必须与您自对弈/分析脚本中的 NUM_PARALLEL_GAMES 一致
MAX_BATCH_SIZE = 512
BOARD_SIZE = 9
NUM_INPUT_CHANNELS = 11  # 仅内容通道数

# 输入/输出文件名
PYTORCH_MODEL_PATH = "model.pth"
ONNX_MODEL_PATH = "model.onnx"
TENSORRT_ENGINE_PATH = "model.plan"


def build_engine():
    """
    加载PyTorch模型，转换为ONNX，然后构建并保存TensorRT引擎。
    """
    # 检查 TensorRT 版本
    print(f"TensorRT version: {trt.__version__}")

    # 1. 加载 PyTorch 模型
    print(f"Loading PyTorch model from {PYTORCH_MODEL_PATH}...")
    try:
        model = PomPomNN()
        model.load_state_dict(torch.load(PYTORCH_MODEL_PATH))
        # 确保模型在评估模式，这会关闭 dropout 等层
        model.cuda().eval()
        print("PyTorch model loaded successfully.")
    except Exception as e:
        print(f"Error loading PyTorch model: {e}")
        sys.exit(1)

    # 2. 导出到 ONNX 格式
    print(f"Exporting model to ONNX format at {ONNX_MODEL_PATH}...")
    # 创建一个符合模型输入的虚拟输入张量
    # 注意：这里的输入通道数是内容通道，坐标通道是在模型内部生成的
    dummy_input = torch.randn(MAX_BATCH_SIZE, NUM_INPUT_CHANNELS, BOARD_SIZE, BOARD_SIZE, device='cuda')

    try:
        torch.onnx.export(
            model,
            dummy_input,
            ONNX_MODEL_PATH,
            verbose=False,
            opset_version=17,  # 推荐使用较新的 opset
            input_names=['input'],
            output_names=['policy_logits', 'value_output', 'soft_policy_logits'],
            dynamic_axes=None  # 我们使用固定的最大批处理尺寸，因此没有动态轴
        )
        print("Model exported to ONNX successfully.")
    except Exception as e:
        print(f"Error exporting to ONNX: {e}")
        print("This error often happens if your model contains operations not supported by ONNX.")
        print("The custom self-attention block might be the cause. Ensure all operations are traceable.")
        sys.exit(1)

    # 3. 构建 TensorRT 引擎
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    explicit_batch = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

    print("Building TensorRT engine. This may take a few minutes...")
    with trt.Builder(TRT_LOGGER) as builder, \
            builder.create_network(explicit_batch) as network, \
            builder.create_builder_config() as config, \
            trt.OnnxParser(network, TRT_LOGGER) as parser:

        # 设置构建器配置
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB workspace
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("FP16 mode enabled.")

        # 解析 ONNX 文件
        print(f"Parsing ONNX model {ONNX_MODEL_PATH}...")
        if not os.path.exists(ONNX_MODEL_PATH):
            print(f"ONNX file not found at {ONNX_MODEL_PATH}")
            sys.exit(1)

        with open(ONNX_MODEL_PATH, 'rb') as model_file:
            if not parser.parse(model_file.read()):
                print('ERROR: Failed to parse the ONNX file.')
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                sys.exit(1)
        print("ONNX model parsed successfully.")

        # --- FIXED ---
        # 设置网络输入尺寸。
        # ONNX模型是基于一个11通道的输入导出的（模型内部会添加2个坐标通道）。
        # 因此，TensorRT网络也必须定义为接收11通道的输入，以与ONNX图的定义匹配。
        network.get_input(0).shape = [MAX_BATCH_SIZE, NUM_INPUT_CHANNELS, BOARD_SIZE, BOARD_SIZE]

        # 构建并序列化引擎
        print("Building serialized engine...")
        serialized_engine = builder.build_serialized_network(network, config)
        if serialized_engine is None:
            print("Failed to build the engine.")
            sys.exit(1)
        print("Engine built successfully.")

        # 保存到文件
        with open(TENSORRT_ENGINE_PATH, "wb") as f:
            f.write(serialized_engine)
        print(f"TensorRT engine saved to {TENSORRT_ENGINE_PATH}")


if __name__ == "__main__":
    build_engine()
