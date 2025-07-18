import ctypes
import os
import platform
import time
import pickle
import gzip
import numpy as np
import torch
import random
import sys
import tensorrt as trt

# 导入接口常量
from popucom_nn_interface import NUM_INPUT_CHANNELS, BOARD_SIZE

# --- 配置 ---
NUM_PARALLEL_GAMES = 512
MAX_BATCH_SIZE = NUM_PARALLEL_GAMES
TENSORRT_ENGINE_PATH = "model.plan"  # 现在加载 .plan 文件
DATA_DIR = "self_play_data"
TOTAL_GAME_CYCLES = 20
BOARD_SQUARES = BOARD_SIZE * BOARD_SIZE
USE_PLAYOUT_CAP_RANDOMIZATION = True
FULL_SEARCH_SIMS = 1000
FAST_SEARCH_SIMS = 100
FULL_SEARCH_PROB = 0.25
TEMPERATURE_START = 1.0
TEMPERATURE_DECAY_MOVES = 10


# --- C 语言接口定义 ---
class Bitboards(ctypes.Structure): _fields_ = [("parts", ctypes.c_uint64 * 2)]


class Board(ctypes.Structure): _fields_ = [("pieces", Bitboards * 2), ("tiles", Bitboards * 2),
                                           ("current_player", ctypes.c_int), ("moves_left", ctypes.c_int * 2)]


def setup_c_library():
    lib_name = "popucom_core.dll" if platform.system() == "Windows" else "popucom_core.so"
    if not os.path.exists(lib_name): raise FileNotFoundError(f"未找到C库 '{lib_name}'")
    c_lib = ctypes.CDLL(os.path.abspath(lib_name))
    c_lib.create_mcts_manager.argtypes = [ctypes.c_int, ctypes.c_bool, ctypes.c_double]
    c_lib.create_mcts_manager.restype = ctypes.c_void_p
    c_lib.mcts_run_simulations_and_get_requests.argtypes = [ctypes.c_void_p, ctypes.POINTER(Board),
                                                            ctypes.POINTER(ctypes.c_int), ctypes.c_int]
    c_lib.mcts_run_simulations_and_get_requests.restype = ctypes.c_int
    c_lib.mcts_feed_results.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                                        ctypes.POINTER(Board)]
    c_lib.boards_to_tensors_c.argtypes = [ctypes.POINTER(Board), ctypes.c_int, ctypes.POINTER(ctypes.c_float)]
    c_lib.destroy_mcts_manager.argtypes = [ctypes.c_void_p]
    c_lib.mcts_get_policy.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_float)]
    c_lib.mcts_get_legal_moves_mask.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_float)]
    c_lib.mcts_get_final_value.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    c_lib.mcts_get_final_value.restype = ctypes.c_float
    c_lib.mcts_get_simulations_done.argtypes = [ctypes.c_void_p, ctypes.c_int]
    c_lib.mcts_get_simulations_done.restype = ctypes.c_int
    c_lib.mcts_make_move.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    c_lib.mcts_is_game_over.argtypes = [ctypes.c_void_p, ctypes.c_int]
    c_lib.mcts_is_game_over.restype = ctypes.c_bool
    c_lib.mcts_get_board_state.argtypes = [ctypes.c_void_p, ctypes.c_int]
    c_lib.mcts_get_board_state.restype = ctypes.POINTER(Board)
    return c_lib


c_lib = setup_c_library()


# --- TensorRT 推理辅助类 ---
class TensorRTModel:
    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        if not os.path.exists(engine_path):
            raise FileNotFoundError(f"Engine file not found at {engine_path}. Please run the pipeline to build it.")
        with open(engine_path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.stream = torch.cuda.Stream()
        self.tensors = {}
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = self.engine.get_tensor_shape(name)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            device_mem = torch.empty(tuple(shape), dtype=torch.from_numpy(np.array([], dtype=dtype)).dtype,
                                     device='cuda')
            self.tensors[name] = device_mem
            self.context.set_tensor_address(name, device_mem.data_ptr())
        self.input_name = self.engine.get_tensor_name(0)
        self.output_names = ['policy_logits', 'value_output', 'soft_policy_logits']

    def __call__(self, input_tensor):
        input_tensor_contiguous = input_tensor.contiguous()
        self.tensors[self.input_name].copy_(input_tensor_contiguous)
        self.context.execute_async_v3(stream_handle=self.stream.cuda_stream)
        self.stream.synchronize()
        policy_logits = self.tensors[self.output_names[0]]
        value_output = self.tensors[self.output_names[1]]
        soft_policy_logits = self.tensors[self.output_names[2]]
        batch_size = input_tensor.shape[0]
        return policy_logits[:batch_size], value_output[:batch_size], soft_policy_logits[:batch_size]


# --- 自对弈运行器 (现在使用 TensorRTModel) ---
class GameBatchRunner:
    def __init__(self, model, device):
        self.device = device
        self.model = model  # 接收一个 TensorRTModel 实例
        self.mcts_manager = c_lib.create_mcts_manager(NUM_PARALLEL_GAMES, True, 0.0)
        self.game_histories = [[] for _ in range(NUM_PARALLEL_GAMES)]
        self.active_games = list(range(NUM_PARALLEL_GAMES))
        self.move_counters = [0] * NUM_PARALLEL_GAMES
        self.current_move_targets = {}
        self.is_full_search_game = {}

    def run(self):
        while self.active_games:
            self._set_search_targets()
            num_requests, board_buffer = self._get_mcts_requests()
            if num_requests > 0:
                self._process_nn_requests(num_requests, board_buffer)
            self._process_completed_searches()
            self.active_games = [idx for idx in self.active_games if
                                 not c_lib.mcts_is_game_over(self.mcts_manager, idx)]
        print(f"一批 {NUM_PARALLEL_GAMES} 局游戏已完成。")
        return self._get_training_data()

    def _process_nn_requests(self, num_requests, board_buffer):
        """使用填充来匹配TensorRT引擎的固定批次大小"""
        input_tensor_np = np.zeros((num_requests, NUM_INPUT_CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        c_lib.boards_to_tensors_c(board_buffer, num_requests,
                                  input_tensor_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
        input_batch = torch.from_numpy(input_tensor_np).to(self.device)

        # 填充数据以匹配最大批次大小
        if num_requests < MAX_BATCH_SIZE:
            padding = torch.zeros(MAX_BATCH_SIZE - num_requests, NUM_INPUT_CHANNELS, BOARD_SIZE, BOARD_SIZE,
                                  device=self.device)
            final_batch = torch.cat([input_batch, padding], dim=0)
        else:
            final_batch = input_batch

        # 使用 TensorRT 模型进行推理
        policies_logits, values, _ = self.model(final_batch)

        # 处理结果
        policies = torch.softmax(policies_logits.float(), dim=1).cpu().numpy()
        values = values.float().cpu().numpy().flatten()

        # 只将有效的结果反馈给C++
        policies_data = policies[:num_requests]
        values_data = values[:num_requests]
        c_lib.mcts_feed_results(
            self.mcts_manager,
            policies_data.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            values_data.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            board_buffer
        )

    def _set_search_targets(self):
        """为所有活跃且没有目标的棋局设置搜索次数"""
        for game_idx in self.active_games:
            if game_idx not in self.current_move_targets:
                if USE_PLAYOUT_CAP_RANDOMIZATION and random.random() > FULL_SEARCH_PROB:
                    self.current_move_targets[game_idx] = FAST_SEARCH_SIMS
                    self.is_full_search_game[game_idx] = False
                else:
                    self.current_move_targets[game_idx] = FULL_SEARCH_SIMS
                    self.is_full_search_game[game_idx] = True

    def _get_mcts_requests(self):
        """从MCTS获取一批需要评估的棋盘状态"""
        board_buffer = (Board * MAX_BATCH_SIZE)()
        request_indices = (ctypes.c_int * MAX_BATCH_SIZE)()
        num_requests = c_lib.mcts_run_simulations_and_get_requests(self.mcts_manager, board_buffer, request_indices,
                                                                   MAX_BATCH_SIZE)
        return num_requests, board_buffer

    def _process_completed_searches(self):
        """检查并处理所有达到模拟次数的游戏"""
        games_that_moved = []
        for game_idx in self.active_games:
            if c_lib.mcts_get_simulations_done(self.mcts_manager, game_idx) >= self.current_move_targets.get(game_idx,
                                                                                                             float('inf')):
                self._select_and_make_move(game_idx)
                games_that_moved.append(game_idx)

        if games_that_moved:
            for game_idx in games_that_moved:
                if game_idx in self.current_move_targets: del self.current_move_targets[game_idx]
                if game_idx in self.is_full_search_game: del self.is_full_search_game[game_idx]

    def _select_and_make_move(self, game_idx):
        """为单个游戏选择并执行一步棋"""
        is_full_search = self.is_full_search_game.get(game_idx, False)

        policy_buffer = (ctypes.c_float * BOARD_SQUARES)()
        c_lib.mcts_get_policy(self.mcts_manager, game_idx, policy_buffer)
        policy_np = np.ctypeslib.as_array(policy_buffer).copy()

        if is_full_search:
            self._save_history(game_idx, policy_np)

        move_count = self.move_counters[game_idx]
        if move_count < TEMPERATURE_DECAY_MOVES:
            move = self._sample_with_temperature(game_idx, policy_np, TEMPERATURE_START)
        else:
            legal_moves_mask = self._get_legal_moves_mask(game_idx)
            masked_policy = policy_np * legal_moves_mask
            if np.sum(masked_policy) > 0:
                move = np.argmax(masked_policy)
            else:
                move = -1

        if move != -1:
            c_lib.mcts_make_move(self.mcts_manager, game_idx, int(move))
            self.move_counters[game_idx] += 1

    def _sample_with_temperature(self, game_idx, policy, temperature):
        """根据温度对策略进行重采样"""
        if temperature == 0:
            return np.argmax(policy)

        policy = np.power(policy, 1.0 / temperature)
        policy /= np.sum(policy)

        legal_moves_mask = self._get_legal_moves_mask(game_idx)
        masked_policy = policy * legal_moves_mask

        if np.sum(masked_policy) < 1e-8:
            legal_indices = np.where(legal_moves_mask > 0.5)[0]
            return np.random.choice(legal_indices) if len(legal_indices) > 0 else -1

        move_probs = masked_policy / np.sum(masked_policy)
        return np.random.choice(range(BOARD_SQUARES), p=move_probs)

    def _get_legal_moves_mask(self, game_idx):
        """获取当前棋局的合法走法掩码"""
        mask_buffer = (ctypes.c_float * BOARD_SQUARES)()
        c_lib.mcts_get_legal_moves_mask(self.mcts_manager, game_idx, mask_buffer)
        return np.ctypeslib.as_array(mask_buffer).copy()

    def _save_history(self, game_idx, policy_np):
        """保存当前状态和策略以用于后续训练"""
        board_state_ptr = c_lib.mcts_get_board_state(self.mcts_manager, game_idx)
        state_tensor_np = np.zeros((1, NUM_INPUT_CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        c_lib.boards_to_tensors_c(board_state_ptr, 1, state_tensor_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
        self.game_histories[game_idx].append(
            (state_tensor_np[0], policy_np, board_state_ptr.contents.current_player)
        )

    def _get_training_data(self):
        """收集所有完成的游戏数据，并附上最终的游戏结果作为价值目标"""
        all_training_data = []
        for game_idx in range(NUM_PARALLEL_GAMES):
            if not self.game_histories[game_idx]: continue

            final_value_for_player_0 = c_lib.mcts_get_final_value(self.mcts_manager, game_idx, 0)

            for state_tensor, policy, player_at_step in self.game_histories[game_idx]:
                value_target = final_value_for_player_0 if player_at_step == 0 else -final_value_for_player_0
                all_training_data.append((state_tensor, policy, value_target))
        return all_training_data

    def __del__(self):
        if hasattr(self, 'mcts_manager') and self.mcts_manager:
            c_lib.destroy_mcts_manager(self.mcts_manager)


if __name__ == "__main__":
    print("开始 TensorRT 自对弈工作脚本...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cpu':
        print("错误: 此脚本需要 CUDA GPU 才能运行 TensorRT 引擎。", file=sys.stderr)
        sys.exit(1)

    try:
        # 直接加载优化后的 TensorRT 引擎
        trt_model = TensorRTModel(TENSORRT_ENGINE_PATH)
        print(f"TensorRT 引擎 ({TENSORRT_ENGINE_PATH}) 加载成功。")
    except Exception as e:
        print(f"错误: 加载 TensorRT 引擎失败: {e}", file=sys.stderr)
        sys.exit(1)

    all_cycles_data = []
    for i in range(TOTAL_GAME_CYCLES):
        print(f"\n--- 开始第 {i + 1}/{TOTAL_GAME_CYCLES} 批次游戏 ---")
        start_time = time.time()
        try:
            runner = GameBatchRunner(trt_model, device)
            cycle_data = runner.run()
            if cycle_data:
                all_cycles_data.extend(cycle_data)
            del runner
        except Exception as e:
            print(f"在批次 {i + 1} 中发生严重错误: {e}", file=sys.stderr)
            import traceback

            traceback.print_exc()
            break
        print(f"批次 {i + 1} 耗时: {time.time() - start_time:.2f} 秒。")

    if all_cycles_data:
        if not os.path.exists(DATA_DIR): os.makedirs(DATA_DIR)
        filename = os.path.join(DATA_DIR, f"selfplay_data_{int(time.time())}.pkl.gz")
        with gzip.open(filename, 'wb') as f:
            pickle.dump(all_cycles_data, f)
        print(f"\n所有数据已合并, 共 {len(all_cycles_data)} 条记录已保存至 {filename}")
    else:
        print("\n所有批次完成，但未生成任何训练数据。")
