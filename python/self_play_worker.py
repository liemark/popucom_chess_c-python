import ctypes
import os
import platform
import time
import pickle
import gzip
import numpy as np
import torch
import random

from popucom_nn_model import PomPomNN
from popucom_nn_interface import NUM_INPUT_CHANNELS, BOARD_SIZE, MAX_MOVES_PER_PLAYER

# --- 全局配置 ---
NUM_PARALLEL_GAMES = 512
MAX_BATCH_SIZE = NUM_PARALLEL_GAMES
MODEL_PATH = "model.pth"
DATA_DIR = "self_play_data"
TOTAL_GAME_CYCLES = 20
BOARD_SQUARES = BOARD_SIZE * BOARD_SIZE

# --- MODIFIED: 重新引入走子上限随机化 (PCR) 配置 ---
USE_PLAYOUT_CAP_RANDOMIZATION = True
FULL_SEARCH_SIMS = 1000  # 完整搜索的模拟次数
FAST_SEARCH_SIMS = 100  # 快速搜索的模拟次数
FULL_SEARCH_PROB = 0.25  # 进行完整搜索的概率

# --- 温度参数 ---
TEMPERATURE_DECAY_MOVES = 10
TEMPERATURE_MOVE_SELECTION = 1.0
TEMPERATURE_END = 0.1


# --- C 语言接口定义 ---
class Bitboards(ctypes.Structure): _fields_ = [("parts", ctypes.c_uint64 * 2)]


class Board(ctypes.Structure): _fields_ = [("pieces", Bitboards * 2), ("tiles", Bitboards * 2),
                                           ("current_player", ctypes.c_int), ("moves_left", ctypes.c_int * 2)]


def setup_c_library():
    lib_name = "popucom_core.dll" if platform.system() == "Windows" else "popucom_core.so"
    if not os.path.exists(lib_name):
        raise FileNotFoundError(f"未找到C库 '{lib_name}'。请重新编译C代码。")
    c_lib = ctypes.CDLL(os.path.abspath(lib_name))
    c_lib.create_mcts_manager.argtypes = [ctypes.c_int, ctypes.c_bool, ctypes.c_double]
    c_lib.create_mcts_manager.restype = ctypes.c_void_p
    c_lib.mcts_set_fpu.argtypes = [ctypes.c_void_p, ctypes.c_double]
    c_lib.mcts_set_fpu.restype = None
    c_lib.boards_to_tensors_c.argtypes = [ctypes.POINTER(Board), ctypes.c_int, ctypes.POINTER(ctypes.c_float)]
    c_lib.mcts_feed_results.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                                        ctypes.POINTER(Board)]
    c_lib.destroy_mcts_manager.argtypes = [ctypes.c_void_p]
    c_lib.mcts_run_simulations_and_get_requests.argtypes = [ctypes.c_void_p, ctypes.POINTER(Board),
                                                            ctypes.POINTER(ctypes.c_int), ctypes.c_int]
    c_lib.mcts_run_simulations_and_get_requests.restype = ctypes.c_int
    c_lib.mcts_get_policy.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_float)]
    c_lib.mcts_get_policy.restype = ctypes.c_bool
    c_lib.mcts_make_move.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    c_lib.mcts_is_game_over.argtypes = [ctypes.c_void_p, ctypes.c_int]
    c_lib.mcts_is_game_over.restype = ctypes.c_bool
    c_lib.mcts_get_final_value.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    c_lib.mcts_get_final_value.restype = ctypes.c_float
    c_lib.mcts_get_board_state.argtypes = [ctypes.c_void_p, ctypes.c_int]
    c_lib.mcts_get_board_state.restype = ctypes.POINTER(Board)
    c_lib.mcts_get_simulations_done.argtypes = [ctypes.c_void_p, ctypes.c_int]
    c_lib.mcts_get_simulations_done.restype = ctypes.c_int
    c_lib.pop_count.argtypes = [ctypes.POINTER(Bitboards)]
    c_lib.pop_count.restype = ctypes.c_int
    c_lib.mcts_set_noise_enabled.argtypes = [ctypes.c_void_p, ctypes.c_bool]
    c_lib.mcts_get_legal_moves_mask.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_float)]
    c_lib.mcts_get_legal_moves_mask.restype = None
    return c_lib


c_lib = setup_c_library()


class GameBatchRunner:
    def __init__(self, model):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device).eval()
        self.mcts_manager = c_lib.create_mcts_manager(NUM_PARALLEL_GAMES, True, 0.0)
        self.game_histories = [[] for _ in range(NUM_PARALLEL_GAMES)]
        self.active_games = list(range(NUM_PARALLEL_GAMES))

        # 状态管理字典
        self.current_move_targets = {}
        self.is_full_search_game = {}

    def run(self):
        """MODIFIED: 引入高效的、非阻塞的PCR循环"""
        while self.active_games:
            # 1. 为所有需要确定搜索模式的游戏设置目标
            for game_idx in self.active_games:
                if game_idx not in self.current_move_targets:
                    if USE_PLAYOUT_CAP_RANDOMIZATION and random.random() > FULL_SEARCH_PROB:
                        self.current_move_targets[game_idx] = FAST_SEARCH_SIMS
                        self.is_full_search_game[game_idx] = False
                    else:
                        self.current_move_targets[game_idx] = FULL_SEARCH_SIMS
                        self.is_full_search_game[game_idx] = True

            # 2. 运行一轮MCTS请求和响应
            board_buffer = (Board * MAX_BATCH_SIZE)()
            request_indices = (ctypes.c_int * MAX_BATCH_SIZE)()
            num_requests = c_lib.mcts_run_simulations_and_get_requests(self.mcts_manager, board_buffer, request_indices,
                                                                       MAX_BATCH_SIZE)

            if num_requests > 0:
                input_tensor_np = np.zeros((num_requests, NUM_INPUT_CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
                c_lib.boards_to_tensors_c(board_buffer, num_requests,
                                          input_tensor_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))

                input_batch = torch.from_numpy(input_tensor_np).to(self.device)
                with torch.no_grad():
                    use_amp = self.device.type == 'cuda'
                    with torch.amp.autocast(device_type=self.device.type, enabled=use_amp):
                        policies_logits, values, _ = self.model(input_batch)
                    policies = torch.softmax(policies_logits.float(), dim=1).cpu().numpy()
                    values = values.float().cpu().numpy().flatten()

                c_lib.mcts_feed_results(self.mcts_manager, policies.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                        values.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), board_buffer)

            # 3. 检查并处理所有达到模拟次数的游戏
            games_that_moved = []
            for game_idx in self.active_games:
                if c_lib.mcts_get_simulations_done(self.mcts_manager, game_idx) >= self.current_move_targets.get(
                        game_idx, float('inf')):
                    self._process_move_for_game(game_idx)
                    games_that_moved.append(game_idx)

            # 4. 清理已移动游戏的状态，并更新活跃游戏列表
            if games_that_moved:
                for game_idx in games_that_moved:
                    if game_idx in self.current_move_targets: del self.current_move_targets[game_idx]
                    if game_idx in self.is_full_search_game: del self.is_full_search_game[game_idx]

                self.active_games = [idx for idx in self.active_games if
                                     not c_lib.mcts_is_game_over(self.mcts_manager, idx)]

        print("一个批次的并行游戏已完成。")
        return self._get_training_data()

    def _process_move_for_game(self, game_idx):
        is_full_search = self.is_full_search_game.get(game_idx, False)
        policy_buffer = (ctypes.c_float * BOARD_SQUARES)()
        c_lib.mcts_get_policy(self.mcts_manager, game_idx, policy_buffer)
        policy_np = np.ctypeslib.as_array(policy_buffer).copy()

        # 仅在完整搜索时记录训练数据
        if is_full_search:
            board_state_ptr = c_lib.mcts_get_board_state(self.mcts_manager, game_idx)
            state_tensor_np = np.zeros((1, NUM_INPUT_CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
            c_lib.boards_to_tensors_c(board_state_ptr, 1,
                                      state_tensor_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
            self.game_histories[game_idx].append(
                (state_tensor_np[0], policy_np, board_state_ptr.contents.current_player))

        legal_moves_mask_buffer = (ctypes.c_float * BOARD_SQUARES)()
        c_lib.mcts_get_legal_moves_mask(self.mcts_manager, game_idx, legal_moves_mask_buffer)
        legal_moves_mask_np = np.ctypeslib.as_array(legal_moves_mask_buffer).copy()

        masked_policy = policy_np * legal_moves_mask_np
        if np.sum(masked_policy) < 1e-8:
            legal_indices = np.where(legal_moves_mask_np > 0.5)[0]
            move = np.random.choice(legal_indices) if len(legal_indices) > 0 else -1
        else:
            move_probs = masked_policy / np.sum(masked_policy)
            move = np.random.choice(range(BOARD_SQUARES), p=move_probs)

        if move != -1:
            c_lib.mcts_make_move(self.mcts_manager, game_idx, int(move))

    def _get_training_data(self):
        all_training_data = []
        for game_idx in range(NUM_PARALLEL_GAMES):
            if not self.game_histories[game_idx]: continue
            for state_tensor, policy, player_at_step in self.game_histories[game_idx]:
                final_value = c_lib.mcts_get_final_value(self.mcts_manager, game_idx, player_at_step)
                all_training_data.append((state_tensor, policy, final_value))
        return all_training_data

    def __del__(self):
        if hasattr(self, 'mcts_manager') and self.mcts_manager:
            c_lib.destroy_mcts_manager(self.mcts_manager)


if __name__ == "__main__":
    print("开始批处理 MCTS 自对弈 (PyTorch + PCR优化版)...")

    try:
        model = PomPomNN()
        model.load_state_dict(torch.load(MODEL_PATH))
        print("模型已加载。")
    except FileNotFoundError:
        model = PomPomNN()
        torch.save(model.state_dict(), MODEL_PATH)
        print("未找到模型，已创建并保存一个随机初始化的新模型。")

    all_cycles_data = []
    for i in range(TOTAL_GAME_CYCLES):
        print(f"\n--- 开始第 {i + 1}/{TOTAL_GAME_CYCLES} 批次游戏 ---")
        try:
            runner = GameBatchRunner(model)
            cycle_data = runner.run()
            if cycle_data:
                all_cycles_data.extend(cycle_data)
                print(f"批次 {i + 1} 完成，获得 {len(cycle_data)} 条数据。当前总数据量: {len(all_cycles_data)}")
            del runner
        except Exception as e:
            print(f"在批次 {i + 1} 中发生严重错误: {e}")
            print("正在终止自对弈...")
            break

    if all_cycles_data:
        if not os.path.exists(DATA_DIR):
            os.makedirs(DATA_DIR)

        filename = os.path.join(DATA_DIR, f"mega_batch_{int(time.time())}.pkl.gz")
        with gzip.open(filename, 'wb') as f:
            pickle.dump(all_cycles_data, f)
        print(f"\n所有完成的批次数据已合并, 共 {len(all_cycles_data)} 条数据已保存至 {filename}")
    else:
        print("\n所有批次完成，但未生成任何数据。")
