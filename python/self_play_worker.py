import ctypes
import os
import platform
import time
import pickle
import gzip
import numpy as np
import torch

# MODIFIED: Import the updated model
from popucom_nn_model import PomPomNN
from popucom_nn_interface import NUM_INPUT_CHANNELS, BOARD_SIZE, MAX_MOVES_PER_PLAYER

# --- 全局配置 ---
MCTS_SIMULATIONS = 1200 # 提高上限
NUM_PARALLEL_GAMES = 512
MAX_BATCH_SIZE = NUM_PARALLEL_GAMES
MODEL_PATH = "model.pth"
DATA_DIR = "self_play_data"
TOTAL_GAME_CYCLES = 7
BOARD_SQUARES = BOARD_SIZE * BOARD_SIZE

# --- 温度参数 ---
TEMPERATURE_DECAY_MOVES = 10
TEMPERATURE_MOVE_SELECTION = 1.0
TEMPERATURE_END = 0.1


# --- C 语言接口定义 ---
class Bitboards(ctypes.Structure): _fields_ = [("parts", ctypes.c_uint64 * 2)]


class Board(ctypes.Structure): _fields_ = [("pieces", Bitboards * 2), ("tiles", Bitboards * 2),
                                           ("current_player", ctypes.c_int), ("moves_left", ctypes.c_int * 2)]


def setup_c_library():
    """加载 C++ 动态库并设置所有函数的参数类型和返回类型。"""
    lib_name = "popucom_core.dll" if platform.system() == "Windows" else "popucom_core.so"
    if not os.path.exists(lib_name):
        raise FileNotFoundError(f"未找到C库 '{lib_name}'。请重新编译C代码。")
    c_lib = ctypes.CDLL(os.path.abspath(lib_name))

    # C函数接口定义
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
    def __init__(self, model, num_games):
        self.num_games = num_games
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device).eval()

        self.mcts_manager = c_lib.create_mcts_manager(num_games, True, 0.02)  # Use a small FPU value
        self.game_histories = [[] for _ in range(num_games)]
        self.move_counts = [0] * num_games
        self.active_games = list(range(num_games))

    def run(self):
        while self.active_games:
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
                        # MODIFIED: Unpack 3 results from the simplified model
                        policies_logits, values, _ = self.model(input_batch)
                    policies = torch.softmax(policies_logits.float(), dim=1).cpu().numpy()
                    values = values.float().cpu().numpy().flatten()

                contiguous_policies = np.ascontiguousarray(policies, dtype=np.float32)
                contiguous_values = np.ascontiguousarray(values, dtype=np.float32)
                c_lib.mcts_feed_results(self.mcts_manager,
                                        contiguous_policies.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                        contiguous_values.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), board_buffer)

            policy_buffer = (ctypes.c_float * BOARD_SQUARES)()
            for game_idx in list(self.active_games):
                if c_lib.mcts_get_simulations_done(self.mcts_manager, game_idx) >= MCTS_SIMULATIONS:
                    c_lib.mcts_get_policy(self.mcts_manager, game_idx, policy_buffer)
                    policy_np = np.ctypeslib.as_array(policy_buffer).copy()

                    board_state_ptr = c_lib.mcts_get_board_state(self.mcts_manager, game_idx)
                    state_tensor_np = np.zeros((1, NUM_INPUT_CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
                    c_lib.boards_to_tensors_c(board_state_ptr, 1,
                                              state_tensor_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))

                    # MODIFIED: The history now only stores state, policy, and player.
                    # The legal moves mask is fetched for move selection but not stored for training.
                    self.game_histories[game_idx].append(
                        (state_tensor_np[0], policy_np, board_state_ptr.contents.current_player)
                    )

                    # --- Move Selection Logic ---
                    # Get legal moves mask for immediate use in move selection
                    legal_moves_mask_buffer = (ctypes.c_float * BOARD_SQUARES)()
                    c_lib.mcts_get_legal_moves_mask(self.mcts_manager, game_idx, legal_moves_mask_buffer)
                    legal_moves_mask_np = np.ctypeslib.as_array(legal_moves_mask_buffer).copy()

                    move = -1
                    if self.move_counts[game_idx] == 0:
                        legal_indices = np.where(legal_moves_mask_np > 0.5)[0]
                        if len(legal_indices) > 0:
                            move = np.random.choice(legal_indices)
                        else:
                            if game_idx in self.active_games: self.active_games.remove(game_idx)
                            continue
                    else:
                        move_selection_temp = TEMPERATURE_MOVE_SELECTION if self.move_counts[
                                                                                game_idx] < TEMPERATURE_DECAY_MOVES else TEMPERATURE_END

                        # Apply policy mask to ensure only legal moves are selected
                        masked_policy = policy_np * legal_moves_mask_np

                        if move_selection_temp > 0:
                            # Use masked policy for temperature-based sampling
                            sum_masked_policy = np.sum(masked_policy)
                            if sum_masked_policy > 1e-8:
                                move_probs = masked_policy / sum_masked_policy
                            else:  # Fallback if all legal moves have zero probability
                                move_probs = legal_moves_mask_np / np.sum(legal_moves_mask_np)

                            # Re-normalize after applying temperature
                            move_probs = move_probs ** (1.0 / move_selection_temp)
                            sum_probs = np.sum(move_probs)
                            if sum_probs > 1e-8:
                                move_probs /= sum_probs
                            else:  # Final fallback
                                move_probs = legal_moves_mask_np / np.sum(legal_moves_mask_np)
                        else:
                            # Temperature is 0, choose the best move from the masked policy
                            move_probs = np.zeros_like(masked_policy)
                            best_move = np.argmax(masked_policy)
                            move_probs[best_move] = 1.0

                        if np.sum(move_probs) > 1e-8:
                            move = np.random.choice(range(BOARD_SQUARES), p=move_probs)
                        else:
                            if game_idx in self.active_games: self.active_games.remove(game_idx)
                            continue

                    c_lib.mcts_make_move(self.mcts_manager, game_idx, int(move))
                    self.move_counts[game_idx] += 1

                    if c_lib.mcts_is_game_over(self.mcts_manager, game_idx):
                        if game_idx in self.active_games: self.active_games.remove(game_idx)

        print("所有并行游戏已完成。")
        all_training_data = []
        for game_idx in range(self.num_games):
            if not self.game_histories[game_idx]: continue

            # MODIFIED: Process the simplified history format
            for state_tensor, policy, player_at_step in self.game_histories[game_idx]:
                final_value = c_lib.mcts_get_final_value(self.mcts_manager, game_idx, player_at_step)
                # MODIFIED: Save the simplified data tuple for training
                all_training_data.append((state_tensor, policy, final_value))

        if all_training_data:
            if not os.path.exists(DATA_DIR):
                os.makedirs(DATA_DIR)
            filename = os.path.join(DATA_DIR, f"batch_{int(time.time())}.pkl.gz")
            with gzip.open(filename, 'wb') as f:
                pickle.dump(all_training_data, f)
            print(f"批处理完成, {len(all_training_data)} 条数据已压缩保存至 {filename}")

    def __del__(self):
        if hasattr(self, 'mcts_manager') and self.mcts_manager:
            c_lib.destroy_mcts_manager(self.mcts_manager)


if __name__ == "__main__":
    print("开始批处理 MCTS 自对弈...")
    try:
        model = PomPomNN()
        model.load_state_dict(torch.load(MODEL_PATH))
        print("模型已加载。")
    except FileNotFoundError:
        model = PomPomNN()
        torch.save(model.state_dict(), MODEL_PATH)
        print("未找到模型，已创建并保存一个随机初始化的新模型。")

    for i in range(TOTAL_GAME_CYCLES):
        print(f"\n--- 开始第 {i + 1}/{TOTAL_GAME_CYCLES} 批次游戏 ---")
        runner = GameBatchRunner(model, NUM_PARALLEL_GAMES)
        runner.run()
