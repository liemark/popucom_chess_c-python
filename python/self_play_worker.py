import ctypes
import os
import platform
import time
import pickle
import numpy as np
import torch

from popucom_nn_model import PomPomNN, BOARD_SIZE, NUM_INPUT_CHANNELS

MAX_MOVES_PER_PLAYER = 25

# --- 全局配置 ---
MCTS_SIMULATIONS = 400
NUM_PARALLEL_GAMES = 128
MAX_BATCH_SIZE = NUM_PARALLEL_GAMES
MODEL_PATH = "model.pth"
DATA_DIR = "self_play_data"
TOTAL_GAME_CYCLES = 6
TEMPERATURE_MOVE_SELECTION = 1.0
TEMPERATURE_DECAY_MOVES = 20
TEMPERATURE_END = 0.1


# --- C 语言接口定义 ---
class Bitboards(ctypes.Structure): _fields_ = [("parts", ctypes.c_uint64 * 2)]


class Board(ctypes.Structure): _fields_ = [("pieces", Bitboards * 2), ("tiles", Bitboards * 2),
                                           ("current_player", ctypes.c_int), ("moves_left", ctypes.c_int * 2)]


def setup_c_library():
    lib_name = "popucom_core.dll" if platform.system() == "Windows" else "popucom_core.so"
    if not os.path.exists(lib_name): raise FileNotFoundError(f"未找到C库 '{lib_name}'")
    c_lib = ctypes.CDLL(os.path.abspath(lib_name))

    func_defs = {
        "create_mcts_manager": (ctypes.c_void_p, [ctypes.c_int]),
        "destroy_mcts_manager": (None, [ctypes.c_void_p]),
        "mcts_run_simulations_and_get_requests": (
        ctypes.c_int, [ctypes.c_void_p, ctypes.POINTER(Board), ctypes.POINTER(ctypes.c_int), ctypes.c_int]),
        "mcts_feed_results": (None, [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                                     ctypes.POINTER(ctypes.c_float)]),
        "mcts_get_policy": (ctypes.c_bool, [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_float)]),
        "mcts_make_move": (None, [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]),
        "mcts_is_game_over": (ctypes.c_bool, [ctypes.c_void_p, ctypes.c_int]),
        "mcts_get_unweighted_simulations_done": (ctypes.c_int, [ctypes.c_void_p, ctypes.c_int]),
        "mcts_get_board_state": (ctypes.POINTER(Board), [ctypes.c_void_p, ctypes.c_int]),
        "get_score_diff": (ctypes.c_int, [ctypes.POINTER(Board)])
    }

    for name, (restype, argtypes) in func_defs.items():
        if hasattr(c_lib, name):
            func = getattr(c_lib, name)
            func.restype, func.argtypes = restype, argtypes
        else:
            print(f"警告: C库中未找到函数 '{name}'")
    return c_lib


c_lib = setup_c_library()


class GameBatchRunner:
    def __init__(self, model, num_games):
        self.num_games = num_games
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device).eval()
        self.mcts_manager = c_lib.create_mcts_manager(num_games)
        if not self.mcts_manager:
            print("\nFATAL ERROR: MCTS manager creation failed (out of memory?).")
            exit(1)
        self.game_histories = [[] for _ in range(num_games)]
        self.move_counts = [0] * num_games
        self.active_games = list(range(num_games))

    def _run_mcts_for_games(self, game_indices, num_sims):
        for i in range(num_sims):
            if not self.active_games: break

            board_buffer = (Board * len(game_indices))()
            request_indices_buffer = (ctypes.c_int * len(game_indices))()
            num_requests = c_lib.mcts_run_simulations_and_get_requests(self.mcts_manager, board_buffer,
                                                                       request_indices_buffer, len(game_indices))

            if num_requests > 0:
                batch_tensors = [self.board_to_tensor(board_buffer[j]) for j in range(num_requests)]
                input_batch = torch.from_numpy(np.array(batch_tensors)).to(self.device)
                with torch.no_grad():
                    policies_logits, values_raw, _, _, uncertainties_raw = self.model(input_batch)
                    policies = torch.softmax(policies_logits, dim=1).cpu().numpy()
                    values = values_raw.cpu().numpy()
                    uncertainties = uncertainties_raw.cpu().numpy()
                c_lib.mcts_feed_results(
                    self.mcts_manager,
                    policies.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    values.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    uncertainties.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
                )

    def run(self):
        while self.active_games:
            self._run_mcts_for_games(self.active_games, MCTS_SIMULATIONS)

            policy_buffer = (ctypes.c_float * (BOARD_SIZE * BOARD_SIZE))()
            for game_idx in list(self.active_games):
                if c_lib.mcts_get_unweighted_simulations_done(self.mcts_manager, game_idx) >= MCTS_SIMULATIONS:
                    c_lib.mcts_get_policy(self.mcts_manager, game_idx, policy_buffer)
                    policy_np = np.ctypeslib.as_array(policy_buffer).copy()

                    board_state_ptr = c_lib.mcts_get_board_state(self.mcts_manager, game_idx)
                    current_player = board_state_ptr.contents.current_player
                    state_tensor = self.board_to_tensor(board_state_ptr.contents)
                    self.game_histories[game_idx].append((state_tensor, policy_np, current_player))

                    # --- FIX: Robust move selection logic ---
                    move_selection_temp = TEMPERATURE_MOVE_SELECTION if self.move_counts[
                                                                            game_idx] < TEMPERATURE_DECAY_MOVES else TEMPERATURE_END
                    if move_selection_temp > 0:
                        move_probs = policy_np ** (1.0 / move_selection_temp)
                    else:
                        move_probs = np.zeros_like(policy_np)
                        if np.sum(policy_np) > 0:
                            move_probs[np.argmax(policy_np)] = 1.0

                    sum_probs = np.sum(move_probs)
                    if sum_probs > 1e-8:
                        # This is the normal path: normalize probabilities and choose a move
                        move_probs /= sum_probs
                        move = np.random.choice(range(BOARD_SIZE * BOARD_SIZE), p=move_probs)
                    else:
                        # This path is taken if no legal moves exist or policy is all zero.
                        # We safely remove the game and continue to the next one.
                        if game_idx in self.active_games:
                            self.active_games.remove(game_idx)
                        continue

                    c_lib.mcts_make_move(self.mcts_manager, game_idx, int(move))
                    self.move_counts[game_idx] += 1

                    if c_lib.mcts_is_game_over(self.mcts_manager, game_idx):
                        if game_idx in self.active_games:
                            self.active_games.remove(game_idx)

        print("所有并行游戏已完成。")
        all_training_data = []
        for game_idx in range(self.num_games):
            final_board_state_ptr = c_lib.mcts_get_board_state(self.mcts_manager, game_idx)
            if not final_board_state_ptr: continue
            raw_score_diff = c_lib.get_score_diff(final_board_state_ptr)

            for state_tensor, policy, player_at_step in self.game_histories[game_idx]:
                normalized_score = float(raw_score_diff) / (BOARD_SIZE * BOARD_SIZE)
                final_value = normalized_score if player_at_step == 0 else -normalized_score
                ownership_target = self.calculate_ownership_target(final_board_state_ptr.contents)
                all_training_data.append((state_tensor, policy, final_value, ownership_target))

        if all_training_data:
            if not os.path.exists(DATA_DIR): os.makedirs(DATA_DIR)
            filename = os.path.join(DATA_DIR, f"batch_{int(time.time())}.pkl")
            with open(filename, 'wb') as f:
                pickle.dump(all_training_data, f)
            print(f"批处理完成, {len(all_training_data)} 条数据已保存至 {filename}")

    def __del__(self):
        if hasattr(self, 'mcts_manager') and self.mcts_manager:
            c_lib.destroy_mcts_manager(self.mcts_manager)

    def board_to_tensor(self, board_c: Board) -> np.ndarray:
        tensor = np.zeros((NUM_INPUT_CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        p, o = board_c.current_player, 1 - board_c.current_player

        def get_plane(bb):
            plane = np.zeros(BOARD_SIZE * BOARD_SIZE, dtype=np.float32)
            for i in range(BOARD_SIZE * BOARD_SIZE):
                if (bb.parts[i // 64] >> (i % 64)) & 1: plane[i] = 1.0
            return plane.reshape((BOARD_SIZE, BOARD_SIZE))

        tensor[0, :, :] = get_plane(board_c.pieces[p]);
        tensor[1, :, :] = get_plane(board_c.pieces[o])
        tensor[2, :, :] = get_plane(board_c.tiles[p]);
        tensor[3, :, :] = get_plane(board_c.tiles[o])
        tensor[4, :, :] = 1. if p == 0 else 0.;
        tensor[5, :, :] = 1. if p == 1 else 0.
        tensor[6, :, :] = float(board_c.moves_left[0]) / MAX_MOVES_PER_PLAYER
        tensor[7, :, :] = float(board_c.moves_left[1]) / MAX_MOVES_PER_PLAYER
        tensor[8, :, :] = float(c_lib.pop_count(ctypes.byref(board_c.tiles[0]))) / (BOARD_SIZE * BOARD_SIZE)
        tensor[9, :, :] = float(c_lib.pop_count(ctypes.byref(board_c.tiles[1]))) / (BOARD_SIZE * BOARD_SIZE)
        all_tiles = Bitboards();
        all_tiles.parts[0] = ~(board_c.tiles[0].parts[0] | board_c.tiles[1].parts[0]);
        all_tiles.parts[1] = ~(board_c.tiles[0].parts[1] | board_c.tiles[1].parts[1])
        tensor[10, :, :] = get_plane(all_tiles)
        return tensor

    def calculate_ownership_target(self, final_board_c: Board) -> np.ndarray:
        ownership = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        for sq in range(BOARD_SIZE * BOARD_SIZE):
            r, c = sq // BOARD_SIZE, sq % BOARD_SIZE
            if (final_board_c.tiles[0].parts[sq // 64] >> (sq % 64)) & 1:
                ownership[r, c] = 1.0
            elif (final_board_c.tiles[1].parts[sq // 64] >> (sq % 64)) & 1:
                ownership[r, c] = -1.0
        return ownership


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
