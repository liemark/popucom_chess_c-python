import ctypes
import os
import platform
import numpy as np
import onnxruntime as ort
import argparse
from collections import defaultdict
import sys

from popucom_nn_interface import NUM_INPUT_CHANNELS, BOARD_SIZE


# --- C 语言接口定义 ---
class Bitboards(ctypes.Structure): _fields_ = [("parts", ctypes.c_uint64 * 2)]


class Board(ctypes.Structure): _fields_ = [("pieces", Bitboards * 2), ("tiles", Bitboards * 2),
                                           ("current_player", ctypes.c_int), ("moves_left", ctypes.c_int * 2)]


def setup_c_library():
    lib_name = "popucom_core.dll" if platform.system() == "Windows" else "popucom_core.so"
    if not os.path.exists(lib_name):
        raise FileNotFoundError(f"未找到C库 '{lib_name}'。请编译C代码。")
    c_lib = ctypes.CDLL(os.path.abspath(lib_name))

    C_FUNCTIONS = {
        "init_board": (None, [ctypes.POINTER(Board)]),
        "make_move": (ctypes.c_bool, [ctypes.POINTER(Board), ctypes.c_int]),
        "get_legal_moves": (Bitboards, [ctypes.POINTER(Board)]),
        "is_bit_set": (ctypes.c_bool, [ctypes.POINTER(Bitboards), ctypes.c_int]),
        "create_mcts_manager": (ctypes.c_void_p, [ctypes.c_int, ctypes.c_bool, ctypes.c_double]),
        "destroy_mcts_manager": (None, [ctypes.c_void_p]),
        "mcts_run_simulations_and_get_requests": (
        ctypes.c_int, [ctypes.c_void_p, ctypes.POINTER(Board), ctypes.POINTER(ctypes.c_int), ctypes.c_int]),
        "mcts_feed_results": (
        None, [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(Board)]),
        "mcts_get_policy": (ctypes.c_bool, [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_float)]),
        "mcts_make_move": (None, [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]),
        "mcts_is_game_over": (ctypes.c_bool, [ctypes.c_void_p, ctypes.c_int]),
        "get_game_result": (ctypes.c_int, [ctypes.POINTER(Board)]),
        "mcts_get_simulations_done": (ctypes.c_int, [ctypes.c_void_p, ctypes.c_int]),
        "mcts_get_board_state": (ctypes.POINTER(Board), [ctypes.c_void_p, ctypes.c_int]),
        "boards_to_tensors_c": (None, [ctypes.POINTER(Board), ctypes.c_int, ctypes.POINTER(ctypes.c_float)]),
        "mcts_reset_for_analysis": (None, [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(Board)]),
        "mcts_get_legal_moves_mask": (None, [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_float)])
    }

    for func_name, (restype, argtypes) in C_FUNCTIONS.items():
        if hasattr(c_lib, func_name):
            func = getattr(c_lib, func_name)
            func.restype = restype
            func.argtypes = argtypes
        else:
            print(f"警告: 在C库中未找到函数 '{func_name}'")

    return c_lib


c_lib = setup_c_library()
BLACK, WHITE = 0, 1
BLACK_WIN, WHITE_WIN, DRAW = 1, 2, 0


class ArenaRunner:
    def __init__(self, session_black, session_white, num_games, simulations, opening_moves_n=0):
        self.num_games = num_games
        self.simulations = simulations
        self.opening_moves_n = opening_moves_n
        self.session_black = session_black
        self.session_white = session_white
        self.mcts_manager = c_lib.create_mcts_manager(num_games, False, 0.0)
        self.active_games = list(range(num_games))
        self.results = []
        self.total_games_initial = num_games
        self.move_counts = [0] * num_games

    def run_matches(self):
        while self.active_games:
            board_buffer = (Board * self.num_games)()
            request_indices = (ctypes.c_int * self.num_games)()
            num_requests = c_lib.mcts_run_simulations_and_get_requests(self.mcts_manager, board_buffer, request_indices,
                                                                       self.num_games)

            if num_requests > 0:
                black_requests, white_requests = [], []
                black_indices, white_indices = [], []
                for i in range(num_requests):
                    board = board_buffer[i]
                    if board.current_player == BLACK:
                        black_requests.append(board)
                        black_indices.append(i)
                    else:
                        white_requests.append(board)
                        white_indices.append(i)

                policies = np.zeros((num_requests, BOARD_SIZE * BOARD_SIZE), dtype=np.float32)
                values = np.zeros(num_requests, dtype=np.float32)
                if black_requests:
                    p, v = self._get_model_outputs(self.session_black, black_requests)
                    policies[black_indices], values[black_indices] = p, v
                if white_requests:
                    p, v = self._get_model_outputs(self.session_white, white_requests)
                    policies[white_indices], values[white_indices] = p, v
                c_lib.mcts_feed_results(self.mcts_manager, policies.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                        values.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), board_buffer)

            games_that_moved = []
            for game_idx in list(self.active_games):
                if c_lib.mcts_get_simulations_done(self.mcts_manager, game_idx) >= self.simulations:
                    policy_buffer = (ctypes.c_float * (BOARD_SIZE * BOARD_SIZE))()
                    c_lib.mcts_get_policy(self.mcts_manager, game_idx, policy_buffer)
                    policy = np.ctypeslib.as_array(policy_buffer).copy()

                    legal_moves_mask_buffer = (ctypes.c_float * (BOARD_SIZE * BOARD_SIZE))()
                    c_lib.mcts_get_legal_moves_mask(self.mcts_manager, game_idx, legal_moves_mask_buffer)
                    legal_moves_mask = np.ctypeslib.as_array(legal_moves_mask_buffer).copy()

                    masked_policy = policy * legal_moves_mask

                    move = -1
                    if np.sum(masked_policy) > 1e-8:
                        if self.move_counts[game_idx] < self.opening_moves_n:
                            # High temperature (effectively random choice among legal moves)
                            move_probs = masked_policy / np.sum(masked_policy)
                            move = np.random.choice(range(BOARD_SIZE * BOARD_SIZE), p=move_probs)
                        else:
                            # Deterministic play
                            move = np.argmax(masked_policy)
                    else:
                        legal_indices = np.where(legal_moves_mask > 0.5)[0]
                        if len(legal_indices) > 0:
                            move = np.random.choice(legal_indices)

                    if move != -1:
                        c_lib.mcts_make_move(self.mcts_manager, game_idx, int(move))
                        self.move_counts[game_idx] += 1
                        games_that_moved.append(game_idx)

            if games_that_moved:
                self.active_games = [idx for idx in self.active_games if
                                     not c_lib.mcts_is_game_over(self.mcts_manager, idx)]

            completed_games = self.total_games_initial - len(self.active_games)
            progress = (completed_games / self.total_games_initial) * 100
            sys.stdout.write(f"\r对局进度: {completed_games}/{self.total_games_initial} ({progress:.1f}%)")
            sys.stdout.flush()

        for i in range(self.num_games):
            board_ptr = c_lib.mcts_get_board_state(self.mcts_manager, i)
            self.results.append(c_lib.get_game_result(board_ptr))

        c_lib.destroy_mcts_manager(self.mcts_manager)
        print()
        return self.results

    def _get_model_outputs(self, session, boards):
        num_boards = len(boards)
        input_tensor_np = np.zeros((num_boards, NUM_INPUT_CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        board_array = (Board * num_boards)(*boards)
        c_lib.boards_to_tensors_c(board_array, num_boards,
                                  input_tensor_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
        ort_inputs = {session.get_inputs()[0].name: input_tensor_np}
        ort_outs = session.run(None, ort_inputs)
        policies_logits, values_output = ort_outs[0], ort_outs[1]
        exp_logits = np.exp(policies_logits - np.max(policies_logits, axis=1, keepdims=True))
        policies = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        values = values_output.flatten()
        return policies, values


def load_onnx_session(path):
    try:
        print(f"正在从 {path} 加载ONNX模型...")
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        session = ort.InferenceSession(path, providers=providers)
        print(f"ONNX Runtime将使用: {session.get_providers()[0]}")
        return session
    except Exception as e:
        print(f"错误: 无法加载ONNX模型 {path}: {e}");
        exit()


def print_results(scores, model_a_name, model_b_name):
    print("\n" + "=" * 40 + "\n 对 练 结 果 报 告\n" + "=" * 40)
    print(f"模型A: {model_a_name}\n模型B: {model_b_name}\n总对局数: {sum(scores.values())}\n" + "-" * 40)
    a_wins, b_wins, draws = scores[model_a_name], scores[model_b_name], scores['draw']
    total = a_wins + b_wins + draws
    if total == 0: print("没有完成任何对局。"); return
    a_win_rate, b_win_rate = a_wins / total * 100, b_wins / total * 100
    print(f"{model_a_name} 胜: {a_wins} ({a_win_rate:.2f}%)")
    print(f"{model_b_name} 胜: {b_wins} ({b_win_rate:.2f}%)")
    print(f"平局: {draws}\n等效胜率: {(a_wins + draws * 0.5) / total * 100:.2f}%\n" + "=" * 40)


def main(args):
    if args.num_games % 2 != 0:
        print("总对局数不是偶数。为保证公平，将自动减1。");
        args.num_games -= 1

    session_a = load_onnx_session(args.model_a_path)
    session_b = load_onnx_session(args.model_b_path)
    games_per_matchup = args.num_games // 2
    total_scores = defaultdict(int)

    opening_moves_n = args.opening_moves
    if opening_moves_n > 0:
        print(f"将使用模型驱动的随机 {opening_moves_n} 步开局。")

    print(f"\n--- 开始第一轮: {args.model_a_path} (黑) vs {args.model_b_path} (白) ---")
    runner1 = ArenaRunner(session_a, session_b, games_per_matchup, args.simulations, opening_moves_n=opening_moves_n)
    results1 = runner1.run_matches()
    for res in results1:
        if res == BLACK_WIN:
            total_scores[args.model_a_path] += 1
        elif res == WHITE_WIN:
            total_scores[args.model_b_path] += 1
        else:
            total_scores['draw'] += 1
    print_results(total_scores, args.model_a_path, args.model_b_path)

    print(f"\n--- 开始第二轮: {args.model_b_path} (黑) vs {args.model_a_path} (白) ---")
    runner2 = ArenaRunner(session_b, session_a, games_per_matchup, args.simulations, opening_moves_n=opening_moves_n)
    results2 = runner2.run_matches()
    for res in results2:
        if res == BLACK_WIN:
            total_scores[args.model_b_path] += 1
        elif res == WHITE_WIN:
            total_scores[args.model_a_path] += 1
        else:
            total_scores['draw'] += 1
    print_results(total_scores, args.model_a_path, args.model_b_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="运行两个泡姆棋模型进行对练。")
    parser.add_argument("--model_a_path", type=str, default="model_a.onnx", help="模型A的路径 (.onnx)")
    parser.add_argument("--model_b_path", type=str, default="model_b.onnx", help="模型B的路径 (.onnx)")
    parser.add_argument("--num_games", type=int, default=100, help="总对局数 (必须是偶数)")
    parser.add_argument("--simulations", type=int, default=200, help="每一步的MCTS模拟次数")
    parser.add_argument("--opening_moves", type=int, default=4, help="模型驱动的随机开局步数。设为0则为纯确定性对弈。")

    args = parser.parse_args()
    main(args)
