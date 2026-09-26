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


def setup_c_library(lib_name=None):
    if lib_name is None:
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
        # [新增] 带不确定度的接口（不确定的 playout 折算为部分访问次数）
        "mcts_feed_results_with_uncertainty": (
        None, [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
               ctypes.POINTER(ctypes.c_float), ctypes.POINTER(Board)]),
        "mcts_get_policy": (ctypes.c_bool, [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_float)]),
        "mcts_make_move": (None, [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]),
        "mcts_is_game_over": (ctypes.c_bool, [ctypes.c_void_p, ctypes.c_int]),
        "get_game_result": (ctypes.c_int, [ctypes.POINTER(Board)]),
        "mcts_get_simulations_done": (ctypes.c_int, [ctypes.c_void_p, ctypes.c_int]),
        "mcts_get_board_state": (ctypes.POINTER(Board), [ctypes.c_void_p, ctypes.c_int]),
        "boards_to_tensors_c": (None, [ctypes.POINTER(Board), ctypes.c_int, ctypes.POINTER(ctypes.c_float)]),
        # [新增] 贴目相关接口
        "boards_to_tensors_with_komi_c": (None, [ctypes.POINTER(Board), ctypes.c_int,
                                                 ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_float)]),
        "mcts_set_komi": (None, [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]),
        "mcts_get_final_value": (ctypes.c_float, [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]),
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
    def __init__(self, session_black, session_white, num_games, simulations, opening_moves_n=0, komi=0,
                 per_side_uncertainty=False):
        self.num_games = num_games
        self.simulations = simulations
        self.opening_moves_n = opening_moves_n
        self.session_black = session_black
        self.session_white = session_white
        self.komi = int(komi)  # [新增] 对局贴目（白给黑），影响搜索终局估值与胜负结算
        # [新增] 检测两侧模型是否带不确定度头（第 4 输出）
        n_out_black = len(session_black.get_outputs())
        n_out_white = len(session_white.get_outputs())
        self._black_uncertainty = n_out_black >= 4
        self._white_uncertainty = n_out_white >= 4
        both = self._black_uncertainty and self._white_uncertainty
        # [修改] 默认行为与原版一致：双方都带头才启用加权接口，否则走旧接口
        # （避免旧模型的 u=0 被误当成"高置信"）。
        # per_side_uncertainty=True（仅配合 compat DLL popucom_core_compat.dll
        # 的 UNCERTAINTY_SENTINEL 分支，见 build_compat.ps1）时：
        # 任一方带头即启用加权接口，无头一侧填负值哨兵（C++ 转为中性反传，
        # 等价旧行为），新旧模型各保留原生行为
        self._per_side = bool(per_side_uncertainty) and not both
        self._use_uncertainty = both or (self._per_side and
                                         (self._black_uncertainty or self._white_uncertainty))
        # 无头一侧的 u 填充值：哨兵模式 -1（compat DLL 转 neutral），普通模式 0（不读取）
        self._no_head_fill = -1.0 if self._per_side else 0.0
        if self._per_side:
            side = []
            if not self._black_uncertainty: side.append("黑方")
            if not self._white_uncertainty: side.append("白方")
            print(f"提示: {'/'.join(side)}为旧模型（无不确定度头），该侧按中性反传（无加权）；"
                  f"带头的一侧仍走不确定度加权（compat DLL 哨兵模式）。")
        elif not self._use_uncertainty:
            print(f"提示: 检测到旧模型（黑方 {n_out_black} 输出 / 白方 {n_out_white} 输出），"
                  f"将使用不带不确定度加权的兼容模式。")
        # [新增] 检测模型输入通道：12 通道（含贴目平面）才填充贴目，旧 11 通道模型保持兼容
        def _input_channels(session):
            try:
                return int(session.get_inputs()[0].shape[1])
            except (TypeError, IndexError, ValueError):
                return NUM_INPUT_CHANNELS
        self._use_komi_channel = (_input_channels(session_black) >= NUM_INPUT_CHANNELS and
                                  _input_channels(session_white) >= NUM_INPUT_CHANNELS)
        if not self._use_komi_channel:
            print("提示: 检测到旧 11 通道模型，输入不填充贴目平面（贴目仅影响终局结算）。")
        self.mcts_manager = c_lib.create_mcts_manager(num_games, False, 0.0)
        for i in range(num_games):
            c_lib.mcts_set_komi(self.mcts_manager, i, self.komi)
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
                # 哨兵模式下默认填 -1（无头一侧 C++ 端转中性反传）；普通模式填 0
                uncertainties = np.full(num_requests, self._no_head_fill, dtype=np.float32)
                if black_requests:
                    p, v, u = self._get_model_outputs(self.session_black, black_requests,
                                                      self._black_uncertainty)
                    policies[black_indices], values[black_indices] = p, v
                    uncertainties[black_indices] = u
                if white_requests:
                    p, v, u = self._get_model_outputs(self.session_white, white_requests,
                                                      self._white_uncertainty)
                    policies[white_indices], values[white_indices] = p, v
                    uncertainties[white_indices] = u
                # 加权接口启用条件见 __init__：默认双方都带头；哨兵模式任一方带头
                if self._use_uncertainty:
                    c_lib.mcts_feed_results_with_uncertainty(
                        self.mcts_manager,
                        policies.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                        values.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                        uncertainties.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), board_buffer)
                else:
                    c_lib.mcts_feed_results(
                        self.mcts_manager,
                        policies.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
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
            # [修改] 贴目感知结算（komi=0 时与旧 get_game_result 等价）
            v_black = c_lib.mcts_get_final_value(self.mcts_manager, i, BLACK)
            if v_black > 0.5: self.results.append(BLACK_WIN)
            elif v_black < -0.5: self.results.append(WHITE_WIN)
            else: self.results.append(DRAW)

        c_lib.destroy_mcts_manager(self.mcts_manager)
        print()
        return self.results

    def _get_model_outputs(self, session, boards, with_uncertainty=True):
        num_boards = len(boards)
        # [修复] 按 session 自身输入通道分配 buffer：旧 11 通道模型不接受 12 通道输入。
        # 混合对打（新 12 通道 vs 旧 11 通道）时逐模型构造，互不干扰
        try:
            n_ch = int(session.get_inputs()[0].shape[1])
        except (TypeError, IndexError, ValueError):
            n_ch = NUM_INPUT_CHANNELS
        input_tensor_np = np.zeros((num_boards, n_ch, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        board_array = (Board * num_boards)(*boards)
        if n_ch >= NUM_INPUT_CHANNELS:
            # 12 通道输入（通道 11 = 贴目平面；无贴目对局时 komi=0，平面全 0）
            komis_np = np.full(num_boards, self.komi, dtype=np.int32)
            c_lib.boards_to_tensors_with_komi_c(
                board_array, num_boards, komis_np.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                input_tensor_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
        else:
            c_lib.boards_to_tensors_c(board_array, num_boards,
                                      input_tensor_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
        ort_inputs = {session.get_inputs()[0].name: input_tensor_np}
        ort_outs = session.run(None, ort_inputs)
        policies_logits, values_output = ort_outs[0], ort_outs[1]
        exp_logits = np.exp(policies_logits - np.max(policies_logits, axis=1, keepdims=True))
        policies = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        values = values_output.flatten()
        # [新增] 第 4 输出为不确定度（Softplus 恒正，abs 防呆）；
        # 旧 3 输出模型无此输出 → 填 self._no_head_fill：
        # 哨兵模式为 -1（compat DLL 转中性反传，等价旧行为），普通模式为 0（接口不读取）
        if with_uncertainty and len(ort_outs) >= 4:
            uncertainties = np.abs(ort_outs[3].flatten()).astype(np.float32)
        else:
            uncertainties = np.full(num_boards, self._no_head_fill, dtype=np.float32)
        return policies, values, uncertainties


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
    if args.komi:
        print(f"对局贴目: 白给黑 {args.komi} 目")
    runner1 = ArenaRunner(session_a, session_b, games_per_matchup, args.simulations,
                          opening_moves_n=opening_moves_n, komi=args.komi)
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
    runner2 = ArenaRunner(session_b, session_a, games_per_matchup, args.simulations,
                          opening_moves_n=opening_moves_n, komi=args.komi)
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
    parser.add_argument("--model_a_path", type=str, default="model.onnx", help="模型A的路径 (.onnx)")
    parser.add_argument("--model_b_path", type=str, default="model_old.onnx", help="模型B的路径 (.onnx)")
    parser.add_argument("--num_games", type=int, default=200, help="总对局数 (必须是偶数)")
    parser.add_argument("--simulations", type=int, default=20, help="每一步的MCTS模拟次数")
    parser.add_argument("--opening_moves", type=int, default=6, help="模型驱动的随机开局步数。设为0则为纯确定性对弈。")
    parser.add_argument("--komi", type=int, default=0, help="对局贴目（白给黑的目数）。0 = 无贴目。")

    args = parser.parse_args()
    main(args)
