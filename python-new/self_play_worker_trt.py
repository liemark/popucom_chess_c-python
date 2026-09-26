import ctypes
import os
import platform
import time
import pickle
import gzip
import numpy as np
import torch
import sys
import tensorrt as trt
import random  # MODIFIED: 导入 random 模块

# 导入接口常量
from popucom_nn_interface import NUM_INPUT_CHANNELS, BOARD_SIZE
from adaptive_komi import AdaptiveKomiSampler

# --- 配置 ---
TENSORRT_ENGINE_PATH = "model.plan"  # TensorRT引擎文件路径
DATA_DIR = "self_play_data"
BOARD_SQUARES = BOARD_SIZE * BOARD_SIZE

# --- 并行与批处理配置 ---
NUM_PARALLEL_GAMES = 512
MAX_BATCH_SIZE = NUM_PARALLEL_GAMES

# --- 训练周期配置 ---
TOTAL_GAME_CYCLES = 7

# --- Playout Cap Randomization (PCR) 配置 ---
# MODIFIED: 采用新的PCR配置
# 在每个落子前，AI会根据以下概率随机选择一个模拟次数。
# 最后仅保存模拟次数最高的局面
PCR_OPTIONS = [
    (300, 1),  # (模拟次数, 概率) -> 75% 的概率搜索 400 次
    #(400, 0.5), # (模拟次数, 概率) -> 25% 的概率搜索 1200 次
]

# REMOVED: 移除了旧的分阶段搜索配置
# OPENING_PHASE_MOVES = 30
# OPENING_SIMS = 300
# ENDGAME_SIMS = 1000

# --- 走法选择温度配置 ---
TEMPERATURE_START = 0.5
TEMPERATURE_DECAY_MOVES = 10


# --- C 语言接口定义 ---
class Bitboards(ctypes.Structure): _fields_ = [("parts", ctypes.c_uint64 * 2)]


class Board(ctypes.Structure): _fields_ = [("pieces", Bitboards * 2), ("tiles", Bitboards * 2),
                                           ("current_player", ctypes.c_int), ("moves_left", ctypes.c_int * 2)]


def setup_c_library():
    lib_name = "popucom_core.dll" if platform.system() == "Windows" else "popucom_core.so"
    if not os.path.exists(lib_name): raise FileNotFoundError(f"未找到C库 '{lib_name}'")
    c_lib = ctypes.CDLL(os.path.abspath(lib_name))
    c_lib.create_mcts_manager.argtypes = [ctypes.c_int, ctypes.c_bool, ctypes.c_double];
    c_lib.create_mcts_manager.restype = ctypes.c_void_p
    c_lib.destroy_mcts_manager.argtypes = [ctypes.c_void_p]
    c_lib.mcts_run_simulations_and_get_requests.argtypes = [ctypes.c_void_p, ctypes.POINTER(Board),
                                                            ctypes.POINTER(ctypes.c_int), ctypes.c_int];
    c_lib.mcts_run_simulations_and_get_requests.restype = ctypes.c_int
    c_lib.mcts_feed_results.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                                        ctypes.POINTER(Board)]
    # [新增] 带不确定度的接口声明
    c_lib.mcts_feed_results_with_uncertainty.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float),
                                                         ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                                                         ctypes.POINTER(Board)]
    c_lib.boards_to_tensors_c.argtypes = [ctypes.POINTER(Board), ctypes.c_int, ctypes.POINTER(ctypes.c_float)]
    # [新增] 贴目相关接口
    c_lib.mcts_set_komi.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    c_lib.mcts_get_score_diff.argtypes = [ctypes.c_void_p, ctypes.c_int]
    c_lib.mcts_get_score_diff.restype = ctypes.c_int
    c_lib.boards_to_tensors_with_komi_c.argtypes = [ctypes.POINTER(Board), ctypes.c_int,
                                                    ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_float)]
    c_lib.mcts_get_policy.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_float)]
    c_lib.mcts_get_legal_moves_mask.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_float)]
    c_lib.mcts_get_final_value.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int];
    c_lib.mcts_get_final_value.restype = ctypes.c_float
    c_lib.mcts_get_simulations_done.argtypes = [ctypes.c_void_p, ctypes.c_int];
    c_lib.mcts_get_simulations_done.restype = ctypes.c_int
    c_lib.mcts_make_move.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    c_lib.mcts_is_game_over.argtypes = [ctypes.c_void_p, ctypes.c_int];
    c_lib.mcts_is_game_over.restype = ctypes.c_bool
    c_lib.mcts_get_board_state.argtypes = [ctypes.c_void_p, ctypes.c_int];
    c_lib.mcts_get_board_state.restype = ctypes.POINTER(Board)
    return c_lib


c_lib = setup_c_library()


# --- TensorRT 推理辅助类 ---
class TensorRTModel:
    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        if not os.path.exists(engine_path):
            raise FileNotFoundError(f"TensorRT引擎文件未找到: {engine_path}")
        with open(engine_path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()
        self.stream = torch.cuda.Stream()

        self.tensors = {}
        self.input_name = ""
        self.output_names = []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = self.engine.get_tensor_shape(name)
            shape[0] = MAX_BATCH_SIZE
            dtype = torch.from_numpy(np.array([], dtype=trt.nptype(self.engine.get_tensor_dtype(name)))).dtype
            device_mem = torch.empty(tuple(shape), dtype=dtype, device='cuda')
            self.tensors[name] = device_mem
            self.context.set_tensor_address(name, device_mem.data_ptr())

            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_name = name
            else:
                self.output_names.append(name)

        # [修复+新增] 不再依赖排序后的下标（旧代码假设的字母序与实际不符，
        # 曾导致策略头与软策略头互换），改为按名字直接查找输出张量。
        # 引擎必须包含 4 个输出（含 uncertainty_output），否则要求重新导出模型。

    def _get_output(self, name):
        if name not in self.tensors:
            raise KeyError(f"TensorRT 引擎输出中不存在 '{name}'，"
                           f"现有输出: {list(self.tensors.keys())}。请重新导出模型。")
        return self.tensors[name]

    def __call__(self, input_tensor):
        input_tensor_contiguous = input_tensor.contiguous()
        self.tensors[self.input_name].copy_(input_tensor_contiguous)
        self.context.execute_async_v3(stream_handle=self.stream.cuda_stream)
        self.stream.synchronize()

        policy_logits = self._get_output('policy_logits')
        value_output = self._get_output('value_output')
        soft_policy_logits = self._get_output('soft_policy_logits')
        uncertainty_output = self._get_output('uncertainty_output')  # [新增] 第 4 输出
        score_output = self._get_output('score_output')  # [新增] 第 5 输出：贴目调整后净胜目数

        batch_size = input_tensor.shape[0]
        return (policy_logits[:batch_size], value_output[:batch_size],
                soft_policy_logits[:batch_size], uncertainty_output[:batch_size],
                score_output[:batch_size])


# --- 自对弈运行器 ---
class GameBatchRunner:
    def __init__(self, model, device, komi_sampler=None):
        self.device = device
        self.model = model
        self.mcts_manager = c_lib.create_mcts_manager(NUM_PARALLEL_GAMES, True, 0.0)
        self.game_histories = [[] for _ in range(NUM_PARALLEL_GAMES)]
        self.active_games = list(range(NUM_PARALLEL_GAMES))
        self.move_counters = [0] * NUM_PARALLEL_GAMES
        self.current_move_targets = {}

        # [新增] 自适应随机贴目：每局开赛前按胜率核加权采样，整局固定
        self.komi_sampler = komi_sampler if komi_sampler is not None else AdaptiveKomiSampler()
        print(self.komi_sampler.status_str())
        self.game_komi = [self.komi_sampler.sample_komi() for _ in range(NUM_PARALLEL_GAMES)]
        for game_idx in range(NUM_PARALLEL_GAMES):
            c_lib.mcts_set_komi(self.mcts_manager, game_idx, self.game_komi[game_idx])

        # MODIFIED: 在类初始化时预处理PCR选项以提高效率
        self.pcr_sims = [opt[0] for opt in PCR_OPTIONS]
        self.pcr_weights = [opt[1] for opt in PCR_OPTIONS]

        # MODIFIED: 计算最大的搜索次数，用于后续过滤数据
        self.max_pcr_sims = max(self.pcr_sims)
        print(f"PCR配置已加载，仅保存搜索次数达到 {self.max_pcr_sims} 次的局面数据。")

    def run(self):
        while self.active_games:
            self._set_search_targets()
            num_requests, board_buffer, request_indices = self._get_mcts_requests()
            if num_requests > 0:
                self._process_nn_requests(num_requests, board_buffer, request_indices)
            self._process_completed_searches()
            self.active_games = [idx for idx in self.active_games if
                                 not c_lib.mcts_is_game_over(self.mcts_manager, idx)]
        print(f"一批 {NUM_PARALLEL_GAMES} 局游戏已完成。")
        return self._get_training_data()

    def _process_nn_requests(self, num_requests, board_buffer, request_indices):
        # [修改] 12 通道输入（通道 11 = 贴目平面），按请求所属对局的贴目填充
        input_tensor_np = np.zeros((num_requests, NUM_INPUT_CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        komis_np = np.array([self.game_komi[request_indices[i]] for i in range(num_requests)], dtype=np.int32)
        c_lib.boards_to_tensors_with_komi_c(
            board_buffer, num_requests, komis_np.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            input_tensor_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
        input_batch = torch.from_numpy(input_tensor_np).to(self.device)

        if num_requests < MAX_BATCH_SIZE:
            padding = torch.zeros(MAX_BATCH_SIZE - num_requests, NUM_INPUT_CHANNELS, BOARD_SIZE, BOARD_SIZE,
                                  device=self.device)
            final_batch = torch.cat([input_batch, padding], dim=0)
        else:
            final_batch = input_batch

        policies_logits, values, _, uncertainties, _scores = self.model(final_batch)
        policies = torch.softmax(policies_logits.float(), dim=1).cpu().numpy()
        values = values.float().cpu().numpy().flatten()
        # [新增] 不确定度（Softplus 理论上恒正，abs 仅作防呆）
        uncertainties = torch.abs(uncertainties.float()).cpu().numpy().flatten()

        c_lib.mcts_feed_results_with_uncertainty(self.mcts_manager,
                                                 policies[:num_requests].ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                                 values[:num_requests].ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                                 uncertainties[:num_requests].ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                                 board_buffer)

    def _set_search_targets(self):
        """MODIFIED: 为所有活跃且没有目标的棋局设置随机化的搜索次数"""
        for game_idx in self.active_games:
            if game_idx not in self.current_move_targets:
                # 使用 random.choices 根据权重随机选择一个模拟次数
                chosen_sims = random.choices(self.pcr_sims, weights=self.pcr_weights, k=1)[0]
                self.current_move_targets[game_idx] = chosen_sims


    def _get_mcts_requests(self):
        board_buffer = (Board * MAX_BATCH_SIZE)()
        request_indices = (ctypes.c_int * MAX_BATCH_SIZE)()
        num_requests = c_lib.mcts_run_simulations_and_get_requests(self.mcts_manager, board_buffer, request_indices,
                                                                   MAX_BATCH_SIZE)
        return num_requests, board_buffer, request_indices

    def _process_completed_searches(self):
        games_that_moved = []
        for game_idx in self.active_games:
            if c_lib.mcts_get_simulations_done(self.mcts_manager, game_idx) >= self.current_move_targets.get(game_idx,
                                                                                                             float('inf')):
                self._select_and_make_move(game_idx)
                games_that_moved.append(game_idx)
        if games_that_moved:
            for game_idx in games_that_moved:
                if game_idx in self.current_move_targets: del self.current_move_targets[game_idx]

    def _select_and_make_move(self, game_idx):
        policy_buffer = (ctypes.c_float * BOARD_SQUARES)();
        c_lib.mcts_get_policy(self.mcts_manager, game_idx, policy_buffer)
        policy_np = np.ctypeslib.as_array(policy_buffer).copy()

        # MODIFIED: 数据过滤逻辑
        # 获取当前这一步所设定的目标搜索次数
        current_target = self.current_move_targets.get(game_idx, 0)

        # 仅当当前步骤的搜索次数是"深搜"（即等于最大PCR次数，如1000次）时，才保存数据。
        # 如果是200次的浅搜，数据将被丢弃，不用于训练，但游戏会继续进行。
        if current_target >= self.max_pcr_sims:
            self._save_history(game_idx, policy_np)

        move_count = self.move_counters[game_idx]
        temp = TEMPERATURE_START if move_count < TEMPERATURE_DECAY_MOVES else 0.0
        move = self._sample_with_temperature(game_idx, policy_np, temp)

        if move != -1:
            c_lib.mcts_make_move(self.mcts_manager, game_idx, int(move))
            self.move_counters[game_idx] += 1

    def _sample_with_temperature(self, game_idx, policy, temperature):
        legal_moves_mask = self._get_legal_moves_mask(game_idx)
        masked_policy = policy * legal_moves_mask
        if np.sum(masked_policy) < 1e-8:
            legal_indices = np.where(legal_moves_mask > 0.5)[0]
            return np.random.choice(legal_indices) if len(legal_indices) > 0 else -1

        if temperature == 0:
            return np.argmax(masked_policy)
        else:
            # 使用更安全的方式避免除零错误
            masked_policy_pow = np.power(masked_policy, 1.0 / temperature)
            sum_probs = np.sum(masked_policy_pow)
            if sum_probs < 1e-8:
                return np.random.choice(np.where(legal_moves_mask > 0.5)[0])
            move_probs = masked_policy_pow / sum_probs
            return np.random.choice(range(BOARD_SQUARES), p=move_probs)

    def _get_legal_moves_mask(self, game_idx):
        mask_buffer = (ctypes.c_float * BOARD_SQUARES)();
        c_lib.mcts_get_legal_moves_mask(self.mcts_manager, game_idx, mask_buffer)
        return np.ctypeslib.as_array(mask_buffer).copy()

    def _save_history(self, game_idx, policy_np):
        board_state_ptr = c_lib.mcts_get_board_state(self.mcts_manager, game_idx)
        # [修改] 12 通道（含本局贴目平面），与推理输入保持一致
        state_tensor_np = np.zeros((1, NUM_INPUT_CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        komi_np = np.array([self.game_komi[game_idx]], dtype=np.int32)
        c_lib.boards_to_tensors_with_komi_c(board_state_ptr, 1, komi_np.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                                             state_tensor_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
        self.game_histories[game_idx].append((state_tensor_np[0], policy_np, board_state_ptr.contents.current_player))

    def _get_training_data(self):
        all_training_data = []
        for game_idx in range(NUM_PARALLEL_GAMES):
            if not self.game_histories[game_idx]: continue
            komi = self.game_komi[game_idx]
            # [新增] 原始目差（黑视角）供自适应采样统计；mcts_get_final_value 已贴目感知
            score_diff = c_lib.mcts_get_score_diff(self.mcts_manager, game_idx)
            self.komi_sampler.record_game(score_diff)
            adjusted_black_margin = score_diff + komi  # 黑视角、贴目调整后净胜目数
            final_value_for_player_0 = c_lib.mcts_get_final_value(self.mcts_manager, game_idx, 0)
            for state_tensor, policy, player_at_step in self.game_histories[game_idx]:
                value_target = final_value_for_player_0 if player_at_step == 0 else -final_value_for_player_0
                # margin：行棋方视角（与 state 的视角翻转一致），score_head 的训练目标
                margin = adjusted_black_margin if player_at_step == 0 else -adjusted_black_margin
                all_training_data.append((state_tensor, policy, value_target, margin, komi))
        # [新增] 持久化贴目统计，跨自对弈会话延续
        self.komi_sampler.save()
        return all_training_data

    def __del__(self):
        if hasattr(self, 'mcts_manager') and self.mcts_manager:
            c_lib.destroy_mcts_manager(self.mcts_manager)


if __name__ == "__main__":
    print("开始 TensorRT 自对弈工作脚本...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cpu':
        print("错误: 此脚本需要 CUDA GPU 才能运行 TensorRT 引擎。", file=sys.stderr);
        sys.exit(1)
    try:
        trt_model = TensorRTModel(TENSORRT_ENGINE_PATH)
        print(f"TensorRT 引擎 ({TENSORRT_ENGINE_PATH}) 加载成功。")
    except Exception as e:
        print(f"错误: 加载 TensorRT 引擎失败: {e}", file=sys.stderr);
        sys.exit(1)

    all_cycles_data = []
    komi_sampler = AdaptiveKomiSampler()  # [新增] 跨批次共享的贴目统计
    # [新增] 从历史棋谱重建贴目统计（特征值已缓存到 komi_index.json，增量更新），
    # 与在线记录同一衰减律，统计渐进、可从棋谱完整恢复
    _n = komi_sampler.rebuild_from_records(DATA_DIR)
    if _n:
        print(f"已从历史棋谱回放 {_n} 局重建贴目统计。")
    for i in range(TOTAL_GAME_CYCLES):
        print(f"\n--- 开始第 {i + 1}/{TOTAL_GAME_CYCLES} 批次游戏 ---")
        start_time = time.time()
        try:
            runner = GameBatchRunner(trt_model, device, komi_sampler)
            cycle_data = runner.run()
            if cycle_data:
                all_cycles_data.extend(cycle_data)
            del runner
        except Exception as e:
            print(f"在批次 {i + 1} 中发生严重错误: {e}", file=sys.stderr)
            import traceback;

            traceback.print_exc();
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

