# profile_worker.py
#
# 一个专门用于分析 self_play_worker.py 性能瓶颈的脚本。
# 这个版本适配了C++优化和PyTorch混合精度API。

import torch
import numpy as np
import time
import ctypes

# 从原始脚本中导入必要的组件和设置
from self_play_worker import (
    setup_c_library,
    GameBatchRunner,
    MCTS_SIMULATIONS,
    NUM_PARALLEL_GAMES,
    MAX_BATCH_SIZE,
    MODEL_PATH,
    BOARD_SQUARES,
    NUM_INPUT_CHANNELS,
    BOARD_SIZE,
    TEMPERATURE_MOVE_SELECTION,
    TEMPERATURE_DECAY_MOVES,
    TEMPERATURE_END,
    Board
)
from popucom_nn_model import PomPomNN


# --- 性能分析版本的 GameBatchRunner ---
class ProfiledGameBatchRunner(GameBatchRunner):
    def run(self):
        """
        运行一个完整的游戏批次，并记录每个关键步骤的耗时。
        """
        timings = {
            "1. MCTS Simulation (C++)": 0.0,
            "2. Data Conversion (C++)": 0.0,
            "3. GPU Inference (NN Forward)": 0.0,
            "4. MCTS Update (C++)": 0.0,
            "5. Move Selection & History": 0.0,
            "Total Loop Time": 0.0
        }
        batch_count = 0

        print("\n--- 开始性能分析运行 ---")
        print(f"将运行 {NUM_PARALLEL_GAMES} 个并行游戏，直到所有游戏结束。")

        while self.active_games:
            loop_start_time = time.perf_counter()
            batch_count += 1

            # --- 步骤 1: MCTS 模拟 (C++) ---
            t_start = time.perf_counter()
            board_buffer = (Board * MAX_BATCH_SIZE)()
            request_indices = (ctypes.c_int * MAX_BATCH_SIZE)()
            num_requests = c_lib.mcts_run_simulations_and_get_requests(
                self.mcts_manager, board_buffer, request_indices, MAX_BATCH_SIZE
            )
            timings["1. MCTS Simulation (C++)"] += time.perf_counter() - t_start

            if num_requests > 0:
                # --- 步骤 2: 数据转换 (C++) ---
                t_start = time.perf_counter()
                input_tensor_np = np.zeros((num_requests, NUM_INPUT_CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
                c_lib.boards_to_tensors_c(
                    board_buffer,
                    num_requests,
                    input_tensor_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
                )
                input_batch = torch.from_numpy(input_tensor_np).to(self.device)
                timings["2. Data Conversion (C++)"] += time.perf_counter() - t_start

                # --- 步骤 3: GPU 推理 ---
                t_start = time.perf_counter()
                with torch.no_grad():
                    use_amp = self.device.type == 'cuda'
                    # 使用 autocast
                    with torch.amp.autocast(device_type=self.device.type, enabled=use_amp):
                        policies_logits, values, _, _ = self.model(input_batch)

                    policies = torch.softmax(policies_logits.float(), dim=1).cpu().numpy()
                    values = values.float().cpu().numpy().flatten()

                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                timings["3. GPU Inference (NN Forward)"] += time.perf_counter() - t_start

                # --- 步骤 4: MCTS 更新 (C++) ---
                t_start = time.perf_counter()
                contiguous_policies = np.ascontiguousarray(policies, dtype=np.float32)
                contiguous_values = np.ascontiguousarray(values, dtype=np.float32)
                c_lib.mcts_feed_results(
                    self.mcts_manager,
                    contiguous_policies.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    contiguous_values.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    board_buffer
                )
                timings["4. MCTS Update (C++)"] += time.perf_counter() - t_start

            # --- 步骤 5: 走法选择 & 历史记录 ---
            t_start = time.perf_counter()
            policy_buffer = (ctypes.c_float * BOARD_SQUARES)()
            for game_idx in list(self.active_games):
                if c_lib.mcts_get_simulations_done(self.mcts_manager, game_idx) >= MCTS_SIMULATIONS:
                    c_lib.mcts_get_policy(self.mcts_manager, game_idx, policy_buffer)
                    policy_np = np.ctypeslib.as_array(policy_buffer).copy()
                    board_state_ptr = c_lib.mcts_get_board_state(self.mcts_manager, game_idx)
                    state_tensor_np = np.zeros((1, NUM_INPUT_CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
                    c_lib.boards_to_tensors_c(board_state_ptr, 1,
                                              state_tensor_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
                    self.game_histories[game_idx].append(
                        (state_tensor_np[0], policy_np, board_state_ptr.contents.current_player))
                    move_selection_temp = TEMPERATURE_MOVE_SELECTION if self.move_counts[
                                                                            game_idx] < TEMPERATURE_DECAY_MOVES else TEMPERATURE_END
                    if move_selection_temp > 0:
                        move_probs = policy_np ** (1.0 / move_selection_temp)
                        sum_probs = np.sum(move_probs)
                        move_probs /= sum_probs if sum_probs > 1e-8 else 1.0
                    else:
                        move_probs = np.zeros_like(policy_np)
                        if np.sum(policy_np) > 0: move_probs[np.argmax(policy_np)] = 1.0
                    if np.sum(move_probs) < 1e-8:
                        if game_idx in self.active_games: self.active_games.remove(game_idx)
                        continue
                    move = np.random.choice(range(BOARD_SQUARES), p=move_probs)
                    c_lib.mcts_make_move(self.mcts_manager, game_idx, int(move))
                    self.move_counts[game_idx] += 1
                    if c_lib.mcts_is_game_over(self.mcts_manager, game_idx):
                        if game_idx in self.active_games: self.active_games.remove(game_idx)
            timings["5. Move Selection & History"] += time.perf_counter() - t_start

            timings["Total Loop Time"] += time.perf_counter() - loop_start_time

        # --- 打印性能报告 ---
        print("\n--- 性能分析报告 ---")
        total_time = timings["Total Loop Time"]
        if total_time == 0:
            print("总时间为0，无法生成报告。")
            return

        print(f"总循环次数: {batch_count}")
        print(f"总耗时: {total_time:.4f} 秒\n")
        print("{:<30} | {:>12} | {:>10}".format("环节", "总耗时 (秒)", "占比 (%)"))
        print("-" * 57)

        for name, duration in timings.items():
            if name != "Total Loop Time":
                percentage = (duration / total_time) * 100
                print("{:<30} | {:>12.4f} | {:>9.2f}%".format(name, duration, percentage))

        print("-" * 57)


if __name__ == "__main__":
    try:
        c_lib = setup_c_library()
        model = PomPomNN()
        model.load_state_dict(torch.load(MODEL_PATH))
        print(f"模型已从 {MODEL_PATH} 加载。")
    except (FileNotFoundError, RuntimeError) as e:
        print(f"警告: {e}。将使用随机初始化的模型进行分析。")
        model = PomPomNN()

    profiler_runner = ProfiledGameBatchRunner(model, NUM_PARALLEL_GAMES)
    profiler_runner.run()
