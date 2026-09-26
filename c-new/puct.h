#ifndef PUCT_H
#define PUCT_H

#include "game.h" // 仍然需要Board等基本定义

// 定义API导出宏，用于Windows DLL
#if defined(_WIN32) || defined(_WIN64)
#define API __declspec(dllexport)
#else
#define API
#endif

#ifdef __cplusplus
extern "C" {
#endif

    // --- C-API 函数声明 ---
    // 这些是暴露给Python的唯一接口

    API void* create_mcts_manager(int num_games, bool enable_noise, double initial_fpu);
    API void destroy_mcts_manager(void* manager_ptr);

    API void mcts_set_noise_enabled(void* manager_ptr, bool enable);
    API void mcts_set_fpu(void* manager_ptr, double new_fpu);

    API int mcts_run_simulations_and_get_requests(void* manager_ptr, Board* board_requests_buffer, int* request_indices_buffer, int max_requests);
    // [保留] 旧接口：不传入不确定度，内部以 weight=1、σ=0 反传（行为与旧版 DLL 等价）
    // 供 GUI / arena 等尚未迁移的调用方继续使用
    API void mcts_feed_results(void* manager_ptr, const float* policies, const float* values, const Board* boards);
    // [新增] 带不确定度的接口：每次 playout 权重 w = 1/(uncertainty + baseline)，
    // 且 σ = uncertainty 作为方差质量随搜索树向上传播
    API void mcts_feed_results_with_uncertainty(void* manager_ptr, const float* policies, const float* values, const float* uncertainties, const Board* boards);

    API bool mcts_get_policy(void* manager_ptr, int game_index, float* policy_buffer);
    API void mcts_make_move(void* manager_ptr, int game_index, int square);
    API int mcts_get_simulations_done(void* manager_ptr, int game_index);
    API const Board* mcts_get_board_state(void* manager_ptr, int game_index);
    API void mcts_reset_for_analysis(void* manager_ptr, int game_index, const Board* board);
    API bool mcts_is_game_over(void* manager_ptr, int game_index);
    API float mcts_get_final_value(void* manager_ptr, int game_index, int player_perspective);

    // [新增] 贴目接口：设置本局贴目（白给黑）；读取终局地板差（黑视角，未含贴目）
    // 注意：贴目会改变搜索终局估值与 mcts_get_final_value 的胜负判定（komi=0 等价旧行为）
    API void mcts_set_komi(void* manager_ptr, int game_index, int komi);
    API int mcts_get_score_diff(void* manager_ptr, int game_index);

    API int mcts_get_analysis_data(void* manager_ptr, int game_index, int* moves_buffer, float* q_values_buffer, int* visit_counts_buffer, float* puct_scores_buffer, int buffer_size);

    // 修复：添加缺失的函数声明
    API void mcts_get_legal_moves_mask(void* manager_ptr, int game_index, float* mask_buffer);

    // [新增] 观察接口：读取根节点各子节点 Q 估计的方差（不确定度沿搜索树传播的结果）
    API int mcts_get_child_variances(void* manager_ptr, int game_index, float* variance_buffer, int buffer_size);

    // 辅助函数
    // [保留] 旧 11 通道接口（GUI / arena 未迁移前继续可用）
    API void boards_to_tensors_c(const Board* boards, int num_boards, float* output_tensor);
    // [新增] 12 通道接口：通道 11 = 贴目平面 komi/KOMI_SCALE（与 popucom_nn_interface.KOMI_SCALE=8 一致）
    API void boards_to_tensors_with_komi_c(const Board* boards, int num_boards, const int* komis, float* output_tensor);

#ifdef __cplusplus
}
#endif

#endif // PUCT_H
