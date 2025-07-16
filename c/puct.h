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
    API void mcts_feed_results(void* manager_ptr, const float* policies, const float* values, const Board* boards);

    API bool mcts_get_policy(void* manager_ptr, int game_index, float* policy_buffer);
    API void mcts_make_move(void* manager_ptr, int game_index, int square);
    API int mcts_get_simulations_done(void* manager_ptr, int game_index);
    API const Board* mcts_get_board_state(void* manager_ptr, int game_index);
    API void mcts_reset_for_analysis(void* manager_ptr, int game_index, const Board* board);
    API bool mcts_is_game_over(void* manager_ptr, int game_index);
    API float mcts_get_final_value(void* manager_ptr, int game_index, int player_perspective);
    API int mcts_get_analysis_data(void* manager_ptr, int game_index, int* moves_buffer, float* q_values_buffer, int* visit_counts_buffer, float* puct_scores_buffer, int buffer_size);

    // 修复：添加缺失的函数声明
    API void mcts_get_legal_moves_mask(void* manager_ptr, int game_index, float* mask_buffer);

    // 辅助函数
    API void boards_to_tensors_c(const Board* boards, int num_boards, float* output_tensor);

#ifdef __cplusplus
}
#endif

#endif // PUCT_H
