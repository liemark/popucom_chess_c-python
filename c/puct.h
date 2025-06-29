#ifndef PUCT_H
#define PUCT_H

#include "game.h"

// Forward declaration
class Arena;

#ifdef __cplusplus
extern "C" {
#endif

	// ... (其他函数声明保持不变) ...
	API void* create_mcts_manager(int num_games);
	API void destroy_mcts_manager(void* manager_ptr);
	API int mcts_run_simulations_and_get_requests(void* manager_ptr, Board* board_requests_buffer, int* request_indices_buffer, int max_requests);
	API void mcts_feed_results(void* manager_ptr, const float* policies, const float* values, const float* uncertainties);
	API bool mcts_get_policy(void* manager_ptr, int game_index, float* policy_buffer);
	API void mcts_make_move(void* manager_ptr, int game_index, int square);
	API bool mcts_is_game_over(void* manager_ptr, int game_index);
	API float mcts_get_final_value(void* manager_ptr, int game_index, int player_perspective);
	API const Board* mcts_get_board_state(void* manager_ptr, int game_index);
	API int mcts_get_simulations_done(void* manager_ptr, int game_index);
	API int mcts_get_analysis_data(void* manager_ptr, int game_index, int* moves_buffer, float* q_values_buffer, int* visit_counts_buffer, float* puct_scores_buffer, int buffer_size);
	API void mcts_reset_for_analysis(void* manager_ptr, int game_index, const Board* board);

	// NEW: API function to enable or disable Dirichlet noise for a specific game instance.
	API void mcts_set_noise_enabled(void* manager_ptr, int game_index, bool enabled);


#ifdef __cplusplus
}
#endif

#endif // PUCT_H
