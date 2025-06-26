#ifndef PUCT_H
#define PUCT_H

#include "game.h"

#ifdef _WIN32
#define API __declspec(dllexport)
#else
#define API
#endif

#ifdef __cplusplus
extern "C" {
#endif

    typedef struct MCTSManager MCTSManager;

    API MCTSManager* create_mcts_manager(int num_parallel_games);
    API void destroy_mcts_manager(MCTSManager* manager_handle);

    API int mcts_run_simulations_and_get_requests(MCTSManager* manager_handle, Board* request_board_output, int* request_game_indices_ptr, int max_requests);

    API void mcts_feed_results(MCTSManager* manager_handle, const float* policies, const float* values);
    API bool mcts_get_policy(MCTSManager* manager_handle, int game_idx, float* policy_output);
    API void mcts_make_move(MCTSManager* manager_handle, int game_idx, int move);
    API bool mcts_is_game_over(MCTSManager* manager_handle, int game_idx);
    API float mcts_get_final_value(MCTSManager* manager_handle, int game_idx, int player_at_step);
    API const Board* mcts_get_board_state(MCTSManager* manager_handle, int game_idx);
    API int mcts_get_simulations_done(MCTSManager* manager_handle, int game_idx);

#ifdef __cplusplus
}
#endif

#endif // PUCT_H