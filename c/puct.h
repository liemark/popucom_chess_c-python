#ifndef PUCT_H
#define PUCT_H

#include "game.h"
#include <vector>

// --- 将所有结构体的完整定义放在头文件中 ---
struct MCTSNode {
    MCTSNode* parent = nullptr;
    std::vector<MCTSNode*> children;
    int move = -1;
    int visit_count = 0;
    double total_action_value = 0.0;
    float prior_probability = 0.0f;
};

struct MCTSTree {
    Board board;
    MCTSNode* root = nullptr;
    bool active = false;
    int simulations_done = 0;
};

struct MCTSManager {
    int num_games = 0;
    std::vector<MCTSTree*> games;
    std::vector<MCTSNode*> request_nodes;
    std::vector<Board> request_board_buffer;
    std::vector<bool> is_root_request_buffer;
};

#if defined(_WIN32) || defined(_WIN64)
#define API __declspec(dllexport)
#else
#define API
#endif

#ifdef __cplusplus
extern "C" {
#endif

    API MCTSManager* create_mcts_manager(int num_parallel_games);
    API void destroy_mcts_manager(MCTSManager* manager_handle);
    API int mcts_run_simulations_and_get_requests(MCTSManager* manager_handle, Board* request_board_output, int* request_game_indices_ptr, int max_requests);
    API void mcts_feed_results(MCTSManager* manager_handle, const float* policies, const float* values);
    API bool mcts_get_policy(MCTSManager* manager_handle, int game_idx, float* policy_output);
    API bool mcts_get_analysis_data(MCTSManager* manager, int game_idx, float* q_values_output, float* policy_output);
    API void mcts_make_move(MCTSManager* manager_handle, int game_idx, int move);
    API bool mcts_is_game_over(MCTSManager* manager_handle, int game_idx);
    API float mcts_get_final_value(MCTSManager* manager_handle, int game_idx, int player_at_step);
    API const Board* mcts_get_board_state(MCTSManager* manager_handle, int game_idx);
    API int mcts_get_simulations_done(MCTSManager* manager_handle, int game_idx);

#ifdef __cplusplus
}
#endif

#endif // PUCT_H
