#ifndef PUCT_H
#define PUCT_H

#include <vector>
#include "game.h" // 包含游戏逻辑的头文件

// 使用现代 C++ 来管理 MCTS 树结构，这比原始指针更安全
// 但API接口仍然是 C 风格的，以便 Python ctypes 可以轻松调用

// --- 内部数据结构 (使用现代C++) ---
// 这些结构体的完整定义放在头文件中，以避免定义冲突
struct MCTSNode {
    MCTSNode* parent;
    std::vector<MCTSNode*> children;
    int move;
    int visit_count;
    double total_action_value;
    float prior_probability;
};

struct MCTSTree {
    Board board;
    MCTSNode* root;
    bool active;
    int simulations_done;
};

struct MCTSManager {
    int num_games;
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

    // API 函数现在使用一个不透明的指针类型，隐藏了C++的实现细节
    // 这使得C++端可以自由使用 std::vector 等特性，而C端/Python端无需关心

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
