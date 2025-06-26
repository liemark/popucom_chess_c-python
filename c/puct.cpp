#include "puct.h"
#include "game.h"

#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <cfloat>
#include <numeric>
#include <stdexcept>

// 包含正确的平台特定头文件
#if defined(_WIN32) || defined(_WIN64)
#include <windows.h>
#else
#include <unistd.h>
#endif


#define C_PUCT 1.0f

// --- 内部辅助函数声明 ---
static MCTSNode* create_node(MCTSNode* parent, int move, float prior_p);
static void free_node_recursive(MCTSNode* node);
static MCTSNode* select_child(MCTSNode* node);
static void expand_node(MCTSNode* node, const Board* board_state, const float* policy);
static void backpropagate(MCTSNode* node, float value);
static float mcts_internal_get_final_value(const Board* final_board, int player_at_step);


extern "C" {

    // --- API 函数实现 ---
    API MCTSManager* create_mcts_manager(int num_parallel_games) {
#if defined(_WIN32) || defined(_WIN64)
        srand(static_cast<unsigned int>(time(nullptr)) ^ GetCurrentProcessId());
#else
        srand(static_cast<unsigned int>(time(nullptr)) ^ getpid());
#endif

        try {
            MCTSManager* manager = new MCTSManager();
            manager->num_games = num_parallel_games;
            manager->games.resize(num_parallel_games, nullptr);
            for (int i = 0; i < num_parallel_games; ++i) {
                manager->games[i] = new MCTSTree();
                init_board(&manager->games[i]->board);
                manager->games[i]->root = create_node(nullptr, -1, 1.0f);
                manager->games[i]->active = true;
                manager->games[i]->simulations_done = 0;
            }
            return manager;
        }
        catch (const std::bad_alloc&) {
            return nullptr;
        }
    }

    API void destroy_mcts_manager(MCTSManager* manager) {
        if (!manager) return;
        for (MCTSTree* tree : manager->games) {
            if (tree) {
                free_node_recursive(tree->root);
                delete tree;
            }
        }
        delete manager;
    }

    API int mcts_run_simulations_and_get_requests(MCTSManager* manager, Board* request_board_output, int* request_game_indices_ptr, int max_requests) {
        if (!manager) return 0;
        manager->request_nodes.clear();
        manager->request_board_buffer.clear();
        manager->is_root_request_buffer.clear();

        for (int i = 0; i < manager->num_games; ++i) {
            if (!manager->games[i] || !manager->games[i]->active) continue;
            if (manager->request_nodes.size() >= static_cast<size_t>(max_requests)) break;

            MCTSTree* tree = manager->games[i];
            MCTSNode* node = tree->root;
            Board current_sim_board;
            copy_board(&tree->board, &current_sim_board);

            while (node && !node->children.empty()) {
                MCTSNode* next_node = select_child(node);
                if (!next_node) break;
                node = next_node;
                make_move(&current_sim_board, node->move);
            }

            if (get_game_result(&current_sim_board) != IN_PROGRESS) {
                // 【核心修复】价值应该是从刚刚完成移动的玩家的视角来计算的
                int player_who_moved = 1 - current_sim_board.current_player;
                float value = mcts_internal_get_final_value(&current_sim_board, player_who_moved);

                // 直接回传这个价值，backpropagate函数会处理后续的正负号翻转
                if (node) backpropagate(node, value);
                tree->simulations_done++;
            }
            else {
                if (node) {
                    manager->is_root_request_buffer.push_back(node == tree->root);
                    manager->request_nodes.push_back(node);
                    manager->request_board_buffer.push_back(current_sim_board);
                    request_game_indices_ptr[manager->request_board_buffer.size() - 1] = i;
                }
            }
        }
        for (size_t i = 0; i < manager->request_board_buffer.size(); ++i) {
            if (request_board_output) {
                memcpy(&request_board_output[i], &manager->request_board_buffer[i], sizeof(Board));
            }
        }
        return static_cast<int>(manager->request_nodes.size());
    }


    API void mcts_feed_results(MCTSManager* manager, const float* policies, const float* values) {
        if (!manager) return;
        for (size_t i = 0; i < manager->request_nodes.size(); ++i) {
            MCTSNode* node = manager->request_nodes[i];
            if (!node) continue;

            const float* original_policy = policies + i * (BOARD_SQUARES);
            float value_for_node = values[i];

            const Board* board_state = &manager->request_board_buffer[i];

            expand_node(node, board_state, original_policy);
            backpropagate(node, value_for_node);

            MCTSNode* root_finder = node;
            while (root_finder && root_finder->parent) root_finder = root_finder->parent;

            for (int j = 0; j < manager->num_games; ++j) {
                if (manager->games[j] && manager->games[j]->root == root_finder) {
                    manager->games[j]->simulations_done++;
                    break;
                }
            }
        }
    }

    API bool mcts_get_policy(MCTSManager* manager, int game_idx, float* policy_output) {
        if (!manager || game_idx >= manager->num_games || !manager->games[game_idx] || !manager->games[game_idx]->active) return false;
        MCTSTree* tree = manager->games[game_idx];
        memset(policy_output, 0, sizeof(float) * BOARD_SQUARES);

        if (!tree->root || tree->root->children.empty()) return true;

        float total_visits = 0;
        for (const auto& child : tree->root->children) {
            if (child) total_visits += child->visit_count;
        }
        if (total_visits > 0) {
            for (const auto& child : tree->root->children) {
                if (child) policy_output[child->move] = (float)child->visit_count / total_visits;
            }
        }
        return true;
    }

    API bool mcts_get_analysis_data(MCTSManager* manager, int game_idx, float* q_values_output, float* policy_output) {
        if (!manager || game_idx >= manager->num_games || !manager->games[game_idx] || !manager->games[game_idx]->active) {
            return false;
        }
        MCTSTree* tree = manager->games[game_idx];
        memset(q_values_output, 0, sizeof(float) * BOARD_SQUARES);
        memset(policy_output, 0, sizeof(float) * BOARD_SQUARES);

        if (!tree->root || tree->root->children.empty()) {
            return true;
        }

        float total_visits = 0;
        for (const auto& child : tree->root->children) {
            if (child) total_visits += child->visit_count;
        }

        if (total_visits > 0) {
            for (const auto& child : tree->root->children) {
                if (child) {
                    int move = child->move;
                    if (move >= 0 && move < BOARD_SQUARES) {
                        policy_output[move] = static_cast<float>(child->visit_count) / total_visits;
                        // 【核心修复】Q值现在应该直接使用，因为backpropagate已经处理了正负号
                        q_values_output[move] = (child->visit_count > 0)
                            ? static_cast<float>(child->total_action_value / child->visit_count)
                            : 0.0f;
                    }
                }
            }
        }
        return true;
    }

    API void mcts_make_move(MCTSManager* manager, int game_idx, int move) {
        if (!manager || game_idx >= manager->num_games || !manager->games[game_idx]) return;
        MCTSTree* tree = manager->games[game_idx];
        MCTSNode* new_root = nullptr;

        if (tree->root) {
            for (auto& child : tree->root->children) {
                if (child && child->move == move) {
                    new_root = child;
                    child = nullptr;
                    break;
                }
            }
            free_node_recursive(tree->root);
        }
        tree->root = new_root ? new_root : create_node(nullptr, -1, 1.0f);
        if (tree->root) tree->root->parent = nullptr;

        make_move(&tree->board, move);
        tree->simulations_done = 0;
    }

    API bool mcts_is_game_over(MCTSManager* manager, int game_idx) {
        if (!manager || game_idx >= manager->num_games || !manager->games[game_idx]) return true;
        return get_game_result(&manager->games[game_idx]->board) != IN_PROGRESS;
    }

    API float mcts_get_final_value(MCTSManager* manager, int game_idx, int player_at_step) {
        if (!manager || game_idx >= manager->num_games || !manager->games[game_idx]) return 0.0f;
        const Board* board = &manager->games[game_idx]->board;
        return mcts_internal_get_final_value(board, player_at_step);
    }

    API const Board* mcts_get_board_state(MCTSManager* manager, int game_idx) {
        if (!manager || game_idx >= manager->num_games || !manager->games[game_idx]) return nullptr;
        return &manager->games[game_idx]->board;
    }

    API int mcts_get_simulations_done(MCTSManager* manager, int game_idx) {
        if (!manager || game_idx >= manager->num_games || !manager->games[game_idx]) return -1;
        return manager->games[game_idx]->simulations_done;
    }

} // extern "C"

// --- 内部辅助函数的具体实现 (使用现代C++) ---
static MCTSNode* create_node(MCTSNode* parent, int move, float prior_p) {
    try {
        MCTSNode* node = new MCTSNode();
        node->parent = parent;
        node->move = move;
        node->prior_probability = prior_p;
        node->visit_count = 0;
        node->total_action_value = 0.0;
        return node;
    }
    catch (const std::bad_alloc&) {
        return nullptr;
    }
}

static void free_node_recursive(MCTSNode* node) {
    if (!node) return;
    for (MCTSNode* child : node->children) {
        free_node_recursive(child);
    }
    delete node;
}

static double calculate_puct_score(const MCTSNode* node) {
    if (node == nullptr || node->parent == nullptr || node->parent->visit_count == 0) return 1e9;
    double q_value = (node->visit_count == 0) ? 0.0 : node->total_action_value / node->visit_count;
    double u_value = C_PUCT * node->prior_probability * sqrt(static_cast<double>(node->parent->visit_count)) / (1.0 + node->visit_count);
    // 【核心修复】PUCT分数应该是从父节点的角度看的。父节点希望最大化自己的收益，
    // 即 -（子节点的Q值） + UCB探索项。子节点的Q值是total_action_value/visit_count
    return -q_value + u_value;
}


static MCTSNode* select_child(MCTSNode* node) {
    if (node == nullptr || node->children.empty()) return nullptr;
    MCTSNode* best_child = nullptr;
    double max_score = -DBL_MAX;
    for (MCTSNode* child : node->children) {
        if (!child) continue;
        double score = calculate_puct_score(child);
        if (score > max_score) {
            max_score = score;
            best_child = child;
        }
    }
    return best_child;
}

static void expand_node(MCTSNode* node, const Board* board_state, const float* policy) {
    if (!node || !board_state || !policy || !node->children.empty()) return;

    Bitboards legal_moves = get_legal_moves(board_state);

    float policy_sum = 0.0f;
    for (int i = 0; i < BOARD_SQUARES; i++) {
        if (is_bit_set(&legal_moves, i)) {
            policy_sum += policy[i];
        }
    }
    if (policy_sum < 1e-6f) policy_sum = 1.0f;

    for (int i = 0; i < BOARD_SQUARES; i++) {
        if (is_bit_set(&legal_moves, i)) {
            MCTSNode* new_child = create_node(node, i, policy[i] / policy_sum);
            if (new_child) node->children.push_back(new_child);
        }
    }
}

static void backpropagate(MCTSNode* node, float value) {
    while (node != nullptr) {
        node->visit_count++;
        node->total_action_value += value;
        value = -value;
        node = node->parent;
    }
}

static float mcts_internal_get_final_value(const Board* final_board, int player_at_step) {
    if (!final_board) return 0.0f;

    int black_tiles = pop_count(&final_board->tiles[BLACK]);
    int white_tiles = pop_count(&final_board->tiles[WHITE]);

    int score_diff;
    if (player_at_step == BLACK) {
        score_diff = black_tiles - white_tiles;
    }
    else { // player_at_step == WHITE
        score_diff = white_tiles - black_tiles;
    }

    return static_cast<float>(score_diff) / static_cast<float>(BOARD_SQUARES);
}
