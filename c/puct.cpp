#include "puct.h"
#include "game.h"

#include <vector>
#include <cmath>
#include <memory>
#include <mutex>
#include <random>
#include <algorithm>
#include <iostream>

// --- MCTS 核心参数 ---
const float C_PUCT = 1.25f;
// Alpha for Dirichlet noise is scaled dynamically based on the number of legal moves
const float DIRICHLET_ALPHA_SCALING = 10.83f;
const float DIRICHLET_EPSILON = 0.25f;
const float MIN_UNCERTAINTY_WEIGHT = 0.1f;

// --- MCTS 节点结构 ---
// This version uses std::unique_ptr for each node, which is slower but, 
// as you've found, more stable in your environment.
struct Node {
    Node* parent = nullptr;
    std::vector<std::unique_ptr<Node>> children; // Child nodes are owned by their parent

    Board board_state;
    int move_leading_to_this_node = -1;
    bool is_expanded = false;

    double visit_count = 0.0;
    double total_action_value = 0.0;
    float prior_probability = 0.0;

    Node(const Board& board, float prior) : prior_probability(prior) {
        copy_board(&board, &board_state);
    }

    double get_puct_value(double total_parent_visits) const {
        double q_value = (visit_count > 1e-6) ? (-total_action_value / visit_count) : 0.0;
        double u_value = C_PUCT * prior_probability * (std::sqrt(total_parent_visits) / (1.0 + visit_count));
        return q_value + u_value;
    }
};

// --- 单个游戏 MCTS 搜索器 ---
class MCTSSearch {
public:
    std::unique_ptr<Node> root;
    int simulations_done_this_turn = 0;
    int game_index;
    Node* pending_evaluation_leaf = nullptr;
    std::mt19937 rng;
    bool use_dirichlet_noise = true; // Flag to control noise. Default ON for self-play.

    MCTSSearch(int index) : game_index(index) {
        std::random_device rd;
        rng.seed(rd());
        Board b;
        init_board(&b);
        // Using make_unique for stability, as requested
        root = std::make_unique<Node>(b, 1.0f);
    }

    void reset(const Board* new_board_state) {
        root = std::make_unique<Node>(*new_board_state, 1.0f);
        simulations_done_this_turn = 0;
        pending_evaluation_leaf = nullptr;
    }

    void run_simulation() {
        if (pending_evaluation_leaf) return;

        Node* leaf = select_leaf();
        if (get_game_result(&leaf->board_state) != IN_PROGRESS) {
            int raw_score_diff = get_score_diff(&leaf->board_state);
            float normalized_score = static_cast<float>(raw_score_diff) / static_cast<float>(BOARD_SQUARES);
            float value_for_leaf_player = (leaf->board_state.current_player == BLACK) ? normalized_score : -normalized_score;
            backpropagate(leaf, value_for_leaf_player, 1.0f); // Final state has a weight of 1
        }
        else {
            pending_evaluation_leaf = leaf;
        }
    }

    void add_dirichlet_noise(std::vector<std::pair<int, float>>& priors) {
        if (priors.empty()) return;

        float dynamic_alpha = DIRICHLET_ALPHA_SCALING / priors.size();
        std::gamma_distribution<float> gamma(dynamic_alpha, 1.0f);
        std::vector<float> noise;
        float noise_sum = 0.0f;

        for (size_t i = 0; i < priors.size(); ++i) {
            float g = gamma(rng);
            noise.push_back(g);
            noise_sum += g;
        }

        if (noise_sum > 1e-6) {
            for (size_t i = 0; i < priors.size(); ++i) {
                priors[i].second = (1.0f - DIRICHLET_EPSILON) * priors[i].second + DIRICHLET_EPSILON * (noise[i] / noise_sum);
            }
        }
    }

    void expand_and_evaluate(const float* policy, float value, float uncertainty) {
        if (!pending_evaluation_leaf) return;

        Node* leaf = pending_evaluation_leaf;
        pending_evaluation_leaf = nullptr;

        Bitboards legal_moves = get_legal_moves(&leaf->board_state);
        std::vector<std::pair<int, float>> priors;
        for (int sq = 0; sq < BOARD_SQUARES; ++sq) {
            if (GET_BIT(legal_moves, sq)) {
                priors.push_back({ sq, policy[sq] });
            }
        }

        // MODIFIED: Apply noise only if the flag is enabled
        if (leaf == root.get() && use_dirichlet_noise) {
            add_dirichlet_noise(priors);
        }

        for (const auto& p : priors) {
            int sq = p.first;
            float prior_prob = p.second;
            Board next_board;
            copy_board(&leaf->board_state, &next_board);
            make_move(&next_board, sq);
            auto new_child = std::make_unique<Node>(next_board, prior_prob);
            new_child->parent = leaf;
            new_child->move_leading_to_this_node = sq;
            leaf->children.push_back(std::move(new_child));
        }

        leaf->is_expanded = true;
        float playout_weight = 1.0f / (MIN_UNCERTAINTY_WEIGHT + uncertainty);
        backpropagate(leaf, value, playout_weight);
    }

    void backpropagate(Node* leaf, float value, float weight) {
        Node* current = leaf;
        float current_value = value;
        while (current) {
            current->visit_count += weight;
            current->total_action_value += current_value * weight;
            current_value *= -1.0f;
            current = current->parent;
        }
    }

    Node* select_leaf() {
        Node* current = root.get();
        while (current->is_expanded) {
            if (current->children.empty()) return current;
            current = get_best_child(current);
        }
        return current;
    }

    Node* get_best_child(Node* parent) {
        double max_puct = -1e9;
        Node* best_child = nullptr;
        for (const auto& child : parent->children) {
            double puct_val = child->get_puct_value(parent->visit_count);
            if (puct_val > max_puct) {
                max_puct = puct_val;
                best_child = child.get();
            }
        }
        return best_child;
    }

    void get_policy(float* policy_buffer) {
        std::fill(policy_buffer, policy_buffer + BOARD_SQUARES, 0.0f);
        if (!root || root->children.empty()) return;
        double total_visits = 0.0;
        for (const auto& child : root->children) {
            total_visits += child->visit_count;
        }
        if (total_visits > 1e-6) {
            for (const auto& child : root->children) {
                policy_buffer[child->move_leading_to_this_node] = static_cast<float>(child->visit_count / total_visits);
            }
        }
    }

    void make_move_on_tree(int square) {
        std::unique_ptr<Node> new_root = nullptr;
        for (auto& child : root->children) {
            if (child->move_leading_to_this_node == square) {
                new_root = std::move(child);
                break;
            }
        }
        if (new_root) {
            root = std::move(new_root);
            root->parent = nullptr;
        }
        else {
            Board new_board;
            copy_board(&root->board_state, &new_board);
            if (make_move(&new_board, square)) {
                root = std::make_unique<Node>(new_board, 1.0f);
            }
        }
        simulations_done_this_turn = 0;
        pending_evaluation_leaf = nullptr;
    }
};

// --- MCTS 管理器 ---
class MCTSManager {
public:
    std::vector<std::unique_ptr<MCTSSearch>> searches;
    std::mutex mtx;
    MCTSManager(int num_games) {
        for (int i = 0; i < num_games; ++i) { searches.push_back(std::make_unique<MCTSSearch>(i)); }
    }
};

// --- C 语言接口实现 ---
extern "C" {
    API void* create_mcts_manager(int num_games) {
        try {
            return new MCTSManager(num_games);
        }
        catch (const std::bad_alloc& /* e */) { // FIX: Unused variable warning
            return nullptr;
        }
    }
    API void destroy_mcts_manager(void* manager_ptr) { delete static_cast<MCTSManager*>(manager_ptr); }

    API int mcts_run_simulations_and_get_requests(void* manager_ptr, Board* board_requests_buffer, int* request_indices_buffer, int max_requests) {
        MCTSManager* manager = static_cast<MCTSManager*>(manager_ptr);
        std::lock_guard<std::mutex> lock(manager->mtx);
        int requests_count = 0;
        for (auto& search : manager->searches) {
            if (requests_count >= max_requests) break;
            if (get_game_result(&search->root->board_state) != IN_PROGRESS) continue;
            if (!search->pending_evaluation_leaf) {
                search->run_simulation();
                if (search->pending_evaluation_leaf) {
                    copy_board(&search->pending_evaluation_leaf->board_state, &board_requests_buffer[requests_count]);
                    request_indices_buffer[requests_count] = search->game_index;
                    requests_count++;
                }
            }
        }
        return requests_count;
    }

    API void mcts_feed_results(void* manager_ptr, const float* policies, const float* values, const float* uncertainties) {
        MCTSManager* manager = static_cast<MCTSManager*>(manager_ptr);
        std::lock_guard<std::mutex> lock(manager->mtx);
        int result_idx = 0;
        for (auto& search : manager->searches) {
            if (search->pending_evaluation_leaf) {
                search->expand_and_evaluate(&policies[result_idx * BOARD_SQUARES], values[result_idx], uncertainties[result_idx]);
                result_idx++;
            }
        }
    }

    API bool mcts_get_policy(void* manager_ptr, int game_index, float* policy_buffer) {
        MCTSManager* manager = static_cast<MCTSManager*>(manager_ptr);
        if (game_index < 0 || game_index >= manager->searches.size()) return false;
        std::lock_guard<std::mutex> lock(manager->mtx);
        manager->searches[game_index]->get_policy(policy_buffer);
        return true;
    }

    API void mcts_make_move(void* manager_ptr, int game_index, int square) {
        MCTSManager* manager = static_cast<MCTSManager*>(manager_ptr);
        if (game_index < 0 || game_index >= manager->searches.size()) return;
        std::lock_guard<std::mutex> lock(manager->mtx);
        manager->searches[game_index]->make_move_on_tree(square);
    }

    API bool mcts_is_game_over(void* manager_ptr, int game_index) {
        MCTSManager* manager = static_cast<MCTSManager*>(manager_ptr);
        if (game_index < 0 || game_index >= manager->searches.size()) return true;
        std::lock_guard<std::mutex> lock(manager->mtx);
        return get_game_result(&manager->searches[game_index]->root->board_state) != IN_PROGRESS;
    }

    API float mcts_get_final_value(void* manager_ptr, int game_index, int player_perspective) {
        MCTSManager* manager = static_cast<MCTSManager*>(manager_ptr);
        if (game_index < 0 || game_index >= manager->searches.size()) return 0.0f;
        std::lock_guard<std::mutex> lock(manager->mtx);
        const Board* board = &manager->searches[game_index]->root->board_state;
        int raw_score_diff = get_score_diff(board);
        float normalized_score = static_cast<float>(raw_score_diff) / static_cast<float>(BOARD_SQUARES);
        return (player_perspective == BLACK) ? normalized_score : -normalized_score;
    }

    API const Board* mcts_get_board_state(void* manager_ptr, int game_index) {
        MCTSManager* manager = static_cast<MCTSManager*>(manager_ptr);
        if (game_index < 0 || game_index >= manager->searches.size()) return nullptr;
        return &manager->searches[game_index]->root.get()->board_state;
    }

    API int mcts_get_simulations_done(void* manager_ptr, int game_index) {
        MCTSManager* manager = static_cast<MCTSManager*>(manager_ptr);
        if (game_index < 0 || game_index >= manager->searches.size()) return 0;
        return static_cast<int>(manager->searches[game_index]->root->visit_count);
    }

    API int mcts_get_analysis_data(void* manager_ptr, int game_index, int* moves_buffer, float* q_values_buffer, int* visit_counts_buffer, float* puct_scores_buffer, int buffer_size) {
        MCTSManager* manager = static_cast<MCTSManager*>(manager_ptr);
        if (game_index < 0 || game_index >= manager->searches.size()) return 0;
        std::lock_guard<std::mutex> lock(manager->mtx);
        Node* root = manager->searches[game_index]->root.get();
        if (!root || !root->is_expanded) return 0;

        int count = 0;
        double parent_visits = root->visit_count;
        for (const auto& child : root->children) {
            if (count >= buffer_size) break;
            moves_buffer[count] = child->move_leading_to_this_node;
            q_values_buffer[count] = (child->visit_count > 1e-6) ? static_cast<float>(-child->total_action_value / child->visit_count) : 0.0f;
            visit_counts_buffer[count] = static_cast<int>(child->visit_count);
            puct_scores_buffer[count] = static_cast<float>(child->get_puct_value(parent_visits));
            count++;
        }
        return count;
    }

    API void mcts_reset_for_analysis(void* manager_ptr, int game_index, const Board* board) {
        MCTSManager* manager = static_cast<MCTSManager*>(manager_ptr);
        if (game_index < 0 || game_index >= manager->searches.size()) return;
        std::lock_guard<std::mutex> lock(manager->mtx);
        manager->searches[game_index]->reset(board);
    }

    // NEW: Implementation for the noise switch
    API void mcts_set_noise_enabled(void* manager_ptr, int game_index, bool enabled) {
        MCTSManager* manager = static_cast<MCTSManager*>(manager_ptr);
        if (game_index < 0 || game_index >= manager->searches.size()) return;
        std::lock_guard<std::mutex> lock(manager->mtx);
        manager->searches[game_index]->use_dirichlet_noise = enabled;
    }
}
