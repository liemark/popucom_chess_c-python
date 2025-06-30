#include "puct.h"
#include "game.h"

#include <vector>
#include <cmath>
#include <memory>
#include <mutex>
#include <random> // 用于狄利克雷噪声
#include <algorithm>
#include <iostream>

// --- MCTS 核心参数 ---
const float C_PUCT = 1.25f;
const float DIRICHLET_ALPHA_BASE = 0.03f;
const float NOISE_RATIO = 0.25f;

// --- MCTS 节点结构 ---
struct Node {
    Node* parent = nullptr;
    std::vector<std::unique_ptr<Node>> children;

    Board board_state;
    int move_leading_to_this_node = -1;
    bool is_expanded = false;

    int visit_count = 0;
    double total_action_value = 0.0; // W(s,a)
    float prior_probability = 0.0;   // P(s,a)

    Node(const Board& board, float prior) : prior_probability(prior) {
        copy_board(&board, &board_state);
    }

    double get_puct_value(int total_parent_visits) const {
        double q_value = (visit_count > 0) ? (-total_action_value / visit_count) : 0.0;
        double u_value = C_PUCT * prior_probability * (std::sqrt(static_cast<double>(total_parent_visits)) / (1.0 + visit_count));
        return q_value + u_value;
    }
};

// --- 单个游戏 MCTS 搜索器 ---
class MCTSSearch {
private:
    std::mt19937 rng{ std::random_device{}() }; // 随机数生成器用于噪声

    // *** 新增：应用狄利克雷噪声 ***
    void apply_dirichlet_noise(Node* node) {
        if (node->children.empty()) return;

        int num_legal_moves = node->children.size();
        if (num_legal_moves <= 1) return;

        // 计算Alpha参数
        double alpha = static_cast<double>(DIRICHLET_ALPHA_BASE * (19 * 19)) / static_cast<double>(num_legal_moves);

        std::gamma_distribution<double> gamma(alpha, 1.0);

        std::vector<double> noise;
        double noise_sum = 0.0;
        for (int i = 0; i < num_legal_moves; ++i) {
            noise.push_back(gamma(rng));
            noise_sum += noise.back();
        }

        if (noise_sum > 1e-9) { // 避免除以零
            for (int i = 0; i < num_legal_moves; ++i) {
                // 将噪声应用到子节点的先验概率上
                float noisy_prior = static_cast<float>(noise[i] / noise_sum);
                node->children[i]->prior_probability = (1.0f - NOISE_RATIO) * node->children[i]->prior_probability + NOISE_RATIO * noisy_prior;
            }
        }
    }


public:
    std::unique_ptr<Node> root;
    int simulations_done = 0;
    int game_index;
    Node* pending_evaluation_leaf = nullptr;
    bool add_dirichlet_noise; // 开关

    MCTSSearch(int index, bool enable_noise) : game_index(index), add_dirichlet_noise(enable_noise) {
        Board b;
        init_board(&b);
        root = std::make_unique<Node>(b, 1.0f);
    }

    void set_noise_enabled(bool enabled) {
        add_dirichlet_noise = enabled;
    }

    void reset(const Board* new_board_state) {
        root = std::make_unique<Node>(*new_board_state, 1.0f);
        simulations_done = 0;
        pending_evaluation_leaf = nullptr;
    }

    void run_simulation() {
        if (pending_evaluation_leaf) return;

        Node* leaf = select_leaf();
        if (get_game_result(&leaf->board_state) != IN_PROGRESS) {
            int raw_score_diff = get_score_diff(&leaf->board_state);
            float normalized_score = static_cast<float>(raw_score_diff) / static_cast<float>(BOARD_SQUARES);
            float value_for_leaf_player = (leaf->board_state.current_player == BLACK) ? normalized_score : -normalized_score;
            backpropagate(leaf, value_for_leaf_player);
        }
        else {
            pending_evaluation_leaf = leaf;
        }
    }

    Node* select_leaf() {
        Node* current = root.get();
        while (current->is_expanded) {
            if (current->children.empty()) {
                return current;
            }
            current = get_best_child(current);
        }
        return current;
    }

    void expand_and_evaluate(const float* policy, float value) {
        if (!pending_evaluation_leaf) return;

        Node* leaf = pending_evaluation_leaf;
        pending_evaluation_leaf = nullptr;

        Bitboards legal_moves = get_legal_moves(&leaf->board_state);

        for (int sq = 0; sq < BOARD_SQUARES; ++sq) {
            if (GET_BIT(legal_moves, sq)) {
                Board next_board;
                copy_board(&leaf->board_state, &next_board);
                make_move(&next_board, sq);

                auto child = std::make_unique<Node>(next_board, policy[sq]);
                child->parent = leaf;
                child->move_leading_to_this_node = sq;
                leaf->children.push_back(std::move(child));
            }
        }
        leaf->is_expanded = true;

        // *** 在根节点上应用噪声 ***
        if (leaf == root.get() && add_dirichlet_noise) {
            apply_dirichlet_noise(leaf);
        }

        backpropagate(leaf, value);
    }

    void backpropagate(Node* leaf, float value) {
        Node* current = leaf;
        float current_value = value;
        while (current) {
            current->visit_count++;
            current->total_action_value += current_value;
            current_value *= -1.0f;
            current = current->parent;
        }
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

        // 使用 softmax temperature of 1.03 (如KataGo所建议)
        const float temperature = 1.03f;
        float total_visits = 0;

        // 为了数值稳定性，先找到最大的访问次数
        int max_visits = 0;
        for (const auto& child : root->children) {
            if (child->visit_count > max_visits) {
                max_visits = child->visit_count;
            }
        }

        float visit_sum = 0.0f;
        std::vector<float> adjusted_visits;
        for (const auto& child : root->children) {
            float adj_visit = std::exp(static_cast<float>(child->visit_count - max_visits) / temperature);
            adjusted_visits.push_back(adj_visit);
            visit_sum += adj_visit;
        }

        if (visit_sum > 1e-9) {
            for (size_t i = 0; i < root->children.size(); ++i) {
                policy_buffer[root->children[i]->move_leading_to_this_node] = adjusted_visits[i] / visit_sum;
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
        simulations_done = 0;
        pending_evaluation_leaf = nullptr;
    }
};

// --- MCTS 管理器 ---
class MCTSManager {
public:
    std::vector<std::unique_ptr<MCTSSearch>> searches;
    std::mutex mtx;
    bool noise_enabled_for_new_games;

    MCTSManager(int num_games, bool enable_noise) : noise_enabled_for_new_games(enable_noise) {
        for (int i = 0; i < num_games; ++i) {
            searches.push_back(std::make_unique<MCTSSearch>(i, noise_enabled_for_new_games));
        }
    }

    void set_noise_enabled_for_all(bool enabled) {
        std::lock_guard<std::mutex> lock(mtx);
        noise_enabled_for_new_games = enabled;
        for (auto& search : searches) {
            search->set_noise_enabled(enabled);
        }
    }
};

// C 接口
extern "C" {
    // *** 修改：增加 enable_noise 参数 ***
    API void* create_mcts_manager(int num_games, bool enable_noise) {
        return new MCTSManager(num_games, enable_noise);
    }

    // *** 新增：动态设置噪声开关 ***
    API void mcts_set_noise_enabled(void* manager_ptr, bool enable) {
        MCTSManager* manager = static_cast<MCTSManager*>(manager_ptr);
        if (manager) {
            manager->set_noise_enabled_for_all(enable);
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

    API void mcts_feed_results(void* manager_ptr, const float* policies, const float* values) {
        MCTSManager* manager = static_cast<MCTSManager*>(manager_ptr);
        std::lock_guard<std::mutex> lock(manager->mtx);
        int result_idx = 0;
        for (auto& search : manager->searches) {
            if (search->pending_evaluation_leaf) {
                search->expand_and_evaluate(&policies[result_idx * BOARD_SQUARES], values[result_idx]);
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

    // ... 其他 C API 函数保持不变 ...
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
        return &manager->searches[game_index]->root->board_state;
    }
    API int mcts_get_simulations_done(void* manager_ptr, int game_index) {
        MCTSManager* manager = static_cast<MCTSManager*>(manager_ptr);
        if (game_index < 0 || game_index >= manager->searches.size()) return 0;
        return manager->searches[game_index]->root->visit_count;
    }

    API int mcts_get_analysis_data(void* manager_ptr, int game_index, int* moves_buffer, float* q_values_buffer, int* visit_counts_buffer, float* puct_scores_buffer, int buffer_size) {
        MCTSManager* manager = static_cast<MCTSManager*>(manager_ptr);
        if (game_index < 0 || game_index >= manager->searches.size()) return 0;
        std::lock_guard<std::mutex> lock(manager->mtx);
        Node* root = manager->searches[game_index]->root.get();
        if (!root || !root->is_expanded) return 0;

        int count = 0;
        int parent_visits = root->visit_count;
        for (const auto& child : root->children) {
            if (count >= buffer_size) break;
            moves_buffer[count] = child->move_leading_to_this_node;
            q_values_buffer[count] = (child->visit_count > 0) ? static_cast<float>(-child->total_action_value / child->visit_count) : 0.0f;
            visit_counts_buffer[count] = child->visit_count;
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
}