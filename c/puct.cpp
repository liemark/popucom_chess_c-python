#include "puct.h"
#include "game.h"

#include <vector>
#include <cmath>
#include <memory>
#include <mutex>
#include <random>
#include <algorithm>
#include <iostream>
#include <limits> // 用于 size_t 的最大值

// --- MCTS Core Parameters ---
// MODIFIED: Reverted to the standard PUCT constant.
const float C_PUCT = 1.0f;
const float NOISE_RATIO = 0.25f;
constexpr size_t INITIAL_NODE_STORE_CAPACITY = 8192;

// 使用一个明确的常量作为无效索引，而不是-1
const size_t INVALID_INDEX = std::numeric_limits<size_t>::max();

// --- Node Structure (Index Pointers Version) ---
struct Node {
    // 使用 size_t 来匹配 vector 的索引类型，修复类型转换警告
    size_t parent_idx = INVALID_INDEX;
    size_t children_start_idx = INVALID_INDEX;

    int num_children = 0;
    int move_leading_to_this_node = -1;
    bool is_expanded = false;
    int visit_count = 0;
    double total_action_value = 0.0;
    float prior_probability = 0.0;

    Node(size_t parent, int move, float prior) : parent_idx(parent), move_leading_to_this_node(move), prior_probability(prior) {}
    Node() = default;

    /** First Play Urgency（FPU)
    * 为未访问节点设置一个小的正Q值（从父节点角度看），以鼓励探索。
    * 为了方便训练，干脆先设置为足够大的值，例如1，效果约等于+∞，强制至少看一次所有点位
    * 正常使用时可以设置为0.02
    */
    // MODIFIED: Reverted to the standard PUCT formula: Q + c_puct * U
    double get_puct_value(int total_parent_visits, double fpu_value) const {
        // Q value: Average action value for this node. From the parent's perspective, so it's -total_action_value.
        // If unvisited, use the FPU value.
        double q_value = (visit_count > 0) ? (-total_action_value / visit_count) : fpu_value;

        // U value: Exploration term based on prior probability and parent's visit count.
        double u_value = C_PUCT * prior_probability * (std::sqrt(static_cast<double>(total_parent_visits)) / (1.0 + visit_count));

        return q_value + u_value;
    }
};

// --- Node Storage ---
class NodeStore {
private:
    std::vector<Node> nodes;
public:
    NodeStore() { nodes.reserve(INITIAL_NODE_STORE_CAPACITY); }
    void clear() { nodes.clear(); nodes.reserve(INITIAL_NODE_STORE_CAPACITY); }
    Node& operator[](size_t index) { return nodes[index]; }
    const Node& operator[](size_t index) const { return nodes[index]; }

    size_t add_node(size_t parent_idx, int move, float prior) {
        nodes.emplace_back(parent_idx, move, prior);
        return nodes.size() - 1;
    }
    void add_children(size_t parent_idx, int count, const std::vector<int>& moves, const float* policy) {
        Node& parent_node = nodes[parent_idx];
        parent_node.children_start_idx = nodes.size();
        parent_node.num_children = count;
        for (int i = 0; i < count; ++i) {
            int move = moves[i];
            nodes.emplace_back(parent_idx, move, policy[move]);
        }
    }
    size_t size() const { return nodes.size(); }
};


// --- MCTS Searcher for a single game ---
class MCTSSearch {
private:
    std::mt19937 rng{ std::random_device{}() };
    void apply_dirichlet_noise(size_t node_idx) {
        Node& node = node_store[node_idx];
        if (node.num_children <= 1) return;
        const double KATA_GO_ALPHA_SCALER = 10.83;
        double alpha = KATA_GO_ALPHA_SCALER / static_cast<double>(node.num_children);
        std::gamma_distribution<double> gamma(alpha, 1.0);
        std::vector<double> noise;
        double noise_sum = 0.0;
        for (int i = 0; i < node.num_children; ++i) {
            noise.push_back(gamma(rng));
            noise_sum += noise.back();
        }
        if (noise_sum > 1e-9) {
            for (int i = 0; i < node.num_children; ++i) {
                Node& child = node_store[node.children_start_idx + i];
                float noisy_prior = static_cast<float>(noise[i] / noise_sum);
                child.prior_probability = (1.0f - NOISE_RATIO) * child.prior_probability + NOISE_RATIO * noisy_prior;
            }
        }
    }

public:
    NodeStore node_store;
    size_t root_idx = INVALID_INDEX;
    Board root_board;
    int game_index;
    size_t pending_evaluation_leaf_idx = INVALID_INDEX;
    bool add_dirichlet_noise;
    double fpu_value;

    MCTSSearch(int index, bool enable_noise, double initial_fpu)
        : game_index(index), add_dirichlet_noise(enable_noise), fpu_value(initial_fpu) {
        init_board(&root_board);
        reset(&root_board);
    }

    void set_fpu(double new_fpu) { fpu_value = new_fpu; }
    void set_noise_enabled(bool enabled) { add_dirichlet_noise = enabled; }
    void reset(const Board* new_board_state) {
        node_store.clear();
        copy_board(new_board_state, &root_board);
        root_idx = node_store.add_node(INVALID_INDEX, -1, 1.0f);
        pending_evaluation_leaf_idx = INVALID_INDEX;
    }
    int get_simulations_done() const {
        if (root_idx != INVALID_INDEX && root_idx < node_store.size()) {
            return node_store[root_idx].visit_count;
        }
        return 0;
    }
    void run_simulation() {
        if (pending_evaluation_leaf_idx != INVALID_INDEX) return;
        Board current_board;
        copy_board(&root_board, &current_board);
        size_t leaf_idx = select_leaf(&current_board);

        // MODIFIED: Use win/loss/draw for terminal node value
        if (get_game_result(&current_board) != IN_PROGRESS) {
            GameResult result = get_game_result(&current_board);
            float value = 0.0f;
            if (result == BLACK_WIN) value = 1.0f;
            else if (result == WHITE_WIN) value = -1.0f;
            // Draw is 0.0f

            // The value is from the perspective of the player whose turn it would be at the leaf.
            // If Black won, the final value is 1.0. If it's Black's turn at the leaf, they see +1.0.
            // If it's White's turn, they see -1.0.
            float value_for_leaf_player = (current_board.current_player == BLACK) ? value : -value;
            backpropagate(leaf_idx, value_for_leaf_player);
        }
        else {
            pending_evaluation_leaf_idx = leaf_idx;
        }
    }
    size_t select_leaf(Board* current_board) {
        size_t current_idx = root_idx;
        while (node_store[current_idx].is_expanded) {
            if (node_store[current_idx].num_children == 0) return current_idx;
            size_t best_child_offset = get_best_child_offset(current_idx);
            if (best_child_offset == INVALID_INDEX) return current_idx;
            current_idx = node_store[current_idx].children_start_idx + best_child_offset;
            make_move(current_board, node_store[current_idx].move_leading_to_this_node);
        }
        return current_idx;
    }
    void expand_and_evaluate(const Board& board_at_leaf, const float* policy, float value) {
        if (pending_evaluation_leaf_idx == INVALID_INDEX) return;
        size_t leaf_idx = pending_evaluation_leaf_idx;
        pending_evaluation_leaf_idx = INVALID_INDEX;
        Bitboards legal_moves_bb = get_legal_moves(&board_at_leaf);
        std::vector<int> legal_moves;
        for (int sq = 0; sq < BOARD_SQUARES; ++sq) {
            if (GET_BIT(legal_moves_bb, sq)) {
                legal_moves.push_back(sq);
            }
        }
        if (!legal_moves.empty()) {
            node_store.add_children(leaf_idx, legal_moves.size(), legal_moves, policy);
        }
        node_store[leaf_idx].is_expanded = true;
        if (leaf_idx == root_idx && add_dirichlet_noise) {
            apply_dirichlet_noise(leaf_idx);
        }
        backpropagate(leaf_idx, value);
    }
    void backpropagate(size_t leaf_idx, float value) {
        size_t current_idx = leaf_idx;
        float current_value = value;
        while (current_idx != INVALID_INDEX) {
            Node& current_node = node_store[current_idx];
            current_node.visit_count++;
            current_node.total_action_value += current_value;
            current_value *= -1.0f;
            current_idx = current_node.parent_idx;
        }
    }

    size_t get_best_child_offset(size_t parent_idx) {
        double max_puct = -1e9;
        size_t best_child_offset = INVALID_INDEX;
        const Node& parent_node = node_store[parent_idx];

        for (int i = 0; i < parent_node.num_children; ++i) {
            const Node& child = node_store[parent_node.children_start_idx + i];
            double puct_val = child.get_puct_value(parent_node.visit_count, this->fpu_value);
            if (puct_val > max_puct) {
                max_puct = puct_val;
                best_child_offset = i;
            }
        }
        return best_child_offset;
    }
    void get_policy(float* policy_buffer) {
        std::fill(policy_buffer, policy_buffer + BOARD_SQUARES, 0.0f);
        const Node& root_node = node_store[root_idx];
        if (root_node.num_children == 0) return;
        const float temperature = 1.03f;
        int max_visits = 0;
        for (int i = 0; i < root_node.num_children; ++i) {
            const Node& child = node_store[root_node.children_start_idx + i];
            if (child.visit_count > max_visits) max_visits = child.visit_count;
        }
        float visit_sum = 0.0f;
        std::vector<float> adjusted_visits;
        adjusted_visits.reserve(root_node.num_children);
        for (int i = 0; i < root_node.num_children; ++i) {
            const Node& child = node_store[root_node.children_start_idx + i];
            float adj_visit = std::exp(static_cast<float>(child.visit_count - max_visits) / temperature);
            adjusted_visits.push_back(adj_visit);
            visit_sum += adj_visit;
        }
        if (visit_sum > 1e-9) {
            for (int i = 0; i < root_node.num_children; ++i) {
                const Node& child = node_store[root_node.children_start_idx + i];
                policy_buffer[child.move_leading_to_this_node] = adjusted_visits[i] / visit_sum;
            }
        }
    }
    void make_move_on_tree(int square) {
        const Node& old_root = node_store[root_idx];
        size_t new_root_idx = INVALID_INDEX;
        for (int i = 0; i < old_root.num_children; ++i) {
            size_t child_idx = old_root.children_start_idx + i;
            if (node_store[child_idx].move_leading_to_this_node == square) {
                new_root_idx = child_idx;
                break;
            }
        }
        make_move(&root_board, square);
        if (new_root_idx != INVALID_INDEX) {
            root_idx = new_root_idx;
            node_store[root_idx].parent_idx = INVALID_INDEX;
        }
        else {
            reset(&root_board);
        }
        pending_evaluation_leaf_idx = INVALID_INDEX;
    }
};

// --- MCTS Manager ---
class MCTSManager {
public:
    std::vector<std::unique_ptr<MCTSSearch>> searches;
    std::mutex mtx;
    MCTSManager(int num_games, bool enable_noise, double initial_fpu) {
        for (int i = 0; i < num_games; ++i) {
            searches.push_back(std::make_unique<MCTSSearch>(i, enable_noise, initial_fpu));
        }
    }
    void set_fpu_for_all(double new_fpu) {
        std::lock_guard<std::mutex> lock(mtx);
        for (auto& search : searches) {
            search->set_fpu(new_fpu);
        }
    }
    void set_noise_enabled_for_all(bool enabled) {
        std::lock_guard<std::mutex> lock(mtx);
        for (auto& search : searches) {
            search->set_noise_enabled(enabled);
        }
    }
};


// C-style API
extern "C" {
    const int NUM_INPUT_CHANNELS = 11;
    const int MAX_MOVES_PER_PLAYER = 25;
    API void boards_to_tensors_c(const Board* boards, int num_boards, float* output_tensor) {
        const int plane_size = BOARD_SQUARES;
        const int tensor_size = NUM_INPUT_CHANNELS * plane_size;
        for (int i = 0; i < num_boards; ++i) {
            const Board& board = boards[i];
            float* current_tensor_ptr = output_tensor + i * tensor_size;
            int p = board.current_player;
            int o = 1 - p;
            auto fill_plane = [&](int plane_idx, const Bitboards& bb) {
                float* plane_ptr = current_tensor_ptr + plane_idx * plane_size;
                for (int sq = 0; sq < BOARD_SQUARES; ++sq) {
                    plane_ptr[sq] = GET_BIT(bb, sq) ? 1.0f : 0.0f;
                }
                };
            auto fill_scalar_plane = [&](int plane_idx, float value) {
                float* plane_ptr = current_tensor_ptr + plane_idx * plane_size;
                std::fill(plane_ptr, plane_ptr + plane_size, value);
                };
            fill_plane(0, board.pieces[p]);
            fill_plane(1, board.pieces[o]);
            fill_plane(2, board.tiles[p]);
            fill_plane(3, board.tiles[o]);
            fill_scalar_plane(4, (p == BLACK) ? 1.0f : 0.0f);
            fill_scalar_plane(5, (p == WHITE) ? 1.0f : 0.0f);
            fill_scalar_plane(6, static_cast<float>(board.moves_left[BLACK]) / MAX_MOVES_PER_PLAYER);
            fill_scalar_plane(7, static_cast<float>(board.moves_left[WHITE]) / MAX_MOVES_PER_PLAYER);
            fill_scalar_plane(8, static_cast<float>(pop_count(&board.tiles[BLACK])) / BOARD_SQUARES);
            fill_scalar_plane(9, static_cast<float>(pop_count(&board.tiles[WHITE])) / BOARD_SQUARES);
            Bitboards all_tiles;
            all_tiles.parts[0] = ~(board.tiles[BLACK].parts[0] | board.tiles[WHITE].parts[0]);
            all_tiles.parts[1] = ~(board.tiles[BLACK].parts[1] | board.tiles[WHITE].parts[1]);
            fill_plane(10, all_tiles);
        }
    }

    API void* create_mcts_manager(int num_games, bool enable_noise, double initial_fpu) {
        return new MCTSManager(num_games, enable_noise, initial_fpu);
    }

    API void mcts_set_fpu(void* manager_ptr, double new_fpu) {
        if (auto* manager = static_cast<MCTSManager*>(manager_ptr)) {
            manager->set_fpu_for_all(new_fpu);
        }
    }

    API void destroy_mcts_manager(void* manager_ptr) { delete static_cast<MCTSManager*>(manager_ptr); }
    API void mcts_set_noise_enabled(void* manager_ptr, bool enable) {
        if (auto* manager = static_cast<MCTSManager*>(manager_ptr)) {
            manager->set_noise_enabled_for_all(enable);
        }
    }
    API int mcts_run_simulations_and_get_requests(void* manager_ptr, Board* board_requests_buffer, int* request_indices_buffer, int max_requests) {
        MCTSManager* manager = static_cast<MCTSManager*>(manager_ptr);
        std::lock_guard<std::mutex> lock(manager->mtx);
        int requests_count = 0;
        for (auto& search : manager->searches) {
            if (requests_count >= max_requests) break;
            if (get_game_result(&search->root_board) != IN_PROGRESS) continue;
            if (search->pending_evaluation_leaf_idx == INVALID_INDEX) {
                search->run_simulation();
                if (search->pending_evaluation_leaf_idx != INVALID_INDEX) {
                    Board leaf_board;
                    copy_board(&search->root_board, &leaf_board);
                    std::vector<int> path;
                    size_t curr = search->pending_evaluation_leaf_idx;
                    while (curr != INVALID_INDEX && search->node_store[curr].parent_idx != INVALID_INDEX) {
                        path.push_back(search->node_store[curr].move_leading_to_this_node);
                        curr = search->node_store[curr].parent_idx;
                    }
                    std::reverse(path.begin(), path.end());
                    for (int move : path) {
                        make_move(&leaf_board, move);
                    }
                    copy_board(&leaf_board, &board_requests_buffer[requests_count]);
                    request_indices_buffer[requests_count] = search->game_index;
                    requests_count++;
                }
            }
        }
        return requests_count;
    }
    API void mcts_feed_results(void* manager_ptr, const float* policies, const float* values, const Board* boards) {
        MCTSManager* manager = static_cast<MCTSManager*>(manager_ptr);
        std::lock_guard<std::mutex> lock(manager->mtx);
        int result_idx = 0;
        for (auto& search : manager->searches) {
            if (search->pending_evaluation_leaf_idx != INVALID_INDEX) {
                search->expand_and_evaluate(boards[result_idx], &policies[result_idx * BOARD_SQUARES], values[result_idx]);
                result_idx++;
            }
        }
    }
    API bool mcts_get_policy(void* manager_ptr, int game_index, float* policy_buffer) {
        MCTSManager* manager = static_cast<MCTSManager*>(manager_ptr);
        if (game_index < 0 || (size_t)game_index >= manager->searches.size()) return false;
        std::lock_guard<std::mutex> lock(manager->mtx);
        manager->searches[game_index]->get_policy(policy_buffer);
        return true;
    }

    API void mcts_get_legal_moves_mask(void* manager_ptr, int game_index, float* mask_buffer) {
        MCTSManager* manager = static_cast<MCTSManager*>(manager_ptr);
        if (game_index < 0 || (size_t)game_index >= manager->searches.size()) {
            std::fill(mask_buffer, mask_buffer + BOARD_SQUARES, 0.0f);
            return;
        }
        std::lock_guard<std::mutex> lock(manager->mtx);
        const Board* board = &manager->searches[game_index]->root_board;
        Bitboards legal_moves_bb = get_legal_moves(board);
        for (int sq = 0; sq < BOARD_SQUARES; ++sq) {
            if (GET_BIT(legal_moves_bb, sq)) {
                mask_buffer[sq] = 1.0f;
            }
            else {
                mask_buffer[sq] = 0.0f;
            }
        }
    }

    API void mcts_make_move(void* manager_ptr, int game_index, int square) {
        MCTSManager* manager = static_cast<MCTSManager*>(manager_ptr);
        if (game_index < 0 || (size_t)game_index >= manager->searches.size()) return;
        std::lock_guard<std::mutex> lock(manager->mtx);
        manager->searches[game_index]->make_move_on_tree(square);
    }
    API int mcts_get_simulations_done(void* manager_ptr, int game_index) {
        MCTSManager* manager = static_cast<MCTSManager*>(manager_ptr);
        if (game_index < 0 || (size_t)game_index >= manager->searches.size()) return 0;
        std::lock_guard<std::mutex> lock(manager->mtx);
        return manager->searches[game_index]->get_simulations_done();
    }
    API const Board* mcts_get_board_state(void* manager_ptr, int game_index) {
        MCTSManager* manager = static_cast<MCTSManager*>(manager_ptr);
        if (game_index < 0 || (size_t)game_index >= manager->searches.size()) return nullptr;
        return &manager->searches[game_index]->root_board;
    }
    API void mcts_reset_for_analysis(void* manager_ptr, int game_index, const Board* board) {
        MCTSManager* manager = static_cast<MCTSManager*>(manager_ptr);
        if (game_index < 0 || (size_t)game_index >= manager->searches.size()) return;
        std::lock_guard<std::mutex> lock(manager->mtx);
        manager->searches[game_index]->reset(board);
    }
    API bool mcts_is_game_over(void* manager_ptr, int game_index) {
        MCTSManager* manager = static_cast<MCTSManager*>(manager_ptr);
        if (game_index < 0 || (size_t)game_index >= manager->searches.size()) return true;
        std::lock_guard<std::mutex> lock(manager->mtx);
        return get_game_result(&manager->searches[game_index]->root_board) != IN_PROGRESS;
    }

    // MODIFIED: Return win/loss/draw value instead of normalized score
    API float mcts_get_final_value(void* manager_ptr, int game_index, int player_perspective) {
        MCTSManager* manager = static_cast<MCTSManager*>(manager_ptr);
        if (game_index < 0 || (size_t)game_index >= manager->searches.size()) return 0.0f;

        std::lock_guard<std::mutex> lock(manager->mtx);
        const Board* board = &manager->searches[game_index]->root_board;
        GameResult result = get_game_result(board);

        float value = 0.0f;
        if (result == BLACK_WIN) value = 1.0f;
        else if (result == WHITE_WIN) value = -1.0f;
        // Draw remains 0.0f

        // Return value from the perspective of the requested player
        return (player_perspective == BLACK) ? value : -value;
    }

    API int mcts_get_analysis_data(void* manager_ptr, int game_index, int* moves_buffer, float* q_values_buffer, int* visit_counts_buffer, float* puct_scores_buffer, int buffer_size) {
        MCTSManager* manager = static_cast<MCTSManager*>(manager_ptr);
        if (game_index < 0 || (size_t)game_index >= manager->searches.size()) return 0;
        std::lock_guard<std::mutex> lock(manager->mtx);
        auto& search = manager->searches[game_index];
        const Node& root_node = search->node_store[search->root_idx];
        if (!root_node.is_expanded) return 0;
        int count = 0;
        int parent_visits = root_node.visit_count;
        for (int i = 0; i < root_node.num_children; ++i) {
            if (count >= buffer_size) break;
            const Node& child = search->node_store[root_node.children_start_idx + i];
            moves_buffer[count] = child.move_leading_to_this_node;
            q_values_buffer[count] = (child.visit_count > 0) ? static_cast<float>(-child.total_action_value / child.visit_count) : 0.0f;
            visit_counts_buffer[count] = child.visit_count;
            puct_scores_buffer[count] = static_cast<float>(child.get_puct_value(parent_visits, search->fpu_value));
            count++;
        }
        return count;
    }
}
