#include "mcts_search.h"
#include "game.h"      // 确保游戏逻辑可用
#include "mcts_node.h" // 确保节点定义可用
#include <cmath>       // for std::sqrt, std::pow
#include <algorithm>   // for std::fill
#include <vector>

// --- MCTS参数定义 ---
// 定义在.h中声明为extern的全局常量
const float C_PUCT = 1.1f;
const float NOISE_RATIO = 0.25f;
const float POLICY_SHARPENING_Q_THRESHOLD = 0.4f;
const float POLICY_SHARPENING_FACTOR = 0.1f;
const size_t INITIAL_NODE_STORE_CAPACITY = 2048;

// --- NodeStore Class Implementation ---

NodeStore::NodeStore() {
    nodes.reserve(INITIAL_NODE_STORE_CAPACITY);
}

void NodeStore::clear() {
    nodes.clear();
    nodes.reserve(INITIAL_NODE_STORE_CAPACITY);
}

Node& NodeStore::operator[](size_t index) {
    return nodes[index];
}

const Node& NodeStore::operator[](size_t index) const {
    return nodes[index];
}

void NodeStore::swap(std::vector<Node>& other) {
    nodes.swap(other);
}

size_t NodeStore::add_node(size_t parent_idx, int move, float prior) {
    nodes.emplace_back(parent_idx, move, prior);
    return nodes.size() - 1;
}

void NodeStore::add_children(size_t parent_idx, int count, const std::vector<int>& moves, const float* policy) {
    Node& parent_node = nodes[parent_idx];
    parent_node.children_start_idx = nodes.size();
    parent_node.num_children = count;
    for (int i = 0; i < count; ++i) {
        int move = moves[i];
        nodes.emplace_back(parent_idx, move, policy[move]);
    }
}

size_t NodeStore::size() const {
    return nodes.size();
}


// --- MCTSSearch Class Implementation ---

MCTSSearch::MCTSSearch(const Board* board_state, bool enable_noise, double initial_fpu, int game_komi)
    : rng(std::random_device{}()), // 初始化随机数生成器
    root_idx(INVALID_INDEX),
    pending_evaluation_leaf_idx(INVALID_INDEX),
    add_dirichlet_noise(enable_noise),
    fpu_value(initial_fpu),
    komi(game_komi) {
    reset(board_state);
}

void MCTSSearch::reset(const Board* new_board_state) {
    node_store.clear();
    copy_board(new_board_state, &root_board);
    root_idx = node_store.add_node(INVALID_INDEX, -1, 1.0f);
    pending_evaluation_leaf_idx = INVALID_INDEX;
}

int MCTSSearch::get_simulations_done() const {
    if (root_idx != INVALID_INDEX && root_idx < node_store.size()) {
        // [修改] 用真实模拟次数（而非加权 visit_count），保持停止准则语义不变
        return node_store[root_idx].playout_count;
    }
    return 0;
}

void MCTSSearch::apply_dirichlet_noise(size_t node_idx) {
    Node& node = node_store[node_idx];
    if (node.num_children <= 1) return;

    const double KATA_GO_ALPHA_SCALER = 10.83;
    double alpha = KATA_GO_ALPHA_SCALER / static_cast<double>(node.num_children);
    std::gamma_distribution<double> gamma(alpha, 1.0);

    std::vector<double> noise;
    noise.reserve(node.num_children);
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

size_t MCTSSearch::select_leaf(Board* current_board) {
    size_t current_idx = root_idx;
    while (node_store[current_idx].is_expanded) {
        if (node_store[current_idx].num_children == 0) {
            return current_idx; // Terminal node found
        }
        size_t best_child_offset = get_best_child_offset(current_idx);
        if (best_child_offset == INVALID_INDEX) {
            return current_idx; // No valid children to explore
        }
        current_idx = node_store[current_idx].children_start_idx + best_child_offset;
        make_move(current_board, node_store[current_idx].move_leading_to_this_node);
    }
    return current_idx;
}

void MCTSSearch::backpropagate(size_t leaf_idx, float value, double weight, double sigma) {
    size_t current_idx = leaf_idx;
    float current_value = value;
    const double sigma_sq = sigma * sigma; // 方差质量（不随视角翻转）
    while (current_idx != INVALID_INDEX) {
        Node& current_node = node_store[current_idx];
        // [修改] 不确定性加权 + 显式方差传播（每节点 O(1) 增量更新）
        const double w = weight;
        current_node.visit_count += w;                       // W:  Σw
        current_node.total_action_value += w * current_value; // S1: Σw·v
        current_node.total_square_value += w * current_value * current_value; // S2: Σw·v²
        current_node.total_variance += w * sigma_sq;        // SV: Σw·σ²
        current_node.playout_count++;                       // 真实模拟次数（停止准则）
        current_value *= -1.0f; // Flip value for the parent's perspective
        current_idx = current_node.parent_idx;
    }
}

size_t MCTSSearch::get_best_child_offset(size_t parent_idx) {
    double max_puct = -1e18;
    size_t best_child_offset = INVALID_INDEX;
    const Node& parent_node = node_store[parent_idx];

    // 1. 获取父节点在该玩家视角的当前平均价值 (Q)
    double p_q = 0.0;
    if (parent_node.visit_count > 0.0) {
        // parent_node 的 total_action_value 是在该节点累加的
        p_q = parent_node.total_action_value / parent_node.visit_count;
    }

    for (int i = 0; i < parent_node.num_children; ++i) {
        const Node& child = node_store[parent_node.children_start_idx + i];

        // 2. 将计算好的父节点 Q 传给子节点
        double puct_val = child.get_puct_value(parent_node.visit_count, p_q);

        if (puct_val > max_puct) {
            max_puct = puct_val;
            best_child_offset = i;
        }
    }
    return best_child_offset;
}

void MCTSSearch::run_simulation(Board& leaf_board_out) {
    if (pending_evaluation_leaf_idx != INVALID_INDEX) return;

    copy_board(&root_board, &leaf_board_out);
    size_t leaf_idx = select_leaf(&leaf_board_out);

    if (get_game_result(&leaf_board_out) != IN_PROGRESS) {
        // [修改] 贴目感知的终局估值：
        // - 双方步数用尽：score_diff + komi > 0 为黑胜（贴目可能翻转胜负/产生平局）
        // - 当前方无合法步：直接判负（与贴目无关，沿用原始规则）
        float value = 0.0f;
        if (leaf_board_out.moves_left[BLACK] <= 0 && leaf_board_out.moves_left[WHITE] <= 0) {
            int adjusted_diff = get_score_diff(&leaf_board_out) + komi;
            value = (adjusted_diff > 0) ? 1.0f : ((adjusted_diff < 0) ? -1.0f : 0.0f);
        }
        else {
            value = (leaf_board_out.current_player == BLACK) ? -1.0f : 1.0f;
        }

        float value_for_leaf_player = (leaf_board_out.current_player == BLACK) ? value : -value;
        // [修改] 终局节点是 100% 确定的：σ = 0，权重取最大值 1/UNCERTAINTY_BASELINE
        double terminal_weight = 1.0 / UNCERTAINTY_BASELINE;
        backpropagate(leaf_idx, value_for_leaf_player, terminal_weight, 0.0);
    }
    else {
        pending_evaluation_leaf_idx = leaf_idx;
    }
}

void MCTSSearch::expand_and_evaluate_impl(const Board& board_at_leaf, const float* policy, float value, double weight, double sigma) {
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

    backpropagate(leaf_idx, value, weight, sigma);
}

void MCTSSearch::expand_and_evaluate(const Board& board_at_leaf, const float* policy, float value, float uncertainty) {
    // [修改] Uncertainty-Weighted Playouts 核心：
    // 不确定的评估折算为"部分访问次数"，确定的评估算"多于一次访问"
    if (uncertainty < 0.0f) uncertainty = 0.0f; // 防呆（Softplus 理论上恒正）
    double weight = 1.0 / (static_cast<double>(uncertainty) + UNCERTAINTY_BASELINE);
    expand_and_evaluate_impl(board_at_leaf, policy, value, weight, static_cast<double>(uncertainty));
}

void MCTSSearch::expand_and_evaluate_neutral(const Board& board_at_leaf, const float* policy, float value) {
    // 旧行为等价入口：每次 playout 恰好计为一次访问，且不注入任何方差质量
    expand_and_evaluate_impl(board_at_leaf, policy, value, 1.0, 0.0);
}

int MCTSSearch::get_child_variances(float* variance_buffer, int buffer_size) {
    if (root_idx == INVALID_INDEX || root_idx >= node_store.size()) return 0;
    const Node& root_node = node_store[root_idx];
    if (!root_node.is_expanded || root_node.num_children <= 0) return 0;

    int count = 0;
    for (int i = 0; i < root_node.num_children && count < buffer_size; ++i) {
        const Node& child = node_store[root_node.children_start_idx + i];
        variance_buffer[count] = (child.visit_count > 0.0) ? static_cast<float>(child.get_variance()) : 0.0f;
        count++;
    }
    return count;
}

void MCTSSearch::get_policy(float* policy_buffer) {
    std::fill(policy_buffer, policy_buffer + BOARD_SQUARES, 0.0f);
    const Node& root_node = node_store[root_idx];
    if (root_node.num_children == 0) return;

    double max_q = -2.0;
    for (int i = 0; i < root_node.num_children; ++i) {
        const Node& child = node_store[root_node.children_start_idx + i];
        if (child.get_q_value() > max_q) {
            max_q = child.get_q_value();
        }
    }

    const float temperature = 1.03f;
    float visit_sum = 0.0f;
    std::vector<float> adjusted_visits;
    adjusted_visits.reserve(root_node.num_children);

    for (int i = 0; i < root_node.num_children; ++i) {
        const Node& child = node_store[root_node.children_start_idx + i];
        double child_q = child.get_q_value();
        float effective_visits = static_cast<float>(child.visit_count);

        if (child_q < max_q - POLICY_SHARPENING_Q_THRESHOLD) {
            effective_visits *= POLICY_SHARPENING_FACTOR;
        }

        float adj_visit = std::pow(effective_visits, 1.0f / temperature);
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
