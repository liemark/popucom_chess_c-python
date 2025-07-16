#include "mcts_node.h"
#include <cmath> // for std::sqrt

// MCTS参数，暂时在此定义，后续会移至更合适的位置 (例如 mcts_search.h)
const float C_PUCT = 1.1f;

Node::Node(size_t parent, int move, float prior)
    : parent_idx(parent),
    move_leading_to_this_node(move),
    prior_probability(prior) {
    // 其他成员变量会被默认初始化
}

double Node::get_q_value() const {
    if (visit_count == 0) {
        return 0.0;
    }
    // Q值是从父节点的视角看的，所以是 -total_action_value
    return -total_action_value / visit_count;
}

double Node::get_puct_value(int total_parent_visits, double fpu_value) const {
    // 如果未访问过，Q值使用FPU (First Play Urgency)值
    double q_value = (visit_count > 0) ? get_q_value() : fpu_value;

    // U值是探索项，基于先验概率和父节点的访问次数
    double u_value = C_PUCT * prior_probability * (std::sqrt(static_cast<double>(total_parent_visits)) / (1.0 + visit_count));

    return q_value + u_value;
}
