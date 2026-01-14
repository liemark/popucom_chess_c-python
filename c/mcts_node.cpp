#include "mcts_node.h"
#include <cmath> // for std::sqrt
#include <algorithm>

extern const float C_PUCT;

// FPU 惩罚因子
// 它表示：未访问节点的初始评估 = 父节点 Q - FPU_REDUCTION
// 正的越多，搜索越集中
const float FPU_REDUCTION = 0.00f;

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


double Node::get_puct_value(int total_parent_visits, double parent_q) const {
    double q_value;
    if (visit_count > 0) {
        q_value = get_q_value();
    }
    else {
        // --- 动态 FPU ---
        // 未访问节点的初始评价应该比当前已知的平均水平(parent_q)稍微差一点。
        // 这样即使在先手劣势(parent_q = -0.6)时，新节点起步也是 -0.85。
        // 只有 Prior (神经网络看好的点) 够高，才能让它被选中。
        q_value = parent_q - FPU_REDUCTION * std::sqrt(std::max(0.01f, prior_probability));
    }

    // U值是探索项，基于先验概率和父节点的访问次数
    double u_value = C_PUCT * prior_probability * (std::sqrt(double(total_parent_visits)) / (1.0 + visit_count));
    return q_value + u_value;
}
