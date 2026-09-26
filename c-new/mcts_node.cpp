#include "mcts_node.h"
#include <cmath> // for std::sqrt
#include <algorithm>

extern const float C_PUCT;

// --- 不确定性加权参数定义（在 mcts_node.h 中 extern 声明）---
// 每次 playout 的权重 w = 1 / (uncertainty + UNCERTAINTY_BASELINE)
// 调参建议：0.01 ~ 0.1 之间；加权后访问总量改变，C_PUCT 后续可能需微调
const double UNCERTAINTY_BASELINE = 0.05;
// 节点方差对 PUCT U 项的调制系数：0 = 关闭（仅传播/记录方差，不改变搜索行为）
const double UNCERTAINTY_PUCT_SCALE = 0.0;

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
    if (visit_count <= 0.0) {
        return 0.0;
    }
    // Q值是从父节点的视角看的，所以是 -total_action_value
    return -total_action_value / visit_count;
}

double Node::get_variance() const {
    if (visit_count <= 0.0) {
        return 0.0;
    }
    double w = visit_count;
    // 本节点视角的均值与方差（方差对视角翻转不变）
    double mean = total_action_value / w;                       // E[v]
    double sample_var = total_square_value / w - mean * mean;   // E[v²] - E[v]²
    if (sample_var < 0.0) sample_var = 0.0;                     // 浮点误差截断
    double subtree_var = total_variance / w;                     // E[σ²] 子树方差质量
    return subtree_var + sample_var;
}


double Node::get_puct_value(double total_parent_visits, double parent_q) const {
    double q_value;
    if (visit_count > 0.0) {
        q_value = get_q_value();
    }
    else {
        // --- 动态 FPU ---
        // 未访问节点的初始评价应该比当前已知的平均水平(parent_q)稍微差一点。
        // 这样即使在先手劣势(parent_q = -0.6)时，新节点起步也是 -0.85。
        // 只有 Prior (神经网络看好的点) 够高，才能让它被选中。
        q_value = parent_q - FPU_REDUCTION * std::sqrt(std::max(0.01f, prior_probability));
    }

    // U值是探索项，基于先验概率和父节点的（加权）访问次数
    // [修改] visit_count 为加权 double；
    // 当 UNCERTAINTY_PUCT_SCALE > 0 时，高方差节点（Q 不可信）的探索被削弱
    double modulation = 1.0;
    if (UNCERTAINTY_PUCT_SCALE > 0.0 && visit_count > 0.0) {
        modulation = 1.0 + UNCERTAINTY_PUCT_SCALE * get_variance();
    }
    double u_value = C_PUCT * prior_probability *
                    (std::sqrt(total_parent_visits) / (1.0 + visit_count * modulation));
    return q_value + u_value;
}
