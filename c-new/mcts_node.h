#ifndef MCTS_NODE_H
#define MCTS_NODE_H

#include <cstddef> // for size_t
#include <limits>  // for std::numeric_limits

// 定义一个明确的常量作为无效索引
const size_t INVALID_INDEX = std::numeric_limits<size_t>::max();

// --- 不确定性加权参数 ---
// [新增] Uncertainty-Weighted MCTS Playouts (KataGo v1.9.0 风格)
// 每次 playout 的权重 w = 1 / (uncertainty + UNCERTAINTY_BASELINE)
// 该基线同时限制了单次 playout 的最大权重 (1/UNCERTAINTY_BASELINE)
// 调参建议：0.01 ~ 0.1 之间；太低会让"确定"局面的一步搜索垄断搜索树
extern const double UNCERTAINTY_BASELINE;
// [新增] 节点方差对 PUCT U 项的调制系数，默认 0 = 关闭（仅观察/记录方差）
// U 分母: 1 + W_child * (1 + UNCERTAINTY_PUCT_SCALE * Var_child)
extern const double UNCERTAINTY_PUCT_SCALE;

/**
 * @struct Node
 * @brief 代表MCTS搜索树中的一个节点。
 * * 存储了节点的父子关系、访问统计、价值评估以及来自神经网络的先验概率。
 * * [修改] 统计数据支持不确定性加权：
 *   - visit_count (W)  为加权访问次数 Σw，不再是整数
 *   - total_action_value (S1) 为 Σw·v
 *   - total_square_value (S2) 为 Σw·v²（新增，用于采样方差）
 *   - total_variance (SV) 为 Σw·σ²（新增，来自子树/NN 的方差质量）
 */
struct Node {
    // 索引，指向父节点和子节点块的起始位置
    size_t parent_idx = INVALID_INDEX;
    size_t children_start_idx = INVALID_INDEX;

    // 节点的属性
    int num_children = 0;
    int move_leading_to_this_node = -1; // 导致从父节点到达此节点的走法
    bool is_expanded = false;           // 该节点是否已被扩展（即已获得神经网络评估）

    // --- MCTS统计数据（不确定性加权）---
    double visit_count = 0.0;           // W: Σw 加权访问次数
    double total_action_value = 0.0;    // S1: Σw·v 从该节点角度看的加权和
    double total_square_value = 0.0;     // S2: Σw·v² 加权平方和（新增）
    double total_variance = 0.0;         // SV: Σw·σ² 方差质量（新增）
    int playout_count = 0;               // 真实模拟次数（停止准则用，新增，恒为整数计数）
    float prior_probability = 0.0;      // 来自神经网络的先验概率

    /**
     * @brief 默认构造函数
     */
    Node() = default;

    /**
     * @brief 构造函数
     * @param parent 父节点的索引
     * @param move 导致此节点的走法
     * @param prior 先验概率
     */
    Node(size_t parent, int move, float prior);

    /**
     * @brief 计算节点的Q值（平均行动价值）。
     * @return 从父节点视角看的该节点的平均价值（加权）。
     */
    double get_q_value() const;

    /**
     * @brief [新增] 计算该节点 Q 估计的方差（不确定度沿搜索树传播的结果）。
     * @return Var = SV/W + max(0, S2/W - Q_own²)
     *         即 子树方差的加权平均 + 采样方差项，方差对视角翻转不变。
     */
    double get_variance() const;

    /**
     * @brief 计算此节点的PUCT值。
     * * PUCT值用于在MCTS的“选择”阶段决定探索哪个子节点。
     * 它平衡了利用（选择Q值高的节点）和探索（选择访问次数少或先验概率高的节点）。
     * * [修改] 入参与内部统计均为 double（加权访问次数），
     * 且 U 项可被节点方差调制（UNCERTAINTY_PUCT_SCALE > 0 时生效）。
     * * @param total_parent_visits 父节点的加权总访问次数。
     * @param fpu_value First Play Urgency，用于未访问节点的Q值。
     * @return 该节点的PUCT分数。
     */
     // 修改：将传入固定的 fpu_value 改为传入父节点的平均 Q 值
    double get_puct_value(double total_parent_visits, double parent_q) const;
};

#endif // MCTS_NODE_H
