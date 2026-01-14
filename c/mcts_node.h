#ifndef MCTS_NODE_H
#define MCTS_NODE_H

#include <cstddef> // for size_t
#include <limits>  // for std::numeric_limits

// 定义一个明确的常量作为无效索引
const size_t INVALID_INDEX = std::numeric_limits<size_t>::max();

/**
 * @struct Node
 * @brief 代表MCTS搜索树中的一个节点。
 * * 存储了节点的父子关系、访问统计、价值评估以及来自神经网络的先验概率。
 */
struct Node {
    // 索引，指向父节点和子节点块的起始位置
    size_t parent_idx = INVALID_INDEX;
    size_t children_start_idx = INVALID_INDEX;

    // 节点的属性
    int num_children = 0;
    int move_leading_to_this_node = -1; // 导致从父节点到达此节点的走法
    bool is_expanded = false;           // 该节点是否已被扩展（即已获得神经网络评估）

    // MCTS统计数据
    int visit_count = 0;                // 该节点被访问的次数
    double total_action_value = 0.0;    // 从该节点角度看的所有子树评估价值的总和
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
     * @return 从父节点视角看的该节点的平均价值。
     */
    double get_q_value() const;

    /**
     * @brief 计算此节点的PUCT值。
     * * PUCT值用于在MCTS的“选择”阶段决定探索哪个子节点。
     * 它平衡了利用（选择Q值高的节点）和探索（选择访问次数少或先验概率高的节点）。
     * * @param total_parent_visits 父节点的总访问次数。
     * @param fpu_value First Play Urgency，用于未访问节点的Q值。
     * @return 该节点的PUCT分数。
     */
     // 修改：将传入固定的 fpu_value 改为传入父节点的平均 Q 值
    double get_puct_value(int total_parent_visits, double parent_q) const;
};

#endif // MCTS_NODE_H
