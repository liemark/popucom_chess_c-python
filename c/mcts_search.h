#ifndef MCTS_SEARCH_H
#define MCTS_SEARCH_H

#include <vector>
#include <memory>
#include <random>
#include "game.h"      // 包含游戏逻辑的定义
#include "mcts_node.h" // 包含Node结构体的定义

// --- MCTS参数定义 ---
// 这些参数现在属于MCTS搜索模块的一部分
extern const float C_PUCT;
extern const float NOISE_RATIO;
extern const float POLICY_SHARPENING_Q_THRESHOLD;
extern const float POLICY_SHARPENING_FACTOR;
extern const size_t INITIAL_NODE_STORE_CAPACITY;

/**
 * @class NodeStore
 * @brief 一个高效的节点存储容器。
 * * 使用一个连续的vector来存储所有的MCTS节点，以提高缓存命中率和内存使用效率。
 * 节点的父子关系通过索引来维护。
 */
class NodeStore {
private:
    std::vector<Node> nodes;

public:
    NodeStore();
    void clear();
    Node& operator[](size_t index);
    const Node& operator[](size_t index) const;
    void swap(std::vector<Node>& other);
    size_t add_node(size_t parent_idx, int move, float prior);
    void add_children(size_t parent_idx, int count, const std::vector<int>& moves, const float* policy);
    size_t size() const;
};

/**
 * @class MCTSSearch
 * @brief 负责单个MCTS（蒙特卡洛树搜索）实例。
 * * 这个类封装了对一个特定游戏局面进行MCTS所需的所有逻辑，
 * 包括选择(Select)、扩展(Expand)、评估(Evaluate)和反向传播(Backpropagate)。
 * 每个MCTSSearch实例都拥有自己的搜索树（通过NodeStore管理）。
 */
class MCTSSearch {
private:
    std::mt19937 rng; // 用于狄利克雷噪声的随机数生成器

    /**
     * @brief 在根节点应用狄利克雷噪声以鼓励探索。
     * @param node_idx 根节点的索引。
     */
    void apply_dirichlet_noise(size_t node_idx);

    /**
     * @brief 从根节点开始，根据PUCT值向下选择一个叶子节点。
     * @param current_board 一个临时的Board对象，用于在选择过程中模拟走子。
     * @return 叶子节点的索引。
     */
    size_t select_leaf(Board* current_board);

    /**
     * @brief 在反向传播阶段更新路径上所有节点的统计数据。
     * @param leaf_idx 开始反向传播的叶子节点索引。
     * @param value 从叶子节点传回的评估价值。
     */
    void backpropagate(size_t leaf_idx, float value);

    /**
     * @brief 根据PUCT值选择一个父节点的最佳子节点。
     * @param parent_idx 父节点的索引。
     * @return 最佳子节点在其兄弟节点中的偏移量。
     */
    size_t get_best_child_offset(size_t parent_idx);

public:
    NodeStore node_store;
    size_t root_idx;
    Board root_board; // 存储当前搜索树的根局面状态
    size_t pending_evaluation_leaf_idx; // 等待神经网络评估的叶子节点索引
    bool add_dirichlet_noise;
    double fpu_value;

    /**
     * @brief 构造函数。
     * @param board_state 搜索树的初始根局面。
     * @param enable_noise 是否在根节点启用狄利克雷噪声。
     * @param initial_fpu First Play Urgency的初始值。
     */
    MCTSSearch(const Board* board_state, bool enable_noise, double initial_fpu);

    /**
     * @brief 重置搜索树以匹配一个新的棋盘状态。
     * @param new_board_state 新的根局面。
     */
    void reset(const Board* new_board_state);

    /**
     * @brief 获取当前搜索已经完成的模拟次数。
     * @return 根节点的访问次数。
     */
    int get_simulations_done() const;

    /**
     * @brief 执行一次完整的MCTS模拟（选择->可能扩展）。
     * @param leaf_board_out [out] 如果需要网络评估，这里会存储叶子节点的棋盘状态。
     */
    void run_simulation(Board& leaf_board_out);

    /**
     * @brief 使用神经网络的评估结果来扩展一个叶子节点。
     * @param board_at_leaf 叶子节点的棋盘状态。
     * @param policy 神经网络输出的策略向量。
     * @param value 神经网络输出的价值。
     */
    void expand_and_evaluate(const Board& board_at_leaf, const float* policy, float value);

    /**
     * @brief 从MCTS的访问次数中计算出最终的策略分布。
     * @param policy_buffer [out] 用于存储计算出的策略的缓冲区。
     */
    void get_policy(float* policy_buffer);
};

#endif // MCTS_SEARCH_H
