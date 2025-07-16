#ifndef MCTS_MANAGER_H
#define MCTS_MANAGER_H

#include "game.h"
#include "mcts_search.h"
#include <vector>
#include <memory>
#include <mutex>
#include <unordered_map>

/**
 * @struct BoardHasher
 * @brief 为Board结构体提供哈希函数，使其可以作为unordered_map的键。
 */
struct BoardHasher {
    std::size_t operator()(const Board& b) const;
};

/**
 * @struct BoardEqual
 * @brief 为Board结构体提供相等比较函数，用于哈希表。
 */
struct BoardEqual {
    bool operator()(const Board& a, const Board& b) const;
};

/**
 * @class MCTSManager
 * @brief 管理多个并行的MCTS搜索实例。
 *
 * 这个类是连接Python和C++核心逻辑的桥梁。它负责：
 * 1. 维护所有并行游戏的当前棋盘状态。
 * 2. 持有一个置换表（transposition_table），用于在不同游戏间共享相同局面的MCTS搜索树，
 * 避免重复计算。
 * 3. 处理来自Python的请求，分发给对应的MCTSSearch实例。
 */
class MCTSManager {
public:
    // 置换表：键是棋盘状态，值是指向对应MCTS搜索树的共享指针。
    std::unordered_map<Board, std::shared_ptr<MCTSSearch>, BoardHasher, BoardEqual> transposition_table;

    // 存储所有并行游戏的当前棋盘状态。
    std::vector<Board> game_boards;

    // 临时存储当前等待神经网络评估的请求。
    std::vector<std::pair<Board, std::shared_ptr<MCTSSearch>>> pending_requests;

    // 互斥锁，用于保证多线程环境下的线程安全。
    std::mutex mtx;

    // 配置参数
    int num_games;
    bool enable_noise;
    double initial_fpu;

    /**
     * @brief 构造函数。
     * @param num_games_p 要管理的并行游戏数量。
     * @param enable_noise_p 是否启用狄利克雷噪声。
     * @param initial_fpu_p FPU的初始值。
     */
    MCTSManager(int num_games_p, bool enable_noise_p, double initial_fpu_p);

    /**
     * @brief 析构函数。
     */
    ~MCTSManager();
};

#endif // MCTS_MANAGER_H
