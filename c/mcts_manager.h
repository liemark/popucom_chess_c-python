#ifndef MCTS_MANAGER_H
#define MCTS_MANAGER_H

#include "game.h"
#include "mcts_search.h"
#include <vector>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <queue> // 用于实现FIFO队列

/**
 * @struct BoardHasher
 * @brief 为Board结构体提供哈希函数。
 */
struct BoardHasher {
    std::size_t operator()(const Board& b) const;
};

/**
 * @struct BoardEqual
 * @brief 为Board结构体提供相等比较函数。
 */
struct BoardEqual {
    bool operator()(const Board& a, const Board& b) const;
};

/**
 * @class MCTSManager
 * @brief 管理多个并行的MCTS搜索实例，并实现一个带大小限制的置换表。
 */
class MCTSManager {
public:
    // 置换表
    std::unordered_map<Board, std::shared_ptr<MCTSSearch>, BoardHasher, BoardEqual> transposition_table;

    // MODIFIED: 用于实现FIFO缓存的队列
    std::queue<Board> tt_insertion_order;

    // MODIFIED: 置换表的最大容量
    const size_t max_tt_size;

    std::vector<Board> game_boards;
    std::vector<std::pair<Board, std::shared_ptr<MCTSSearch>>> pending_requests;
    std::mutex mtx;

    int num_games;
    bool enable_noise;
    double initial_fpu;

    /**
     * @brief 构造函数。
     * @param num_games_p 要管理的并行游戏数量。
     * @param enable_noise_p 是否启用狄利克雷噪声。
     * @param initial_fpu_p FPU的初始值。
     * @param tt_size 置换表的最大容量。
     */
    MCTSManager(int num_games_p, bool enable_noise_p, double initial_fpu_p, size_t tt_size);

    ~MCTSManager();
};

#endif // MCTS_MANAGER_H
