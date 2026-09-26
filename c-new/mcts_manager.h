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
 * @struct TTKey
 * @brief [新增] 置换表键：局面 + 贴目。
 * 不同贴目下同一局面的价值/搜索行为不同，搜索树不可跨贴目共享，
 * 否则 Q 值会互相污染（贴目可翻转终局胜负）。
 */
struct TTKey {
    Board board;
    int komi;
    TTKey() : komi(0) {}
    TTKey(const Board& b, int k) : board(b), komi(k) {}
};

struct TTKeyHasher {
    std::size_t operator()(const TTKey& k) const;
};

struct TTKeyEqual {
    bool operator()(const TTKey& a, const TTKey& b) const;
};

/**
 * @class MCTSManager
 * @brief 管理多个并行的MCTS搜索实例，并实现一个带大小限制的置换表。
 */
class MCTSManager {
public:
    // 置换表 [修改]：键为 (Board, komi)，不同贴目的搜索树相互隔离
    std::unordered_map<TTKey, std::shared_ptr<MCTSSearch>, TTKeyHasher, TTKeyEqual> transposition_table;

    // MODIFIED: 用于实现FIFO缓存的队列
    std::queue<TTKey> tt_insertion_order;

    // MODIFIED: 置换表的最大容量
    const size_t max_tt_size;

    std::vector<Board> game_boards;
    std::vector<int> game_komi; // [新增] 每局贴目（白给黑的目数，0 = 无贴目）
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
