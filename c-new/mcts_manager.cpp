#include "mcts_manager.h"
#include "game.h"
#include <cstring>

// --- BoardHasher Implementation ---
std::size_t BoardHasher::operator()(const Board& b) const {
    size_t h1 = b.pieces[0].parts[0] ^ b.tiles[0].parts[0];
    size_t h2 = b.pieces[0].parts[1] ^ b.tiles[0].parts[1];
    size_t h3 = b.pieces[1].parts[0] ^ b.tiles[1].parts[0];
    size_t h4 = b.pieces[1].parts[1] ^ b.tiles[1].parts[1];
    size_t h5 = b.current_player;
    size_t h6 = b.moves_left[0];
    size_t h7 = b.moves_left[1];
    return h1 ^ (h2 << 1) ^ (h3 << 2) ^ (h4 << 3) ^ (h5 << 4) ^ (h6 << 5) ^ (h7 << 6);
}

// --- BoardEqual Implementation ---
bool BoardEqual::operator()(const Board& a, const Board& b) const {
    return memcmp(&a, &b, sizeof(Board)) == 0;
}

// --- TTKeyHasher / TTKeyEqual Implementation ---
// [新增] 贴目参与哈希与相等判定：不同贴目 = 不同的搜索问题
std::size_t TTKeyHasher::operator()(const TTKey& k) const {
    BoardHasher board_hasher;
    // 黄金比例散列混合贴目（int 上的avalanche）
    return board_hasher(k.board) ^ (static_cast<std::size_t>(static_cast<unsigned>(k.komi)) * 0x9E3779B97F4A7C15ULL);
}

bool TTKeyEqual::operator()(const TTKey& a, const TTKey& b) const {
    return a.komi == b.komi && BoardEqual()(a.board, b.board);
}

// --- MCTSManager Implementation ---
MCTSManager::MCTSManager(int num_games_p, bool enable_noise_p, double initial_fpu_p, size_t tt_size)
    : max_tt_size(tt_size), // MODIFIED: 初始化最大容量
    num_games(num_games_p),
    enable_noise(enable_noise_p),
    initial_fpu(initial_fpu_p) {
    game_boards.resize(num_games);
    game_komi.assign(num_games, 0); // [新增] 默认无贴目（与旧行为等价）
    for (int i = 0; i < num_games; ++i) {
        init_board(&game_boards[i]);
    }
}

MCTSManager::~MCTSManager() {
    // Destructor remains empty, smart pointers handle memory.
}
