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

// --- MCTSManager Implementation ---
MCTSManager::MCTSManager(int num_games_p, bool enable_noise_p, double initial_fpu_p, size_t tt_size)
    : max_tt_size(tt_size), // MODIFIED: 初始化最大容量
    num_games(num_games_p),
    enable_noise(enable_noise_p),
    initial_fpu(initial_fpu_p) {
    game_boards.resize(num_games);
    for (int i = 0; i < num_games; ++i) {
        init_board(&game_boards[i]);
    }
}

MCTSManager::~MCTSManager() {
    // Destructor remains empty, smart pointers handle memory.
}
