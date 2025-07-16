#include "mcts_manager.h"
#include "game.h"       // For init_board
#include <cstring>      // For memcmp

// --- BoardHasher Implementation ---
std::size_t BoardHasher::operator()(const Board& b) const {
    // A simple but effective hash function combining all parts of the board state.
    // It uses bitwise XOR and left shifts to mix the bits from different components.
    size_t h1 = b.pieces[0].parts[0] ^ b.tiles[0].parts[0];
    size_t h2 = b.pieces[0].parts[1] ^ b.tiles[0].parts[1];
    size_t h3 = b.pieces[1].parts[0] ^ b.tiles[1].parts[0];
    size_t h4 = b.pieces[1].parts[1] ^ b.tiles[1].parts[1];
    size_t h5 = b.current_player;
    size_t h6 = b.moves_left[0];
    size_t h7 = b.moves_left[1];

    // Combine all hash parts
    return h1 ^ (h2 << 1) ^ (h3 << 2) ^ (h4 << 3) ^ (h5 << 4) ^ (h6 << 5) ^ (h7 << 6);
}

// --- BoardEqual Implementation ---
bool BoardEqual::operator()(const Board& a, const Board& b) const {
    // memcmp is a fast way to check if two structs have identical byte-for-byte content.
    return memcmp(&a, &b, sizeof(Board)) == 0;
}

// --- MCTSManager Implementation ---
MCTSManager::MCTSManager(int num_games_p, bool enable_noise_p, double initial_fpu_p)
    : num_games(num_games_p),
    enable_noise(enable_noise_p),
    initial_fpu(initial_fpu_p) {
    // Resize the vector to hold all game states
    game_boards.resize(num_games);
    // Initialize each game board to the starting position
    for (int i = 0; i < num_games; ++i) {
        init_board(&game_boards[i]);
    }
}

MCTSManager::~MCTSManager() {
    // The smart pointers (std::shared_ptr) in the transposition_table
    // will automatically manage the memory of the MCTSSearch objects.
    // So, the destructor can be empty.
}
