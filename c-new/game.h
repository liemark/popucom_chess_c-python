#ifndef GAME_H
#define GAME_H

#include <cstdint>
#include <stdbool.h>

#if defined(_WIN32) || defined(_WIN64)
#define API __declspec(dllexport)
#else
#define API
#endif

typedef struct {
    uint64_t parts[2];
} Bitboards;

typedef struct {
    Bitboards pieces[2];
    Bitboards tiles[2];
    int current_player;
    int moves_left[2];
} Board;

#define BOARD_WIDTH 9
#define BOARD_HEIGHT 9
#define BOARD_SQUARES (BOARD_WIDTH * BOARD_HEIGHT)
#define TOTAL_MOVES 25

enum Player { BLACK = 0, WHITE = 1 };
enum GameResult { IN_PROGRESS = -1, DRAW = 0, BLACK_WIN = 1, WHITE_WIN = 2 };

#define get_row(sq) ((sq) / BOARD_WIDTH)
#define get_col(sq) ((sq) % BOARD_WIDTH)
#define get_sq(r, c) ((r) * BOARD_WIDTH + (c))
#define is_valid(r, c) ((r) >= 0 && (r) < BOARD_HEIGHT && (c) >= 0 && (c) < BOARD_WIDTH)

#define GET_BIT(bb, sq) (((sq) >= 0 && (sq) < BOARD_SQUARES) ? (((bb).parts[(sq) / 64] >> ((sq) % 64)) & 1ULL) : 0)
#define SET_BIT(bb, sq) do { if ((sq) >= 0 && (sq) < BOARD_SQUARES) (bb).parts[(sq) / 64] |= (1ULL << ((sq) % 64)); } while(0)
#define CLEAR_BIT(bb, sq) do { if ((sq) >= 0 && (sq) < BOARD_SQUARES) (bb).parts[(sq) / 64] &= ~(1ULL << ((sq) % 64)); } while(0)

#ifdef __cplusplus
extern "C" {
#endif

    API void init_board(Board* board);
    API void copy_board(const Board* src, Board* dest);
    API void print_board(const Board* board);
    API Bitboards get_legal_moves(const Board* board);
    API enum GameResult get_game_result(const Board* board);
    API bool make_move(Board* board, int square);
    API int pop_count(const Bitboards* bb);
    API bool is_bit_set(const Bitboards* bb, int sq);
    API int get_score_diff(const Board* board); // NEW: Export the score difference function.

#ifdef __cplusplus
}
#endif

#endif // GAME_H
