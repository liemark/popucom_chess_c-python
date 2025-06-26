#ifndef GAME_H
#define GAME_H

#include <cstdint>   // For uint64_t
#include <stdbool.h> // For bool type in C-compatible headers

// --- API 宏定义，用于导出函数 ---
#if defined(_WIN32) || defined(_WIN64)
#define API __declspec(dllexport)
#else
#define API
#endif

// --- 结构体定义 ---
typedef struct {
    uint64_t parts[2];
} Bitboards;

typedef struct {
    Bitboards pieces[2]; // 0 for Black, 1 for White
    Bitboards tiles[2];  // 0 for Black, 1 for White
    int current_player;
    int moves_left[2];
} Board;

// --- 常量和枚举定义 ---
#define BOARD_WIDTH 9
#define BOARD_HEIGHT 9
#define BOARD_SQUARES (BOARD_WIDTH * BOARD_HEIGHT) // 81
#define TOTAL_MOVES 25

enum Player { BLACK = 0, WHITE = 1 };
enum GameResult { IN_PROGRESS = -1, DRAW = 0, BLACK_WIN = 1, WHITE_WIN = 2 };

// --- 宏定义 (供C++内部高效使用) ---
#define get_row(sq) ((sq) / BOARD_WIDTH)
#define get_col(sq) ((sq) % BOARD_WIDTH)
#define get_sq(r, c) ((r) * BOARD_WIDTH + (c))
#define is_valid(r, c) ((r) >= 0 && (r) < BOARD_HEIGHT && (c) >= 0 && (c) < BOARD_WIDTH)

// 统一宏名称
#define GET_BIT(bb, sq) (((sq) >= 0 && (sq) < BOARD_SQUARES) ? (((bb).parts[(sq) / 64] >> ((sq) % 64)) & 1ULL) : 0)
#define SET_BIT(bb, sq) do { if ((sq) >= 0 && (sq) < BOARD_SQUARES) (bb).parts[(sq) / 64] |= (1ULL << ((sq) % 64)); } while(0)
#define CLEAR_BIT(bb, sq) do { if ((sq) >= 0 && (sq) < BOARD_SQUARES) (bb).parts[(sq) / 64] &= ~(1ULL << ((sq) % 64)); } while(0)

// extern "C" 确保 C++ 编译器以 C 语言的方式导出这些函数
#ifdef __cplusplus
extern "C" {
#endif

    // --- 函数声明 ---
    API void init_board(Board* board);
    API void copy_board(const Board* src, Board* dest);
    API void print_board(const Board* board);
    API Bitboards get_legal_moves(const Board* board);
    API enum GameResult get_game_result(const Board* board);
    API bool make_move(Board* board, int square);
    API int pop_count(const Bitboards* bb);
    API bool is_bit_set(const Bitboards* bb, int sq);

#ifdef __cplusplus
}
#endif

#endif // GAME_H
