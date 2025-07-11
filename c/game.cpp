#include "game.h"
#include <cstring>
#include <iostream>

// For pop_count optimization
#if defined(__GNUC__) || defined(__clang__)
#include <x86intrin.h>
#endif
#if defined(_MSC_VER)
#include <intrin.h>
#endif


extern "C" {

    int pop_count(const Bitboards* bb) {
        if (!bb) return 0;
        // OPTIMIZED: Use compiler intrinsics for popcount, which are much faster.
#if defined(__GNUC__) || defined(__clang__)
        return __builtin_popcountll(bb->parts[0]) + __builtin_popcountll(bb->parts[1]);
#elif defined(_MSC_VER)
        return (int)__popcnt64(bb->parts[0]) + (int)__popcnt64(bb->parts[1]);
#else
        // Fallback to the classic algorithm if no intrinsic is available.
        int count = 0;
        uint64_t p1 = bb->parts[0];
        uint64_t p2 = bb->parts[1];
        while (p1) { p1 &= (p1 - 1); count++; }
        while (p2) { p2 &= (p2 - 1); count++; }
        return count;
#endif
    }

    // NEW: Function to get the raw score difference (Black - White)
    int get_score_diff(const Board* board) {
        if (!board) return 0;
        return pop_count(&board->tiles[BLACK]) - pop_count(&board->tiles[WHITE]);
    }

    bool is_bit_set(const Bitboards* bb, int sq) {
        if (!bb) return false;
        return GET_BIT(*bb, sq) != 0;
    }

    void init_board(Board* board) {
        if (!board) return;
        memset(board, 0, sizeof(Board));
        board->current_player = BLACK;
        board->moves_left[BLACK] = TOTAL_MOVES;
        board->moves_left[WHITE] = TOTAL_MOVES;
    }

    void copy_board(const Board* src, Board* dest) {
        if (!src || !dest) return;
        memcpy(dest, src, sizeof(Board));
    }

    void print_board(const Board* board) {
        if (!board) return;
        std::cout << "  a b c d e f g h i" << std::endl;
        for (int r = 0; r < BOARD_HEIGHT; ++r) {
            std::cout << r + 1 << " ";
            for (int c = 0; c < BOARD_WIDTH; ++c) {
                int sq = get_sq(r, c);
                if (GET_BIT(board->pieces[BLACK], sq))      std::cout << "X ";
                else if (GET_BIT(board->pieces[WHITE], sq)) std::cout << "O ";
                else if (GET_BIT(board->tiles[BLACK], sq))  std::cout << ". ";
                else if (GET_BIT(board->tiles[WHITE], sq))  std::cout << "o ";
                else                                        std::cout << "- ";
            }
            std::cout << std::endl;
        }
        std::cout << "Player: " << (board->current_player == BLACK ? "Black(X)" : "White(O)")
            << ", Moves left: B:" << board->moves_left[BLACK]
            << " W:" << board->moves_left[WHITE] << std::endl;
    }

    Bitboards get_legal_moves(const Board* board) {
        Bitboards legal_moves = { 0, 0 };
        if (!board) return legal_moves;

        Bitboards all_pieces = { 0, 0 };
        const Bitboards opponent_tiles = board->tiles[1 - board->current_player];

        all_pieces.parts[0] = board->pieces[BLACK].parts[0] | board->pieces[WHITE].parts[0];
        all_pieces.parts[1] = board->pieces[BLACK].parts[1] | board->pieces[WHITE].parts[1];

        legal_moves.parts[0] = ~all_pieces.parts[0] & ~opponent_tiles.parts[0];
        legal_moves.parts[1] = ~all_pieces.parts[1] & ~opponent_tiles.parts[1];

        uint64_t mask_part1 = ~0ULL;
        uint64_t mask_part2 = 0;

        if (BOARD_SQUARES < 64) {
            mask_part1 = (1ULL << BOARD_SQUARES) - 1;
        }
        else if (BOARD_SQUARES > 64) {
            int bits_in_part2 = BOARD_SQUARES - 64;
            if (bits_in_part2 > 0 && bits_in_part2 < 64) {
                mask_part2 = (1ULL << bits_in_part2) - 1;
            }
            else if (bits_in_part2 >= 64) {
                mask_part2 = ~0ULL;
            }
        }

        legal_moves.parts[0] &= mask_part1;
        legal_moves.parts[1] &= mask_part2;

        return legal_moves;
    }

    enum GameResult get_game_result(const Board* board) {
        if (!board) return DRAW;
        if (board->moves_left[BLACK] <= 0 && board->moves_left[WHITE] <= 0) {
            int score_diff = get_score_diff(board);
            if (score_diff > 0) return BLACK_WIN;
            if (score_diff < 0) return WHITE_WIN;
            return DRAW;
        }

        Bitboards legal_moves = get_legal_moves(board);
        if (legal_moves.parts[0] == 0 && legal_moves.parts[1] == 0) {
            return (board->current_player == BLACK) ? WHITE_WIN : BLACK_WIN;
        }
        return IN_PROGRESS;
    }

    bool make_move(Board* board, int square) {
        if (!board || square < 0 || square >= BOARD_SQUARES) return false;

        Bitboards legal_moves = get_legal_moves(board);
        if (!GET_BIT(legal_moves, square)) {
            return false;
        }

        int player = board->current_player;
        int opponent = 1 - player;

        SET_BIT(board->pieces[player], square);

        Bitboards elimination_mask = { 0, 0 };
        Bitboards coloring_lines_mask = { 0, 0 };
        bool elimination_occurred = false;

        int dr[] = { 0, 1, -1, 1 };
        int dc[] = { 1, 0, 1, 1 };

        for (int i = 0; i < 4; ++i) {
            int line_len = 1;
            Bitboards current_line = { 0, 0 };
            SET_BIT(current_line, square);

            for (int dir = -1; dir <= 1; dir += 2) {
                for (int k = 1; k < BOARD_WIDTH; ++k) {
                    int r = get_row(square) + dir * k * dr[i];
                    int c = get_col(square) + dir * k * dc[i];
                    if (is_valid(r, c) && GET_BIT(board->pieces[player], get_sq(r, c))) {
                        line_len++;
                        SET_BIT(current_line, get_sq(r, c));
                    }
                    else {
                        break;
                    }
                }
            }

            if (line_len >= 3) {
                elimination_occurred = true;
                elimination_mask.parts[0] |= current_line.parts[0];
                elimination_mask.parts[1] |= current_line.parts[1];
                SET_BIT(coloring_lines_mask, i);
            }
        }

        if (elimination_occurred) {
            board->pieces[player].parts[0] &= ~elimination_mask.parts[0];
            board->pieces[player].parts[1] &= ~elimination_mask.parts[1];

            Bitboards final_color_mask = { 0, 0 };
            final_color_mask.parts[0] = elimination_mask.parts[0];
            final_color_mask.parts[1] = elimination_mask.parts[1];

            for (int i = 0; i < 4; ++i) {
                if (GET_BIT(coloring_lines_mask, i)) {
                    for (int dir = -1; dir <= 1; dir += 2) {
                        for (int k = 1; k < BOARD_WIDTH; ++k) {
                            int r = get_row(square) + dir * k * dr[i];
                            int c = get_col(square) + dir * k * dc[i];
                            if (!is_valid(r, c)) break;
                            int ray_sq = get_sq(r, c);
                            if (GET_BIT(board->pieces[opponent], ray_sq)) break;
                            SET_BIT(final_color_mask, ray_sq);
                        }
                    }
                }
            }

            board->tiles[player].parts[0] |= final_color_mask.parts[0];
            board->tiles[player].parts[1] |= final_color_mask.parts[1];
            board->tiles[opponent].parts[0] &= ~final_color_mask.parts[0];
            board->tiles[opponent].parts[1] &= ~final_color_mask.parts[1];
        }

        board->moves_left[player]--;
        board->current_player = opponent;

        return true;
    }

} // extern "C"
