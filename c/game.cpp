#include "game.h"
#include <cstring> // For memset/memcpy
#include <iostream>

extern "C" {

    // --- pop_count 函数实现 ---
    int pop_count(const Bitboards* bb) {
        if (!bb) return 0;
        int count = 0;
        uint64_t p1 = bb->parts[0];
        uint64_t p2 = bb->parts[1];

        while (p1) {
            p1 &= (p1 - 1);
            count++;
        }
        while (p2) {
            p2 &= (p2 - 1);
            count++;
        }
        return count;
    }

    // --- is_bit_set 函数的实现 ---
    bool is_bit_set(const Bitboards* bb, int sq) {
        if (!bb) return false;
        return GET_BIT(*bb, sq) != 0;
    }

    // --- 公共函数实现 ---

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
                if (is_bit_set(&board->pieces[BLACK], sq))      std::cout << "X ";
                else if (is_bit_set(&board->pieces[WHITE], sq)) std::cout << "O ";
                else if (is_bit_set(&board->tiles[BLACK], sq))  std::cout << ". ";
                else if (is_bit_set(&board->tiles[WHITE], sq))  std::cout << "o ";
                else                                            std::cout << "- ";
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
            int black_score = pop_count(&board->tiles[BLACK]);
            int white_score = pop_count(&board->tiles[WHITE]);
            if (black_score > white_score) return BLACK_WIN;
            if (white_score > black_score) return WHITE_WIN;
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
        if (!is_bit_set(&legal_moves, square)) {
            return false;
        }

        int player = board->current_player;
        int opponent = 1 - player;

        SET_BIT(board->pieces[player], square);

        Bitboards elimination_mask = { 0, 0 };
        Bitboards coloring_lines_mask = { 0, 0 }; // 用来标记哪些方向(0-3)发生了消除
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
                    if (is_valid(r, c) && is_bit_set(&board->pieces[player], get_sq(r, c))) {
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
                SET_BIT(coloring_lines_mask, i); // 标记这个方向
            }
        }

        if (elimination_occurred) {
            board->pieces[player].parts[0] &= ~elimination_mask.parts[0];
            board->pieces[player].parts[1] &= ~elimination_mask.parts[1];

            Bitboards final_color_mask = { 0, 0 };
            final_color_mask.parts[0] = elimination_mask.parts[0];
            final_color_mask.parts[1] = elimination_mask.parts[1];

            for (int i = 0; i < 4; ++i) {
                if (is_bit_set(&coloring_lines_mask, i)) {
                    // 正向射线
                    for (int k = 1; k < BOARD_WIDTH; ++k) {
                        int r = get_row(square) + k * dr[i];
                        int c = get_col(square) + k * dc[i];
                        if (!is_valid(r, c)) break;
                        int ray_sq = get_sq(r, c);
                        if (is_bit_set(&board->pieces[opponent], ray_sq)) break;
                        SET_BIT(final_color_mask, ray_sq);
                    }
                    // 反向射线
                    for (int k = 1; k < BOARD_WIDTH; ++k) {
                        int r = get_row(square) - k * dr[i];
                        int c = get_col(square) - k * dc[i];
                        if (!is_valid(r, c)) break;
                        int ray_sq = get_sq(r, c);
                        if (is_bit_set(&board->pieces[opponent], ray_sq)) break;
                        SET_BIT(final_color_mask, ray_sq);
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
