import ctypes
import os
import platform

# --- Constants from C code ---
BOARD_WIDTH = 9
BOARD_SQUARES = 81
BLACK = 0
WHITE = 1

# --- Game Result Enums from C ---
IN_PROGRESS = 0
BLACK_WIN = 1
WHITE_WIN = 2
DRAW = 3


# --- C Library Loading ---
def load_library():
    """Loads the compiled C library based on the operating system."""
    lib_name = ""
    if platform.system() == "Windows":
        lib_name = "popucom_core.dll"
    elif platform.system() == "Linux":
        lib_name = "popucom_core.so"
    elif platform.system() == "Darwin":  # macOS
        lib_name = "popucom_core.dylib"
    else:
        raise RuntimeError("Unsupported operating system")

    if not os.path.exists(lib_name):
        print(f"错误: 未找到库文件 '{lib_name}'。")
        print("请确保您已经成功编译了C语言核心，并且该文件与此脚本在同一目录下。")
        exit()

    return ctypes.CDLL(os.path.abspath(lib_name))


c_lib = load_library()
print(f"成功加载 C 语言核心库: {c_lib._name}")


# --- C Struct Replication in Python ---
class Bitboards(ctypes.Structure):
    _fields_ = [("parts", ctypes.c_uint64 * 2)]


class Board(ctypes.Structure):
    _fields_ = [("pieces", Bitboards * 2),
                ("tiles", Bitboards * 2),
                ("current_player", ctypes.c_int),
                ("moves_left", ctypes.c_int * 2)]


# --- C Function Prototypes Definition ---
# This step is crucial to tell Python how to call C functions correctly.

# void init_board(Board* board);
init_board = c_lib.init_board
init_board.argtypes = [ctypes.POINTER(Board)]
init_board.restype = None

# void print_board(const Board* board);
print_board = c_lib.print_board
print_board.argtypes = [ctypes.POINTER(Board)]
print_board.restype = None

# bool make_move(Board* board, int square);
make_move = c_lib.make_move
make_move.argtypes = [ctypes.POINTER(Board), ctypes.c_int]
make_move.restype = ctypes.c_bool

# Bitboards get_legal_moves(const Board* board);
get_legal_moves = c_lib.get_legal_moves
get_legal_moves.argtypes = [ctypes.POINTER(Board)]
get_legal_moves.restype = Bitboards  # Returns a struct by value

# enum GameResult get_game_result(const Board* board);
get_game_result = c_lib.get_game_result
get_game_result.argtypes = [ctypes.POINTER(Board)]
get_game_result.restype = ctypes.c_int


# --- Python Helper Functions ---
def get_bit_py(bitboard: Bitboards, square: int) -> bool:
    """A Python helper to check if a bit is set in our Bitboards struct."""
    if not 0 <= square < BOARD_SQUARES:
        return False
    # This logic must match the C macro GET_BIT
    return (bitboard.parts[square // 64] >> (square % 64)) & 1 == 1


def print_welcome_message():
    """Prints a welcome message and instructions."""
    print("\n欢迎来到泡姆棋 (Popucom Chess) C语言核心版!")
    print("=========================================")
    print("两位玩家轮流下棋。黑方 (X) 先行。")
    print("输入棋盘上 0 到 80 的数字来落子。")
    print("棋盘坐标对应如下:")
    for r in range(BOARD_WIDTH):
        print(" ".join([f"{r * BOARD_WIDTH + c:2d}" for c in range(BOARD_WIDTH)]))
    print("=========================================\n")


# --- Main Game Loop ---
def main():
    """Main function to run the two-player game."""
    board = Board()
    init_board(ctypes.byref(board))

    print_welcome_message()

    while True:
        # 1. Check game state and print the board
        game_result = get_game_result(ctypes.byref(board))
        if game_result != IN_PROGRESS:
            print("\n--- 游戏结束 ---")
            if game_result == BLACK_WIN:
                print("结果: 黑方 (X) 获胜!")
            elif game_result == WHITE_WIN:
                print("结果: 白方 (O) 获胜!")
            elif game_result == DRAW:
                print("结果: 平局!")
            break  # Exit the game loop

        print_board(ctypes.byref(board))

        # 2. Get legal moves from C core
        legal_moves_bb = get_legal_moves(ctypes.byref(board))

        # 3. Prompt current player for input
        player_name = "黑方 (X)" if board.current_player == BLACK else "白方 (O)"

        while True:  # Input validation loop
            try:
                user_input = input(f"[{player_name}] 请输入您的落子位置 (0-80): ")
                square = int(user_input)

                # Check if the move is legal using our Python helper
                if get_bit_py(legal_moves_bb, square):
                    break  # Valid move, exit validation loop
                else:
                    print("无效落子! 该位置不能落子，请重新选择。")
            except (ValueError, IndexError):
                print("输入无效! 请输入一个 0 到 80 之间的数字。")

        # 4. Make the move by calling the C function
        success = make_move(ctypes.byref(board), square)
        if not success:
            # This should technically not happen due to the check above, but it's good practice
            print("错误: C核心返回落子失败。")

        print("\n" + "=" * 40 + "\n")  # Separator for the next turn


if __name__ == "__main__":
    main()
