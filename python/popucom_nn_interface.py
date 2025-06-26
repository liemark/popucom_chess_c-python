# popucom_nn_interface.py

# --- Configuration ---
BOARD_SIZE = 9
MAX_MOVES_PER_PLAYER = 25 # 每位玩家的最大步数

# --- Input Tensor Channel Definitions ---
# This defines the "contract" for the input tensor that the C MCTS search
# will prepare and the Python PyTorch model will consume.
#
# Channels are from the perspective of the current player. The C++ code
# is responsible for flipping the board if the current player is White.
#
# Channel 0: Current player's pieces
# Channel 1: Opponent's pieces
# Channel 2: Current player's colored tiles
# Channel 3: Opponent's colored tiles
#
# --- Global State Channels ---
# These channels are constant across the entire board (broadcasted).
#
# Channel 4: Player to move is Black (1.0 if black to move, else 0.0)
# Channel 5: Player to move is White (1.0 if white to move, else 0.0)
# Channel 6: Black's moves left (normalized value)
# Channel 7: White's moves left (normalized value)
# Channel 8: Black's tile count (normalized value)
# Channel 9: White's tile count (normalized value)
# Channel 10: Unpainted tiles (1 for unpainted, 0 otherwise)
#
# The model will internally add 2 more channels for X and Y coordinates.
NUM_INPUT_CHANNELS = 11
