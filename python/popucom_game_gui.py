import tkinter as tk
from tkinter import messagebox, Scale, Frame, Label, Button, Radiobutton, StringVar
import os
import torch
import numpy as np
import random
import threading
import time
import platform
import ctypes

# --- 模块导入 ---
try:
    from popucom_nn_model import PomPomNN
except ImportError as e:
    messagebox.showerror("导入错误",
                         f"无法导入神经网络模型 'PomPomNN'。请确保 'popucom_nn_model.py' 文件存在。\n错误: {e}")
    exit()


# --- C++ 库接口定义 ---
class Bitboards(ctypes.Structure):
    _fields_ = [("parts", ctypes.c_uint64 * 2)]


class Board(ctypes.Structure):
    _fields_ = [
        ("pieces", Bitboards * 2),
        ("tiles", Bitboards * 2),
        ("current_player", ctypes.c_int),
        ("moves_left", ctypes.c_int * 2)
    ]


def setup_c_library():
    """加载 C++ 动态库并设置所有函数的接口。"""
    lib_name = "popucom_core.dll" if platform.system() == "Windows" else "popucom_core.so"
    if not os.path.exists(lib_name):
        raise FileNotFoundError(f"未找到 C++ 库 '{lib_name}'。请确保已成功编译 C++ 代码。")

    c_lib = ctypes.CDLL(os.path.abspath(lib_name))

    # --- MCTS (puct.h) 函数接口 ---
    c_lib.create_mcts_manager.argtypes = [ctypes.c_int]
    c_lib.create_mcts_manager.restype = ctypes.c_void_p
    c_lib.destroy_mcts_manager.argtypes = [ctypes.c_void_p]
    c_lib.mcts_run_simulations_and_get_requests.argtypes = [ctypes.c_void_p, ctypes.POINTER(Board),
                                                            ctypes.POINTER(ctypes.c_int), ctypes.c_int]
    c_lib.mcts_run_simulations_and_get_requests.restype = ctypes.c_int
    c_lib.mcts_feed_results.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float)]
    c_lib.mcts_get_policy.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_float)]
    c_lib.mcts_get_policy.restype = ctypes.c_bool
    c_lib.mcts_make_move.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    c_lib.mcts_is_game_over.argtypes = [ctypes.c_void_p, ctypes.c_int]
    c_lib.mcts_is_game_over.restype = ctypes.c_bool
    c_lib.mcts_get_final_value.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    c_lib.mcts_get_final_value.restype = ctypes.c_float
    c_lib.mcts_get_board_state.argtypes = [ctypes.c_void_p, ctypes.c_int]
    c_lib.mcts_get_board_state.restype = ctypes.POINTER(Board)
    c_lib.mcts_get_simulations_done.argtypes = [ctypes.c_void_p, ctypes.c_int]
    c_lib.mcts_get_simulations_done.restype = ctypes.c_int

    # --- 游戏逻辑 (game.h) 函数接口 ---
    c_lib.get_legal_moves.argtypes = [ctypes.POINTER(Board)]
    c_lib.get_legal_moves.restype = Bitboards
    c_lib.get_game_result.argtypes = [ctypes.POINTER(Board)]
    c_lib.get_game_result.restype = ctypes.c_int
    c_lib.is_bit_set.argtypes = [ctypes.POINTER(Bitboards), ctypes.c_int]
    c_lib.is_bit_set.restype = ctypes.c_bool
    c_lib.pop_count.argtypes = [ctypes.POINTER(Bitboards)]
    c_lib.pop_count.restype = ctypes.c_int

    return c_lib


# --- 全局常量 ---
BOARD_SIZE = 9
BOARD_SQUARES = BOARD_SIZE * BOARD_SIZE
BLACK_PLAYER, WHITE_PLAYER = 0, 1

# --- GUI 颜色配置 ---
BOARD_BACKGROUND_COLOR = "#D2B48C"
GRID_LINE_COLOR = "#8B4513"
UNPAINTED_FLOOR_COLOR = "#FFF8DC"
BLACK_PAINTED_FLOOR_COLOR = "#FFDAB9"
WHITE_PAINTED_FLOOR_COLOR = "#90EE90"
BLACK_PIECE_COLOR = "#FF0000"
WHITE_PIECE_COLOR = "#008000"


# --- GUI 主类 ---
class PomPomGameGUI:
    def __init__(self, master):
        self.master = master
        master.title("泡姆棋 (C++ 核心)")

        try:
            self.c_lib = setup_c_library()
        except Exception as e:
            messagebox.showerror("C++ 库加载失败", f"无法加载或设置 C++ 库接口。\n错误: {e}")
            master.destroy()
            return

        self.mcts_manager = None
        self.game_running = False
        self.game_mode = StringVar(value="human_vs_ai")
        self.human_player_choice = StringVar(value="human_black")
        self.human_player = BLACK_PLAYER
        self.move_history = []
        self.analysis_policy = None

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ai_model = PomPomNN().to(self.device)
        self._load_ai_model()
        self.ai_model.eval()

        self._setup_gui()
        self._reset_and_start_new_game()

    def _setup_gui(self):
        self.cell_size = 60
        self.canvas = tk.Canvas(self.master, width=self.cell_size * BOARD_SIZE, height=self.cell_size * BOARD_SIZE,
                                bg=BOARD_BACKGROUND_COLOR)
        self.canvas.pack(pady=10)
        self.canvas.bind("<Button-1>", self._handle_click)
        self.master.bind("<Configure>", self._on_resize)

        self.status_label = Label(self.master, text="请开始新游戏", font=("Arial", 14, "bold"))
        self.status_label.pack(pady=(10, 0))

        # 新增: 用于显示步数和涂色数量的信息标签
        self.info_label = Label(self.master, text="", font=("Arial", 10))
        self.info_label.pack(pady=(0, 10))

        control_frame = Frame(self.master)
        control_frame.pack(pady=5)

        Radiobutton(control_frame, text="人机对战", variable=self.game_mode, value="human_vs_ai",
                    command=self._reset_and_start_new_game).pack(side=tk.LEFT)
        Radiobutton(control_frame, text="人人对战", variable=self.game_mode, value="human_vs_human",
                    command=self._reset_and_start_new_game).pack(side=tk.LEFT)
        Radiobutton(control_frame, text="机机对战", variable=self.game_mode, value="ai_vs_ai",
                    command=self._reset_and_start_new_game).pack(side=tk.LEFT)

        self.role_frame = Frame(self.master)
        self.role_frame.pack(pady=5)
        Radiobutton(self.role_frame, text="我执红 (先手)", variable=self.human_player_choice, value="human_black",
                    command=self._reset_and_start_new_game).pack(side=tk.LEFT)
        Radiobutton(self.role_frame, text="AI执红 (先手)", variable=self.human_player_choice, value="ai_black",
                    command=self._reset_and_start_new_game).pack(side=tk.LEFT)

        action_frame = Frame(self.master)
        action_frame.pack(pady=5)
        Button(action_frame, text="新游戏", command=self._reset_and_start_new_game).pack(side=tk.LEFT, padx=5)
        self.undo_button = Button(action_frame, text="悔棋", command=self._undo_move)
        self.undo_button.pack(side=tk.LEFT, padx=5)
        self.analyze_button = Button(action_frame, text="分析", command=self._analyze_board)
        self.analyze_button.pack(side=tk.LEFT, padx=5)

        ai_frame = Frame(self.master)
        ai_frame.pack(pady=10)
        Label(ai_frame, text="AI 搜索次数:").pack(side=tk.LEFT)
        self.ai_sims_slider = Scale(ai_frame, from_=50, to=5000, orient=tk.HORIZONTAL, resolution=50, length=200)
        self.ai_sims_slider.set(800)
        self.ai_sims_slider.pack(side=tk.LEFT)

        Label(ai_frame, text="AI 温度:").pack(side=tk.LEFT, padx=(10, 0))
        self.temperature_slider = Scale(ai_frame, from_=0.0, to=2.0, orient=tk.HORIZONTAL, resolution=0.1, length=150)
        self.temperature_slider.set(0.1)
        self.temperature_slider.pack(side=tk.LEFT)

    def _load_ai_model(self):
        model_path = "model.pth"
        if os.path.exists(model_path):
            try:
                self.ai_model.load_state_dict(torch.load(model_path, map_location=self.device))
                print(f"成功从 '{model_path}' 加载模型。")
            except Exception as e:
                messagebox.showwarning("模型加载警告", f"无法加载模型: {e}\nAI将使用随机权重。")
        else:
            print("未找到模型文件。AI将使用随机权重。")

    def _reset_and_start_new_game(self):
        if self.mcts_manager:
            self.c_lib.destroy_mcts_manager(self.mcts_manager)

        self.mcts_manager = self.c_lib.create_mcts_manager(1)
        self.game_running = True
        self.move_history.clear()
        self.analysis_policy = None

        self.human_player = BLACK_PLAYER if self.human_player_choice.get() == "human_black" else WHITE_PLAYER

        is_human_vs_ai = self.game_mode.get() == "human_vs_ai"
        for widget in self.role_frame.winfo_children():
            widget.config(state=tk.NORMAL if is_human_vs_ai else tk.DISABLED)

        self._start_game_flow()

    def _start_game_flow(self):
        self._draw_board()
        self._update_status()

        if not self.game_running:
            return

        mode = self.game_mode.get()
        board_ptr = self._get_board_state()
        if not board_ptr: return
        current_player = board_ptr.contents.current_player

        is_ai_turn = (mode == "ai_vs_ai") or (mode == "human_vs_ai" and current_player != self.human_player)

        if is_ai_turn:
            self.master.after(200, self._ai_turn)

    def _get_board_state(self):
        if not self.mcts_manager: return None
        return self.c_lib.mcts_get_board_state(self.mcts_manager, 0)

    def _draw_board(self):
        self.canvas.delete("all")
        board_ptr = self._get_board_state()
        if not board_ptr: return
        board = board_ptr.contents

        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                x1, y1 = c * self.cell_size, r * self.cell_size
                x2, y2 = x1 + self.cell_size, y1 + self.cell_size
                sq = r * BOARD_SIZE + c

                floor_color = UNPAINTED_FLOOR_COLOR
                if self.c_lib.is_bit_set(ctypes.byref(board.tiles[BLACK_PLAYER]), sq):
                    floor_color = BLACK_PAINTED_FLOOR_COLOR
                elif self.c_lib.is_bit_set(ctypes.byref(board.tiles[WHITE_PLAYER]), sq):
                    floor_color = WHITE_PAINTED_FLOOR_COLOR
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=floor_color, outline="")

        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                x1, y1 = c * self.cell_size, r * self.cell_size
                x2, y2 = x1 + self.cell_size, y1 + self.cell_size
                center_x, center_y = x1 + self.cell_size / 2, y1 + self.cell_size / 2
                radius = self.cell_size / 2 * 0.8
                sq = r * BOARD_SIZE + c

                self.canvas.create_rectangle(x1, y1, x2, y2, outline=GRID_LINE_COLOR)

                piece_color = None
                if self.c_lib.is_bit_set(ctypes.byref(board.pieces[BLACK_PLAYER]), sq):
                    piece_color = BLACK_PIECE_COLOR
                elif self.c_lib.is_bit_set(ctypes.byref(board.pieces[WHITE_PLAYER]), sq):
                    piece_color = WHITE_PIECE_COLOR
                if piece_color:
                    self.canvas.create_oval(center_x - radius, center_y - radius, center_x + radius, center_y + radius,
                                            fill=piece_color)

        if self.analysis_policy is not None:
            max_prob = np.max(self.analysis_policy) if np.sum(self.analysis_policy) > 0 else 0
            for r in range(BOARD_SIZE):
                for c in range(BOARD_SIZE):
                    prob = self.analysis_policy[r * BOARD_SIZE + c]
                    if prob > 0.001:
                        center_x = c * self.cell_size + self.cell_size / 2
                        center_y = r * self.cell_size + self.cell_size / 2
                        text_color = "darkgreen" if prob >= max_prob * 0.9 else "blue"
                        font_size = max(8, int(self.cell_size / 6))
                        self.canvas.create_text(center_x, center_y, text=f"{prob:.2f}", fill=text_color,
                                                font=("Arial", font_size, "bold"))

    def _update_status(self):
        board_ptr = self._get_board_state()
        if not board_ptr: return
        board = board_ptr.contents

        result = self.c_lib.get_game_result(board_ptr)

        # 更新信息标签
        black_moves_left = board.moves_left[BLACK_PLAYER]
        white_moves_left = board.moves_left[WHITE_PLAYER]
        black_tiles = self.c_lib.pop_count(ctypes.byref(board.tiles[BLACK_PLAYER]))
        white_tiles = self.c_lib.pop_count(ctypes.byref(board.tiles[WHITE_PLAYER]))
        info_text = (f"红方剩余行动力: {black_moves_left} | 绿方剩余行动力: {white_moves_left}\n"
                     f"红方涂色数量: {black_tiles} | 绿方涂色数量: {white_tiles}")
        self.info_label.config(text=info_text)

        if result != -1:  # IN_PROGRESS
            self.game_running = False
            winner_text = "平局！"
            if result == 1:
                winner_text = "红方胜利！"
            elif result == 2:
                winner_text = "绿方胜利！"
            self.status_label.config(text=f"游戏结束 - {winner_text}")
            return

        player = board.current_player
        player_name = "红方" if player == BLACK_PLAYER else "绿方"
        player_color = BLACK_PIECE_COLOR if player == BLACK_PLAYER else WHITE_PIECE_COLOR
        mode = self.game_mode.get()

        turn_info = ""
        if mode == "human_vs_human":
            turn_info = f"轮到 {player_name}"
        elif mode == "human_vs_ai":
            turn_info = f"轮到您 ({player_name})" if player == self.human_player else f"AI ({player_name}) 思考中..."
        elif mode == "ai_vs_ai":
            turn_info = f"AI ({player_name}) 思考中..."

        self.status_label.config(text=turn_info, fg=player_color)

    def _handle_click(self, event):
        if not self.game_running: return
        mode = self.game_mode.get()
        board_ptr = self._get_board_state()
        if not board_ptr: return

        is_human_turn = (mode == "human_vs_human") or (
                    mode == "human_vs_ai" and board_ptr.contents.current_player == self.human_player)
        if not is_human_turn:
            return

        col = int(event.x // self.cell_size)
        row = int(event.y // self.cell_size)
        move = row * BOARD_SIZE + col

        legal_moves_bb = self.c_lib.get_legal_moves(board_ptr)
        if self.c_lib.is_bit_set(ctypes.byref(legal_moves_bb), move):
            self._make_move(move)
        else:
            messagebox.showwarning("非法落子", "不能在此处落子。")

    def _make_move(self, move):
        self.analysis_policy = None
        self.move_history.append(move)
        self.c_lib.mcts_make_move(self.mcts_manager, 0, move)
        self._start_game_flow()

    def _undo_move(self):
        if len(self.move_history) < 1:
            messagebox.showinfo("悔棋", "无法再悔棋。")
            return

        steps_to_undo = 1 if self.game_mode.get() == "human_vs_human" else 2

        if len(self.move_history) < steps_to_undo:
            messagebox.showinfo("悔棋", "无法再悔棋。")
            return

        self.move_history = self.move_history[:-steps_to_undo]
        temp_history = list(self.move_history)

        if self.mcts_manager: self.c_lib.destroy_mcts_manager(self.mcts_manager)
        self.mcts_manager = self.c_lib.create_mcts_manager(1)

        for move in temp_history:
            self.c_lib.mcts_make_move(self.mcts_manager, 0, move)

        self.game_running = True
        self._start_game_flow()

    def _ai_turn(self):
        if not self.game_running: return
        self.status_label.config(text="AI 思考中...")
        self.master.update_idletasks()

        sims = self.ai_sims_slider.get()
        temp = self.temperature_slider.get()

        threading.Thread(target=self._ai_worker, args=(sims, temp)).start()

    def _ai_worker(self, simulations, temperature):
        board_buffer = (Board * 1)()
        request_indices = (ctypes.c_int * 1)()

        for _ in range(simulations):
            if not self.game_running: return
            num_reqs = self.c_lib.mcts_run_simulations_and_get_requests(self.mcts_manager, board_buffer,
                                                                        request_indices, 1)

            if num_reqs > 0:
                board_py = self._board_to_tensor(board_buffer[0])
                input_tensor = torch.from_numpy(board_py).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    policy_logits, value, _ = self.ai_model(input_tensor)
                    policy = torch.softmax(policy_logits, dim=1).cpu().numpy().flatten()
                    value_np = value.cpu().numpy().flatten()

                self.c_lib.mcts_feed_results(self.mcts_manager, policy.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                             value_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))

        if not self.game_running: return

        policy_buffer = (ctypes.c_float * BOARD_SQUARES)()
        self.c_lib.mcts_get_policy(self.mcts_manager, 0, policy_buffer)
        policy_np = np.ctypeslib.as_array(policy_buffer)

        if temperature > 0:
            policy_np += 1e-8
            move_probs = policy_np ** (1.0 / temperature)
            move_probs /= np.sum(move_probs)
        else:
            move_probs = np.zeros_like(policy_np)
            if np.sum(policy_np) > 0:
                move_probs[np.argmax(policy_np)] = 1.0

        if np.sum(move_probs) < 1e-8:
            self.master.after(0, self._handle_no_ai_move)
            return

        move = np.random.choice(range(BOARD_SQUARES), p=move_probs)
        self.master.after(0, self._make_move, int(move))

    def _analyze_board(self):
        if not self.game_running: return
        self.analyze_button.config(text="分析中...", state=tk.DISABLED)
        sims = self.ai_sims_slider.get()

        def analysis_worker():
            temp_manager = self.c_lib.create_mcts_manager(1)
            current_board_ptr = self._get_board_state()
            if not current_board_ptr:
                self.master.after(0, self._finish_analysis)
                return

            self.c_lib.copy_board(current_board_ptr, self.c_lib.mcts_get_board_state(temp_manager, 0))

            board_buffer = (Board * 1)()
            request_indices = (ctypes.c_int * 1)()
            for _ in range(sims):
                num_reqs = self.c_lib.mcts_run_simulations_and_get_requests(temp_manager, board_buffer, request_indices,
                                                                            1)
                if num_reqs > 0:
                    board_py = self._board_to_tensor(board_buffer[0])
                    input_tensor = torch.from_numpy(board_py).unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        policy_logits, value, _ = self.ai_model(input_tensor)
                        policy = torch.softmax(policy_logits, dim=1).cpu().numpy().flatten()
                        value_np = value.cpu().numpy().flatten()
                    self.c_lib.mcts_feed_results(temp_manager, policy.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                                 value_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))

            policy_buffer = (ctypes.c_float * BOARD_SQUARES)()
            self.c_lib.mcts_get_policy(temp_manager, 0, policy_buffer)
            self.analysis_policy = np.ctypeslib.as_array(policy_buffer).copy()
            self.c_lib.destroy_mcts_manager(temp_manager)

            self.master.after(0, self._finish_analysis)

        threading.Thread(target=analysis_worker).start()

    def _finish_analysis(self):
        self.analyze_button.config(text="分析", state=tk.NORMAL)
        self._draw_board()

    def _handle_no_ai_move(self):
        self.game_running = False
        messagebox.showinfo("游戏结束", "AI 无棋可走，游戏结束。")
        self._update_status()

    def _board_to_tensor(self, board_c: Board) -> np.ndarray:
        # 这个函数的逻辑必须与自对弈和训练脚本中的完全一致
        tensor = np.zeros((11, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        p, o = board_c.current_player, 1 - board_c.current_player

        def get_plane(bb):
            plane = np.zeros(BOARD_SQUARES, dtype=np.float32)
            for i in range(BOARD_SQUARES):
                if self.c_lib.is_bit_set(ctypes.byref(bb), i): plane[i] = 1.0
            return plane.reshape((BOARD_SIZE, BOARD_SIZE))

        tensor[0, :, :] = get_plane(board_c.pieces[p])
        tensor[1, :, :] = get_plane(board_c.pieces[o])
        tensor[2, :, :] = get_plane(board_c.tiles[p])
        tensor[3, :, :] = get_plane(board_c.tiles[o])
        tensor[4, :, :] = 1. if p == 0 else 0.
        tensor[5, :, :] = 1. if p == 1 else 0.
        tensor[6, :, :] = float(board_c.moves_left[0]) / 25.0  # 使用新的 TOTAL_MOVES
        tensor[7, :, :] = float(board_c.moves_left[1]) / 25.0  # 使用新的 TOTAL_MOVES
        tensor[8, :, :] = float(self.c_lib.pop_count(ctypes.byref(board_c.tiles[0]))) / BOARD_SQUARES
        tensor[9, :, :] = float(self.c_lib.pop_count(ctypes.byref(board_c.tiles[1]))) / BOARD_SQUARES

        all_tiles = Bitboards()
        all_tiles.parts[0] = ~ (board_c.tiles[0].parts[0] | board_c.tiles[1].parts[0])
        all_tiles.parts[1] = ~ (board_c.tiles[0].parts[1] | board_c.tiles[1].parts[1])
        tensor[10, :, :] = get_plane(all_tiles)
        return tensor

    def _on_resize(self, event):
        new_width = self.canvas.winfo_width()
        new_height = self.canvas.winfo_height()
        min_dim = min(new_width, new_height)
        if min_dim > BOARD_SIZE * 10:
            self.cell_size = min_dim // BOARD_SIZE
            self._draw_board()


if __name__ == "__main__":
    root = tk.Tk()
    app = PomPomGameGUI(root)
    root.mainloop()
