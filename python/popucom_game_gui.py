import tkinter as tk
from tkinter import ttk, messagebox, Scale, font, Checkbutton, BooleanVar
import os
import torch
import numpy as np
import threading
import time
import ctypes
import platform
from torch.amp import autocast

# 确保从正确的文件导入
from popucom_nn_model import PomPomNN
from popucom_nn_interface import BOARD_SIZE, NUM_INPUT_CHANNELS, MAX_MOVES_PER_PLAYER


# --- C 语言接口定义 ---
class Bitboards(ctypes.Structure): _fields_ = [("parts", ctypes.c_uint64 * 2)]


class Board(ctypes.Structure): _fields_ = [("pieces", Bitboards * 2), ("tiles", Bitboards * 2),
                                           ("current_player", ctypes.c_int), ("moves_left", ctypes.c_int * 2)]


# 更新C函数定义字典
C_FUNCTIONS = {
    "init_board": (None, [ctypes.POINTER(Board)]),
    "copy_board": (None, [ctypes.POINTER(Board), ctypes.POINTER(Board)]),
    "get_legal_moves": (Bitboards, [ctypes.POINTER(Board)]),
    "get_game_result": (ctypes.c_int, [ctypes.POINTER(Board)]),
    "get_score_diff": (ctypes.c_int, [ctypes.POINTER(Board)]),
    "make_move": (ctypes.c_bool, [ctypes.POINTER(Board), ctypes.c_int]),
    "pop_count": (ctypes.c_int, [ctypes.POINTER(Bitboards)]),
    "is_bit_set": (ctypes.c_bool, [ctypes.POINTER(Bitboards), ctypes.c_int]),
    "create_mcts_manager": (ctypes.c_void_p, [ctypes.c_int, ctypes.c_bool, ctypes.c_double]),
    "mcts_set_noise_enabled": (None, [ctypes.c_void_p, ctypes.c_bool]),
    "mcts_set_fpu": (None, [ctypes.c_void_p, ctypes.c_double]),
    "boards_to_tensors_c": (None, [ctypes.POINTER(Board), ctypes.c_int, ctypes.POINTER(ctypes.c_float)]),
    "destroy_mcts_manager": (None, [ctypes.c_void_p]),
    "mcts_run_simulations_and_get_requests": (
        ctypes.c_int, [ctypes.c_void_p, ctypes.POINTER(Board), ctypes.POINTER(ctypes.c_int), ctypes.c_int]),
    "mcts_feed_results": (
        None, [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(Board)]),
    "mcts_get_policy": (ctypes.c_bool, [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_float)]),
    "mcts_make_move": (None, [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]),
    "mcts_get_simulations_done": (ctypes.c_int, [ctypes.c_void_p, ctypes.c_int]),
    "mcts_get_analysis_data": (ctypes.c_int, [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int),
                                              ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_int),
                                              ctypes.POINTER(ctypes.c_float), ctypes.c_int]),
    "mcts_reset_for_analysis": (None, [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(Board)])
}


def setup_c_library():
    lib_name = "popucom_core.dll" if platform.system() == "Windows" else "popucom_core.so"
    if not os.path.exists(lib_name): raise FileNotFoundError(f"未找到C库 '{lib_name}'。请编译C++代码。")
    c_lib = ctypes.CDLL(os.path.abspath(lib_name))
    for func_name, (restype, argtypes) in C_FUNCTIONS.items():
        if hasattr(c_lib, func_name):
            func = getattr(c_lib, func_name)
            func.restype = restype
            func.argtypes = argtypes
        else:
            print(f"警告: 在C库中未找到函数 '{func_name}'")
    return c_lib


try:
    c_lib = setup_c_library()
except FileNotFoundError as e:
    messagebox.showerror("库加载错误", str(e));
    exit()

# --- 全局常量 ---
MODEL_PATH, BLACK_PLAYER, WHITE_PLAYER = "model.pth", 0, 1
IN_PROGRESS, DRAW, BLACK_WIN, WHITE_WIN = -1, 0, 1, 2
BOARD_BACKGROUND_COLOR, GRID_LINE_COLOR = "#D2B48C", "#8B4513"
UNPAINTED_FLOOR_COLOR, BLACK_PAINTED_FLOOR_COLOR, WHITE_PAINTED_FLOOR_COLOR = "#FFF8DC", "#FFDAB9", "#90EE90"
BLACK_PIECE_COLOR, WHITE_PIECE_COLOR = "red", "green"
DEFAULT_FPU_GUI = 0.0


class GameTreeNode:
    """Represents a node in the game tree."""

    def __init__(self, board_state, move_sq=None, parent=None):
        self.board_state = board_state
        self.move_sq = move_sq
        self.parent = parent
        self.children = {}  # key: move_sq, value: GameTreeNode


class PomPomGUI:
    def __init__(self, master):
        self.master = master
        master.title("泡姆棋")
        self.cell_size = 60
        self.game_running = False
        self.game_mode = tk.StringVar(value="human_vs_ai")
        self.human_player_choice = tk.StringVar(value="human_black")
        self.human_player, self.ai_player = BLACK_PLAYER, WHITE_PLAYER
        self.board_c, self._last_move_coords = Board(), None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ai_model = self._load_ai_model()
        self.dirichlet_noise_enabled = BooleanVar(value=False)
        self.mcts_manager_gui = c_lib.create_mcts_manager(1, self.dirichlet_noise_enabled.get(), DEFAULT_FPU_GUI)
        self.ai_thread = None
        self.analysis_data, self.analysis_in_progress = {}, False
        self.best_puct_move, self.best_visit_move = None, None

        self.game_tree_root = None
        self.current_node = None
        self.listbox_nodes = []

        self._setup_gui()
        self._reset_and_start_new_game()

    def _load_ai_model(self):
        try:
            model = PomPomNN()
            model.load_state_dict(torch.load(MODEL_PATH, map_location=self.device))
            model.to(self.device).eval()
            print(f"AI模型已从 {MODEL_PATH} 加载。")
            return model
        except Exception as e:
            messagebox.showwarning("模型加载失败", f"无法加载模型: {e}\nAI将使用随机权重的网络。")
            return PomPomNN().to(self.device).eval()

    def _setup_gui(self):
        top_frame = tk.Frame(self.master)
        top_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=(10, 0))

        main_board_frame = tk.Frame(self.master)
        main_board_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True, padx=10, pady=10)

        control_frame = tk.Frame(top_frame)
        control_frame.pack(side=tk.TOP, fill=tk.X)

        mode_frame = tk.LabelFrame(control_frame, text="游戏模式", padx=5, pady=5)
        mode_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5)
        for text, value in [("人机对战", "human_vs_ai"), ("人人对战", "human_vs_human"), ("机机对战", "ai_vs_ai")]:
            tk.Radiobutton(mode_frame, text=text, variable=self.game_mode, value=value,
                           command=self._on_mode_change).pack(anchor='w')

        self.role_frame = tk.LabelFrame(control_frame, text="执子选择", padx=5, pady=5)
        self.role_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5)
        for text, value in [("我执红 (先手)", "human_black"), ("我执绿 (后手)", "ai_black")]:
            tk.Radiobutton(self.role_frame, text=text, variable=self.human_player_choice, value=value,
                           command=self._on_mode_change).pack(anchor='w')

        ai_frame = tk.LabelFrame(control_frame, text="AI 设置", padx=5, pady=5)
        ai_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

        tk.Label(ai_frame, text="搜索模拟次数:").grid(row=0, column=0, sticky='w')
        self.ai_sims_slider = Scale(ai_frame, from_=50, to=5000, orient=tk.HORIZONTAL, resolution=50, length=200);
        self.ai_sims_slider.set(800);
        self.ai_sims_slider.grid(row=0, column=1, sticky='ew')

        tk.Label(ai_frame, text="落子温度:").grid(row=1, column=0, sticky='w')
        self.temperature_slider = Scale(ai_frame, from_=0.0, to=2.0, orient=tk.HORIZONTAL, resolution=0.1, length=200);
        self.temperature_slider.set(0.0);
        self.temperature_slider.grid(row=1, column=1, sticky='ew')

        tk.Label(ai_frame, text="节点初始分数/探索性:").grid(row=2, column=0, sticky='w')
        self.fpu_slider = Scale(ai_frame, from_=-0.5, to=0.5, orient=tk.HORIZONTAL, resolution=0.01, length=100,
                                command=self._on_fpu_change);
        self.fpu_slider.set(DEFAULT_FPU_GUI);
        self.fpu_slider.grid(row=2, column=1, sticky='ew')

        self.noise_checkbox = Checkbutton(ai_frame, text="开启狄利克雷噪声",
                                          variable=self.dirichlet_noise_enabled, command=self._on_noise_toggle)
        self.noise_checkbox.grid(row=3, column=0, columnspan=2, sticky='w', pady=(5, 0))
        ai_frame.columnconfigure(1, weight=1)

        btn_frame = tk.Frame(control_frame);
        btn_frame.pack(side=tk.RIGHT, padx=5, fill=tk.Y)
        self.new_game_button = tk.Button(btn_frame, text="新对局/清空", command=self._reset_and_start_new_game,
                                         width=10);
        self.new_game_button.pack(pady=2)
        self.undo_button = tk.Button(btn_frame, text="返回上步", command=self._undo_move, width=10);
        self.undo_button.pack(pady=2)
        self.analyze_button = tk.Button(btn_frame, text="分析", command=self._toggle_analysis, width=10);
        self.analyze_button.pack(pady=2)

        status_frame = tk.Frame(top_frame, pady=5);
        status_frame.pack(fill=tk.X, side=tk.BOTTOM)
        self.status_label = tk.Label(status_frame, text="初始化...", font=("Arial", 14, "bold"));
        self.status_label.pack()
        self.moves_label = tk.Label(status_frame, text="", font=("Arial", 11));
        self.moves_label.pack()

        canvas_frame = tk.Frame(main_board_frame)
        canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(canvas_frame, bg=BOARD_BACKGROUND_COLOR);
        self.canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.canvas.bind("<Button-1>", self._handle_click);
        canvas_frame.bind("<Configure>", self._on_resize)

        self._setup_listbox_view(main_board_frame)

        self.master.protocol("WM_DELETE_WINDOW", self._on_closing)

    def _setup_listbox_view(self, parent_frame):
        """Sets up the game view using a Listbox, which handles scrolling correctly."""
        listbox_container = tk.Frame(parent_frame, bd=1, relief=tk.SUNKEN)
        listbox_container.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0))

        tk.Label(listbox_container, text="对局树", font=("Arial", 12, "bold")).pack(pady=(5, 0))

        list_frame = tk.Frame(listbox_container)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        list_frame.grid_rowconfigure(0, weight=1)
        list_frame.grid_columnconfigure(0, weight=1)

        self.game_view = tk.Listbox(list_frame, font=("Courier", 10), selectmode=tk.SINGLE)

        ysb = ttk.Scrollbar(list_frame, orient='vertical', command=self.game_view.yview)
        xsb = ttk.Scrollbar(list_frame, orient='horizontal', command=self.game_view.xview)
        self.game_view.configure(yscrollcommand=ysb.set, xscrollcommand=xsb.set)

        self.game_view.grid(row=0, column=0, sticky='nsew')
        ysb.grid(row=0, column=1, sticky='ns')
        xsb.grid(row=1, column=0, sticky='ew')

        self.game_view.bind("<<ListboxSelect>>", self._on_listbox_select)

    def _update_game_view(self):
        """Clears and rebuilds the Listbox based on the current game tree."""
        self.game_view.delete(0, tk.END)
        self.listbox_nodes.clear()

        def build_list(node, depth=0):
            prefix = " " * depth
            move_text = f"{prefix}{self._get_move_text(node)}"

            self.game_view.insert(tk.END, move_text)
            self.listbox_nodes.append(node)

            for move_sq in sorted(node.children.keys()):
                build_list(node.children[move_sq], depth + 1)

        build_list(self.game_tree_root)
        self._highlight_current_node_in_view()

    def _get_move_text(self, node):
        """Generates the text for a move, e.g., '1. f3' or '1... h5'."""
        if node.parent is None:
            return "游戏开始"

        board_before_move = node.parent.board_state
        turn_player = board_before_move.current_player

        total_moves_made = (MAX_MOVES_PER_PLAYER * 2) - (
                    board_before_move.moves_left[0] + board_before_move.moves_left[1])
        turn_number = total_moves_made // 2 + 1

        move_prefix = f"{turn_number}." if turn_player == BLACK_PLAYER else f"{turn_number}..."
        return f"{move_prefix} {self._coords_to_alg(node.move_sq)}"

    def _highlight_current_node_in_view(self):
        """Finds and highlights the current node in the listbox."""
        try:
            idx = self.listbox_nodes.index(self.current_node)
            self.game_view.selection_clear(0, tk.END)
            self.game_view.selection_set(idx)
            self.game_view.activate(idx)
            self.game_view.see(idx)
        except ValueError:
            pass

    def _on_resize(self, event):
        new_size = min(event.width, event.height)
        new_cell_size = new_size // BOARD_SIZE - 1

        if new_cell_size < 20 or abs(new_cell_size - self.cell_size) < 2:
            return

        self.cell_size = new_cell_size
        self.draw_board()

    def _coords_to_alg(self, sq):
        if sq is None: return "Start"
        col = sq % BOARD_SIZE
        row = sq // BOARD_SIZE
        return f"{'abcdefghi'[col]}{row + 1}"

    def _on_fpu_change(self, value):
        if self.mcts_manager_gui:
            c_lib.mcts_set_fpu(self.mcts_manager_gui, float(value))

    def _on_noise_toggle(self):
        is_enabled = self.dirichlet_noise_enabled.get()
        if self.mcts_manager_gui: c_lib.mcts_set_noise_enabled(self.mcts_manager_gui, is_enabled)

    def _on_closing(self):
        if self.ai_thread and self.ai_thread.is_alive(): self.game_running = False
        if self.mcts_manager_gui: c_lib.destroy_mcts_manager(self.mcts_manager_gui)
        self.master.destroy()

    def _on_mode_change(self):
        mode = self.game_mode.get()
        self.role_frame.pack(side=tk.LEFT, fill=tk.Y,
                             padx=5) if mode == "human_vs_ai" else self.role_frame.pack_forget()

        choice = self.human_player_choice.get()
        self.human_player, self.ai_player = (BLACK_PLAYER, WHITE_PLAYER) if choice == "human_black" else (
        WHITE_PLAYER, BLACK_PLAYER)
        self._continue_game_flow()

    def _reset_and_start_new_game(self):
        if self.ai_thread and self.ai_thread.is_alive(): self.game_running = False; self.ai_thread.join(timeout=0.5)
        c_lib.init_board(ctypes.byref(self.board_c))

        board_copy = Board()
        c_lib.copy_board(ctypes.byref(self.board_c), ctypes.byref(board_copy))
        self.game_tree_root = GameTreeNode(board_copy)
        self.current_node = self.game_tree_root

        self._update_game_view()

        if self.mcts_manager_gui: c_lib.mcts_reset_for_analysis(self.mcts_manager_gui, 0, ctypes.byref(self.board_c))
        self._last_move_coords, self.analysis_data, self.best_puct_move, self.best_visit_move = None, {}, None, None

        choice = self.human_player_choice.get()
        self.human_player, self.ai_player = (BLACK_PLAYER, WHITE_PLAYER) if choice == "human_black" else (
        WHITE_PLAYER, BLACK_PLAYER)

        self.game_running = True
        self._continue_game_flow()

    def _continue_game_flow(self):
        self.draw_board()
        self._update_status_labels()
        self._toggle_ui_elements(True)
        game_result = c_lib.get_game_result(ctypes.byref(self.board_c))
        if game_result != IN_PROGRESS: self._handle_game_over(game_result); return
        mode, player = self.game_mode.get(), self.board_c.current_player
        is_ai_turn = (mode == "ai_vs_ai") or (mode == "human_vs_ai" and player == self.ai_player)
        if is_ai_turn and self.game_running:
            self._toggle_ui_elements(False)
            self.ai_thread = threading.Thread(target=self._ai_turn_logic, daemon=True);
            self.ai_thread.start()

    def _handle_game_over(self, result):
        self.game_running = False
        self._update_status_labels()
        score_diff = abs(c_lib.get_score_diff(ctypes.byref(self.board_c)))
        msg = "平局！"
        if result == BLACK_WIN:
            msg = f"红方胜利！领先 {score_diff} 格。"
        elif result == WHITE_WIN:
            msg = f"绿方胜利！领先 {score_diff} 格。"
        messagebox.showinfo("游戏结束", msg)
        self._toggle_ui_elements(True)

    def draw_board(self):
        self.canvas.delete("all")
        board_pixel_size = BOARD_SIZE * self.cell_size
        self.canvas.config(width=board_pixel_size, height=board_pixel_size)

        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                sq = r * BOARD_SIZE + c
                x1, y1, x2, y2 = c * self.cell_size, r * self.cell_size, (c + 1) * self.cell_size, (
                            r + 1) * self.cell_size
                color = UNPAINTED_FLOOR_COLOR
                if c_lib.is_bit_set(ctypes.byref(self.board_c.tiles[BLACK_PLAYER]), sq):
                    color = BLACK_PAINTED_FLOOR_COLOR
                elif c_lib.is_bit_set(ctypes.byref(self.board_c.tiles[WHITE_PLAYER]), sq):
                    color = WHITE_PAINTED_FLOOR_COLOR
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline=GRID_LINE_COLOR, width=1)

        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                sq = r * BOARD_SIZE + c
                center_x, center_y, radius = c * self.cell_size + self.cell_size / 2, r * self.cell_size + self.cell_size / 2, self.cell_size / 2.2
                p_color = None
                if c_lib.is_bit_set(ctypes.byref(self.board_c.pieces[BLACK_PLAYER]), sq):
                    p_color = BLACK_PIECE_COLOR
                elif c_lib.is_bit_set(ctypes.byref(self.board_c.pieces[WHITE_PLAYER]), sq):
                    p_color = WHITE_PIECE_COLOR
                if p_color:
                    self.canvas.create_oval(center_x - radius, center_y - radius, center_x + radius, center_y + radius,
                                            fill=p_color, outline="black", width=1.5)
                    if self._last_move_coords == (r, c): self.canvas.create_oval(center_x - radius / 4,
                                                                                 center_y - radius / 4,
                                                                                 center_x + radius / 4,
                                                                                 center_y + radius / 4, fill="white",
                                                                                 outline="black")

        if self.analysis_data:
            fnt = font.Font(family='Helvetica', size=max(4, int(self.cell_size / 16)), weight='bold')
            for (r, c), data in self.analysis_data.items():
                center_x, center_y = c * self.cell_size + self.cell_size / 2, r * self.cell_size + self.cell_size / 2
                win_rate = (data['q'] + 1) / 2 * 100
                win_rate_text = f"胜率: {win_rate:.2f}%"
                puct_text = f"PUCT: {data['puct']:.4f}"
                visits_text = f"访问: {data['visits']}"
                self.canvas.create_text(center_x, center_y - self.cell_size * 0.25, text=win_rate_text,
                                        fill="blue" if data['q'] > 0 else "purple", font=fnt)
                self.canvas.create_text(center_x, center_y, text=visits_text, fill="black", font=fnt)
                self.canvas.create_text(center_x, center_y + self.cell_size * 0.25, text=puct_text, fill="#006400",
                                        font=fnt)

        if self.best_puct_move:
            r, c = self.best_puct_move;
            x1, y1, x2, y2 = c * self.cell_size, r * self.cell_size, (c + 1) * self.cell_size, (r + 1) * self.cell_size
            self.canvas.create_rectangle(x1 + 6, y1 + 6, x2 - 6, y2 - 6, outline="orange", width=3, dash=(4, 4))
        if self.best_visit_move:
            r, c = self.best_visit_move;
            x1, y1, x2, y2 = c * self.cell_size, r * self.cell_size, (c + 1) * self.cell_size, (r + 1) * self.cell_size
            self.canvas.create_rectangle(x1 + 3, y1 + 3, x2 - 3, y2 - 3, outline="red", width=3)

    def _update_status_labels(self):
        player, name, color = self.board_c.current_player, "红方" if self.board_c.current_player == 0 else "绿方", BLACK_PIECE_COLOR if self.board_c.current_player == 0 else WHITE_PIECE_COLOR
        mode, turn_info = self.game_mode.get(), ""
        if mode == "human_vs_human":
            turn_info = " (人类)"
        elif mode == "human_vs_ai":
            turn_info = " (您)" if player == self.human_player else " (AI思考中...)"
        elif mode == "ai_vs_ai":
            turn_info = " (AI思考中...)"
        self.status_label.config(text=f"当前回合: {name}{turn_info}" if self.game_running else "游戏结束",
                                 fg=color if self.game_running else "black")
        score_diff = c_lib.get_score_diff(ctypes.byref(self.board_c))
        self.moves_label.config(
            text=f"红方剩余: {self.board_c.moves_left[0]} | 绿方剩余: {self.board_c.moves_left[1]}\n当前分数 (红-绿): {score_diff}")

    def _toggle_ui_elements(self, enabled):
        state = tk.NORMAL if enabled else tk.DISABLED
        self.new_game_button.config(state=state)
        self.undo_button.config(state=tk.NORMAL if self.current_node and self.current_node.parent else tk.DISABLED)
        self.analyze_button.config(state=state)

    def _handle_click(self, event):
        mode = self.game_mode.get()
        is_human = (mode == "human_vs_human") or (
                    mode == "human_vs_ai" and self.board_c.current_player == self.human_player)
        if not self.game_running or not is_human: return
        if self.cell_size == 0: return
        c, r = event.x // self.cell_size, event.y // self.cell_size
        if 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE: self._process_move(r * BOARD_SIZE + c)

    def _process_move(self, sq, from_ai=False):
        legal_bb = c_lib.get_legal_moves(ctypes.byref(self.board_c))
        if not c_lib.is_bit_set(ctypes.byref(legal_bb), sq):
            if not from_ai: messagebox.showwarning("非法落子", "该位置不符合落子规则。")
            return

        # BUG FIX: Correctly advance the current_node pointer.
        if sq not in self.current_node.children:
            # If the move is new, create a new node
            board_after_move = Board()
            # The board state for the new node should be calculated from the current node's board state
            c_lib.copy_board(ctypes.byref(self.current_node.board_state), ctypes.byref(board_after_move))
            c_lib.make_move(ctypes.byref(board_after_move), sq)

            new_node = GameTreeNode(board_after_move, move_sq=sq, parent=self.current_node)
            self.current_node.children[sq] = new_node

        # Now, advance to the child node (whether it was new or existing)
        self.current_node = self.current_node.children[sq]

        c_lib.copy_board(ctypes.byref(self.current_node.board_state), ctypes.byref(self.board_c))
        self._last_move_coords = (sq // BOARD_SIZE, sq % BOARD_SIZE)
        self.analysis_data, self.best_puct_move, self.best_visit_move = {}, None, None
        if self.mcts_manager_gui: c_lib.mcts_reset_for_analysis(self.mcts_manager_gui, 0, ctypes.byref(self.board_c))

        self._update_game_view()
        self._continue_game_flow()

    def _undo_move(self):
        if self.current_node and self.current_node.parent:
            self._navigate_to_node(self.current_node.parent)

    def _on_listbox_select(self, event):
        selection = self.game_view.curselection()
        if not selection: return

        selected_index = selection[0]
        if 0 <= selected_index < len(self.listbox_nodes):
            node_to_visit = self.listbox_nodes[selected_index]
            if node_to_visit != self.current_node:
                self._navigate_to_node(node_to_visit)

    def _navigate_to_node(self, node):
        self.current_node = node
        c_lib.copy_board(ctypes.byref(self.current_node.board_state), ctypes.byref(self.board_c))

        if self.current_node.move_sq is not None:
            sq = self.current_node.move_sq
            self._last_move_coords = (sq // BOARD_SIZE, sq % BOARD_SIZE)
        else:
            self._last_move_coords = None

        self.analysis_data, self.best_puct_move, self.best_visit_move = {}, None, None
        if self.mcts_manager_gui: c_lib.mcts_reset_for_analysis(self.mcts_manager_gui, 0, ctypes.byref(self.board_c))
        self.game_running = True

        self._update_game_view()
        self._continue_game_flow()

    def _toggle_analysis(self):
        if self.analysis_data: self.analysis_data, self.best_puct_move, self.best_visit_move = {}, None, None; self.draw_board(); return
        if not self.game_running: return
        self._toggle_ui_elements(False);
        self.analyze_button.config(text="分析中...")
        threading.Thread(target=self._run_analysis_thread, daemon=True).start()

    def _run_analysis_thread(self):
        sims = self.ai_sims_slider.get();
        if self.mcts_manager_gui: c_lib.mcts_reset_for_analysis(self.mcts_manager_gui, 0, ctypes.byref(self.board_c))
        self._run_mcts_loop(sims, is_analysis=True)
        moves, q_vals, visits, pucts = (ctypes.c_int * 81)(), (ctypes.c_float * 81)(), (ctypes.c_int * 81)(), (
                    ctypes.c_float * 81)()
        num_moves = c_lib.mcts_get_analysis_data(self.mcts_manager_gui, 0, moves, q_vals, visits, pucts, 81)
        temp_data, best_p_move, best_puct, best_v_move, best_visits = {}, None, -float('inf'), None, -1
        for i in range(num_moves):
            r, c = moves[i] // BOARD_SIZE, moves[i] % BOARD_SIZE
            temp_data[(r, c)] = {"q": q_vals[i], "visits": visits[i], "puct": pucts[i]}
            if pucts[i] > best_puct: best_puct, best_p_move = pucts[i], (r, c)
            if visits[i] > best_visits: best_visits, best_v_move = visits[i], (r, c)

        def update_gui():
            self.analysis_data, self.best_puct_move, self.best_visit_move = temp_data, best_p_move, best_v_move
            self.draw_board();
            self._toggle_ui_elements(True);
            self.analyze_button.config(text="分析")

        self.master.after(0, update_gui)

    def _ai_turn_logic(self):
        if self.game_mode.get() == "ai_vs_ai": time.sleep(0.3)
        sims = self.ai_sims_slider.get()
        self._run_mcts_loop(sims)
        policy_buffer = (ctypes.c_float * 81)();
        c_lib.mcts_get_policy(self.mcts_manager_gui, 0, policy_buffer)
        policy_np = np.ctypeslib.as_array(policy_buffer)
        temp = self.temperature_slider.get()
        if temp > 0:
            probs = policy_np ** (1.0 / temp)
            legal_bb = c_lib.get_legal_moves(ctypes.byref(self.board_c))
            for sq in range(81):
                if not c_lib.is_bit_set(ctypes.byref(legal_bb), sq): probs[sq] = 0
            sum_p = np.sum(probs)
            if sum_p > 1e-8:
                move = np.random.choice(81, p=probs / sum_p)
            else:
                legal_idx = [sq for sq in range(81) if c_lib.is_bit_set(ctypes.byref(legal_bb), sq)]
                move = np.random.choice(legal_idx) if legal_idx and self.game_running else -1
        else:
            move = np.argmax(policy_np)
        if self.game_running and move != -1: self.master.after(0, self._process_move, move, True)

    def _run_mcts_loop(self, num_sims, is_analysis=False):
        while c_lib.mcts_get_simulations_done(self.mcts_manager_gui, 0) < num_sims:
            if not self.game_running and not is_analysis: return
            boards, indices = (Board * 1)(), (ctypes.c_int * 1)()
            if c_lib.mcts_run_simulations_and_get_requests(self.mcts_manager_gui, boards, indices, 1) > 0:
                input_tensor_np = np.zeros((1, NUM_INPUT_CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
                c_lib.boards_to_tensors_c(boards, 1, input_tensor_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
                input_batch = torch.from_numpy(input_tensor_np).to(self.device)
                with torch.no_grad():
                    use_amp = self.device.type == 'cuda'
                    with torch.amp.autocast(device_type=self.device.type, enabled=use_amp):
                        p_logits, val, _ = self.ai_model(input_batch)
                    policy = torch.softmax(p_logits.float(), dim=1).cpu().numpy()
                    value = val.float().item()
                c_lib.mcts_feed_results(self.mcts_manager_gui,
                                        np.ascontiguousarray(policy, dtype=np.float32).ctypes.data_as(
                                            ctypes.POINTER(ctypes.c_float)), ctypes.byref(ctypes.c_float(value)),
                                        boards)


if __name__ == "__main__":
    root = tk.Tk()
    app = PomPomGUI(root)
    root.mainloop()
