import tkinter as tk
from tkinter import messagebox, Scale, font
import os
import torch
import numpy as np
import threading
import time
import ctypes
import platform

from popucom_nn_model import PomPomNN, BOARD_SIZE, NUM_INPUT_CHANNELS
from self_play_worker import MAX_MOVES_PER_PLAYER

# --- C 语言接口定义 ---
class Bitboards(ctypes.Structure): _fields_ = [("parts", ctypes.c_uint64 * 2)]
class Board(ctypes.Structure):
    _fields_ = [
        ("pieces", Bitboards * 2), ("tiles", Bitboards * 2),
        ("current_player", ctypes.c_int), ("moves_left", ctypes.c_int * 2)
    ]

C_FUNCTIONS = {
    "init_board": (None, [ctypes.POINTER(Board)]),
    "copy_board": (None, [ctypes.POINTER(Board), ctypes.POINTER(Board)]),
    "get_legal_moves": (Bitboards, [ctypes.POINTER(Board)]),
    "get_game_result": (ctypes.c_int, [ctypes.POINTER(Board)]),
    "get_score_diff": (ctypes.c_int, [ctypes.POINTER(Board)]),
    "make_move": (ctypes.c_bool, [ctypes.POINTER(Board), ctypes.c_int]),
    "pop_count": (ctypes.c_int, [ctypes.POINTER(Bitboards)]),
    "create_mcts_manager": (ctypes.c_void_p, [ctypes.c_int]),
    "destroy_mcts_manager": (None, [ctypes.c_void_p]),
    "mcts_run_simulations_and_get_requests": (
    ctypes.c_int, [ctypes.c_void_p, ctypes.POINTER(Board), ctypes.POINTER(ctypes.c_int), ctypes.c_int]),
    "mcts_feed_results": (None, [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                                 ctypes.POINTER(ctypes.c_float)]),
    "mcts_get_policy": (ctypes.c_bool, [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_float)]),
    "mcts_make_move": (None, [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]),
    "mcts_get_simulations_done": (ctypes.c_int, [ctypes.c_void_p, ctypes.c_int]),
    "mcts_get_analysis_data": (ctypes.c_int, [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int),
                                              ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_int),
                                              ctypes.POINTER(ctypes.c_float), ctypes.c_int]),
    "mcts_reset_for_analysis": (None, [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(Board)]),
    "mcts_set_noise_enabled": (None, [ctypes.c_void_p, ctypes.c_int, ctypes.c_bool]),
}


def setup_c_library():
    lib_name = "popucom_core.dll" if platform.system() == "Windows" else "popucom_core.so"
    if not os.path.exists(lib_name): raise FileNotFoundError(f"未找到C库 '{lib_name}'。请编译C++代码。")
    c_lib = ctypes.CDLL(os.path.abspath(lib_name))
    for func_name, (restype, argtypes) in C_FUNCTIONS.items():
        if hasattr(c_lib, func_name):
            func = getattr(c_lib, func_name)
            func.restype, func.argtypes = restype, argtypes
        else:
            print(f"警告: 在C库中未找到函数 '{func_name}'")
    return c_lib


try:
    c_lib = setup_c_library()
except FileNotFoundError as e:
    messagebox.showerror("库加载错误", str(e)); exit()

# --- 全局常量 ---
MODEL_PATH, BLACK_PLAYER, WHITE_PLAYER = "model.pth", 0, 1
IN_PROGRESS, DRAW, BLACK_WIN, WHITE_WIN = -1, 0, 1, 2
BOARD_BACKGROUND_COLOR, GRID_LINE_COLOR = "#D2B48C", "#8B4513"
UNPAINTED_FLOOR_COLOR, BLACK_PAINTED_FLOOR_COLOR, WHITE_PAINTED_FLOOR_COLOR = "#FFF8DC", "#FFDAB9", "#90EE90"
BLACK_PIECE_COLOR, WHITE_PIECE_COLOR = "red", "green"


class PomPomGUI:
    def __init__(self, master):
        self.master = master
        master.title("泡姆棋 (C++ 内核, 分数版)")
        self.cell_size, self.canvas_width, self.canvas_height = 120, BOARD_SIZE * 120, BOARD_SIZE * 120
        self.game_running, self.game_mode, self.human_player_choice = False, tk.StringVar(
            value="human_vs_ai"), tk.StringVar(value="human_black")
        self.human_player, self.ai_player = BLACK_PLAYER, WHITE_PLAYER
        self.board_c, self.game_history, self._last_move_coords = Board(), [], None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ai_model, self.mcts_manager_gui, self.ai_thread = self._load_ai_model(), c_lib.create_mcts_manager(1), None
        self.analysis_data, self.analysis_in_progress = {}, False
        self.best_puct_move, self.best_visit_move = None, None
        self.root_uncertainty = None  # NEW: To store uncertainty of the current position

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
        control_frame = tk.Frame(self.master)
        control_frame.pack(side=tk.TOP, pady=10, fill=tk.X, padx=10)
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
        self.temperature_slider.set(0.1);
        self.temperature_slider.grid(row=1, column=1, sticky='ew')

        self.noise_enabled_var = tk.BooleanVar(value=False)
        noise_check = tk.Checkbutton(ai_frame, text="开启探索性噪声 (仅AI)", variable=self.noise_enabled_var)
        noise_check.grid(row=2, column=0, columnspan=2, sticky='w')

        ai_frame.columnconfigure(1, weight=1)

        btn_frame = tk.Frame(control_frame)
        btn_frame.pack(side=tk.RIGHT, padx=5)
        self.new_game_button = tk.Button(btn_frame, text="新游戏", command=self._reset_and_start_new_game, width=10)
        self.new_game_button.pack(pady=2)
        self.undo_button = tk.Button(btn_frame, text="悔棋", command=self._undo_move, width=10)
        self.undo_button.pack(pady=2)
        self.analyze_button = tk.Button(btn_frame, text="分析", command=self._toggle_analysis, width=10)
        self.analyze_button.pack(pady=2)

        status_frame = tk.Frame(self.master, pady=5);
        status_frame.pack(fill=tk.X, padx=10)
        self.status_label = tk.Label(status_frame, text="初始化...", font=("Arial", 14, "bold"));
        self.status_label.pack()
        self.moves_label = tk.Label(status_frame, text="", font=("Arial", 11));
        self.moves_label.pack()
        self.canvas = tk.Canvas(self.master, width=self.canvas_width, height=self.canvas_height,
                                bg=BOARD_BACKGROUND_COLOR);
        self.canvas.pack(padx=10, pady=10)
        self.canvas.bind("<Button-1>", self._handle_click);
        self.master.protocol("WM_DELETE_WINDOW", self._on_closing)

    def _on_closing(self):
        if self.ai_thread and self.ai_thread.is_alive(): self.game_running = False
        c_lib.destroy_mcts_manager(self.mcts_manager_gui);
        self.master.destroy()

    def _on_mode_change(self):
        mode, choice = self.game_mode.get(), self.human_player_choice.get()
        self.role_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5)
        if mode == "human_vs_ai":
            if choice == "human_black":
                self.human_player, self.ai_player = BLACK_PLAYER, WHITE_PLAYER
            else:
                self.human_player, self.ai_player = WHITE_PLAYER, BLACK_PLAYER
        else:
            self.role_frame.pack_forget()
        self._continue_game_flow()

    def _reset_and_start_new_game(self):
        if self.ai_thread and self.ai_thread.is_alive(): self.game_running = False; self.ai_thread.join(timeout=0.5)
        c_lib.init_board(ctypes.byref(self.board_c));
        c_lib.mcts_reset_for_analysis(self.mcts_manager_gui, 0, ctypes.byref(self.board_c))
        self.game_history.clear();
        self._last_move_coords, self.analysis_data, self.best_puct_move, self.best_visit_move, self.root_uncertainty = None, {}, None, None, None
        self.game_running = True;
        self._continue_game_flow()

    def _continue_game_flow(self):
        self.draw_board();
        self._update_status_labels();
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
        self.game_running = False;
        self._update_status_labels()
        score_diff = abs(c_lib.get_score_diff(ctypes.byref(self.board_c)))
        if result == BLACK_WIN:
            msg = f"红方胜利！领先 {score_diff} 格。"
        elif result == WHITE_WIN:
            msg = f"绿方胜利！领先 {score_diff} 格。"
        else:
            msg = "平局！"
        messagebox.showinfo("游戏结束", msg);
        self._toggle_ui_elements(True)

    def draw_board(self):
        self.canvas.delete("all")
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                sq = r * BOARD_SIZE + c;
                x1, y1, x2, y2 = c * self.cell_size, r * self.cell_size, (c + 1) * self.cell_size, (
                            r + 1) * self.cell_size
                color = UNPAINTED_FLOOR_COLOR
                if (self.board_c.tiles[BLACK_PLAYER].parts[sq // 64] >> (sq % 64)) & 1:
                    color = BLACK_PAINTED_FLOOR_COLOR
                elif (self.board_c.tiles[WHITE_PLAYER].parts[sq // 64] >> (sq % 64)) & 1:
                    color = WHITE_PAINTED_FLOOR_COLOR
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline=GRID_LINE_COLOR, width=1)
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                sq = r * BOARD_SIZE + c;
                center_x, center_y, radius = c * self.cell_size + 59.5, r * self.cell_size + 59.5, 55
                p_color = None
                if (self.board_c.pieces[BLACK_PLAYER].parts[sq // 64] >> (sq % 64)) & 1:
                    p_color = BLACK_PIECE_COLOR
                elif (self.board_c.pieces[WHITE_PLAYER].parts[sq // 64] >> (sq % 64)) & 1:
                    p_color = WHITE_PIECE_COLOR
                if p_color:
                    self.canvas.create_oval(center_x - radius, center_y - radius, center_x + radius, center_y + radius,
                                            fill=p_color, outline="black", width=1.5)
                    if self._last_move_coords == (r, c): self.canvas.create_oval(center_x - 10, center_y - 10,
                                                                                 center_x + 10, center_y + 10,
                                                                                 fill="white", outline="black")
        if self.analysis_data:
            fnt = font.Font(family='Helvetica', size=max(8, int(self.cell_size / 16)), weight='bold')
            for (r, c), data in self.analysis_data.items():
                center_x, center_y = c * self.cell_size + 59.5, r * self.cell_size + 59.5
                score_lead = data['q'] * BOARD_SIZE * BOARD_SIZE;
                puct_score = data['puct']
                score_text = f"领先: {score_lead:+.1f}";
                puct_text = f"总分: {puct_score:+.2f}";
                visits_text = f"访问: {data['visits']}"
                self.canvas.create_text(center_x, center_y - 20, text=score_text,
                                        fill="blue" if score_lead > 0 else "purple", font=fnt)
                self.canvas.create_text(center_x, center_y, text=puct_text, fill="#006400", font=fnt)
                self.canvas.create_text(center_x, center_y + 20, text=visits_text, fill="black", font=fnt)
        if self.best_puct_move:
            r, c = self.best_puct_move;
            x1, y1, x2, y2 = c * self.cell_size, r * self.cell_size, (c + 1) * self.cell_size, (r + 1) * self.cell_size
            self.canvas.create_rectangle(x1 + 6, y1 + 6, x2 - 6, y2 - 6, outline="blue", width=3)
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
        if self.game_running:
            self.status_label.config(text=f"当前回合: {name}{turn_info}", fg=color)
        else:
            self.status_label.config(text="游戏结束", fg="black")

        score_diff = c_lib.get_score_diff(ctypes.byref(self.board_c))
        # NEW: Add uncertainty display
        uncertainty_text = ""
        if self.root_uncertainty is not None:
            uncertainty_text = f" | AI判断不确定性: {self.root_uncertainty:.4f}"

        self.moves_label.config(
            text=f"红方剩余: {self.board_c.moves_left[0]} | 绿方剩余: {self.board_c.moves_left[1]}\n"
                 f"当前分数 (红-绿): {score_diff}{uncertainty_text}")

    def _toggle_ui_elements(self, enabled):
        state = tk.NORMAL if enabled else tk.DISABLED
        self.new_game_button.config(state=state)
        self.undo_button.config(state=tk.NORMAL if self.game_history and enabled else tk.DISABLED)
        self.analyze_button.config(state=state)

    def _handle_click(self, event):
        mode = self.game_mode.get()
        is_human = (mode == "human_vs_human") or (
                    mode == "human_vs_ai" and self.board_c.current_player == self.human_player)
        if not self.game_running or not is_human: return
        c, r = event.x // self.cell_size, event.y // self.cell_size
        if 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE: self._process_move(r * BOARD_SIZE + c)

    def _process_move(self, sq):
        board_copy = Board();
        c_lib.copy_board(ctypes.byref(self.board_c), ctypes.byref(board_copy));
        self.game_history.append(board_copy)
        if not c_lib.make_move(ctypes.byref(self.board_c), sq): self.game_history.pop(); messagebox.showwarning(
            "非法落子", "该位置不符合落子规则。"); return
        self._last_move_coords, self.analysis_data, self.best_puct_move, self.best_visit_move, self.root_uncertainty = (
        sq // BOARD_SIZE, sq % BOARD_SIZE), {}, None, None, None
        c_lib.mcts_make_move(self.mcts_manager_gui, 0, sq);
        self._continue_game_flow()

    def _undo_move(self):
        if not self.game_history: return
        steps = 2 if self.game_mode.get() == "human_vs_ai" and len(self.game_history) >= 2 else 1
        for _ in range(steps):
            if self.game_history: self.board_c = self.game_history.pop()
        c_lib.mcts_reset_for_analysis(self.mcts_manager_gui, 0, ctypes.byref(self.board_c))
        self.analysis_data, self.best_puct_move, self.best_visit_move, self._last_move_coords, self.game_running, self.root_uncertainty = {}, None, None, None, True, None
        self._continue_game_flow()

    def _toggle_analysis(self):
        if self.analysis_data: self.analysis_data, self.best_puct_move, self.best_visit_move, self.root_uncertainty = {}, None, None, None; self.draw_board(); self._update_status_labels(); return
        if not self.game_running: return
        self._toggle_ui_elements(False);
        self.analyze_button.config(text="分析中...")
        threading.Thread(target=self._run_analysis_thread, daemon=True).start()

    def _run_analysis_thread(self):
        noise_is_on = self.noise_enabled_var.get()
        c_lib.mcts_set_noise_enabled(self.mcts_manager_gui, 0, noise_is_on)
        sims = self.ai_sims_slider.get();
        c_lib.mcts_reset_for_analysis(self.mcts_manager_gui, 0, ctypes.byref(self.board_c))

        # NEW: Perform one evaluation at the root to get the uncertainty
        tensor = self._board_to_tensor(self.board_c)
        input_batch = torch.from_numpy(np.array([tensor])).to(self.device)
        with torch.no_grad():
            _, _, _, _, uncertainties_raw = self.ai_model(input_batch)
        self.root_uncertainty = uncertainties_raw.item()

        self._run_mcts_loop(sims, is_analysis=True)
        moves, q_vals, visits, pucts = (ctypes.c_int * 81)(), (ctypes.c_float * 81)(), (ctypes.c_int * 81)(), (
                    ctypes.c_float * 81)()
        num_moves = c_lib.mcts_get_analysis_data(self.mcts_manager_gui, 0, moves, q_vals, visits, pucts, 81)
        temp_data, best_p_move, best_puct, best_v_move, best_visits = {}, None, -float('inf'), None, -1
        for i in range(num_moves):
            sq, r, c, q, v, p = moves[i], moves[i] // 9, moves[i] % 9, q_vals[i], visits[i], pucts[i]
            temp_data[(r, c)] = {"q": q, "visits": v, "puct": p}
            if p > best_puct: best_puct, best_p_move = p, (r, c)
            if v > best_visits: best_visits, best_v_move = v, (r, c)

        def update_gui():
            self.analysis_data, self.best_puct_move, self.best_visit_move = temp_data, best_p_move, best_v_move
            self.draw_board();
            self._toggle_ui_elements(True);
            self.analyze_button.config(text="分析")
            self._update_status_labels()  # Refresh status to show uncertainty

        self.master.after(0, update_gui)

    def _ai_turn_logic(self):
        noise_is_on = self.noise_enabled_var.get()
        c_lib.mcts_set_noise_enabled(self.mcts_manager_gui, 0, noise_is_on)
        if self.game_mode.get() == "ai_vs_ai": time.sleep(0.3)
        sims = self.ai_sims_slider.get();
        self._run_mcts_loop(sims)
        policy_buffer = (ctypes.c_float * 81)();
        c_lib.mcts_get_policy(self.mcts_manager_gui, 0, policy_buffer)
        policy_np = np.ctypeslib.as_array(policy_buffer)
        temp = self.temperature_slider.get()
        if temp > 0:
            probs = policy_np ** (1.0 / temp)
            legal_bb = c_lib.get_legal_moves(ctypes.byref(self.board_c))
            for sq in range(81):
                if not ((legal_bb.parts[sq // 64] >> (sq % 64)) & 1): probs[sq] = 0
            sum_p = np.sum(probs)
            if sum_p > 1e-8:
                move = np.random.choice(81, p=probs / sum_p)
            else:
                legal_idx = [sq for sq in range(81) if ((legal_bb.parts[sq // 64] >> (sq % 64)) & 1)]
                move = np.random.choice(legal_idx) if legal_idx and self.game_running else -1
        else:
            move = np.argmax(policy_np)
        if self.game_running and move != -1: self.master.after(0, self._process_move, move)

    def _run_mcts_loop(self, num_sims, is_analysis=False):
        while c_lib.mcts_get_simulations_done(self.mcts_manager_gui, 0) < num_sims:
            if not self.game_running and not is_analysis: return
            boards, indices = (Board * 1)(), (ctypes.c_int * 1)()
            if c_lib.mcts_run_simulations_and_get_requests(self.mcts_manager_gui, boards, indices, 1) > 0:
                tensor = self._board_to_tensor(boards[0])
                input_batch = torch.from_numpy(np.array([tensor])).to(self.device)
                with torch.no_grad():
                    p_logits, val_raw, _, _, uncertainties_raw = self.ai_model(input_batch)
                    policy = torch.softmax(p_logits, dim=1).cpu().numpy()
                    val = val_raw.item()
                    uncertainties = uncertainties_raw.cpu().numpy().flatten()
                c_lib.mcts_feed_results(
                    self.mcts_manager_gui,
                    np.ascontiguousarray(policy, dtype=np.float32).ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    np.ascontiguousarray([val], dtype=np.float32).ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    np.ascontiguousarray(uncertainties, dtype=np.float32).ctypes.data_as(ctypes.POINTER(ctypes.c_float))
                )

    def _board_to_tensor(self, board_c: Board) -> np.ndarray:
        tensor = np.zeros((NUM_INPUT_CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        p = board_c.current_player
        o = 1 - p

        def get_plane(bb):
            plane = np.zeros(81, dtype=np.float32)
            for i in range(81):
                if (bb.parts[i // 64] >> (i % 64)) & 1: plane[i] = 1.0
            return plane.reshape((9, 9))

        tensor[0, :, :], tensor[1, :, :] = get_plane(board_c.pieces[p]), get_plane(board_c.pieces[o])
        tensor[2, :, :], tensor[3, :, :] = get_plane(board_c.tiles[p]), get_plane(board_c.tiles[o])
        tensor[4, :, :], tensor[5, :, :] = (1., 0.) if p == 0 else (0., 1.)
        tensor[6, :, :] = float(board_c.moves_left[0]) / MAX_MOVES_PER_PLAYER
        tensor[7, :, :] = float(board_c.moves_left[1]) / MAX_MOVES_PER_PLAYER
        tensor[8, :, :] = float(c_lib.pop_count(ctypes.byref(board_c.tiles[0]))) / 81
        tensor[9, :, :] = float(c_lib.pop_count(ctypes.byref(board_c.tiles[1]))) / 81
        all_tiles = Bitboards();
        all_tiles.parts[0] = ~(board_c.tiles[0].parts[0] | board_c.tiles[1].parts[0]);
        all_tiles.parts[1] = ~(board_c.tiles[0].parts[1] | board_c.tiles[1].parts[1])
        tensor[10, :, :] = get_plane(all_tiles)
        return tensor


if __name__ == "__main__":
    root = tk.Tk()
    root.resizable(False, False)
    app = PomPomGUI(root)
    root.mainloop()
