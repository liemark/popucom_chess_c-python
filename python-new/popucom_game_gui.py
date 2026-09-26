import tkinter as tk
from tkinter import ttk, messagebox, Scale, font, Checkbutton, BooleanVar
import os
import numpy as np
import threading
import time
import ctypes
import platform
import random
import argparse

from popucom_nn_interface import BOARD_SIZE, NUM_INPUT_CHANNELS, MAX_MOVES_PER_PLAYER
from adaptive_komi import AdaptiveKomiSampler, MIN_KOMI as _MIN_KOMI, MAX_KOMI as _MAX_KOMI


# --- C 语言接口定义 ---
class Bitboards(ctypes.Structure): _fields_ = [("parts", ctypes.c_uint64 * 2)]


class Board(ctypes.Structure): _fields_ = [("pieces", Bitboards * 2), ("tiles", Bitboards * 2),
                                           ("current_player", ctypes.c_int), ("moves_left", ctypes.c_int * 2)]


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
    "boards_to_tensors_with_komi_c": (None, [ctypes.POINTER(Board), ctypes.c_int,
                                             ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_float)]),
    "mcts_set_komi": (None, [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]),
    "destroy_mcts_manager": (None, [ctypes.c_void_p]),
    "mcts_run_simulations_and_get_requests": (
        ctypes.c_int, [ctypes.c_void_p, ctypes.POINTER(Board), ctypes.POINTER(ctypes.c_int), ctypes.c_int]),
    "mcts_feed_results": (
    None, [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(Board)]),
    "mcts_feed_results_with_uncertainty": (
    None, [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
           ctypes.POINTER(ctypes.c_float), ctypes.POINTER(Board)]),
    "mcts_get_policy": (ctypes.c_bool, [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_float)]),
    "mcts_make_move": (None, [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]),
    "mcts_get_simulations_done": (ctypes.c_int, [ctypes.c_void_p, ctypes.c_int]),
    "mcts_get_analysis_data": (ctypes.c_int, [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int),
                                              ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_int),
                                              ctypes.POINTER(ctypes.c_float), ctypes.c_int]),
    "mcts_reset_for_analysis": (None, [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(Board)]),
    "mcts_get_legal_moves_mask": (None, [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_float)])
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
    messagebox.showerror("库加载错误", str(e))
    exit()

# --- 全局常量 ---
BLACK_PLAYER, WHITE_PLAYER = 0, 1
IN_PROGRESS, DRAW, BLACK_WIN, WHITE_WIN = -1, 0, 1, 2
BOARD_BACKGROUND_COLOR, GRID_LINE_COLOR = "#D2B48C", "#8B4513"
UNPAINTED_FLOOR_COLOR, BLACK_PAINTED_FLOOR_COLOR, WHITE_PAINTED_FLOOR_COLOR = "#FFF8DC", "#FFDAB9", "#90EE90"
BLACK_PIECE_COLOR, WHITE_PIECE_COLOR = "red", "green"
DEFAULT_FPU_GUI = 0.0


# --- 推理后端抽象 ---
# 统一接口: infer(input_np) -> (policy[B,81] float32, value[B], uncertainty[B] or None, score[B] or None)
# 输入为 (B, NUM_INPUT_CHANNELS, BOARD_SIZE, BOARD_SIZE) 的 float32 numpy 数组。
# uncertainty / score 为 None 表示该模型不带对应输出头（走兼容路径）。
class TorchBackend:
    name = "torch"

    def __init__(self, model_path="model.pth"):
        import torch  # 延迟导入：onnx 用户无需安装 torch
        from popucom_nn_model import PomPomNN
        self.torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = PomPomNN()
        # 防呆：ONNX 文件被误喂给 torch 后端（torch.load 会报难懂的解析错误）
        # ONNX protobuf 首字节恒为 0x08；torch 格式为 zip(PK) 或旧式 pickle(\x80)
        try:
            with open(model_path, 'rb') as f:
                head = f.read(16)
            if head[:1] == b'\x08':
                raise ValueError(f"'{model_path}' 是 ONNX 模型文件，应使用 onnx 后端加载。")
        except OSError:
            pass
        try:
            # strict=False：兼容不含新输出头的旧权重，新头随机初始化
            # PyTorch 2.6 默认 weights_only=True：先按安全模式加载，失败再回退
            # （自有训练产物，来源可信）
            try:
                state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
            except Exception:
                state_dict = torch.load(model_path, map_location=self.device, weights_only=False)
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if missing: print(f"注意: 以下参数在权重中不存在，将随机初始化: {missing}")
            if unexpected: print(f"注意: 权重中存在多余参数，已忽略: {unexpected}")
            print(f"AI模型已从 {model_path} 加载。")
            self.load_ok = True
        except Exception as e:
            print(f"警告: 无法加载 {model_path}: {e}，AI将使用随机权重的网络。")
            self.load_ok = False
        self.model = model.to(self.device).eval()

    def infer(self, input_np):
        torch = self.torch
        t = torch.from_numpy(input_np).to(self.device)
        with torch.no_grad():
            with torch.amp.autocast(device_type=self.device.type, enabled=(self.device.type == 'cuda')):
                p_logits, val, _, unc, score = self.model(t)
            policy = torch.softmax(p_logits.float(), dim=1).cpu().numpy()
            values = val.float().flatten().cpu().numpy()
            uncs = torch.abs(unc.float()).flatten().cpu().numpy()
            scores = score.float().flatten().cpu().numpy()
        return policy, values, uncs, scores


class OnnxBackend:
    name = "onnx"

    def __init__(self, model_path="model.onnx"):
        import onnxruntime as ort  # 延迟导入
        print(f"正在从 {model_path} 加载ONNX模型...")
        providers = ['CPUExecutionProvider', 'CUDAExecutionProvider']
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        print(f"ONNX Runtime将使用: {self.session.get_providers()[0]}")

    def infer(self, input_np):
        outs = self.session.run(None, {self.input_name: input_np})
        logits = outs[0]
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        policy = (exp_logits / np.sum(exp_logits, axis=1, keepdims=True)).astype(np.float32)
        values = np.asarray(outs[1]).flatten()
        # 旧 3 输出模型：uncertainty/score 为 None，调用方走兼容接口
        uncs = np.abs(np.asarray(outs[3])).flatten() if len(outs) >= 4 else None
        scores = np.asarray(outs[4]).flatten() if len(outs) >= 5 else None
        return policy, values, uncs, scores


class TrtBackend:
    name = "trt"

    def __init__(self, model_path="model.plan"):
        import torch  # TensorRTModel 依赖 torch + CUDA
        from self_play_worker_trt import TensorRTModel
        self.torch = torch
        self.model = TensorRTModel(model_path)
        print(f"TensorRT 引擎 ({model_path}) 已为GUI加载。")

    def infer(self, input_np):
        # TensorRT 引擎按固定 max_batch_size 导出，此处逐样本推理最稳妥
        torch = self.torch
        pols, vals, uncs, scs = [], [], [], []
        with torch.no_grad():
            for i in range(input_np.shape[0]):
                t = torch.from_numpy(input_np[i:i + 1]).to('cuda')
                p_logits, val, _, unc, score = self.model(t)
                pols.append(torch.softmax(p_logits.float(), dim=1).cpu().numpy())
                vals.append(val.float().flatten().cpu().numpy())
                uncs.append(torch.abs(unc.float()).flatten().cpu().numpy())
                scs.append(score.float().flatten().cpu().numpy())
        return (np.concatenate(pols, 0), np.concatenate(vals, 0),
                np.concatenate(uncs, 0), np.concatenate(scs, 0))


BACKEND_DEFAULT_MODEL = {'torch': 'model.pth', 'onnx': 'model.onnx', 'trt': 'model.plan'}


def _detect_backend_name(path):
    """按扩展名 + 文件魔数判断权重文件应使用的后端类型（torch/onnx/trt）"""
    ext = os.path.splitext(path)[1].lower()
    if ext == '.onnx': return 'onnx'
    if ext == '.plan': return 'trt'
    if ext in ('.pth', '.pt'): return 'torch'
    try:
        with open(path, 'rb') as f:
            head = f.read(512)
    except OSError:
        return None
    if head[:2] == b'PK' or head[:1] == b'\x80': return 'torch'  # zip / 旧式 pickle
    if head[:1] == b'\x08': return 'onnx'  # ONNX protobuf（ir_version 字段 tag）
    return None


def create_backend(name, model_path=None):
    model_path = model_path or BACKEND_DEFAULT_MODEL[name]
    if name == 'torch': return TorchBackend(model_path)
    if name == 'onnx': return OnnxBackend(model_path)
    if name == 'trt': return TrtBackend(model_path)
    raise ValueError(f"未知后端: {name}")


class GameTreeNode:
    """Represents a node in the game tree."""

    def __init__(self, board_state, move_sq=None, parent=None):
        self.board_state = board_state
        self.move_sq = move_sq
        self.parent = parent
        self.children = {}  # key: move_sq, value: GameTreeNode


class PomPomGUI:
    def __init__(self, master, backend):
        self.master = master
        master.title(f"泡姆棋 [{backend.name}]")
        self.backend = backend
        self.default_model_path = BACKEND_DEFAULT_MODEL.get(backend.name, "model.pth")
        self.cell_size = 60
        self.game_running = False
        self.game_mode = tk.StringVar(value="human_vs_ai")
        self.human_player_choice = tk.StringVar(value="human_black")
        self.human_player, self.ai_player = BLACK_PLAYER, WHITE_PLAYER
        self.board_c, self._last_move_coords = Board(), None
        self.dirichlet_noise_enabled = BooleanVar(value=False)
        self.mcts_manager_gui = c_lib.create_mcts_manager(1, self.dirichlet_noise_enabled.get(), DEFAULT_FPU_GUI)
        # [新增] 贴目：默认取自适应统计的 k*（白给红），可在 UI 中调整
        try:
            _k_star = AdaptiveKomiSampler().current_k_star()
        except Exception:
            _k_star = 0
        self.komi_var = tk.IntVar(value=_k_star)
        c_lib.mcts_set_komi(self.mcts_manager_gui, 0, _k_star)
        self.ai_thread = None
        self.analysis_data, self.analysis_in_progress = {}, False
        self.best_puct_move, self.best_visit_move = None, None

        self.game_tree_root = None
        self.current_node = None
        self.listbox_nodes = []

        self._setup_gui()
        self._reset_and_start_new_game()

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
        self.ai_sims_slider = Scale(ai_frame, from_=50, to=10000, orient=tk.HORIZONTAL, resolution=50, length=200)
        self.ai_sims_slider.set(1200)
        self.ai_sims_slider.grid(row=0, column=1, sticky='ew')

        tk.Label(ai_frame, text="落子温度:").grid(row=1, column=0, sticky='w')
        self.temperature_slider = Scale(ai_frame, from_=0.0, to=2.0, orient=tk.HORIZONTAL, resolution=0.1, length=200)
        self.temperature_slider.set(0.0)
        self.temperature_slider.grid(row=1, column=1, sticky='ew')

        tk.Label(ai_frame, text="节点初始分数/探索性:").grid(row=2, column=0, sticky='w')
        self.fpu_slider = Scale(ai_frame, from_=-0.5, to=0.5, orient=tk.HORIZONTAL, resolution=0.01, length=100,
                                command=self._on_fpu_change)
        self.fpu_slider.set(DEFAULT_FPU_GUI)
        self.fpu_slider.grid(row=2, column=1, sticky='ew')

        self.noise_checkbox = Checkbutton(ai_frame, text="开启狄利克雷噪声",
                                          variable=self.dirichlet_noise_enabled, command=self._on_noise_toggle)
        self.noise_checkbox.grid(row=3, column=0, columnspan=2, sticky='w', pady=(5, 0))

        # 贴目设置（白给红的目数，0 = 无贴目）
        tk.Label(ai_frame, text="贴目 (白给红):").grid(row=4, column=0, sticky='w', pady=(5, 0))
        self.komi_spinbox = ttk.Spinbox(ai_frame, from_=_MIN_KOMI, to=_MAX_KOMI, textvariable=self.komi_var,
                                        width=5, command=self._on_komi_change)
        self.komi_spinbox.grid(row=4, column=1, sticky='w', pady=(5, 0))

        # [新增] 模型权重选择（多种训练权重并存时可手动切换）
        tk.Label(ai_frame, text="模型权重:").grid(row=5, column=0, sticky='w', pady=(5, 0))
        self.model_path_var = tk.StringVar(value=self.default_model_path)
        model_entry = tk.Entry(ai_frame, textvariable=self.model_path_var)
        model_entry.grid(row=5, column=1, sticky='ew', pady=(5, 0))
        self.reload_model_button = tk.Button(ai_frame, text="加载", width=6, command=self._reload_backend)
        self.reload_model_button.grid(row=5, column=2, padx=(4, 0), pady=(5, 0))
        ai_frame.columnconfigure(1, weight=1)

        btn_frame = tk.Frame(control_frame)
        btn_frame.pack(side=tk.RIGHT, padx=5, fill=tk.Y)
        self.new_game_button = tk.Button(btn_frame, text="新对局/清空", command=self._reset_and_start_new_game,
                                         width=10)
        self.new_game_button.pack(pady=2)
        self.undo_button = tk.Button(btn_frame, text="返回上步", command=self._undo_move, width=10)
        self.undo_button.pack(pady=2)
        self.analyze_button = tk.Button(btn_frame, text="分析", command=self._toggle_analysis, width=10)
        self.analyze_button.pack(pady=2)

        status_frame = tk.Frame(top_frame, pady=5)
        status_frame.pack(fill=tk.X, side=tk.BOTTOM)
        self.status_label = tk.Label(status_frame, text="初始化...", font=("Arial", 14, "bold"))
        self.status_label.pack()
        self.moves_label = tk.Label(status_frame, text="", font=("Arial", 11))
        self.moves_label.pack()

        canvas_frame = tk.Frame(main_board_frame)
        canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(canvas_frame, bg=BOARD_BACKGROUND_COLOR)
        self.canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.canvas.bind("<Button-1>", self._handle_click)
        canvas_frame.bind("<Configure>", self._on_resize)

        self._setup_listbox_view(main_board_frame)

        self.master.protocol("WM_DELETE_WINDOW", self._on_closing)

    def _setup_listbox_view(self, parent_frame):
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
        self.game_view.delete(0, tk.END)
        self.listbox_nodes.clear()

        def build_list(node, depth=0):
            move_text = f"{self._get_move_text(node)}"
            self.game_view.insert(tk.END, move_text)
            self.listbox_nodes.append(node)
            for _, child_node in sorted(node.children.items()):
                build_list(child_node, depth + 1)

        build_list(self.game_tree_root)
        self._highlight_current_node_in_view()

    def _get_move_text(self, node):
        if node.parent is None: return "游戏开始"
        board_before_move = node.parent.board_state
        turn_player = board_before_move.current_player
        total_moves_made = (MAX_MOVES_PER_PLAYER * 2) - (
                    board_before_move.moves_left[0] + board_before_move.moves_left[1])
        turn_number = total_moves_made // 2 + 1
        move_prefix = f"{turn_number}." if turn_player == BLACK_PLAYER else f"{turn_number}..."
        return f"{move_prefix} {self._coords_to_alg(node.move_sq)}"

    def _highlight_current_node_in_view(self):
        try:
            if self.current_node in self.listbox_nodes:
                idx = self.listbox_nodes.index(self.current_node)
                self.game_view.selection_clear(0, tk.END)
                self.game_view.selection_set(idx)
                self.game_view.activate(idx)
                self.game_view.see(idx)
        except (ValueError, tk.TclError):
            pass

    def _on_resize(self, event):
        new_size = min(event.width, event.height)
        new_cell_size = new_size // BOARD_SIZE - 1
        if new_cell_size < 20 or abs(new_cell_size - self.cell_size) < 2: return
        self.cell_size = new_cell_size
        self.draw_board()

    def _coords_to_alg(self, sq):
        if sq is None: return "Start"
        return f"{'abcdefghi'[sq % BOARD_SIZE]}{sq // BOARD_SIZE + 1}"

    def _on_fpu_change(self, value):
        if self.mcts_manager_gui: c_lib.mcts_set_fpu(self.mcts_manager_gui, float(value))

    def _on_noise_toggle(self):
        if self.mcts_manager_gui: c_lib.mcts_set_noise_enabled(self.mcts_manager_gui,
                                                               self.dirichlet_noise_enabled.get())

    def _stop_ai_thread(self):
        """A helper function to safely stop any running AI thread."""
        if self.ai_thread and self.ai_thread.is_alive():
            self.game_running = False
            self.ai_thread.join(timeout=0.5)
        self.game_running = True  # Reset flag for the new state

    def _on_closing(self):
        self._stop_ai_thread()
        if self.mcts_manager_gui: c_lib.destroy_mcts_manager(self.mcts_manager_gui)
        self.master.destroy()

    def _on_mode_change(self):
        self._stop_ai_thread()
        mode = self.game_mode.get()
        self.role_frame.pack(side=tk.LEFT, fill=tk.Y,
                             padx=5) if mode == "human_vs_ai" else self.role_frame.pack_forget()
        self._update_human_player()
        self._continue_game_flow()

    def _update_human_player(self):
        choice = self.human_player_choice.get()
        self.human_player, self.ai_player = (BLACK_PLAYER, WHITE_PLAYER) if choice == "human_black" else (
        WHITE_PLAYER, BLACK_PLAYER)

    def _reset_and_start_new_game(self):
        self._stop_ai_thread()
        c_lib.init_board(ctypes.byref(self.board_c))
        # 同步当前贴目到搜索（贴目改变搜索终局估值）
        c_lib.mcts_set_komi(self.mcts_manager_gui, 0, int(self.komi_var.get()))
        board_copy = Board()
        c_lib.copy_board(ctypes.byref(self.board_c), ctypes.byref(board_copy))
        self.game_tree_root = GameTreeNode(board_copy)
        self._navigate_to_node(self.game_tree_root)

    def _on_komi_change(self):
        """贴目修改：同步到 C++ 搜索；若在残局/分析中需要重新搜索才能生效"""
        try:
            k = int(self.komi_var.get())
        except (tk.TclError, ValueError):
            return
        c_lib.mcts_set_komi(self.mcts_manager_gui, 0, k)

    def _reload_backend(self):
        """[新增] 手动加载权重文件（多种训练权重并存时可切换）。
        按文件类型自动识别并切换后端：.pth/.pt→torch，.onnx→onnx，.plan→trt。"""
        path = self.model_path_var.get().strip()
        if not path or not os.path.exists(path):
            messagebox.showwarning("权重加载失败", f"文件不存在: {path}")
            return
        self._stop_ai_thread()
        old_name = self.backend.name
        try:
            backend_name = _detect_backend_name(path) or old_name
            self.backend = create_backend(backend_name, path)
            self.master.title(f"泡姆棋 [{backend_name}]")
            if not getattr(self.backend, 'load_ok', True):
                messagebox.showwarning("权重加载失败", f"无法加载 '{path}'（详见控制台），已回退随机权重。")
                return
            msg = f"已加载: {path}"
            if backend_name != old_name:
                msg += f"\n（后端已切换: {old_name} → {backend_name}）"
            messagebox.showinfo("权重加载成功", msg)
        except Exception as e:
            messagebox.showerror("权重加载失败", str(e))

    def _continue_game_flow(self):
        self.draw_board()
        self._update_status_labels()
        self._toggle_ui_elements(True)
        game_result = c_lib.get_game_result(ctypes.byref(self.board_c))
        if game_result != IN_PROGRESS:
            self._handle_game_over(game_result)
            return
        mode = self.game_mode.get()
        player = self.board_c.current_player
        is_ai_turn = (mode == "ai_vs_ai") or (mode == "human_vs_ai" and player == self.ai_player)
        if is_ai_turn and self.game_running:
            self._toggle_ui_elements(False)
            self.ai_thread = threading.Thread(target=self._ai_turn_logic, daemon=True)
            self.ai_thread.start()

    def _handle_game_over(self, result):
        self.game_running = False
        self._update_status_labels()
        score_diff = c_lib.get_score_diff(ctypes.byref(self.board_c))
        komi = int(self.komi_var.get())
        # 贴目感知终局：双方步数用尽时按 (score_diff + komi) 判胜负，无步判负与贴目无关
        if (self.board_c.moves_left[0] <= 0 and self.board_c.moves_left[1] <= 0):
            adjusted = score_diff + komi
            if adjusted > 0: result = BLACK_WIN
            elif adjusted < 0: result = WHITE_WIN
            else: result = DRAW
            score_diff = abs(adjusted)
            komi_note = f"（含贴目 {komi}）" if komi else ""
        else:
            komi_note = ""
        msg = "平局！"
        if result == BLACK_WIN:
            msg = f"红方胜利！领先 {score_diff} 格{komi_note}。"
        elif result == WHITE_WIN:
            msg = f"绿方胜利！领先 {score_diff} 格{komi_note}。"
        messagebox.showinfo("游戏结束", msg)
        self._toggle_ui_elements(True)

    def draw_board(self):
        self.canvas.delete("all")
        board_pixel_size = BOARD_SIZE * self.cell_size
        self.canvas.config(width=board_pixel_size, height=board_pixel_size)
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                sq = r * BOARD_SIZE + c
                x1, y1 = c * self.cell_size, r * self.cell_size
                x2, y2 = x1 + self.cell_size, y1 + self.cell_size
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
                if self._last_move_coords == (r, c):
                    self.canvas.create_oval(center_x - radius / 4, center_y - radius / 4, center_x + radius / 4,
                                           center_y + radius / 4, fill="white", outline="black")
        if self.analysis_data:
            fnt = font.Font(family='Helvetica', size=max(5, int(self.cell_size / 16)), weight='bold')
            for (r, c), data in self.analysis_data.items():
                center_x, center_y = c * self.cell_size + self.cell_size / 2, r * self.cell_size + self.cell_size / 2
                win_rate = (data['q'] + 1) / 2 * 100
                # [新增] 4 行标注：胜率 / 访问 / 目差 / PUCT；PUCT 行放不下时自动省略
                lines = [(f"胜率: {win_rate:.1f}%", "blue" if data['q'] > 0 else "purple"),
                         (f"访问: {data['visits']}", "black")]
                if data.get('margin') is not None:
                    m = data['margin']
                    lines.append((f"目差: {m:+.1f}",
                                  "red" if m > 0 else ("blue" if m < 0 else "gray")))
                puct_text = f"PUCT: {data['puct']:.3f}"
                if fnt.measure(puct_text) <= self.cell_size - 2:
                    lines.append((puct_text, "#006400"))
                for i, (txt, col) in enumerate(lines):
                    y = center_y + (i - (len(lines) - 1) / 2) * self.cell_size * 0.22
                    self.canvas.create_text(center_x, y, text=txt, fill=col, font=fnt)
        if self.best_puct_move:
            r, c = self.best_puct_move
            self.canvas.create_rectangle(c * self.cell_size + 6, r * self.cell_size + 6, (c + 1) * self.cell_size - 6,
                                         (r + 1) * self.cell_size - 6, outline="orange", width=3, dash=(4, 4))
        if self.best_visit_move:
            r, c = self.best_visit_move
            self.canvas.create_rectangle(c * self.cell_size + 3, r * self.cell_size + 3, (c + 1) * self.cell_size - 3,
                                         (r + 1) * self.cell_size - 3, outline="red", width=3)

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
        is_human_turn = (mode == "human_vs_human") or (
                    mode == "human_vs_ai" and self.board_c.current_player == self.human_player)
        if not self.game_running or not is_human_turn or (self.ai_thread and self.ai_thread.is_alive()): return
        if self.cell_size == 0: return
        c, r = event.x // self.cell_size, event.y // self.cell_size
        if 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE: self._process_move(r * BOARD_SIZE + c)

    def _process_move(self, sq, from_ai=False):
        legal_bb = c_lib.get_legal_moves(ctypes.byref(self.board_c))
        if not c_lib.is_bit_set(ctypes.byref(legal_bb), sq):
            if not from_ai: messagebox.showwarning("非法落子", "该位置不符合落子规则。")
            return
        if sq not in self.current_node.children:
            board_after_move = Board()
            c_lib.copy_board(ctypes.byref(self.current_node.board_state), ctypes.byref(board_after_move))
            c_lib.make_move(ctypes.byref(board_after_move), sq)
            new_node = GameTreeNode(board_after_move, move_sq=sq, parent=self.current_node)
            self.current_node.children[sq] = new_node
        self._navigate_to_node(self.current_node.children[sq])

    def _undo_move(self):
        """Intelligent undo logic: H-vs-AI 模式回退到人类上一个回合。"""
        self._stop_ai_thread()
        if not self.current_node or not self.current_node.parent: return

        mode = self.game_mode.get()
        if mode == "human_vs_ai":
            target_node = self.current_node.parent
            if target_node and target_node.board_state.current_player == self.human_player:
                if target_node.parent: target_node = target_node.parent
            while target_node and target_node.parent and target_node.board_state.current_player != self.human_player:
                target_node = target_node.parent
            if target_node: self._navigate_to_node(target_node)
        else:
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
        """Central function to change the current game state to a specific node."""
        self._stop_ai_thread()
        self.current_node = node
        c_lib.copy_board(ctypes.byref(self.current_node.board_state), ctypes.byref(self.board_c))
        if self.current_node.move_sq is not None:
            self._last_move_coords = (self.current_node.move_sq // BOARD_SIZE, self.current_node.move_sq % BOARD_SIZE)
        else:
            self._last_move_coords = None
        self.analysis_data, self.best_puct_move, self.best_visit_move = {}, None, None
        if self.mcts_manager_gui: c_lib.mcts_reset_for_analysis(self.mcts_manager_gui, 0, ctypes.byref(self.board_c))
        self.game_running = (c_lib.get_game_result(ctypes.byref(self.board_c)) == IN_PROGRESS)
        self._update_game_view()
        self._continue_game_flow()

    def _toggle_analysis(self):
        if self.analysis_data:
            self.analysis_data, self.best_puct_move, self.best_visit_move = {}, None, None
            self.draw_board()
            return
        if not self.game_running: return
        if self.ai_thread and self.ai_thread.is_alive(): return
        self._toggle_ui_elements(False)
        self.analyze_button.config(text="分析中...")
        self.ai_thread = threading.Thread(target=self._run_analysis_thread, daemon=True)
        self.ai_thread.start()

    def _run_analysis_thread(self):
        try:
            sims = self.ai_sims_slider.get()
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

            # [新增] 候选目差：对每个候选走法的子局面做一次批量推理。
            # score_head 输出为"子局面行棋方（=当前方的对手）视角、贴目调整后净胜目数"，
            # 取反即得当前行棋方视角：轮到谁分析，就显示"该方预计净胜几目"。
            # 训练早期若引擎认为先手 +6，则先手回合各格显示 +6，后手回合显示 -6。
            if temp_data:
                try:
                    cand_cells = list(temp_data.keys())
                    n = len(cand_cells)
                    boards_np = (Board * n)()
                    for idx, (r, c) in enumerate(cand_cells):
                        b = Board()
                        c_lib.copy_board(ctypes.byref(self.board_c), ctypes.byref(b))
                        c_lib.make_move(ctypes.byref(b), r * BOARD_SIZE + c)
                        boards_np[idx] = b
                    input_np = np.zeros((n, NUM_INPUT_CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
                    komi_np = np.array([int(self.komi_var.get())] * n, dtype=np.int32)
                    c_lib.boards_to_tensors_with_komi_c(
                        boards_np, n, komi_np.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                        input_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
                    _, _, _, cand_scores = self.backend.infer(input_np)
                    if cand_scores is not None:
                        for idx, cell in enumerate(cand_cells):
                            temp_data[cell]['margin'] = -float(cand_scores[idx])
                except Exception as e:
                    print(f"候选目差推理失败（不影响分析显示）: {e}")

            def update_gui_after_analysis():
                self.analysis_data = temp_data
                self.best_puct_move = best_p_move
                self.best_visit_move = best_v_move
                self.draw_board()

            self.master.after(0, update_gui_after_analysis)
        except Exception as e:
            print(f"分析出错: {e}")
            self.master.after(0, lambda: self.analyze_button.config(text="分析(错误)"))
        finally:
            self.master.after(0, lambda: (self._toggle_ui_elements(True), self.analyze_button.config(text="分析")))

    def _ai_turn_logic(self):
        if self.game_mode.get() == "ai_vs_ai": time.sleep(0.3)
        sims = self.ai_sims_slider.get()
        self._run_mcts_loop(sims)
        if not self.game_running: return
        policy_buffer = (ctypes.c_float * 81)()
        c_lib.mcts_get_policy(self.mcts_manager_gui, 0, policy_buffer)
        policy_np = np.ctypeslib.as_array(policy_buffer)
        temp = self.temperature_slider.get()
        move = -1
        legal_bb = c_lib.get_legal_moves(ctypes.byref(self.board_c))
        if temp > 0:
            probs = policy_np ** (1.0 / temp)
            for sq in range(81):
                if not c_lib.is_bit_set(ctypes.byref(legal_bb), sq): probs[sq] = 0
            sum_p = np.sum(probs)
            if sum_p > 1e-8: move = np.random.choice(81, p=probs / sum_p)
        else:
            best_prob = -1
            for sq in range(81):
                if c_lib.is_bit_set(ctypes.byref(legal_bb), sq) and policy_np[sq] > best_prob:
                    best_prob, move = policy_np[sq], sq
        if move == -1:  # 兜底：策略全零时随机选一个合法步
            legal_idx = [sq for sq in range(81) if c_lib.is_bit_set(ctypes.byref(legal_bb), sq)]
            if legal_idx: move = random.choice(legal_idx)

        if self.game_running and move != -1:
            self.master.after(0, self._process_move, move, True)

    def _run_mcts_loop(self, num_sims, is_analysis=False):
        while c_lib.mcts_get_simulations_done(self.mcts_manager_gui, 0) < num_sims:
            if not self.game_running and not is_analysis: return
            boards, indices = (Board * 1)(), (ctypes.c_int * 1)()
            if c_lib.mcts_run_simulations_and_get_requests(self.mcts_manager_gui, boards, indices, 1) > 0:
                # 12 通道输入（通道 11 = 贴目平面）
                input_tensor_np = np.zeros((1, NUM_INPUT_CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
                komi_np = np.array([int(self.komi_var.get())], dtype=np.int32)
                c_lib.boards_to_tensors_with_komi_c(
                    boards, 1, komi_np.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                    input_tensor_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
                # 统一后端推理：numpy 进 numpy 出（后端差异在此消化）
                policy, values, uncs, scores = self.backend.infer(input_tensor_np)
                value = float(values[0])
                if uncs is not None:
                    uncertainty = float(abs(uncs[0]))
                    c_lib.mcts_feed_results_with_uncertainty(self.mcts_manager_gui,
                                        np.ascontiguousarray(policy, dtype=np.float32).ctypes.data_as(
                                            ctypes.POINTER(ctypes.c_float)), ctypes.byref(ctypes.c_float(value)),
                                        ctypes.byref(ctypes.c_float(uncertainty)), boards)
                else:
                    # 旧模型（无不确定度头）走兼容接口
                    c_lib.mcts_feed_results(self.mcts_manager_gui,
                                        np.ascontiguousarray(policy, dtype=np.float32).ctypes.data_as(
                                            ctypes.POINTER(ctypes.c_float)), ctypes.byref(ctypes.c_float(value)),
                                        boards)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="泡姆棋 GUI（统一推理后端）")
    parser.add_argument('--backend', choices=['torch', 'onnx', 'trt'], default='torch',
                        help="推理后端：torch=model.pth | onnx=model.onnx | trt=model.plan")
    parser.add_argument('--model', default=None,
                        help="模型权重文件路径（默认取后端对应文件；GUI 内也可随时切换）")
    args = parser.parse_args()
    try:
        backend = create_backend(args.backend, args.model)
    except Exception as e:
        _root = tk.Tk()
        _root.withdraw()
        messagebox.showerror("后端加载失败", f"无法加载 '{args.backend}' 后端: {e}")
        exit(1)
    root = tk.Tk()
    app = PomPomGUI(root, backend)
    root.mainloop()
