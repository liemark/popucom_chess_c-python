# adaptive_komi.py
# 自适应随机贴目 (Adaptive Random Komi, 胜率核加权采样)
#
# 泡姆泡姆棋后手(白)必胜，训练时由白给黑贴 k 目以平衡局面：
#   贴目调整后黑胜  <=>  score_diff + k > 0   (score_diff = 黑地板数 - 白地板数)
#   k 可为负（黑倒贴白）——理论公平贴目虽为正，但训练初期引擎远弱于最优，
#   实测平衡点 k* 由历史目差分布决定，可能落在负区间，采样须覆盖
#
# 核心思想（胜率核加权，替代简单三角分布）：
#   1. 一局棋的终局目差 d 唯一决定它在任何贴目 k 下的胜负 (d + k > 0)，
#      因此历史对局的目差分布就是【所有】贴目下初始局面胜率的完整经验估计：
#      p̂(k) = P(d + k > 0)，无需试玩不同贴目，也无需模型估计（无偏、零开销）
#   2. 采样权重 w(k) ∝ exp( -（p̂(k) - 0.5)² / (2σ²) )
#      - 目差分散 → 接近 50% 胜率的贴目区间宽 → 采样自然铺开（多样性）
#      - 目差集中 → 采样自动收缩到公平贴目附近（避免采样"已决定"的对局，
#        即从第 1 步起价值标签恒为 ±1 的对局，其学习信号弱）
#      - 权重分布形状由真实对局数据决定，而非固定几何形状
#   3. FLOOR 探索下限保证极端贴目仍有采样机会
#   4. 冷启动（统计局数不足）时均匀采样（含负贴目），快速积累目差统计
#
# 时间平滑（对抗训练中的目差统计振荡）：
#   1. 指数衰减加权直方图：每局全部历史权重 × DECAY 再记入新样本，
#      替代定长滚动窗口——窗口截断会造成"整批换血"式的统计跳变，
#      而指数衰减是逐局渐变，且天然适应策略随迭代漂移的非平稳性。
#      衰减率动态：N_eff 从 250（初期短记忆）随累计局数增长到 2000
#      （后期长记忆；引擎变强 → 目差分布集中 → p̂(k) 陡 → 需更多样本精定位 k*）。
#   2. 采样分布 EMA：实际采样的概率分布向目标核权重缓慢靠拢
#      （每局只融合 SMOOTH_A=0.3），切断
#      "贴目组合批间剧变 → 目差分布漂移 → k* 再变" 的反馈环。
#
# 备注：劣势方残局"赌一把"是正确博弈行为（反正要输，价值恒 -1），
# 会使终局目差厚尾——贴目采样只依赖目差符号（胜率是阶跃函数），
# 对长尾稳健；目差回归侧由训练的 Huber 损失(delta=3)抑制离群值。
#
# 目数为整数（地板数量），贴目同为整数；统计持久化到 komi_stats.json。

import json
import os
import math
import random
import gzip
import pickle
import re
import glob
import numpy as np

STATS_FILE = "komi_stats.json"
MIN_KOMI = -8               # [新增] 贴目下限（负值 = 补偿后手）。引擎不强/游戏后手占优时
                             # 平衡贴目可能为负；不设下限会把 k* 卡死在 0，对局一边倒
MAX_KOMI = 40               # 贴目上限（9x9 满盘 81 格，40 已远超合理范围）
SIGMA = 0.10                # 胜率核宽度：p=0.5→w=1，p=0.7→w≈0.14，p=0.9→w≈3e-4
FLOOR = 0.01                # 探索下限（相对峰值），保证极端贴目仍有采样机会
COLD_START_GAMES = 16       # 有效局数少于该值时进入冷启动均匀采样
COLD_START_MAX = 8          # 冷启动均匀采样的贴目上限（与 MIN_KOMI 对称覆盖正负两侧）
# [修改] 衰减率改为随累计局数增长的动态窗口：
#   N_eff(n) = min(N_EFF_MAX, N_EFF_MIN + n / N_EFF_GROWTH)，DECAY = 1 - 1/N_eff
#   - 初期 N_EFF_MIN=250（≈旧 DECAY=0.996）：策略漂移快，短记忆快速适应
#   - 后期 N_EFF_MAX=2000（DECAY=0.9995）：引擎变强、目差分布集中，p̂(k) 曲线
#     变陡，需要更多有效样本精确定位 k*（对应"训练滑窗越来越大"）
#   - 每记录 N_EFF_GROWTH 局窗口 +1；250→2000 需约 14k 局
N_EFF_MIN = 250             # 初期有效窗口（稳态有效局数）
N_EFF_MAX = 2000            # 后期有效窗口上限
N_EFF_GROWTH = 8            # 窗口增长斜率：每记录 8 局有效窗口 +1
DECAY = 1.0 - 1.0 / N_EFF_MIN  # 初期衰减率（仅供引用/演示；实际用 _decay_now()）
SMOOTH_A = 0.3              # 采样分布 EMA 融合系数（每局向目标权重靠拢 30%）
PRUNE_WEIGHT = 1e-4         # 直方图中权重低于该值的目差项定期清理

RECORDS_INDEX = "komi_index.json"  # 棋谱特征值索引（存于数据目录）
# [修改] 覆盖动态衰减最慢情形（N_EFF_MAX=2000 → DECAY=0.9995）至权重 <1e-6
# （9.2 个自然对数半衰期单位：9*ln(10)/ln(1/0.9995) ≈ 41000；对齐 PRUNE 清理阈
# 值 1e-4 并留余量，30000 局已使最旧局权重 <1e-6，远低于清理线）
MAX_REPLAY_GAMES = 30000


def _file_epoch(path):
    """从棋谱文件名提取时间戳（selfplay_data_1790372968 / mega_batch_...），失败回退 mtime"""
    m = re.search(r'(\d{8,})', os.path.basename(path))
    return int(m.group(1)) if m else int(os.path.getmtime(path))


def _extract_game_diffs(path):
    """解压一份棋谱，提取每局终局目差（黑视角，按局完成顺序）。
    棋谱为扁平样本列表 (state, policy, value, margin, komi)：
      - margin 为整局常量（行棋方视角、贴目调整后净胜目数），
        行棋方由输入通道 4（"黑行棋"全 1 平面）判定
      - 局边界：komi/|margin| 变化，或剩余步数通道（6/7 黑/白）回升——
        局内剩余步数只减不增，回升必为新局（双判据取或，避免相邻局
        恰好同 komi 同 |margin| 时被误合并）
    返回每局的 score_diff 列表；旧格式（无贴目字段）返回 None。"""
    with gzip.open(path, 'rb') as f:
        data = pickle.load(f)
    if not data or not hasattr(data[0], '__len__') or len(data[0]) < 5:
        return None  # 旧 3 元组棋谱：无 margin/komi，无法提取
    diffs, key, prev_ml = [], None, None
    for sample in data:
        state, _policy, _value, margin, komi = sample
        m = float(margin)
        k = int(komi)
        st = np.asarray(state)
        # 黑/白剩余步数（归一化 broadcast 平面，取 .max() 即标量值）
        ml = (float(st[6].max()), float(st[7].max())) if st.shape[0] > 7 else (0.0, 0.0)
        new_game = ((k, abs(m)) != key) or (
            prev_ml is not None and (ml[0] > prev_ml[0] + 1e-6 or ml[1] > prev_ml[1] + 1e-6))
        if new_game:
            key = (k, abs(m))
            p_black = st.shape[0] > 4 and bool(st[4].any())
            diffs.append(int(round(m * (1.0 if p_black else -1.0))) - k)
        prev_ml = ml
    return diffs


class AdaptiveKomiSampler:
    def __init__(self, stats_path=STATS_FILE):
        self.stats_path = stats_path
        self.hist = {}            # 目差 -> 指数加权计数
        self.total = 0.0          # 加权总局数
        self.games_seen = 0        # [新增] 累计记录局数（驱动动态衰减窗口）
        self.sampling_dist = None  # 采样分布 EMA（dict {贴目: 权重}）
        self._load()

    # ---------- 持久化 ----------
    def _load(self):
        if os.path.exists(self.stats_path):
            try:
                with open(self.stats_path, 'r', encoding='utf-8') as f:
                    obj = json.load(f)
                if "hist" in obj:
                    # 新格式：指数加权直方图 + 采样分布 EMA
                    self.hist = {int(k): float(v) for k, v in obj["hist"].items()}
                    self.total = float(obj.get("total", sum(self.hist.values())))
                    self.games_seen = int(obj.get("games_seen", 0))  # [新增] 动态窗口驱动
                    sd = obj.get("sampling_dist")
                    # [修改] 允许负贴目后改为 dict {贴目: 权重}；旧格式（list）丢弃，EMA 自动重建
                    if isinstance(sd, dict):
                        self.sampling_dist = {int(k): float(w) for k, w in sd.items()}
                else:
                    # 兼容旧格式：普通目差列表 -> 等权直方图
                    for d in obj.get("diffs", []):
                        self.hist[int(d)] = self.hist.get(int(d), 0.0) + 1.0
                    self.total = float(len(obj.get("diffs", [])))
            except Exception as e:
                print(f"警告: 读取贴目统计失败，从零开始: {e}")

    def save(self):
        try:
            obj = {"hist": self.hist, "total": self.total, "games_seen": self.games_seen}
            if self.sampling_dist is not None:
                obj["sampling_dist"] = self.sampling_dist
            with open(self.stats_path, 'w', encoding='utf-8') as f:
                json.dump(obj, f)
        except Exception as e:
            print(f"警告: 保存贴目统计失败: {e}")

    # ---------- 历史棋谱重建 ----------
    def rebuild_from_records(self, records_dir="self_play_data"):
        """[新增] 从历史自对弈棋谱重建贴目统计。
        工作流：
          1. 每份棋谱的"特征值"（每局终局目差，有序）解压提取一次后缓存到
             数据目录下的 komi_index.json（按 mtime 增量更新，之后不再解压）
          2. 按文件时间从旧到新逐局回放指数衰减（与在线 record_game 同一衰减律），
             只回放最近 MAX_REPLAY_GAMES 局——更旧的局在数值上已无影响，
             因此统计与在线记录完全同构，只是数据源更完整、可追溯
        返回回放的总局数。"""
        if not os.path.isdir(records_dir):
            return 0
        index_path = os.path.join(records_dir, RECORDS_INDEX)
        index = {}
        if os.path.exists(index_path):
            try:
                with open(index_path, 'r', encoding='utf-8') as f:
                    index = json.load(f)
            except Exception:
                index = {}

        files = sorted(glob.glob(os.path.join(records_dir, "*.pkl.gz")), key=_file_epoch)
        valid_names, changed = set(), False
        for fp in files:
            name = os.path.basename(fp)
            valid_names.add(name)
            mtime = os.path.getmtime(fp)
            ent = index.get(name)
            if ent and ent.get("mtime") == mtime:
                continue  # 索引缓存有效
            try:
                diffs = _extract_game_diffs(fp)
            except Exception as e:
                print(f"警告: 提取棋谱特征值失败 {name}: {e}")
                continue
            if diffs is None:
                continue  # 旧格式棋谱（无贴目字段），跳过
            index[name] = {"mtime": mtime, "diffs": diffs}
            changed = True

        # 清理已删除文件的索引项，保存索引
        if set(index.keys()) - valid_names:
            index = {n: v for n, v in index.items() if n in valid_names}
            changed = True
        if changed:
            try:
                with open(index_path, 'w', encoding='utf-8') as f:
                    json.dump(index, f)
            except Exception as e:
                print(f"警告: 保存棋谱索引失败: {e}")

        # 按时间序拼接所有局的目差，回放最近 MAX_REPLAY_GAMES 局
        all_diffs = []
        for name in sorted(index.keys(), key=lambda n: _file_epoch(os.path.join(records_dir, n))):
            all_diffs.extend(index[name]["diffs"])
        tail = all_diffs[-MAX_REPLAY_GAMES:]
        self.hist, self.total, self.games_seen = {}, 0.0, 0
        for d in tail:
            self.record_game(d)  # games_seen 随回放同步增长，窗口扩张与在线记录同构
        self.save()
        return len(tail)

    # ---------- 统计与估计 ----------
    def _decay_now(self):
        """[新增] 当前衰减率：有效窗口随累计局数增长（引擎变强→滑窗变大）。
        N_eff = min(N_EFF_MAX, N_EFF_MIN + games_seen / N_EFF_GROWTH)。"""
        n_eff = min(N_EFF_MAX, N_EFF_MIN + self.games_seen / N_EFF_GROWTH)
        return 1.0 - 1.0 / n_eff

    def effective_window(self):
        """[新增] 当前有效窗口（稳态有效局数），仅供展示"""
        return 1.0 / (1.0 - self._decay_now())

    def record_game(self, score_diff):
        """一局结束后记录终局地板差（黑视角，黑-白）。
        全部历史权重先按当前动态衰减率衰减，再记入新样本——逐局渐变，无窗口跳变。
        窗口随累计局数增长：初期短记忆（快速适应策略漂移），
        后期长记忆（精确定位陡峭的 p̂(k) 曲线上的 k*）。"""
        d = int(score_diff)
        decay = self._decay_now()
        if self.hist:
            for k in self.hist:
                self.hist[k] *= decay
            self.total *= decay
            # 定期清理可忽略项，防止直方图无限膨胀
            if len(self.hist) > 128:
                for k in [k for k, w in self.hist.items() if w < PRUNE_WEIGHT]:
                    del self.hist[k]
        self.hist[d] = self.hist.get(d, 0.0) + 1.0
        self.total += 1.0
        self.games_seen += 1

    def _weighted_win(self, komi):
        """加权"黑胜"期望（平局记 0.5）"""
        s = 0.0
        for d, w in self.hist.items():
            adj = d + komi
            s += w if adj > 0 else (0.5 * w if adj == 0 else 0.0)
        return s

    def win_rate(self, komi):
        """若固定贴 komi 目，黑方（贴目调整后）胜率；平局记 0.5"""
        if self.total <= 0.0:
            return None
        return self._weighted_win(komi) / self.total

    def weighted_median(self):
        """目差分布的加权中位数（仅展示用）"""
        if not self.hist:
            return 0
        acc, half = 0.0, self.total / 2.0
        for d in sorted(self.hist):
            acc += self.hist[d]
            if acc >= half:
                return d
        return max(self.hist)

    def current_k_star(self):
        """最接近 50% 胜率的最小贴目（仅用于展示/GUI 默认值）"""
        if self.total <= 0.0:
            return 0
        best_k, best_dist = MIN_KOMI, float('inf')
        for k in range(MIN_KOMI, MAX_KOMI + 1):
            dist = abs(self.win_rate(k) - 0.5)
            if dist < best_dist - 1e-12:  # 严格更优，保留最小 k
                best_dist, best_k = dist, k
        return best_k

    # ---------- 采样 ----------
    def _target_weights(self):
        """各贴目的核加权：峰值在胜率 50% 处，形状由目差分布决定。
        返回 dict {贴目: 权重}，覆盖 [MIN_KOMI, MAX_KOMI]。"""
        ws = {}
        for k in range(MIN_KOMI, MAX_KOMI + 1):
            p = self.win_rate(k)
            w = math.exp(-((p - 0.5) ** 2) / (2.0 * SIGMA * SIGMA)) if p is not None else 0.0
            ws[k] = max(w, FLOOR)
        return ws

    def sample_komi(self):
        if self.total < COLD_START_GAMES:
            return random.randint(MIN_KOMI, COLD_START_MAX)  # 冷启动：均匀采样积累统计（含负贴目）
        target = self._target_weights()
        # 采样分布 EMA：每局只向目标分布靠拢 SMOOTH_A，贴目组合批间平滑过渡
        # [修改] 改为 dict {k: w}，天然支持负下标；旧 list 分布加载时已被置 None
        if self.sampling_dist is None:
            self.sampling_dist = dict(target)
        else:
            self.sampling_dist = {k: SMOOTH_A * t + (1.0 - SMOOTH_A) * self.sampling_dist.get(k, 0.0)
                                  for k, t in target.items()}
        total = sum(self.sampling_dist.values())
        r = random.random() * total
        acc = 0.0
        for k in sorted(self.sampling_dist):
            acc += self.sampling_dist[k]
            if r <= acc:
                return k
        return MAX_KOMI

    # ---------- 展示 ----------
    def status_str(self):
        n = int(round(self.total))
        if self.total < COLD_START_GAMES:
            return f"贴目统计: ~{n} 局(指数加权) | 冷启动均匀采样 [{MIN_KOMI}, {COLD_START_MAX}]"
        k_star = self.current_k_star()
        sd = self.sampling_dist or self._target_weights()
        total = sum(sd.values())
        top = sorted(sd.keys(), key=lambda k: -sd[k])[:5]
        parts = [f"{k}:{sd[k] / total * 100:.0f}%" for k in top]
        med = self.weighted_median()
        win = self.effective_window()
        return (f"贴目统计: ~{n} 局(指数加权) | 有效窗口 {win:.0f} 局(动态,累计 {self.games_seen}) | 目差中位数 {med} | "
                f"k*={k_star} (黑胜率 {self.win_rate(k_star) * 100:.1f}%) | 采样权重 Top5: {', '.join(parts)}")


if __name__ == "__main__":
    from collections import Counter

    s = AdaptiveKomiSampler()
    print(s.status_str())

    # 场景 A：目差集中（后手优势稳定在 ~12 目）→ 采样应自动收缩
    s_a = AdaptiveKomiSampler.__new__(AdaptiveKomiSampler)
    s_a.hist, s_a.total, s_a.sampling_dist = {}, 0.0, None
    for d in [-12, -11, -12, -13, -12, -12, -11, -12, -13, -12, -12, -11]:
        s_a.record_game(d)
    print("集中:", s_a.status_str())
    print("采样分布:", dict(sorted(Counter(s_a.sample_komi() for _ in range(2000)).items())))

    # 场景 B：目差分散（-20 ~ -5）→ 采样应自动铺开
    s_b = AdaptiveKomiSampler.__new__(AdaptiveKomiSampler)
    s_b.hist, s_b.total, s_b.sampling_dist = {}, 0.0, None
    for d in [-20, -8, -15, -5, -12, -18, -9, -13, -6, -16, -11, -7]:
        s_b.record_game(d)
    print("分散:", s_b.status_str())
    print("采样分布:", dict(sorted(Counter(s_b.sample_komi() for _ in range(2000)).items())))

    # 场景 C：模拟策略漂移——前 300 局目差 ~+4，之后模型突变目差 ~-4。
    # 演示 k* 随新数据渐进过渡（无窗口换血式跳变），以及批间(每 100 局)的平滑程度。
    random.seed(7)
    s_c = AdaptiveKomiSampler.__new__(AdaptiveKomiSampler)
    s_c.hist, s_c.total, s_c.sampling_dist = {}, 0.0, None
    for i in range(700):
        if i < 300:
            d = random.randint(2, 7)
        else:
            d = random.randint(-7, -2)
        s_c.record_game(d)
        if (i + 1) % 100 == 0:
            print(f"第 {i + 1} 局后 | 目差中位数 {s_c.weighted_median():+d} | "
                  f"k*={s_c.current_k_star()} | 黑胜率(k*) {s_c.win_rate(s_c.current_k_star()) * 100:.1f}%")
            # 每 100 局采样一次贴目（模拟每局开赛采样）
            _ = s_c.sample_komi()
