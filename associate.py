import os
import csv
import math
from dataclasses import dataclass
from collections import defaultdict, deque

import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    raise ImportError("请先安装 scipy: pip install scipy")


# =========================
# 1. 数据结构
# =========================
@dataclass
class Detection:
    frame_id: int
    track_id: int
    cls_id: int
    x1: float
    y1: float
    x2: float
    y2: float
    cx: float
    cy_bottom: float
    conf: float

    @property
    def pos(self):
        return np.array([self.cx, self.cy_bottom], dtype=np.float32)

    @property
    def w(self):
        return float(self.x2 - self.x1)

    @property
    def h(self):
        return float(self.y2 - self.y1)


# =========================
# 2. 读取轨迹 txt
# 支持两种格式：
#   custom: frame,track_id,cls,x1,y1,x2,y2,cx,cy_bottom,conf
#   mot:    frame,id,x,y,w,h,score,class,-1
# =========================
def load_track_txt(txt_path, txt_format="custom", cls_filter=None):
    tracks_by_frame = defaultdict(list)

    with open(txt_path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            parts = [x.strip() for x in line.split(",")]

            try:
                if txt_format == "custom":
                    # frame,track_id,cls,x1,y1,x2,y2,cx,cy_bottom,conf
                    if len(parts) < 10:
                        raise ValueError(f"第 {ln} 行列数不足，custom 格式至少需要 10 列")
                    frame_id = int(float(parts[0]))
                    track_id = int(float(parts[1]))
                    cls_id = int(float(parts[2]))
                    x1 = float(parts[3])
                    y1 = float(parts[4])
                    x2 = float(parts[5])
                    y2 = float(parts[6])
                    cx = float(parts[7])
                    cy_bottom = float(parts[8])
                    conf = float(parts[9])

                elif txt_format == "mot":
                    # frame,id,x,y,w,h,score,class,-1
                    if len(parts) < 8:
                        raise ValueError(f"第 {ln} 行列数不足，mot 格式至少需要 8 列")
                    frame_id = int(float(parts[0]))
                    track_id = int(float(parts[1]))
                    x1 = float(parts[2])
                    y1 = float(parts[3])
                    w = float(parts[4])
                    h = float(parts[5])
                    conf = float(parts[6])
                    cls_id = int(float(parts[7]))
                    x2 = x1 + w
                    y2 = y1 + h
                    cx = (x1 + x2) / 2.0
                    cy_bottom = y2

                else:
                    raise ValueError("txt_format 只能是 'custom' 或 'mot'")

                if cls_filter is not None and cls_id != cls_filter:
                    continue

                det = Detection(
                    frame_id=frame_id,
                    track_id=track_id,
                    cls_id=cls_id,
                    x1=x1, y1=y1, x2=x2, y2=y2,
                    cx=cx, cy_bottom=cy_bottom,
                    conf=conf
                )
                tracks_by_frame[frame_id].append(det)

            except Exception as e:
                raise ValueError(f"解析 {txt_path} 第 {ln} 行失败：{line}\n错误信息：{e}")

    return dict(sorted(tracks_by_frame.items(), key=lambda x: x[0]))


# =========================
# 3. 工具函数
# =========================
def angle_between(v1, v2):
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    c = float(np.dot(v1, v2) / (n1 * n2 + 1e-12))
    c = np.clip(c, -1.0, 1.0)
    return float(math.acos(c))


def resample_1d(seq, out_len=8):
    seq = np.asarray(seq, dtype=np.float32)
    if len(seq) == 0:
        return np.zeros(out_len, dtype=np.float32)
    if len(seq) == 1:
        return np.full(out_len, float(seq[0]), dtype=np.float32)

    x_old = np.linspace(0, 1, len(seq))
    x_new = np.linspace(0, 1, out_len)
    return np.interp(x_new, x_old, seq).astype(np.float32)


def normalize_seq(seq, out_len=8, clip_val=3.0):
    seq = resample_1d(seq, out_len)
    scale = np.mean(np.abs(seq)) + 1e-6
    seq = seq / scale
    seq = np.clip(seq, -clip_val, clip_val)
    return seq.astype(np.float32)


def seq_similarity(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    mae = float(np.mean(np.abs(a - b)))
    return math.exp(-mae)


def current_speed_from_history(hist):
    if len(hist) < 2:
        return 0.0
    pts = np.array([[d.cx, d.cy_bottom] for d in hist], dtype=np.float32)
    diffs = np.diff(pts, axis=0)
    sp = np.linalg.norm(diffs, axis=1)
    if len(sp) == 0:
        return 0.0
    return float(np.mean(sp[-min(3, len(sp)):]))

# =========================
# 4. 轨迹时间特征
# =========================
def build_traj_feature(hist, seq_len=8):
    """
    hist: deque[Detection]
    只做视角相对稳的特征，不直接用绝对方向角
    """
    if len(hist) < 2:
        return {
            "speed_seq": np.zeros(seq_len, dtype=np.float32),
            "acc_seq": np.zeros(seq_len, dtype=np.float32),
            "turn_seq": np.zeros(seq_len, dtype=np.float32),
            "stop_ratio": 1.0,
            "life_ratio": min(len(hist) / 20.0, 1.0),
        }

    pts = np.array([[d.cx, d.cy_bottom] for d in hist], dtype=np.float32)
    motion = np.diff(pts, axis=0)  # [T-1, 2]
    speed = np.linalg.norm(motion, axis=1)

    if len(speed) >= 2:
        acc = np.diff(speed)
    else:
        acc = np.array([], dtype=np.float32)

    turns = []
    if len(motion) >= 2:
        for k in range(len(motion) - 1):
            ang = angle_between(motion[k], motion[k + 1]) / math.pi  # 归一化到 [0,1]
            turns.append(ang)
    turns = np.asarray(turns, dtype=np.float32)

    # 停走比例：阈值取轨迹自身速度统计的相对阈值
    pos_speed = speed[speed > 1e-6]
    if len(pos_speed) > 0:
        stop_th = max(1.5, 0.35 * float(np.median(pos_speed)))
        stop_ratio = float(np.mean(speed < stop_th))
    else:
        stop_ratio = 1.0

    feat = {
        "speed_seq": normalize_seq(speed, out_len=seq_len),
        "acc_seq": normalize_seq(acc, out_len=seq_len),
        "turn_seq": resample_1d(turns, out_len=seq_len),   # 已经在 [0,1]
        "stop_ratio": stop_ratio,
        "life_ratio": min(len(hist) / 20.0, 1.0),
    }
    return feat


def traj_similarity(fa, fb):
    s1 = seq_similarity(fa["speed_seq"], fb["speed_seq"])
    s2 = seq_similarity(fa["acc_seq"], fb["acc_seq"])
    s3 = seq_similarity(fa["turn_seq"], fb["turn_seq"])
    s4 = math.exp(-abs(fa["stop_ratio"] - fb["stop_ratio"]) * 3.0)
    s5 = math.exp(-abs(fa["life_ratio"] - fb["life_ratio"]) * 2.0)

    return 0.30 * s1 + 0.25 * s2 + 0.20 * s3 + 0.15 * s4 + 0.10 * s5


# =========================
# 5. 当前帧位置关系特征
# =========================
def estimate_local_axes(current_dets, histories):
    """
    用当前活跃轨迹的最近运动估计局部主方向 u 和横向方向 w
    只在每个视角内部估计，不做跨视角单应/标定
    """
    vecs = []
    for det in current_dets:
        hist = histories[det.track_id]
        if len(hist) >= 2:
            pts = np.array([[d.cx, d.cy_bottom] for d in hist], dtype=np.float32)
            diffs = np.diff(pts, axis=0)
            v = np.mean(diffs[-min(3, len(diffs)):], axis=0)
            if np.linalg.norm(v) > 1e-6:
                vecs.append(v)

    if len(vecs) >= 2:
        X = np.array(vecs, dtype=np.float32)
        X = X - np.mean(X, axis=0, keepdims=True)
        _, _, vh = np.linalg.svd(X, full_matrices=False)
        u = vh[0]
    elif len(current_dets) >= 2:
        pts = np.array([[d.cx, d.cy_bottom] for d in current_dets], dtype=np.float32)
        pts = pts - np.mean(pts, axis=0, keepdims=True)
        _, _, vh = np.linalg.svd(pts, full_matrices=False)
        u = vh[0]
    else:
        u = np.array([1.0, 0.0], dtype=np.float32)

    n = np.linalg.norm(u)
    if n < 1e-6:
        u = np.array([1.0, 0.0], dtype=np.float32)
    else:
        u = u / n

    w = np.array([-u[1], u[0]], dtype=np.float32)
    return u, w


def build_relation_features(current_dets, histories, k_neighbors=3):
    """
    对当前帧每条活跃轨迹构造“位置关系指纹”
    """
    if len(current_dets) == 0:
        return {}

    u, w = estimate_local_axes(current_dets, histories)

    ids = [d.track_id for d in current_dets]
    pts = np.array([[d.cx, d.cy_bottom] for d in current_dets], dtype=np.float32)
    speeds = np.array([current_speed_from_history(histories[d.track_id]) for d in current_dets], dtype=np.float32)

    s_coord = pts @ u
    l_coord = pts @ w

    n = len(current_dets)

    # 纵向/横向排序 rank
    order_s = np.argsort(s_coord)
    rank_s = np.empty(n, dtype=np.int32)
    rank_s[order_s] = np.arange(n)

    order_l = np.argsort(l_coord)
    rank_l = np.empty(n, dtype=np.int32)
    rank_l[order_l] = np.arange(n)

    feats = {}

    for i, det in enumerate(current_dets):
        if n == 1:
            feats[det.track_id] = {
                "srank": 0.5,
                "lrank": 0.5,
                "front": 0.5,
                "back": 0.5,
                "left": 0.5,
                "right": 0.5,
                "dist_vec": np.ones(k_neighbors, dtype=np.float32),
                "rel_speed": 0.0,
            }
            continue

        diff = pts - pts[i]
        dist = np.linalg.norm(diff, axis=1)

        other_idx = np.where(np.arange(n) != i)[0]
        sort_idx = other_idx[np.argsort(dist[other_idx])]
        nn_idx = sort_idx[:k_neighbors]

        ds = dist[nn_idx]
        s_delta = s_coord[nn_idx] - s_coord[i]
        l_delta = l_coord[nn_idx] - l_coord[i]

        # 前后左右比例
        front = float(np.mean(s_delta > 0)) if len(nn_idx) > 0 else 0.5
        back = float(np.mean(s_delta < 0)) if len(nn_idx) > 0 else 0.5
        left = float(np.mean(l_delta < 0)) if len(nn_idx) > 0 else 0.5
        right = float(np.mean(l_delta > 0)) if len(nn_idx) > 0 else 0.5

        # 距离排序向量：只比较相对模式，不比较绝对像素值
        if len(ds) > 0:
            dist_vec = ds / (np.mean(ds) + 1e-6)
        else:
            dist_vec = np.ones(0, dtype=np.float32)

        if len(dist_vec) < k_neighbors:
            dist_vec = np.concatenate([
                dist_vec.astype(np.float32),
                np.full(k_neighbors - len(dist_vec), 2.0, dtype=np.float32)
            ], axis=0)

        # 相对速度差
        if len(nn_idx) > 0:
            rel_speed = float(np.mean(np.abs(speeds[nn_idx] - speeds[i]) / (np.mean(speeds[nn_idx]) + 1e-6)))
            rel_speed = min(rel_speed, 3.0)
        else:
            rel_speed = 0.0

        feats[det.track_id] = {
            "srank": float(rank_s[i] / max(n - 1, 1)),
            "lrank": float(rank_l[i] / max(n - 1, 1)),
            "front": front,
            "back": back,
            "left": left,
            "right": right,
            "dist_vec": dist_vec.astype(np.float32),
            "rel_speed": rel_speed,
        }

    return feats


def flip_relation_feature(f, flip_s=False, flip_l=False):
    g = {
        "srank": f["srank"],
        "lrank": f["lrank"],
        "front": f["front"],
        "back": f["back"],
        "left": f["left"],
        "right": f["right"],
        "dist_vec": f["dist_vec"].copy(),
        "rel_speed": f["rel_speed"],
    }

    if flip_s:
        g["srank"] = 1.0 - g["srank"]
        g["front"], g["back"] = g["back"], g["front"]

    if flip_l:
        g["lrank"] = 1.0 - g["lrank"]
        g["left"], g["right"] = g["right"], g["left"]

    return g


def relation_similarity(fa, fb):
    """
    两视角的 PCA 主轴方向可能相反，所以对 fb 做 4 种翻转组合，取最大相似度
    """
    best = 0.0

    for flip_s in [False, True]:
        for flip_l in [False, True]:
            g = flip_relation_feature(fb, flip_s=flip_s, flip_l=flip_l)

            d_rank = abs(fa["srank"] - g["srank"]) + abs(fa["lrank"] - g["lrank"])
            d_side = (
                abs(fa["front"] - g["front"]) +
                abs(fa["back"] - g["back"]) +
                abs(fa["left"] - g["left"]) +
                abs(fa["right"] - g["right"])
            )
            d_speed = abs(fa["rel_speed"] - g["rel_speed"])
            d_dist = float(np.mean(np.abs(fa["dist_vec"] - g["dist_vec"])))

            sim_struct = math.exp(-(0.35 * d_rank + 0.45 * d_side + 0.20 * d_speed))
            sim_dist = math.exp(-d_dist)

            sim = 0.60 * sim_struct + 0.40 * sim_dist
            best = max(best, sim)

    return best


# =========================
# 6. 匹配器
# =========================
class CrossViewMatcher:
    def __init__(
        self,
        window_size=20,
        seq_len=8,
        k_neighbors=3,
        alpha=0.35,     # 轨迹时间特征权重
        beta=0.55,      # 位置关系特征权重
        gamma=0.10,     # 历史稳定项权重
        min_score=0.45,
        pair_ttl=5
    ):
        self.window_size = window_size
        self.seq_len = seq_len
        self.k_neighbors = k_neighbors
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.min_score = min_score
        self.pair_ttl = pair_ttl

        self.hist_A = defaultdict(lambda: deque(maxlen=self.window_size))
        self.hist_B = defaultdict(lambda: deque(maxlen=self.window_size))
        self.pair_memory = {}   # (idA, idB) -> {"last_frame": int, "hits": int}

    def temporal_bonus(self, id_a, id_b, frame_id):
        item = self.pair_memory.get((id_a, id_b), None)
        if item is None:
            return 0.0

        age = frame_id - item["last_frame"]
        if age == 1:
            return min(1.0, 0.35 + 0.08 * item["hits"])
        elif age <= 3:
            return 0.35
        elif age <= self.pair_ttl:
            return 0.15
        else:
            return 0.0

    def prune_pair_memory(self, frame_id):
        new_mem = {}
        for k, v in self.pair_memory.items():
            if frame_id - v["last_frame"] <= self.pair_ttl:
                new_mem[k] = v
        self.pair_memory = new_mem

    def update_histories(self, dets_a, dets_b):
        for det in dets_a:
            self.hist_A[det.track_id].append(det)
        for det in dets_b:
            self.hist_B[det.track_id].append(det)

    def match_one_frame(self, frame_id, dets_a, dets_b):
        self.update_histories(dets_a, dets_b)
        self.prune_pair_memory(frame_id)

        if len(dets_a) == 0 or len(dets_b) == 0:
            return []

        # 当前帧特征
        traj_feat_a = {d.track_id: build_traj_feature(self.hist_A[d.track_id], seq_len=self.seq_len) for d in dets_a}
        traj_feat_b = {d.track_id: build_traj_feature(self.hist_B[d.track_id], seq_len=self.seq_len) for d in dets_b}

        rel_feat_a = build_relation_features(dets_a, self.hist_A, k_neighbors=self.k_neighbors)
        rel_feat_b = build_relation_features(dets_b, self.hist_B, k_neighbors=self.k_neighbors)

        na = len(dets_a)
        nb = len(dets_b)

        score_total = np.zeros((na, nb), dtype=np.float32)
        score_traj = np.zeros((na, nb), dtype=np.float32)
        score_rel = np.zeros((na, nb), dtype=np.float32)
        score_temp = np.zeros((na, nb), dtype=np.float32)

        # 计算相似度矩阵
        for i, da in enumerate(dets_a):
            for j, db in enumerate(dets_b):
                st = traj_similarity(traj_feat_a[da.track_id], traj_feat_b[db.track_id])
                sr = relation_similarity(rel_feat_a[da.track_id], rel_feat_b[db.track_id])
                sm = self.temporal_bonus(da.track_id, db.track_id, frame_id)

                s = self.alpha * st + self.beta * sr + self.gamma * sm

                score_traj[i, j] = st
                score_rel[i, j] = sr
                score_temp[i, j] = sm
                score_total[i, j] = s

        # Hungarian
        cost = 1.0 - score_total
        row_ind, col_ind = linear_sum_assignment(cost)

        matches = []
        for r, c in zip(row_ind, col_ind):
            s = float(score_total[r, c])
            if s < self.min_score:
                continue

            id_a = dets_a[r].track_id
            id_b = dets_b[c].track_id

            old = self.pair_memory.get((id_a, id_b), {"hits": 0})
            self.pair_memory[(id_a, id_b)] = {
                "last_frame": frame_id,
                "hits": min(old["hits"] + 1, 20)
            }

            matches.append({
                "frame_id": frame_id,
                "track_id_a": id_a,
                "track_id_b": id_b,
                "score_total": round(float(score_total[r, c]), 6),
                "score_traj": round(float(score_traj[r, c]), 6),
                "score_rel": round(float(score_rel[r, c]), 6),
                "score_temp": round(float(score_temp[r, c]), 6),
            })

        return matches

    def run(self, tracks_a_by_frame, tracks_b_by_frame):
        all_frames = sorted(set(tracks_a_by_frame.keys()) | set(tracks_b_by_frame.keys()))
        results = []

        for frame_id in all_frames:
            dets_a = tracks_a_by_frame.get(frame_id, [])
            dets_b = tracks_b_by_frame.get(frame_id, [])

            frame_matches = self.match_one_frame(frame_id, dets_a, dets_b)
            results.extend(frame_matches)

            if frame_id % 100 == 0:
                print(f"[Info] 已处理到 frame {frame_id}, 当前累计匹配 {len(results)} 条")

        return results


# =========================
# 7. 保存结果
# =========================
def save_matches_csv(matches, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True) if os.path.dirname(save_path) else None

    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "frame_id",
            "track_id_a",
            "track_id_b",
            "score_total",
            "score_traj",
            "score_rel",
            "score_temp"
        ])
        for m in matches:
            writer.writerow([
                m["frame_id"],
                m["track_id_a"],
                m["track_id_b"],
                m["score_total"],
                m["score_traj"],
                m["score_rel"],
                m["score_temp"],
            ])

    print(f"匹配结果已保存到: {save_path}")


# =========================
# 8. 主程序
# =========================
if __name__ == "__main__":
    # -------- 你的输入 --------
    txt_a = "experient_fig/doublesight/droneA_tracks.txt"
    txt_b = "experient_fig/doublesight/droneB_tracks.txt"

    # 如果你之前保存的是：
    # frame,track_id,cls,x1,y1,x2,y2,cx,cy_bottom,conf
    txt_format = "custom"

    # 如果你的 txt 里已经只保留了车辆，这里可以设 None
    # 如果还是混合类别，但你只想匹配 cls=3，就设成 3
    cls_filter = None   # 或 3

    # -------- 加载轨迹 --------
    tracks_a = load_track_txt(txt_a, txt_format=txt_format, cls_filter=cls_filter)
    tracks_b = load_track_txt(txt_b, txt_format=txt_format, cls_filter=cls_filter)

    print(f"A 视角共 {len(tracks_a)} 帧")
    print(f"B 视角共 {len(tracks_b)} 帧")

    # -------- 建立匹配器 --------
    matcher = CrossViewMatcher(
        window_size=20,   # 每条轨迹保留最近 20 帧
        seq_len=8,        # 时间序列重采样长度
        k_neighbors=8,    # 关系特征取 3 个最近邻
        alpha=0.35,
        beta=0.55,
        gamma=0.10,
        min_score=0.7,
        pair_ttl=5
    )

    # -------- 执行匹配 --------
    matches = matcher.run(tracks_a, tracks_b)

    # -------- 保存结果 --------
    save_matches_csv(matches, "results/exp4/cross_match.csv")
