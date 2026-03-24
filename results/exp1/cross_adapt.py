import os
import csv
import cv2
import glob
import math
from dataclasses import dataclass
from collections import defaultdict, deque

import numpy as np
from scipy.optimize import linear_sum_assignment


# =========================================================
# 1. 数据结构
# =========================================================
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
    def box_h(self):
        return float(self.y2 - self.y1)

    @property
    def box_w(self):
        return float(self.x2 - self.x1)


# =========================================================
# 2. 工具
# =========================================================
def natural_sort_key(path):
    name = os.path.basename(path)
    stem = os.path.splitext(name)[0]
    try:
        return int(stem)
    except ValueError:
        return stem


def get_first_image_size(frames_dir):
    exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(frames_dir, ext)))
    files = sorted(files, key=natural_sort_key)

    if not files:
        raise FileNotFoundError(f"在 {frames_dir} 下没有找到图片")

    img = cv2.imread(files[0])
    if img is None:
        raise RuntimeError(f"无法读取图片: {files[0]}")
    h, w = img.shape[:2]
    return w, h


def make_dir(path):
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


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


def normalize_seq(seq, out_len=8):
    seq = resample_1d(seq, out_len=out_len)
    scale = np.mean(np.abs(seq)) + 1e-6
    seq = seq / scale
    seq = np.clip(seq, -3.0, 3.0)
    return seq.astype(np.float32)


def seq_similarity(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    mae = float(np.mean(np.abs(a - b)))
    return math.exp(-mae)


# =========================================================
# 3. 读取轨迹 txt
# custom: frame,track_id,cls,x1,y1,x2,y2,cx,cy_bottom,conf
# mot:    frame,id,x,y,w,h,score,class,-1
# =========================================================
def load_track_txt(txt_path, txt_format="custom", cls_filter=None):
    tracks_by_frame = defaultdict(list)

    with open(txt_path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            parts = [x.strip() for x in line.split(",")]

            if txt_format == "custom":
                if len(parts) < 10:
                    raise ValueError(f"{txt_path} 第 {ln} 行不足 10 列")
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
                if len(parts) < 8:
                    raise ValueError(f"{txt_path} 第 {ln} 行不足 8 列")
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

            tracks_by_frame[frame_id].append(
                Detection(
                    frame_id=frame_id,
                    track_id=track_id,
                    cls_id=cls_id,
                    x1=x1, y1=y1, x2=x2, y2=y2,
                    cx=cx, cy_bottom=cy_bottom,
                    conf=conf
                )
            )

    return dict(sorted(tracks_by_frame.items(), key=lambda x: x[0]))


# =========================================================
# 4. 轨迹时间特征
# =========================================================
def build_traj_feature(hist, seq_len=8):
    if len(hist) < 2:
        return {
            "speed_seq": np.zeros(seq_len, dtype=np.float32),
            "acc_seq": np.zeros(seq_len, dtype=np.float32),
            "turn_seq": np.zeros(seq_len, dtype=np.float32),
            "stop_ratio": 1.0,
            "life_ratio": min(len(hist) / 20.0, 1.0),
        }

    pts = np.array([[d.cx, d.cy_bottom] for d in hist], dtype=np.float32)
    motion = np.diff(pts, axis=0)
    speed = np.linalg.norm(motion, axis=1)

    if len(speed) >= 2:
        acc = np.diff(speed)
    else:
        acc = np.array([], dtype=np.float32)

    turns = []
    if len(motion) >= 2:
        for i in range(len(motion) - 1):
            ang = angle_between(motion[i], motion[i + 1]) / math.pi
            turns.append(ang)
    turns = np.asarray(turns, dtype=np.float32)

    pos_speed = speed[speed > 1e-6]
    if len(pos_speed) > 0:
        stop_th = max(1.5, 0.35 * float(np.median(pos_speed)))
        stop_ratio = float(np.mean(speed < stop_th))
    else:
        stop_ratio = 1.0

    return {
        "speed_seq": normalize_seq(speed, out_len=seq_len),
        "acc_seq": normalize_seq(acc, out_len=seq_len),
        "turn_seq": resample_1d(turns, out_len=seq_len),
        "stop_ratio": stop_ratio,
        "life_ratio": min(len(hist) / 20.0, 1.0),
    }


def traj_similarity(fa, fb):
    s1 = seq_similarity(fa["speed_seq"], fb["speed_seq"])
    s2 = seq_similarity(fa["acc_seq"], fb["acc_seq"])
    s3 = seq_similarity(fa["turn_seq"], fb["turn_seq"])
    s4 = math.exp(-abs(fa["stop_ratio"] - fb["stop_ratio"]) * 3.0)
    s5 = math.exp(-abs(fa["life_ratio"] - fb["life_ratio"]) * 2.0)
    return 0.30 * s1 + 0.25 * s2 + 0.20 * s3 + 0.15 * s4 + 0.10 * s5


# =========================================================
# 5. 关系特征
# 关键变化：
# - anchor_dets: 真正参与匹配的下半部分目标
# - context_dets: 用于计算关系特征的扩展上下文目标
# =========================================================
def current_speed_from_history(hist):
    if len(hist) < 2:
        return 0.0
    pts = np.array([[d.cx, d.cy_bottom] for d in hist], dtype=np.float32)
    diffs = np.diff(pts, axis=0)
    speed = np.linalg.norm(diffs, axis=1)
    if len(speed) == 0:
        return 0.0
    return float(np.mean(speed[-min(3, len(speed)):]))


def estimate_local_axes(context_dets, histories):
    vecs = []
    for det in context_dets:
        hist = histories[det.track_id]
        if len(hist) >= 2:
            pts = np.array([[d.cx, d.cy_bottom] for d in hist], dtype=np.float32)
            diffs = np.diff(pts, axis=0)
            if len(diffs) > 0:
                v = np.mean(diffs[-min(3, len(diffs)):], axis=0)
                if np.linalg.norm(v) > 1e-6:
                    vecs.append(v)

    if len(vecs) >= 2:
        X = np.array(vecs, dtype=np.float32)
        X = X - np.mean(X, axis=0, keepdims=True)
        _, _, vh = np.linalg.svd(X, full_matrices=False)
        u = vh[0]
    elif len(context_dets) >= 2:
        pts = np.array([[d.cx, d.cy_bottom] for d in context_dets], dtype=np.float32)
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


def build_relation_features(anchor_dets, context_dets, histories, k_neighbors=3):
    if len(anchor_dets) == 0 or len(context_dets) == 0:
        return {}

    u, w = estimate_local_axes(context_dets, histories)

    ctx_pts = np.array([[d.cx, d.cy_bottom] for d in context_dets], dtype=np.float32)
    ctx_ids = [d.track_id for d in context_dets]
    ctx_speeds = np.array(
        [current_speed_from_history(histories[d.track_id]) for d in context_dets],
        dtype=np.float32
    )

    s_coord = ctx_pts @ u
    l_coord = ctx_pts @ w

    n = len(context_dets)

    order_s = np.argsort(s_coord)
    rank_s = np.empty(n, dtype=np.int32)
    rank_s[order_s] = np.arange(n)

    order_l = np.argsort(l_coord)
    rank_l = np.empty(n, dtype=np.int32)
    rank_l[order_l] = np.arange(n)

    idx_map = {tid: i for i, tid in enumerate(ctx_ids)}
    feats = {}

    for det in anchor_dets:
        if det.track_id not in idx_map:
            continue

        i = idx_map[det.track_id]
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

        diff = ctx_pts - ctx_pts[i]
        dist = np.linalg.norm(diff, axis=1)

        other_idx = np.array([j for j in range(n) if j != i], dtype=np.int32)
        if len(other_idx) > 0:
            nn_idx = other_idx[np.argsort(dist[other_idx])[:k_neighbors]]
        else:
            nn_idx = np.array([], dtype=np.int32)

        ds = dist[nn_idx] if len(nn_idx) > 0 else np.array([], dtype=np.float32)
        s_delta = s_coord[nn_idx] - s_coord[i] if len(nn_idx) > 0 else np.array([], dtype=np.float32)
        l_delta = l_coord[nn_idx] - l_coord[i] if len(nn_idx) > 0 else np.array([], dtype=np.float32)

        front = float(np.mean(s_delta > 0)) if len(nn_idx) > 0 else 0.5
        back = float(np.mean(s_delta < 0)) if len(nn_idx) > 0 else 0.5
        left = float(np.mean(l_delta < 0)) if len(nn_idx) > 0 else 0.5
        right = float(np.mean(l_delta > 0)) if len(nn_idx) > 0 else 0.5

        if len(ds) > 0:
            dist_vec = ds / (np.mean(ds) + 1e-6)
        else:
            dist_vec = np.ones(0, dtype=np.float32)

        if len(dist_vec) < k_neighbors:
            dist_vec = np.concatenate([
                dist_vec.astype(np.float32),
                np.full(k_neighbors - len(dist_vec), 2.0, dtype=np.float32)
            ], axis=0)

        if len(nn_idx) > 0:
            rel_speed = float(np.mean(np.abs(ctx_speeds[nn_idx] - ctx_speeds[i]) / (np.mean(ctx_speeds[nn_idx]) + 1e-6)))
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
    best = 0.0
    best_rank_gap = 1.0
    best_lane_gap = 1.0

    for flip_s in [False, True]:
        for flip_l in [False, True]:
            g = flip_relation_feature(fb, flip_s=flip_s, flip_l=flip_l)

            d_rank = abs(fa["srank"] - g["srank"])
            d_lane = abs(fa["lrank"] - g["lrank"])
            d_side = (
                abs(fa["front"] - g["front"]) +
                abs(fa["back"] - g["back"]) +
                abs(fa["left"] - g["left"]) +
                abs(fa["right"] - g["right"])
            )
            d_speed = abs(fa["rel_speed"] - g["rel_speed"])
            d_dist = float(np.mean(np.abs(fa["dist_vec"] - g["dist_vec"])))

            sim_struct = math.exp(-(0.45 * d_rank + 0.20 * d_lane + 0.25 * d_side + 0.10 * d_speed))
            sim_dist = math.exp(-d_dist)
            sim = 0.68 * sim_struct + 0.32 * sim_dist

            if sim > best:
                best = sim
                best_rank_gap = d_rank
                best_lane_gap = d_lane

    return best, best_rank_gap, best_lane_gap


# =========================================================
# 6. 改进匹配器
# =========================================================
class CrossViewMatcherLowerAnchor:
    def __init__(
        self,
        img_w_a, img_h_a,
        img_w_b, img_h_b,
        window_size=20,
        seq_len=8,
        k_neighbors=3,

        min_conf=0.60,
        min_track_len=8,

        # 公共区域：A 右，B 左
        a_right_min=0.35,
        b_left_max=0.65,

        # -------- 锚点区域（真正参与匹配）--------
        # 下半部分，但不要贴底边
        anchor_y_min=0.50,
        anchor_bottom_exclude=0.08,   # 距离底边 8% 内不参与

        # -------- 上下文区域（只用于关系特征）--------
        # 比下半部分略扩一点，但不要放到太远端
        context_y_min=0.35,
        context_bottom_exclude=0.06,
        min_context_box_h_ratio=0.035,  # 过滤太远太小的目标

        # 分数权重：降低轨迹时间特征，提高关系分数
        alpha=0.25,   # traj
        beta=0.40,    # relation
        gamma=0.25,   # region prior
        delta=0.10,   # temporal

        min_accept_score=0.70,
        ambiguity_margin=0.08,
        confirm_frames=3,
        pair_ttl=5
    ):
        self.img_w_a = img_w_a
        self.img_h_a = img_h_a
        self.img_w_b = img_w_b
        self.img_h_b = img_h_b

        self.window_size = window_size
        self.seq_len = seq_len
        self.k_neighbors = k_neighbors

        self.min_conf = min_conf
        self.min_track_len = min_track_len

        self.a_right_min = a_right_min
        self.b_left_max = b_left_max

        self.anchor_y_min = anchor_y_min
        self.anchor_bottom_exclude = anchor_bottom_exclude

        self.context_y_min = context_y_min
        self.context_bottom_exclude = context_bottom_exclude
        self.min_context_box_h_ratio = min_context_box_h_ratio

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta

        self.min_accept_score = min_accept_score
        self.ambiguity_margin = ambiguity_margin
        self.confirm_frames = confirm_frames
        self.pair_ttl = pair_ttl

        self.hist_A = defaultdict(lambda: deque(maxlen=self.window_size))
        self.hist_B = defaultdict(lambda: deque(maxlen=self.window_size))

        self.pair_memory = {}

    # -------------------------
    # 基础历史更新
    # -------------------------
    def update_histories(self, dets_a, dets_b):
        for d in dets_a:
            self.hist_A[d.track_id].append(d)
        for d in dets_b:
            self.hist_B[d.track_id].append(d)

    def prune_pair_memory(self, frame_id):
        keep = {}
        for k, v in self.pair_memory.items():
            if frame_id - v["last_frame"] <= self.pair_ttl:
                keep[k] = v
        self.pair_memory = keep

    # -------------------------
    # 公共区域判断
    # -------------------------
    def in_public_region_a(self, det):
        return (det.cx / max(self.img_w_a, 1)) >= self.a_right_min

    def in_public_region_b(self, det):
        return (det.cx / max(self.img_w_b, 1)) <= self.b_left_max

    # -------------------------
    # 锚点区判断：下半部分 + 不贴底边
    # -------------------------
    def in_anchor_region_a(self, det):
        y_norm = det.cy_bottom / max(self.img_h_a, 1)
        return self.anchor_y_min <= y_norm <= (1.0 - self.anchor_bottom_exclude)

    def in_anchor_region_b(self, det):
        y_norm = det.cy_bottom / max(self.img_h_b, 1)
        return self.anchor_y_min <= y_norm <= (1.0 - self.anchor_bottom_exclude)

    # -------------------------
    # 上下文区判断：范围更大，但过滤远端小目标
    # -------------------------
    def in_context_region_a(self, det):
        y_norm = det.cy_bottom / max(self.img_h_a, 1)
        h_ratio = det.box_h / max(self.img_h_a, 1)
        return (
            self.context_y_min <= y_norm <= (1.0 - self.context_bottom_exclude)
            and h_ratio >= self.min_context_box_h_ratio
        )

    def in_context_region_b(self, det):
        y_norm = det.cy_bottom / max(self.img_h_b, 1)
        h_ratio = det.box_h / max(self.img_h_b, 1)
        return (
            self.context_y_min <= y_norm <= (1.0 - self.context_bottom_exclude)
            and h_ratio >= self.min_context_box_h_ratio
        )

    # -------------------------
    # 是否有资格作为锚点
    # -------------------------
    def eligible_anchor_a(self, det):
        return (
            det.conf >= self.min_conf
            and len(self.hist_A[det.track_id]) >= self.min_track_len
            and self.in_public_region_a(det)
            and self.in_anchor_region_a(det)
        )

    def eligible_anchor_b(self, det):
        return (
            det.conf >= self.min_conf
            and len(self.hist_B[det.track_id]) >= self.min_track_len
            and self.in_public_region_b(det)
            and self.in_anchor_region_b(det)
        )

    # -------------------------
    # 是否有资格作为上下文参考
    # -------------------------
    def eligible_context_a(self, det):
        return (
            det.conf >= self.min_conf
            and len(self.hist_A[det.track_id]) >= max(4, self.min_track_len // 2)
            and self.in_public_region_a(det)
            and self.in_context_region_a(det)
        )

    def eligible_context_b(self, det):
        return (
            det.conf >= self.min_conf
            and len(self.hist_B[det.track_id]) >= max(4, self.min_track_len // 2)
            and self.in_public_region_b(det)
            and self.in_context_region_b(det)
        )

    # -------------------------
    # 区域先验分数
    # A 越靠右越好，B 越靠左越好
    # 下半部分中段偏下更优，但过近底边已被过滤
    # -------------------------
    def region_prior_score(self, det_a, det_b):
        xa = det_a.cx / max(self.img_w_a, 1)
        xb = det_b.cx / max(self.img_w_b, 1)
        ya = det_a.cy_bottom / max(self.img_h_a, 1)
        yb = det_b.cy_bottom / max(self.img_h_b, 1)

        sa = np.clip((xa - self.a_right_min) / max(1.0 - self.a_right_min, 1e-6), 0.0, 1.0)
        sb = np.clip((self.b_left_max - xb) / max(self.b_left_max, 1e-6), 0.0, 1.0)

        # y 位置在 anchor 区内越靠下越稳定，但太底部已被截掉
        y_max = 1.0 - self.anchor_bottom_exclude
        ya_score = np.clip((ya - self.anchor_y_min) / max(y_max - self.anchor_y_min, 1e-6), 0.0, 1.0)
        yb_score = np.clip((yb - self.anchor_y_min) / max(y_max - self.anchor_y_min, 1e-6), 0.0, 1.0)

        return float(0.35 * sa + 0.35 * sb + 0.15 * ya_score + 0.15 * yb_score)

    # -------------------------
    # 时间稳定奖励
    # -------------------------
    def temporal_bonus(self, id_a, id_b, frame_id):
        item = self.pair_memory.get((id_a, id_b), None)
        if item is None:
            return 0.0

        gap = frame_id - item["last_frame"]
        if gap == 1:
            return min(1.0, 0.30 + 0.10 * item["consecutive"])
        elif gap <= 3:
            return 0.20
        else:
            return 0.0

    # -------------------------
    # 单帧匹配
    # -------------------------
    def match_one_frame(self, frame_id, dets_a_all, dets_b_all):
        self.update_histories(dets_a_all, dets_b_all)
        self.prune_pair_memory(frame_id)

        # 1) 先构建上下文集
        context_a = [d for d in dets_a_all if self.eligible_context_a(d)]
        context_b = [d for d in dets_b_all if self.eligible_context_b(d)]

        # 2) 再构建锚点集
        anchors_a = [d for d in dets_a_all if self.eligible_anchor_a(d)]
        anchors_b = [d for d in dets_b_all if self.eligible_anchor_b(d)]

        if len(anchors_a) == 0 or len(anchors_b) == 0:
            return []

        # 如果上下文太少，关系特征会很弱，但仍允许退化运行
        traj_feat_a = {d.track_id: build_traj_feature(self.hist_A[d.track_id], self.seq_len) for d in anchors_a}
        traj_feat_b = {d.track_id: build_traj_feature(self.hist_B[d.track_id], self.seq_len) for d in anchors_b}

        rel_feat_a = build_relation_features(anchors_a, context_a, self.hist_A, self.k_neighbors)
        rel_feat_b = build_relation_features(anchors_b, context_b, self.hist_B, self.k_neighbors)

        na = len(anchors_a)
        nb = len(anchors_b)

        score_total = np.full((na, nb), -1.0, dtype=np.float32)
        score_traj = np.zeros((na, nb), dtype=np.float32)
        score_rel = np.zeros((na, nb), dtype=np.float32)
        score_region = np.zeros((na, nb), dtype=np.float32)
        score_temp = np.zeros((na, nb), dtype=np.float32)
        valid_mask = np.zeros((na, nb), dtype=bool)

        # 3) 计算候选对分数
        for i, da in enumerate(anchors_a):
            for j, db in enumerate(anchors_b):
                if da.cls_id != db.cls_id:
                    continue
                if da.track_id not in rel_feat_a or db.track_id not in rel_feat_b:
                    continue

                st = traj_similarity(traj_feat_a[da.track_id], traj_feat_b[db.track_id])
                sr, rank_gap, lane_gap = relation_similarity(rel_feat_a[da.track_id], rel_feat_b[db.track_id])
                sg = self.region_prior_score(da, db)
                sm = self.temporal_bonus(da.track_id, db.track_id, frame_id)

                # 粗门控
                if rank_gap > 0.40:
                    continue
                if lane_gap > 0.55:
                    continue

                s = self.alpha * st + self.beta * sr + self.gamma * sg + self.delta * sm

                valid_mask[i, j] = True
                score_total[i, j] = s
                score_traj[i, j] = st
                score_rel[i, j] = sr
                score_region[i, j] = sg
                score_temp[i, j] = sm

        if not np.any(valid_mask):
            return []

        # 4) Hungarian
        huge = 1e6
        cost = np.full((na, nb), huge, dtype=np.float32)
        cost[valid_mask] = 1.0 - score_total[valid_mask]

        row_ind, col_ind = linear_sum_assignment(cost)

        # 5) top2 歧义抑制
        row_best = np.full(na, -1.0, dtype=np.float32)
        row_second = np.full(na, -1.0, dtype=np.float32)
        for i in range(na):
            vals = score_total[i][valid_mask[i]]
            if len(vals) > 0:
                vals = np.sort(vals)[::-1]
                row_best[i] = vals[0]
                row_second[i] = vals[1] if len(vals) > 1 else -1.0

        col_best = np.full(nb, -1.0, dtype=np.float32)
        col_second = np.full(nb, -1.0, dtype=np.float32)
        for j in range(nb):
            vals = score_total[:, j][valid_mask[:, j]]
            if len(vals) > 0:
                vals = np.sort(vals)[::-1]
                col_best[j] = vals[0]
                col_second[j] = vals[1] if len(vals) > 1 else -1.0

        # 6) 接收 + 连续确认
        confirmed_matches = []

        for r, c in zip(row_ind, col_ind):
            if not valid_mask[r, c]:
                continue

            s = float(score_total[r, c])
            if s < self.min_accept_score:
                continue

            margin_row = s - row_second[r] if row_second[r] >= 0 else s
            margin_col = s - col_second[c] if col_second[c] >= 0 else s
            if margin_row < self.ambiguity_margin:
                continue
            if margin_col < self.ambiguity_margin:
                continue

            da = anchors_a[r]
            db = anchors_b[c]
            pair = (da.track_id, db.track_id)

            old = self.pair_memory.get(pair, None)
            if old is not None and old["last_frame"] == frame_id - 1:
                consecutive = old["consecutive"] + 1
            else:
                consecutive = 1

            self.pair_memory[pair] = {
                "last_frame": frame_id,
                "consecutive": consecutive
            }

            if consecutive >= self.confirm_frames:
                confirmed_matches.append({
                    "frame_id": frame_id,
                    "track_id_a": da.track_id,
                    "track_id_b": db.track_id,
                    "score_total": round(s, 6),
                    "score_traj": round(float(score_traj[r, c]), 6),
                    "score_rel": round(float(score_rel[r, c]), 6),
                    "score_region": round(float(score_region[r, c]), 6),
                    "score_temp": round(float(score_temp[r, c]), 6),
                    "conf_a": round(float(da.conf), 6),
                    "conf_b": round(float(db.conf), 6),
                    "consecutive_hits": consecutive
                })

        return confirmed_matches

    # -------------------------
    # 总运行
    # -------------------------
    def run(self, tracks_a_by_frame, tracks_b_by_frame):
        all_frames = sorted(set(tracks_a_by_frame.keys()) | set(tracks_b_by_frame.keys()))
        results = []

        for frame_id in all_frames:
            dets_a = tracks_a_by_frame.get(frame_id, [])
            dets_b = tracks_b_by_frame.get(frame_id, [])

            frame_matches = self.match_one_frame(frame_id, dets_a, dets_b)
            results.extend(frame_matches)

            if frame_id % 100 == 0:
                print(f"[Info] frame={frame_id}, 已确认匹配 {len(results)} 条")

        return results


# =========================================================
# 7. 保存
# =========================================================
def save_matches_csv(matches, save_path):
    make_dir(os.path.dirname(save_path))
    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "frame_id",
            "track_id_a",
            "track_id_b",
            "score_total",
            "score_traj",
            "score_rel",
            "score_region",
            "score_temp",
            "conf_a",
            "conf_b",
            "consecutive_hits"
        ])
        for m in matches:
            writer.writerow([
                m["frame_id"],
                m["track_id_a"],
                m["track_id_b"],
                m["score_total"],
                m["score_traj"],
                m["score_rel"],
                m["score_region"],
                m["score_temp"],
                m["conf_a"],
                m["conf_b"],
                m["consecutive_hits"]
            ])

    print(f"[Done] 匹配结果已保存到: {save_path}")


# =========================================================
# 8. 主程序
# =========================================================
if __name__ == "__main__":
    # ========= 你改这里 =========
    frames_dir_a = r"experient_fig/doublesight/frame2"
    frames_dir_b = r"experient_fig/doublesight/frame1"

    txt_a = r"results/droneB_tracks.txt"
    txt_b = r"results/droneA_tracks.txt"

    txt_format = "custom"   # 或 "mot"
    cls_filter = 3          # 你现在是车辆 class=3
    out_csv = r"results/cross_match_lower_anchor.csv"

    # 读取图像尺寸
    img_w_a, img_h_a = get_first_image_size(frames_dir_a)
    img_w_b, img_h_b = get_first_image_size(frames_dir_b)

    print(f"A 图像尺寸: {img_w_a} x {img_h_a}")
    print(f"B 图像尺寸: {img_w_b} x {img_h_b}")

    # 读取轨迹
    tracks_a = load_track_txt(txt_a, txt_format=txt_format, cls_filter=cls_filter)
    tracks_b = load_track_txt(txt_b, txt_format=txt_format, cls_filter=cls_filter)

    print(f"A 共 {len(tracks_a)} 帧轨迹")
    print(f"B 共 {len(tracks_b)} 帧轨迹")

    matcher = CrossViewMatcherLowerAnchor(
        img_w_a=img_w_a,
        img_h_a=img_h_a,
        img_w_b=img_w_b,
        img_h_b=img_h_b,

        window_size=20,
        seq_len=8,
        k_neighbors=3,

        min_conf=0.60,
        min_track_len=8,

        # 公共区域：A 右，B 左
        a_right_min=0.35,
        b_left_max=0.65,

        # 锚点区：只在下半部分做正式匹配
        anchor_y_min=0.50,
        anchor_bottom_exclude=0.08,

        # 上下文区：关系特征可向上扩，但排除太远小目标
        context_y_min=0.35,
        context_bottom_exclude=0.06,
        min_context_box_h_ratio=0.035,

        # 降低轨迹权重，提高关系权重
        alpha=0.25,
        beta=0.40,
        gamma=0.25,
        delta=0.10,

        min_accept_score=0.70,
        ambiguity_margin=0.08,
        confirm_frames=3,
        pair_ttl=5
    )

    matches = matcher.run(tracks_a, tracks_b)
    save_matches_csv(matches, out_csv)