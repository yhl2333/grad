import os
import csv
import math
from dataclasses import dataclass
from collections import defaultdict, deque

import numpy as np


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

    @property
    def area(self):
        return float(max(0.0, self.w) * max(0.0, self.h))


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
def safe_log_ratio(a, b, eps=1e-6):
    a = max(float(a), eps)
    b = max(float(b), eps)
    return abs(math.log(a / b))


def exp_sim(diff, scale):
    return math.exp(-float(diff) * float(scale))


def infer_canvas_size(tracks_by_frame):
    max_x = 0.0
    max_y = 0.0
    for _, dets in tracks_by_frame.items():
        for d in dets:
            max_x = max(max_x, d.x2, d.cx)
            max_y = max(max_y, d.y2, d.cy_bottom)
    w = max(1, int(math.ceil(max_x + 1)))
    h = max(1, int(math.ceil(max_y + 1)))
    return w, h


def weighted_mean(xs, ws):
    xs = np.asarray(xs, dtype=np.float32)
    ws = np.asarray(ws, dtype=np.float32)
    s = float(np.sum(ws))
    if s < 1e-6:
        return float(np.mean(xs)) if len(xs) > 0 else 0.0
    return float(np.sum(xs * ws) / s)


# =========================
# 4. 配置
# =========================
@dataclass
class FanSeqChainConfig:
    # 历史窗口：这里只用于 life_ratio
    window_size: int = 20

    # 只保留面积前多少比例
    size_keep_ratio: float = 0.50

    # 基础置信度过滤
    min_conf: float = 0.0

    # 参考点（近似无穷远点）
    vp_center_mix: float = 0.75
    vp_y_offset_ratio: float = 0.15

    # ---------- [IMPROVE-1] 普通候选门控 ----------
    max_rank_gap: float = 0.40
    max_y_gap: float = 0.30
    max_h_ratio: float = 3.0
    max_area_ratio: float = 6.0

    # ---------- [IMPROVE-2] 近端大框更严格门控 ----------
    near_y_thresh: float = 0.55
    near_max_rank_gap: float = 0.22
    near_max_y_gap: float = 0.12
    near_max_h_ratio: float = 1.80
    near_max_area_ratio: float = 2.50

    # 单对候选最小分数
    min_pair_score: float = 0.55

    # ---------- [IMPROVE-3] 转移一致性参数 ----------
    transition_lambda: float = 0.22
    trans_rank_scale: float = 8.0
    trans_y_scale: float = 10.0
    trans_h_scale: float = 2.5
    trans_area_scale: float = 1.8


# =========================
# 5. 只保留面积前50%
# =========================
def keep_top_size_ratio(dets, cfg: FanSeqChainConfig):
    if len(dets) == 0:
        return []

    dets = [d for d in dets if d.conf >= cfg.min_conf]
    if len(dets) == 0:
        return []

    dets_sorted = sorted(dets, key=lambda d: d.area, reverse=True)
    keep_n = max(1, int(math.ceil(len(dets_sorted) * cfg.size_keep_ratio)))
    return dets_sorted[:keep_n]


# =========================
# 6. 构造角度序列 + 描述子
# =========================
def build_fan_sequence_features(current_dets, histories, img_w, img_h, cfg: FanSeqChainConfig):
    """
    返回：
        dets_sorted: 按角度从左到右排序后的 Detection 列表
        feat_by_id:  每个 track_id 的局部描述子
        vp:          参考点
    """
    if len(current_dets) == 0:
        return [], {}, (img_w * 0.5, -cfg.vp_y_offset_ratio * img_h)

    cxs = [d.cx for d in current_dets]
    hs = [max(d.h, 1.0) for d in current_dets]

    # 参考点：以上中为主，稍微参考当前目标分布
    mean_cx = weighted_mean(cxs, hs)
    vx = cfg.vp_center_mix * (img_w * 0.5) + (1.0 - cfg.vp_center_mix) * mean_cx
    vy = -cfg.vp_y_offset_ratio * img_h

    pts = np.array([[d.cx, d.cy_bottom] for d in current_dets], dtype=np.float32)
    thetas = np.array([math.atan2(p[1] - vy, p[0] - vx) for p in pts], dtype=np.float32)

    order = np.argsort(thetas)
    dets_sorted = [current_dets[i] for i in order]
    pts_sorted = pts[order]
    theta_sorted = thetas[order]

    n = len(dets_sorted)

    hs_sorted = np.array([max(d.h, 1.0) for d in dets_sorted], dtype=np.float32)
    areas_sorted = np.array([max(d.area, 1.0) for d in dets_sorted], dtype=np.float32)

    mean_h = float(np.mean(hs_sorted)) + 1e-6
    mean_area = float(np.mean(areas_sorted)) + 1e-6

    if n >= 2:
        total_span = float(theta_sorted[-1] - theta_sorted[0]) + 1e-6
    else:
        total_span = 1.0

    adj_dists = []
    if n >= 2:
        for i in range(n - 1):
            adj_dists.append(float(np.linalg.norm(pts_sorted[i + 1] - pts_sorted[i])))
    mean_adj_dist = float(np.mean(adj_dists)) + 1e-6 if len(adj_dists) > 0 else 1.0

    feat_by_id = {}

    for i, d in enumerate(dets_sorted):
        has_left = 1.0 if i > 0 else 0.0
        has_right = 1.0 if i < n - 1 else 0.0

        if i > 0:
            dtheta_l = float(theta_sorted[i] - theta_sorted[i - 1]) / total_span
            ddist_l = float(np.linalg.norm(pts_sorted[i] - pts_sorted[i - 1])) / mean_adj_dist
            h_ratio_l = float(hs_sorted[i] / (hs_sorted[i - 1] + 1e-6))
        else:
            dtheta_l = 1.0
            ddist_l = 2.0
            h_ratio_l = 1.0

        if i < n - 1:
            dtheta_r = float(theta_sorted[i + 1] - theta_sorted[i]) / total_span
            ddist_r = float(np.linalg.norm(pts_sorted[i + 1] - pts_sorted[i])) / mean_adj_dist
            h_ratio_r = float(hs_sorted[i] / (hs_sorted[i + 1] + 1e-6))
        else:
            dtheta_r = 1.0
            ddist_r = 2.0
            h_ratio_r = 1.0

        life_ratio = min(len(histories[d.track_id]) / max(cfg.window_size, 1), 1.0)

        center_bias = abs(d.cx - img_w * 0.5) / (img_w * 0.5 + 1e-6)
        centrality = math.exp(-center_bias / 0.60)

        feat_by_id[d.track_id] = {
            "track_id": d.track_id,
            "rank": float(i / max(n - 1, 1)),
            "y_norm": float(d.cy_bottom / max(img_h, 1.0)),
            "h_norm": float(d.h / mean_h),
            "area_norm": float(d.area / mean_area),
            "center_bias": float(center_bias),
            "centrality": float(centrality),
            "life_ratio": float(life_ratio),
            "has_left": float(has_left),
            "has_right": float(has_right),
            "dtheta_l": float(dtheta_l),
            "dtheta_r": float(dtheta_r),
            "ddist_l": float(ddist_l),
            "ddist_r": float(ddist_r),
            "h_ratio_l": float(h_ratio_l),
            "h_ratio_r": float(h_ratio_r),
            "conf": float(d.conf),
        }

    return dets_sorted, feat_by_id, (vx, vy)


# =========================
# 7. 单点描述子相似度
# =========================
def fan_descriptor_similarity(fa, fb):
    s_rank = exp_sim(abs(fa["rank"] - fb["rank"]), 5.0)
    s_y = exp_sim(abs(fa["y_norm"] - fb["y_norm"]), 6.0)

    s_h = exp_sim(safe_log_ratio(fa["h_norm"], fb["h_norm"]), 2.6)
    s_area = exp_sim(safe_log_ratio(fa["area_norm"], fb["area_norm"]), 1.8)

    dtheta_diff = 0.5 * abs(fa["dtheta_l"] - fb["dtheta_l"]) + 0.5 * abs(fa["dtheta_r"] - fb["dtheta_r"])
    s_theta = exp_sim(dtheta_diff, 8.0)

    ddist_diff = 0.5 * abs(fa["ddist_l"] - fb["ddist_l"]) + 0.5 * abs(fa["ddist_r"] - fb["ddist_r"])
    s_dist = exp_sim(ddist_diff, 1.8)

    ratio_diff = 0.5 * safe_log_ratio(fa["h_ratio_l"], fb["h_ratio_l"]) + \
                 0.5 * safe_log_ratio(fa["h_ratio_r"], fb["h_ratio_r"])
    s_ratio = exp_sim(ratio_diff, 1.6)

    boundary_diff = abs(fa["has_left"] - fb["has_left"]) + abs(fa["has_right"] - fb["has_right"])
    s_boundary = exp_sim(boundary_diff, 1.5)

    s_center = exp_sim(abs(fa["center_bias"] - fb["center_bias"]), 3.0)
    s_life = exp_sim(abs(fa["life_ratio"] - fb["life_ratio"]), 3.0)

    score = (
        0.14 * s_rank +
        0.12 * s_y +
        0.13 * s_h +
        0.08 * s_area +
        0.18 * s_theta +
        0.14 * s_dist +
        0.09 * s_ratio +
        0.05 * s_boundary +
        0.04 * s_center +
        0.03 * s_life
    )
    return float(score)

def fan_descriptor_similarity_breakdown(fa, fb):
    s_rank = exp_sim(abs(fa["rank"] - fb["rank"]), 5.0)
    s_y = exp_sim(abs(fa["y_norm"] - fb["y_norm"]), 6.0)

    s_h = exp_sim(safe_log_ratio(fa["h_norm"], fb["h_norm"]), 2.6)
    s_area = exp_sim(safe_log_ratio(fa["area_norm"], fb["area_norm"]), 1.8)

    dtheta_diff = 0.5 * abs(fa["dtheta_l"] - fb["dtheta_l"]) + 0.5 * abs(fa["dtheta_r"] - fb["dtheta_r"])
    s_theta = exp_sim(dtheta_diff, 8.0)

    ddist_diff = 0.5 * abs(fa["ddist_l"] - fb["ddist_l"]) + 0.5 * abs(fa["ddist_r"] - fb["ddist_r"])
    s_dist = exp_sim(ddist_diff, 1.8)

    ratio_diff = 0.5 * safe_log_ratio(fa["h_ratio_l"], fb["h_ratio_l"]) + \
                 0.5 * safe_log_ratio(fa["h_ratio_r"], fb["h_ratio_r"])
    s_ratio = exp_sim(ratio_diff, 1.6)

    boundary_diff = abs(fa["has_left"] - fb["has_left"]) + abs(fa["has_right"] - fb["has_right"])
    s_boundary = exp_sim(boundary_diff, 1.5)

    s_center = exp_sim(abs(fa["center_bias"] - fb["center_bias"]), 3.0)
    s_life = exp_sim(abs(fa["life_ratio"] - fb["life_ratio"]), 3.0)

    score = (
        0.14 * s_rank +
        0.12 * s_y +
        0.13 * s_h +
        0.08 * s_area +
        0.18 * s_theta +
        0.14 * s_dist +
        0.09 * s_ratio +
        0.05 * s_boundary +
        0.04 * s_center +
        0.03 * s_life
    )

    detail = {
        "s_rank": s_rank,
        "s_y": s_y,
        "s_h": s_h,
        "s_area": s_area,
        "s_theta": s_theta,
        "s_dist": s_dist,
        "s_ratio": s_ratio,
        "s_boundary": s_boundary,
        "s_center": s_center,
        "s_life": s_life,
        "score_total": score,
    }
    return score, detail
# =========================
# 8. 候选门控
# =========================
def is_valid_candidate(fa, fb, cfg: FanSeqChainConfig):
    """
    [IMPROVE-2]
    对近端目标使用更严格门控。
    """
    h_ratio = max(fa["h_norm"], fb["h_norm"]) / (min(fa["h_norm"], fb["h_norm"]) + 1e-6)
    area_ratio = max(fa["area_norm"], fb["area_norm"]) / (min(fa["area_norm"], fb["area_norm"]) + 1e-6)

    is_near = (fa["y_norm"] > cfg.near_y_thresh) or (fb["y_norm"] > cfg.near_y_thresh)

    if is_near:
        if abs(fa["rank"] - fb["rank"]) > cfg.near_max_rank_gap:
            return False
        if abs(fa["y_norm"] - fb["y_norm"]) > cfg.near_max_y_gap:
            return False
        if h_ratio > cfg.near_max_h_ratio:
            return False
        if area_ratio > cfg.near_max_area_ratio:
            return False
    else:
        if abs(fa["rank"] - fb["rank"]) > cfg.max_rank_gap:
            return False
        if abs(fa["y_norm"] - fb["y_norm"]) > cfg.max_y_gap:
            return False
        if h_ratio > cfg.max_h_ratio:
            return False
        if area_ratio > cfg.max_area_ratio:
            return False

    return True


# =========================
# 9. 转移一致性分数
# =========================
def transition_consistency_bonus(prev_fa, prev_fb, cur_fa, cur_fb, cfg: FanSeqChainConfig):
    """
    [IMPROVE-3]
    比较连续两个匹配之间的“变化是否一致”。
    目标：抑制 A1 / A40 这种单点分数不低，但放到整条链里不顺的错配。
    """
    # rank 增量一致性
    dr_a = cur_fa["rank"] - prev_fa["rank"]
    dr_b = cur_fb["rank"] - prev_fb["rank"]
    s_rank_step = exp_sim(abs(dr_a - dr_b), cfg.trans_rank_scale)

    # y 增量一致性
    dy_a = cur_fa["y_norm"] - prev_fa["y_norm"]
    dy_b = cur_fb["y_norm"] - prev_fb["y_norm"]
    s_y_step = exp_sim(abs(dy_a - dy_b), cfg.trans_y_scale)

    # 高度变化一致性
    dh_a = safe_log_ratio(cur_fa["h_norm"], prev_fa["h_norm"])
    dh_b = safe_log_ratio(cur_fb["h_norm"], prev_fb["h_norm"])
    s_h_step = exp_sim(abs(dh_a - dh_b), cfg.trans_h_scale)

    # 面积变化一致性
    da_a = safe_log_ratio(cur_fa["area_norm"], prev_fa["area_norm"])
    da_b = safe_log_ratio(cur_fb["area_norm"], prev_fb["area_norm"])
    s_area_step = exp_sim(abs(da_a - da_b), cfg.trans_area_scale)

    trans = (
        0.35 * s_rank_step +
        0.35 * s_y_step +
        0.15 * s_h_step +
        0.15 * s_area_step
    )
    return float(trans)


# =========================
# 10. 带转移约束的最大权序列匹配
# =========================
def weighted_chain_sequence_match(seq_a, seq_b, feat_a, feat_b, cfg: FanSeqChainConfig):
    """
    [IMPROVE-4]
    不再使用“纯单点加分”的 LCS。
    改为：
        - 节点：候选配对 (i, j)
        - 权重：单点配对分数
        - 边：从 (p, q) -> (i, j), 且 p < i, q < j
        - 边权：转移一致性项

    最终求一条单调递增路径，使得：
        节点分数之和 + 转移一致性收益 最大
    """
    na = len(seq_a)
    nb = len(seq_b)

    if na == 0 or nb == 0:
        return [], 0.0

    # 先生成所有候选节点
    candidates = []
    for i, da in enumerate(seq_a):
        fa = feat_a[da.track_id]
        for j, db in enumerate(seq_b):
            fb = feat_b[db.track_id]

            if not is_valid_candidate(fa, fb, cfg):
                continue

            s = fan_descriptor_similarity(fa, fb)
            if s < cfg.min_pair_score:
                continue

            candidates.append({
                "i": i,
                "j": j,
                "track_id_a": da.track_id,
                "track_id_b": db.track_id,
                "pair_score": float(s),
                "fa": fa,
                "fb": fb,
            })

    if len(candidates) == 0:
        return [], 0.0

    # 按序列顺序排序
    candidates.sort(key=lambda x: (x["i"], x["j"]))

    k = len(candidates)
    dp = np.zeros(k, dtype=np.float32)
    prev_idx = np.full(k, -1, dtype=np.int32)
    trans_used = np.zeros(k, dtype=np.float32)

    # 每个候选单独成链
    for idx in range(k):
        dp[idx] = candidates[idx]["pair_score"]
        prev_idx[idx] = -1
        trans_used[idx] = 0.0

    # 最长路径 / 最大权路径
    for cur in range(k):
        ci = candidates[cur]["i"]
        cj = candidates[cur]["j"]

        for pre in range(cur):
            pi = candidates[pre]["i"]
            pj = candidates[pre]["j"]

            # 必须保持单调顺序
            if not (pi < ci and pj < cj):
                continue

            trans_raw = transition_consistency_bonus(
                candidates[pre]["fa"], candidates[pre]["fb"],
                candidates[cur]["fa"], candidates[cur]["fb"],
                cfg
            )

            # 转成有奖有罚的转移项
            # trans_raw in [0,1]
            # 0.5 左右近似中性；高于 0.5 奖励，低于 0.5 惩罚
            trans_term = cfg.transition_lambda * (2.0 * trans_raw - 1.0)

            cand_total = dp[pre] + candidates[cur]["pair_score"] + trans_term
            if cand_total > dp[cur]:
                dp[cur] = cand_total
                prev_idx[cur] = pre
                trans_used[cur] = trans_term

    # 找全局最优链尾
    best_end = int(np.argmax(dp))
    best_total_score = float(dp[best_end])

    # 回溯整条链
    chain = []
    cur = best_end
    while cur != -1:
        node = candidates[cur]
        chain.append({
            "i": node["i"],
            "j": node["j"],
            "track_id_a": node["track_id_a"],
            "track_id_b": node["track_id_b"],
            "pair_score": float(node["pair_score"]),
            "trans_term": float(trans_used[cur]),
        })
        cur = int(prev_idx[cur])

    chain.reverse()
    return chain, best_total_score


# =========================
# 11. 匹配器
# =========================
class FanSeqChainMatcher:
    def __init__(self, cfg: FanSeqChainConfig):
        self.cfg = cfg
        self.hist_A = defaultdict(lambda: deque(maxlen=self.cfg.window_size))
        self.hist_B = defaultdict(lambda: deque(maxlen=self.cfg.window_size))

    def update_histories(self, dets_a, dets_b):
        for d in dets_a:
            self.hist_A[d.track_id].append(d)
        for d in dets_b:
            self.hist_B[d.track_id].append(d)

    def match_one_frame(self, frame_id, dets_a, dets_b, img_w_a, img_h_a, img_w_b, img_h_b):
        # 更新历史，供 life_ratio 使用
        self.update_histories(dets_a, dets_b)

        # 只保留面积前50%
        valid_a = keep_top_size_ratio(dets_a, self.cfg)
        valid_b = keep_top_size_ratio(dets_b, self.cfg)

        if len(valid_a) == 0 or len(valid_b) == 0:
            return []

        # 各自构造角度序列
        seq_a, feat_a, vp_a = build_fan_sequence_features(valid_a, self.hist_A, img_w_a, img_h_a, self.cfg)
        seq_b, feat_b, vp_b = build_fan_sequence_features(valid_b, self.hist_B, img_w_b, img_h_b, self.cfg)


        # =========================
        # DEBUG: 打印指定帧、指定A id 与 B 中各目标的分项得分
        # =========================
        debug_frame = 224          # 改成你想看的帧
        debug_a_ids = {27, 16}     # 改成你想看的 A 侧 track_id

        if frame_id == debug_frame:
            print("\n" + "=" * 100)
            print(f"[DEBUG] frame_id = {frame_id}")
            print(f"[DEBUG] A侧有效序列id: {[d.track_id for d in seq_a]}")
            print(f"[DEBUG] B侧有效序列id: {[d.track_id for d in seq_b]}")
            print("=" * 100)

            for da in seq_a:
                if da.track_id not in debug_a_ids:
                    continue

                fa = feat_a[da.track_id]
                print(f"\n[DEBUG] 查看 A:{da.track_id} 与 B中所有目标的匹配分数")
                print("-" * 100)

                rows = []
                for db in seq_b:
                    fb = feat_b[db.track_id]

                    valid = is_valid_candidate(fa, fb, self.cfg)
                    score, detail = fan_descriptor_similarity_breakdown(fa, fb)

                    rows.append({
                        "b_id": db.track_id,
                        "valid": valid,
                        "score_total": score,
                        "detail": detail
                    })

                # 按总分从高到低打印
                rows.sort(key=lambda x: x["score_total"], reverse=True)

                for row in rows:
                    b_id = row["b_id"]
                    valid = row["valid"]
                    detail = row["detail"]

                    print(
                        f"A:{da.track_id} -> B:{b_id} | "
                        f"valid={valid} | "
                        f"total={detail['score_total']:.4f} | "
                        f"rank={detail['s_rank']:.4f}, "
                        f"y={detail['s_y']:.4f}, "
                        f"h={detail['s_h']:.4f}, "
                        f"area={detail['s_area']:.4f}, "
                        f"theta={detail['s_theta']:.4f}, "
                        f"dist={detail['s_dist']:.4f}, "
                        f"ratio={detail['s_ratio']:.4f}, "
                        f"boundary={detail['s_boundary']:.4f}, "
                        f"center={detail['s_center']:.4f}, "
                        f"life={detail['s_life']:.4f}"
                    )



        if len(seq_a) == 0 or len(seq_b) == 0:
            return []

        # 带转移约束的最大权序列匹配
        chain, best_total_score = weighted_chain_sequence_match(seq_a, seq_b, feat_a, feat_b, self.cfg)

        matches = []
        for item in chain:
            local_total = float(item["pair_score"] + item["trans_term"])
            matches.append({
                "frame_id": frame_id,
                "track_id_a": item["track_id_a"],
                "track_id_b": item["track_id_b"],
                "score_total": round(local_total, 6),
                "score_graph": round(float(item["pair_score"]), 6),
                # 这里沿用原字段名，实际装的是“转移项”
                "score_temp": round(float(item["trans_term"]), 6),
                "score_center": 0.0,
                "num_valid_a": len(seq_a),
                "num_valid_b": len(seq_b),
                "best_total_score": round(float(best_total_score), 6),
            })

        return matches

    def run(self, tracks_a_by_frame, tracks_b_by_frame, img_size_a, img_size_b):
        img_w_a, img_h_a = img_size_a
        img_w_b, img_h_b = img_size_b

        all_frames = sorted(set(tracks_a_by_frame.keys()) | set(tracks_b_by_frame.keys()))
        results = []

        for frame_id in all_frames:
            dets_a = tracks_a_by_frame.get(frame_id, [])
            dets_b = tracks_b_by_frame.get(frame_id, [])

            frame_matches = self.match_one_frame(
                frame_id, dets_a, dets_b,
                img_w_a, img_h_a, img_w_b, img_h_b
            )
            results.extend(frame_matches)

            if frame_id % 100 == 0:
                print(f"[Info] 已处理到 frame {frame_id}, 当前累计匹配 {len(results)} 条")

        return results


# =========================
# 12. 保存结果
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
            "score_graph",
            "score_temp",
            "score_center",
            "num_valid_a",
            "num_valid_b",
            "best_total_score",
        ])
        for m in matches:
            writer.writerow([
                m["frame_id"],
                m["track_id_a"],
                m["track_id_b"],
                m["score_total"],
                m["score_graph"],
                m["score_temp"],
                m["score_center"],
                m["num_valid_a"],
                m["num_valid_b"],
                m["best_total_score"],
            ])

    print(f"匹配结果已保存到: {save_path}")


# =========================
# 13. 主程序
# =========================
if __name__ == "__main__":
    # -------- 你的输入 --------
    txt_a = "experient_fig/doublesight/droneA_tracks.txt"
    txt_b = "experient_fig/doublesight/droneB_tracks.txt"

    txt_format = "custom"
    cls_filter = None

    # -------- 加载轨迹 --------
    tracks_a = load_track_txt(txt_a, txt_format=txt_format, cls_filter=cls_filter)
    tracks_b = load_track_txt(txt_b, txt_format=txt_format, cls_filter=cls_filter)

    print(f"A 视角共 {len(tracks_a)} 帧")
    print(f"B 视角共 {len(tracks_b)} 帧")

    # -------- 推断画面尺寸 --------
    img_size_a = infer_canvas_size(tracks_a)
    img_size_b = infer_canvas_size(tracks_b)
    print(f"A 视角推断尺寸: {img_size_a}")
    print(f"B 视角推断尺寸: {img_size_b}")

    # -------- 配置 --------
    cfg = FanSeqChainConfig(
        window_size=20,

        # 只保留面积前 50%
        size_keep_ratio=0.50,

        # 基础过滤
        min_conf=0.0,

        # 参考点
        vp_center_mix=0.75,
        vp_y_offset_ratio=0.15,

        # 普通候选门控
        max_rank_gap=0.40,
        max_y_gap=0.30,
        max_h_ratio=3.0,
        max_area_ratio=6.0,

        # 近端大框严格门控
        near_y_thresh=0.55,
        near_max_rank_gap=0.22,
        near_max_y_gap=0.12,
        near_max_h_ratio=1.80,
        near_max_area_ratio=2.50,

        # 单对最低分
        min_pair_score=0.55,

        # 转移一致性
        transition_lambda=0.22,
        trans_rank_scale=8.0,
        trans_y_scale=10.0,
        trans_h_scale=2.5,
        trans_area_scale=1.8,
    )

    # -------- 建立匹配器 --------
    matcher = FanSeqChainMatcher(cfg)

    # -------- 执行匹配 --------
    matches = matcher.run(tracks_a, tracks_b, img_size_a, img_size_b)

    # -------- 保存结果 --------
    save_matches_csv(matches, "results/exp8_new_new/fan_seq_chain_match.csv")