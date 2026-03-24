import os
import csv
import math
from dataclasses import dataclass
from collections import defaultdict, deque
from typing import Optional

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
    reid_feat: Optional[np.ndarray] = None

    # ===== Kalman sidecar（可选）=====
    pred_cx: float = np.nan
    pred_cy_bottom: float = np.nan

    # ===== 由 当前框 - 预测框 得到的速度方向 =====
    speed_dx: float = np.nan
    speed_dy: float = np.nan
    speed_proj: float = np.nan
    speed_sign: int = 0   # -1 / 0 / +1

    @property
    def pos(self):
        return np.array([self.cx, self.cy_bottom], dtype=np.float32)

    @property
    def pred_pos(self):
        return np.array([self.pred_cx, self.pred_cy_bottom], dtype=np.float32)

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
# 2. 读取 ReID npz
# =========================
def load_reid_npz(npz_path):
    if npz_path is None:
        return {}

    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"ReID特征文件不存在: {npz_path}")

    data = np.load(npz_path)
    frame_ids = data["frame_ids"]
    track_ids = data["track_ids"]
    feats = data["feats"]

    reid_map = {}
    for i in range(len(frame_ids)):
        key = (int(frame_ids[i]), int(track_ids[i]))
        feat = np.asarray(feats[i], dtype=np.float32).reshape(-1)
        norm = np.linalg.norm(feat)
        if norm > 1e-12:
            feat = feat / norm
        else:
            feat = None
        reid_map[key] = feat

    print(f"[Info] 已加载 ReID 特征: {npz_path}, 共 {len(reid_map)} 条")
    return reid_map


# =========================
# 3. 读取 Kalman sidecar txt
# =========================
def load_kf_txt(kf_txt_path):
    """
    sidecar 格式至少应包含:
    frame_id,track_id,pred_x1,pred_y1,pred_x2,pred_y2,pred_cx,pred_cy_bottom,vx,vy,vw,vh
    """
    if kf_txt_path is None:
        return {}

    if not os.path.exists(kf_txt_path):
        raise FileNotFoundError(f"Kalman sidecar 文件不存在: {kf_txt_path}")

    kf_map = {}

    with open(kf_txt_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required_cols = ["frame_id", "track_id", "pred_cx", "pred_cy_bottom"]
        for c in required_cols:
            if c not in reader.fieldnames:
                raise ValueError(f"{kf_txt_path} 缺少字段: {c}")

        for row in reader:
            frame_id = int(float(row["frame_id"]))
            track_id = int(float(row["track_id"]))
            pred_cx = float(row["pred_cx"])
            pred_cy_bottom = float(row["pred_cy_bottom"])
            kf_map[(frame_id, track_id)] = {
                "pred_cx": pred_cx,
                "pred_cy_bottom": pred_cy_bottom,
            }

    print(f"[Info] 已加载 Kalman sidecar: {kf_txt_path}, 共 {len(kf_map)} 条")
    return kf_map


# =========================
# 4. 读取轨迹 txt
# =========================
def load_track_txt(txt_path, txt_format="custom", cls_filter=None, reid_map=None, kf_map=None):
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

                feat = None
                if reid_map is not None:
                    feat = reid_map.get((frame_id, track_id), None)

                pred_cx = np.nan
                pred_cy_bottom = np.nan
                if kf_map is not None:
                    kf_row = kf_map.get((frame_id, track_id), None)
                    if kf_row is not None:
                        pred_cx = float(kf_row["pred_cx"])
                        pred_cy_bottom = float(kf_row["pred_cy_bottom"])

                det = Detection(
                    frame_id=frame_id,
                    track_id=track_id,
                    cls_id=cls_id,
                    x1=x1, y1=y1, x2=x2, y2=y2,
                    cx=cx, cy_bottom=cy_bottom,
                    conf=conf,
                    reid_feat=feat,
                    pred_cx=pred_cx,
                    pred_cy_bottom=pred_cy_bottom,
                )
                tracks_by_frame[frame_id].append(det)

            except Exception as e:
                raise ValueError(f"解析 {txt_path} 第 {ln} 行失败：{line}\n错误信息：{e}")

    return dict(sorted(tracks_by_frame.items(), key=lambda x: x[0]))


# =========================
# 5. 工具函数
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


def reid_similarity(feat_a, feat_b):
    """
    返回 [0, 1] 的 ReID 余弦相似度
    如果任一侧缺失特征，则返回 None
    """
    if feat_a is None or feat_b is None:
        return None

    a = np.asarray(feat_a, dtype=np.float32).reshape(-1)
    b = np.asarray(feat_b, dtype=np.float32).reshape(-1)

    if a.size == 0 or b.size == 0 or a.shape != b.shape:
        return None

    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return None

    a = a / na
    b = b / nb

    cos = float(np.clip(np.dot(a, b), -1.0, 1.0))
    return 0.5 * (cos + 1.0)


def normalize_vec2(v):
    arr = np.asarray(v, dtype=np.float32).reshape(2)
    n = np.linalg.norm(arr)
    if n < 1e-12:
        raise ValueError(f"非法方向轴，长度为 0: {v}")
    return arr / n


# =========================
# 6. 配置
# =========================
@dataclass
class FanSeqChainConfig:
    # 历史窗口
    window_size: int = 20

    # 只保留面积前多少比例
    size_keep_ratio: float = 0.50

    # 基础置信度过滤
    min_conf: float = 0.0

    # 参考点
    vp_center_mix: float = 0.75
    vp_y_offset_ratio: float = 0.15

    # ReID top-k
    reid_topk: int = 5
    min_reid_score: float = 0.50

    # 几何筛选最低分
    min_refine_score: float = 0.45

    # 候选门控
    max_rank_gap: float = 0.40
    max_y_gap: float = 0.30
    max_h_ratio: float = 3.0
    max_area_ratio: float = 6.0

    # 近端更严格
    near_y_thresh: float = 0.55
    near_max_rank_gap: float = 0.22
    near_max_y_gap: float = 0.12
    near_max_h_ratio: float = 1.80
    near_max_area_ratio: float = 2.50

    # 几何筛选各项权重
    refine_pos_weight: float = 0.35
    refine_area_weight: float = 0.20
    refine_h_weight: float = 0.20
    refine_dist_weight: float = 0.25

    # =========================
    # 速度方向门控（可开关）
    # 方向 = 当前框 - 预测框
    # 关闭时，结果与原逻辑一致
    # =========================
    enable_speed_direction: bool = False

    # A/B 两侧分别用哪条图像轴判断方向
    speed_axis_a: tuple = (0.0, 1.0)
    speed_axis_b: tuple = (0.0, 1.0)

    # 两视角符号关系:
    #   "same"     -> sign_a == sign_b
    #   "opposite" -> sign_a == -sign_b
    speed_relation: str = "same"

    # 投影绝对值太小视为方向未知
    min_speed_proj_abs: float = 1.0

    # 方向未知时是否允许继续进入匹配
    allow_unknown_speed_direction: bool = True


# =========================
# 7. 只保留面积前 N%
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
# 8. 速度方向标注
#    方向 = 当前框 - 预测框
# =========================
def annotate_speed_direction(tracks_by_frame, axis, cfg: FanSeqChainConfig):
    axis_unit = normalize_vec2(axis)

    for _, dets in tracks_by_frame.items():
        for d in dets:
            if np.isnan(d.pred_cx) or np.isnan(d.pred_cy_bottom):
                d.speed_dx = np.nan
                d.speed_dy = np.nan
                d.speed_proj = np.nan
                d.speed_sign = 0
                continue

            dx = float(d.cx - d.pred_cx)
            dy = float(d.cy_bottom - d.pred_cy_bottom)
            proj = float(dx * axis_unit[0] + dy * axis_unit[1])

            d.speed_dx = dx
            d.speed_dy = dy
            d.speed_proj = proj

            if abs(proj) < cfg.min_speed_proj_abs:
                d.speed_sign = 0
            else:
                d.speed_sign = 1 if proj > 0 else -1


def is_valid_speed_candidate(da: Detection, db: Detection, cfg: FanSeqChainConfig):
    """
    速度方向门控：
    关闭时永远 True，保证结果和原逻辑一致
    """
    if not cfg.enable_speed_direction:
        return True

    sa = int(da.speed_sign)
    sb = int(db.speed_sign)

    if sa == 0 or sb == 0:
        return bool(cfg.allow_unknown_speed_direction)

    if cfg.speed_relation == "same":
        return sa == sb
    elif cfg.speed_relation == "opposite":
        return sa == -sb
    else:
        raise ValueError("speed_relation 只能是 'same' 或 'opposite'")


# =========================
# 9. 构造角度序列 + 描述子
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
            "h_norm": float(d.h ),
            "area_norm": float(d.area),
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
# 10. 几何筛选分数
# =========================
def geometry_refine_similarity_breakdown(fa, fb, cfg: FanSeqChainConfig):
    s_rank = exp_sim(abs(fa["rank"] - fb["rank"]), 5.0)
    s_y = exp_sim(abs(fa["y_norm"] - fb["y_norm"]), 6.0)
    s_pos = 0.5 * s_rank + 0.5 * s_y

    s_area = exp_sim(safe_log_ratio(fa["area_norm"], fb["area_norm"]), 1.8)
    s_h = exp_sim(safe_log_ratio(fa["h_norm"], fb["h_norm"]), 2.6)

    ddist_diff = 0.5 * abs(fa["ddist_l"] - fb["ddist_l"]) + 0.5 * abs(fa["ddist_r"] - fb["ddist_r"])
    s_dist = exp_sim(ddist_diff, 1.8)

    score = (
        cfg.refine_pos_weight * s_pos +
        cfg.refine_area_weight * s_area +
        cfg.refine_h_weight * s_h +
        cfg.refine_dist_weight * s_dist
    )

    detail = {
        "s_rank": float(s_rank),
        "s_y": float(s_y),
        "s_pos": float(s_pos),
        "s_area": float(s_area),
        "s_h": float(s_h),
        "s_dist": float(s_dist),
        "score_total": float(score),
    }
    return float(score), detail


# =========================
# 11. 几何门控
# =========================
def is_valid_geometry_candidate(fa, fb, cfg: FanSeqChainConfig):
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
# 12. ReID top-k + 几何筛选 + 可选速度方向门控
# 关闭速度开关时，与原逻辑一致
# =========================
def match_by_reid_topk_then_geometry(seq_a, seq_b, feat_a, feat_b, cfg: FanSeqChainConfig):
    if len(seq_a) == 0 or len(seq_b) == 0:
        return []

    all_candidates = []

    for da in seq_a:
        fa = feat_a[da.track_id]

        reid_rows = []
        for db in seq_b:
            # ===== 新增：速度方向门控，关闭时无效 =====
            if not is_valid_speed_candidate(da, db, cfg):
                continue

            reid_score = reid_similarity(da.reid_feat, db.reid_feat)
            if reid_score is None:
                continue
            if reid_score < cfg.min_reid_score:
                continue

            reid_rows.append((db, reid_score))

        if len(reid_rows) == 0:
            continue

        # 先按 ReID 排序，再取 top-k
        reid_rows.sort(key=lambda x: x[1], reverse=True)
        topk_rows = reid_rows[:cfg.reid_topk]

        for db, reid_score in topk_rows:
            fb = feat_b[db.track_id]

            if not is_valid_geometry_candidate(fa, fb, cfg):
                continue

            refine_score, detail = geometry_refine_similarity_breakdown(fa, fb, cfg)
            if refine_score < cfg.min_refine_score:
                continue

            cand = {
                "track_id_a": da.track_id,
                "track_id_b": db.track_id,
                "score_reid": float(reid_score),
                "score_total": float(refine_score),   # 保持旧字段名，兼容可视化脚本
                "score_refine": float(refine_score),

                "rank_a": float(fa["rank"]),
                "rank_b": float(fb["rank"]),
                "y_norm_a": float(fa["y_norm"]),
                "y_norm_b": float(fb["y_norm"]),
                "h_norm_a": float(fa["h_norm"]),
                "h_norm_b": float(fb["h_norm"]),
                "area_norm_a": float(fa["area_norm"]),
                "area_norm_b": float(fb["area_norm"]),

                "s_rank": float(detail["s_rank"]),
                "s_y": float(detail["s_y"]),
                "s_pos": float(detail["s_pos"]),
                "s_area": float(detail["s_area"]),
                "s_h": float(detail["s_h"]),
                "s_dist": float(detail["s_dist"]),

                # 额外导出，便于你调试速度门控
                "speed_sign_a": int(da.speed_sign),
                "speed_sign_b": int(db.speed_sign),
                "speed_proj_a": float(da.speed_proj) if not np.isnan(da.speed_proj) else np.nan,
                "speed_proj_b": float(db.speed_proj) if not np.isnan(db.speed_proj) else np.nan,
                "speed_dx_a": float(da.speed_dx) if not np.isnan(da.speed_dx) else np.nan,
                "speed_dy_a": float(da.speed_dy) if not np.isnan(da.speed_dy) else np.nan,
                "speed_dx_b": float(db.speed_dx) if not np.isnan(db.speed_dx) else np.nan,
                "speed_dy_b": float(db.speed_dy) if not np.isnan(db.speed_dy) else np.nan,
            }
            all_candidates.append(cand)

    # 一对一贪心：按 score_total 再按 score_reid
    all_candidates.sort(key=lambda x: (x["score_total"], x["score_reid"]), reverse=True)

    used_a = set()
    used_b = set()
    matched = []

    for c in all_candidates:
        ta = c["track_id_a"]
        tb = c["track_id_b"]
        if ta in used_a or tb in used_b:
            continue
        matched.append(c)
        used_a.add(ta)
        used_b.add(tb)

    return matched


# =========================
# 13. 主匹配器
# =========================
class FanSeqChainMatcher:
    def __init__(self, cfg: FanSeqChainConfig):
        self.cfg = cfg
        self.histories_a = defaultdict(lambda: deque(maxlen=cfg.window_size))
        self.histories_b = defaultdict(lambda: deque(maxlen=cfg.window_size))

    def prepare_speed_direction(self, tracks_a_by_frame, tracks_b_by_frame):
        if not self.cfg.enable_speed_direction:
            return
        annotate_speed_direction(tracks_a_by_frame, self.cfg.speed_axis_a, self.cfg)
        annotate_speed_direction(tracks_b_by_frame, self.cfg.speed_axis_b, self.cfg)

    def match_one_frame(self, frame_id, dets_a, dets_b, img_w_a, img_h_a, img_w_b, img_h_b):
        cur_a = keep_top_size_ratio(dets_a, self.cfg)
        cur_b = keep_top_size_ratio(dets_b, self.cfg)

        if len(cur_a) == 0 or len(cur_b) == 0:
            return []

        seq_a, feat_a, _ = build_fan_sequence_features(
            cur_a, self.histories_a, img_w_a, img_h_a, self.cfg
        )
        seq_b, feat_b, _ = build_fan_sequence_features(
            cur_b, self.histories_b, img_w_b, img_h_b, self.cfg
        )

        # =========================
        # DEBUG: 打印指定帧、指定A目标与B所有目标的ReID分数
        # =========================
        debug_frame = 224
        debug_a_ids = {72,62,40}

        if frame_id == debug_frame:
            print("\n" + "=" * 120)
            print(f"[DEBUG] frame_id = {frame_id}")
            print(f"[DEBUG] A侧有效序列id: {[d.track_id for d in seq_a]}")
            print(f"[DEBUG] B侧有效序列id: {[d.track_id for d in seq_b]}")
            print("=" * 120)

            for da in seq_a:
                if da.track_id not in debug_a_ids:
                    continue

                fa = feat_a[da.track_id]
                print(f"\n[DEBUG] 查看 A:{da.track_id} 与 B中所有目标的分数")
                print("-" * 120)

                rows = []
                for db in seq_b:
                    fb = feat_b[db.track_id]

                    # 1) 速度方向门控
                    speed_ok = is_valid_speed_candidate(da, db, self.cfg)

                    # 2) ReID
                    reid_score = reid_similarity(da.reid_feat, db.reid_feat)

                    # 3) 几何门控
                    geom_ok = is_valid_geometry_candidate(fa, fb, self.cfg)

                    # 4) 几何分
                    refine_score, detail = geometry_refine_similarity_breakdown(fa, fb, self.cfg)

                    # 5) 最终是否可进入候选
                    final_ok = True
                    if not speed_ok:
                        final_ok = False
                    if reid_score is None or reid_score < self.cfg.min_reid_score:
                        final_ok = False
                    if not geom_ok:
                        final_ok = False
                    if refine_score < self.cfg.min_refine_score:
                        final_ok = False

                    rows.append({
                        "b_id": db.track_id,
                        "speed_ok": speed_ok,
                        "reid_score": reid_score,
                        "geom_ok": geom_ok,
                        "refine_score": refine_score,
                        "final_ok": final_ok,
                           # ===== refine 小分 =====
                        "s_rank": detail["s_rank"],
                        "s_y": detail["s_y"],
                        "s_pos": detail["s_pos"],
                        "s_area": detail["s_area"],
                        "s_h": detail["s_h"],
                        "s_dist": detail["s_dist"],
                        "speed_sign_a": da.speed_sign,
                        "speed_sign_b": db.speed_sign,
                        "speed_proj_a": da.speed_proj,
                        "speed_proj_b": db.speed_proj,
                    })

                # 按 ReID 从高到低排，方便看
                def sort_key(x):
                    s = x["reid_score"]
                    return -1.0 if s is None else s

                rows.sort(key=sort_key, reverse=True)

                for row in rows:
                    reid_str = "None" if row["reid_score"] is None else f"{row['reid_score']:.4f}"
                    refine_str = f"{row['refine_score']:.4f}"

                    proj_a_str = "nan" if np.isnan(row["speed_proj_a"]) else f"{row['speed_proj_a']:.4f}"
                    proj_b_str = "nan" if np.isnan(row["speed_proj_b"]) else f"{row['speed_proj_b']:.4f}"

                    print(
                        f"frame={frame_id} | "
                        f"A:{da.track_id} -> B:{row['b_id']} | "
                        f"reid={reid_str} | "
                        f"speed_ok={row['speed_ok']} | "
                        f"geom_ok={row['geom_ok']} | "
                        f"refine={refine_str} | "
                        f"[s_rank={row['s_rank']:.4f}, "
                        f"s_y={row['s_y']:.4f}, "
                        f"s_pos={row['s_pos']:.4f}, "
                        f"s_area={row['s_area']:.4f}, "
                        f"s_h={row['s_h']:.4f}, "
                        f"s_dist={row['s_dist']:.4f}] | "
                        f"final_ok={row['final_ok']} | "
                        f"signA={row['speed_sign_a']} projA={proj_a_str} | "
                        f"signB={row['speed_sign_b']} projB={proj_b_str}"
                    )

                print("-" * 120)




        frame_matches = match_by_reid_topk_then_geometry(seq_a, seq_b, feat_a, feat_b, self.cfg)

        results = []
        for m in frame_matches:
            out = dict(m)
            out["frame_id"] = int(frame_id)
            results.append(out)
        return results

    def update_histories(self, dets_a, dets_b):
        for d in dets_a:
            self.histories_a[d.track_id].append(d)
        for d in dets_b:
            self.histories_b[d.track_id].append(d)

    def run(self, tracks_a_by_frame, tracks_b_by_frame, img_size_a, img_size_b):
        img_w_a, img_h_a = img_size_a
        img_w_b, img_h_b = img_size_b

        # 只在开关开启时才计算速度方向
        self.prepare_speed_direction(tracks_a_by_frame, tracks_b_by_frame)

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

            # 更新历史放在本帧之后
            self.update_histories(dets_a, dets_b)

            if frame_id % 100 == 0:
                print(f"[Info] 已处理到 frame {frame_id}, 当前累计匹配 {len(results)} 条")

        return results


# =========================
# 14. 保存结果
# =========================
def save_matches_csv(matches, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True) if os.path.dirname(save_path) else None

    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "frame_id",
            "track_id_a",
            "track_id_b",
            "score_total",   # 保持旧字段名
            "score_reid",
            "score_refine",

            "rank_a", "rank_b",
            "y_norm_a", "y_norm_b",
            "h_norm_a", "h_norm_b",
            "area_norm_a", "area_norm_b",

            "s_rank", "s_y", "s_pos", "s_area", "s_h", "s_dist",

            # 额外附加，不影响旧脚本按字段读取
            "speed_sign_a", "speed_sign_b",
            "speed_proj_a", "speed_proj_b",
            "speed_dx_a", "speed_dy_a",
            "speed_dx_b", "speed_dy_b",
        ])

        for m in matches:
            writer.writerow([
                m["frame_id"],
                m["track_id_a"],
                m["track_id_b"],
                m["score_total"],
                m["score_reid"],
                m["score_refine"],

                m["rank_a"], m["rank_b"],
                m["y_norm_a"], m["y_norm_b"],
                m["h_norm_a"], m["h_norm_b"],
                m["area_norm_a"], m["area_norm_b"],

                m["s_rank"], m["s_y"], m["s_pos"], m["s_area"], m["s_h"], m["s_dist"],

                m["speed_sign_a"], m["speed_sign_b"],
                m["speed_proj_a"], m["speed_proj_b"],
                m["speed_dx_a"], m["speed_dy_a"],
                m["speed_dx_b"], m["speed_dy_b"],
            ])

    print(f"[Info] 匹配结果已保存到: {save_path}")


# =========================
# 15. 主程序
# =========================
if __name__ == "__main__":
    # -------- 输入 --------
    txt_a = "experient_fig/doublesight_reid/droneA_tracks.txt"
    txt_b = "experient_fig/doublesight_reid/droneB_tracks.txt"

    reid_npz_a = "experient_fig/doublesight_reid/droneA_reid.npz"
    reid_npz_b = "experient_fig/doublesight_reid/droneB_reid.npz"

    # Kalman sidecar（不开速度方向时可为 None）
    # kf_txt_a = None
    # kf_txt_b = None
    # 启用速度方向时改成：
    kf_txt_a = "experient_fig/doublesight_reidkalm/droneA_kf.txt"
    kf_txt_b = "experient_fig/doublesight_reidkalm/droneB_kf.txt"

    txt_format = "custom"
    cls_filter = None

    # -------- 加载 sidecar --------
    reid_a = load_reid_npz(reid_npz_a)
    reid_b = load_reid_npz(reid_npz_b)

    kf_map_a = load_kf_txt(kf_txt_a) if kf_txt_a is not None else {}
    kf_map_b = load_kf_txt(kf_txt_b) if kf_txt_b is not None else {}

    # -------- 加载轨迹 --------
    tracks_a = load_track_txt(
        txt_a,
        txt_format=txt_format,
        cls_filter=cls_filter,
        reid_map=reid_a,
        kf_map=kf_map_a
    )
    tracks_b = load_track_txt(
        txt_b,
        txt_format=txt_format,
        cls_filter=cls_filter,
        reid_map=reid_b,
        kf_map=kf_map_b
    )

    print(f"A 视角共 {len(tracks_a)} 帧")
    print(f"B 视角共 {len(tracks_b)} 帧")

    # -------- 画面尺寸 --------
    img_size_a = infer_canvas_size(tracks_a)
    img_size_b = infer_canvas_size(tracks_b)
    print(f"[Info] A尺寸推断: {img_size_a}")
    print(f"[Info] B尺寸推断: {img_size_b}")

    # -------- 配置 --------
    cfg = FanSeqChainConfig(
        window_size=20,

        size_keep_ratio=0.50,
        min_conf=0.0,

        vp_center_mix=0.75,
        vp_y_offset_ratio=0.15,

        reid_topk=5,
        min_reid_score=0.50,

        min_refine_score=0.45,

        max_rank_gap=0.40,
        max_y_gap=0.30,
        max_h_ratio=3.0,
        max_area_ratio=6.0,

        near_y_thresh=0.55,
        near_max_rank_gap=0.22,
        near_max_y_gap=0.12,
        near_max_h_ratio=1.80,
        near_max_area_ratio=2.50,

        refine_pos_weight=0.35,
        refine_area_weight=0.20,
        refine_h_weight=0.20,
        refine_dist_weight=0.25,

        # =========================
        # 速度方向开关
        # False: 与原逻辑一致
        # True : 加一道“速度方向不一致就不匹配”的门控
        # =========================
        enable_speed_direction=True,

        # 下面只有 enable_speed_direction=True 时才生效
        speed_axis_a=(0.0, 1.0),
        speed_axis_b=(0.0, 1.0),
        speed_relation="same",
        min_speed_proj_abs=1.0,
        allow_unknown_speed_direction=True,
    )

    # -------- 执行匹配 --------
    matcher = FanSeqChainMatcher(cfg)
    matches = matcher.run(tracks_a, tracks_b, img_size_a, img_size_b)

    # -------- 保存 --------
    save_matches_csv(matches, "results/exp13_reidkalam/cross_match.csv")