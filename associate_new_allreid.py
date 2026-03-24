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
        reid_map[key] = feat

    print(f"[Info] 已加载 ReID 特征: {npz_path}, 共 {len(reid_map)} 条")
    return reid_map


# =========================
# 3. 读取轨迹 txt
# =========================
def load_track_txt(txt_path, txt_format="custom", cls_filter=None, reid_map=None):
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

                det = Detection(
                    frame_id=frame_id,
                    track_id=track_id,
                    cls_id=cls_id,
                    x1=x1, y1=y1, x2=x2, y2=y2,
                    cx=cx, cy_bottom=cy_bottom,
                    conf=conf,
                    reid_feat=feat
                )
                tracks_by_frame[frame_id].append(det)

            except Exception as e:
                raise ValueError(f"解析 {txt_path} 第 {ln} 行失败：{line}\n错误信息：{e}")

    return dict(sorted(tracks_by_frame.items(), key=lambda x: x[0]))


# =========================
# 4. 工具函数
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


# =========================
# 5. 配置
# =========================
@dataclass
class FanSeqChainConfig:
    # 历史窗口（这里只为保留历史描述能力）
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
# 6. 只保留面积前50%
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
# 7. 构造角度序列 + 描述子
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
# 8. 几何筛选分数
# 只用于 ReID top-k 内部筛选
# =========================
def geometry_refine_similarity_breakdown(fa, fb, cfg: FanSeqChainConfig):
    # 位置相似：这里用 rank + y 共同表示
    s_rank = exp_sim(abs(fa["rank"] - fb["rank"]), 5.0)
    s_y = exp_sim(abs(fa["y_norm"] - fb["y_norm"]), 6.0)
    s_pos = 0.5 * s_rank + 0.5 * s_y

    # 面积相似
    s_area = exp_sim(safe_log_ratio(fa["area_norm"], fb["area_norm"]), 1.8)

    # 高度相似
    s_h = exp_sim(safe_log_ratio(fa["h_norm"], fb["h_norm"]), 2.6)

    # 邻域距离相似
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
# 9. 几何门控
# 只在 ReID top-k 内部使用
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
# 10. ReID top-k + 几何筛选
# 说明：
#   1) 先按 ReID 排序
#   2) 每个 A 只取前 top-k 个 B
#   3) 在 top-k 内按几何筛选分数排序
#   4) 最终做一对一贪心
# =========================
def match_by_reid_topk_then_geometry(seq_a, seq_b, feat_a, feat_b, cfg: FanSeqChainConfig):
    if len(seq_a) == 0 or len(seq_b) == 0:
        return []

    all_candidates = []

    for da in seq_a:
        fa = feat_a[da.track_id]

        reid_rows = []
        for db in seq_b:
            reid_score = reid_similarity(da.reid_feat, db.reid_feat)
            if reid_score is None:
                continue
            if reid_score < cfg.min_reid_score:
                continue

            reid_rows.append({
                "db": db,
                "reid_score": float(reid_score),
            })

        if len(reid_rows) == 0:
            continue

        reid_rows.sort(key=lambda x: x["reid_score"], reverse=True)
        shortlist = reid_rows[:cfg.reid_topk]

        for rank_in_topk, row in enumerate(shortlist, start=1):
            db = row["db"]
            reid_score = row["reid_score"]
            fb = feat_b[db.track_id]

            geo_valid = is_valid_geometry_candidate(fa, fb, cfg)
            refine_score, refine_detail = geometry_refine_similarity_breakdown(fa, fb, cfg)

            if not geo_valid:
                continue
            if refine_score < cfg.min_refine_score:
                continue

            all_candidates.append({
                "frame_id": da.frame_id,
                "track_id_a": da.track_id,
                "track_id_b": db.track_id,
                "reid_score": float(reid_score),
                "refine_score": float(refine_score),
                "topk_rank_reid": int(rank_in_topk),

                "score_pos": float(refine_detail["s_pos"]),
                "score_area": float(refine_detail["s_area"]),
                "score_h": float(refine_detail["s_h"]),
                "score_dist": float(refine_detail["s_dist"]),
                "score_rank": float(refine_detail["s_rank"]),
                "score_y": float(refine_detail["s_y"]),
            })

    if len(all_candidates) == 0:
        return []

    # 先按几何筛选分数排序，再用 reid 分数做次级排序
    all_candidates.sort(
        key=lambda x: (x["refine_score"], x["reid_score"]),
        reverse=True
    )

    used_a = set()
    used_b = set()
    matches = []

    for cand in all_candidates:
        a_id = cand["track_id_a"]
        b_id = cand["track_id_b"]

        if a_id in used_a or b_id in used_b:
            continue

        used_a.add(a_id)
        used_b.add(b_id)
        matches.append({
            "frame_id": cand["frame_id"],
            "track_id_a": a_id,
            "track_id_b": b_id,
            "score_total": round(float(cand["refine_score"]), 6),   # 最终匹配依据：几何筛选分
            "score_reid": round(float(cand["reid_score"]), 6),      # ReID只用于 top-k shortlist
            "score_pos": round(float(cand["score_pos"]), 6),
            "score_area": round(float(cand["score_area"]), 6),
            "score_h": round(float(cand["score_h"]), 6),
            "score_dist": round(float(cand["score_dist"]), 6),
            "score_rank": round(float(cand["score_rank"]), 6),
            "score_y": round(float(cand["score_y"]), 6),
            "topk_rank_reid": int(cand["topk_rank_reid"]),
        })

    return matches


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
        self.update_histories(dets_a, dets_b)

        valid_a = keep_top_size_ratio(dets_a, self.cfg)
        valid_b = keep_top_size_ratio(dets_b, self.cfg)

        if len(valid_a) == 0 or len(valid_b) == 0:
            return []

        seq_a, feat_a, _ = build_fan_sequence_features(valid_a, self.hist_A, img_w_a, img_h_a, self.cfg)
        seq_b, feat_b, _ = build_fan_sequence_features(valid_b, self.hist_B, img_w_b, img_h_b, self.cfg)

        if len(seq_a) == 0 or len(seq_b) == 0:
            return []

        # =========================
        # DEBUG: 打印指定帧、指定A id 的 ReID top-k + 几何筛选结果
        # =========================
        debug_frame = 224
        debug_a_ids = {44, 77}

        if frame_id == debug_frame:
            print("\n" + "=" * 140)
            print(f"[DEBUG] frame_id = {frame_id}")
            print(f"[DEBUG] A侧有效序列id: {[d.track_id for d in seq_a]}")
            print(f"[DEBUG] B侧有效序列id: {[d.track_id for d in seq_b]}")
            print("=" * 140)

            for da in seq_a:
                if da.track_id not in debug_a_ids:
                    continue

                fa = feat_a[da.track_id]
                print(f"\n[DEBUG] 查看 A:{da.track_id} 的 ReID top-{self.cfg.reid_topk} 及几何筛选")
                print("-" * 140)

                rows = []
                for db in seq_b:
                    reid_score = reid_similarity(da.reid_feat, db.reid_feat)
                    if reid_score is None:
                        continue

                    fb = feat_b[db.track_id]
                    geo_valid = is_valid_geometry_candidate(fa, fb, self.cfg)
                    refine_score, detail = geometry_refine_similarity_breakdown(fa, fb, self.cfg)

                    rows.append({
                        "b_id": db.track_id,
                        "reid_score": reid_score,
                        "geo_valid": geo_valid,
                        "refine_score": refine_score,
                        "detail": detail
                    })

                rows.sort(key=lambda x: x["reid_score"], reverse=True)
                rows = rows[:self.cfg.reid_topk]

                for k, row in enumerate(rows, start=1):
                    detail = row["detail"]
                    print(
                        f"Top{k} | "
                        f"A:{da.track_id} -> B:{row['b_id']} | "
                        f"reid={row['reid_score']:.4f} | "
                        f"geo_valid={row['geo_valid']} | "
                        f"refine={row['refine_score']:.4f} | "
                        f"pos={detail['s_pos']:.4f}, "
                        f"area={detail['s_area']:.4f}, "
                        f"h={detail['s_h']:.4f}, "
                        f"dist={detail['s_dist']:.4f}, "
                        f"rank={detail['s_rank']:.4f}, "
                        f"y={detail['s_y']:.4f}"
                    )

        matches = match_by_reid_topk_then_geometry(seq_a, seq_b, feat_a, feat_b, self.cfg)

        for m in matches:
            m["num_valid_a"] = len(seq_a)
            m["num_valid_b"] = len(seq_b)

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
            "score_reid",
            "score_pos",
            "score_area",
            "score_h",
            "score_dist",
            "score_rank",
            "score_y",
            "topk_rank_reid",
            "num_valid_a",
            "num_valid_b",
        ])
        for m in matches:
            writer.writerow([
                m["frame_id"],
                m["track_id_a"],
                m["track_id_b"],
                m["score_total"],
                m["score_reid"],
                m["score_pos"],
                m["score_area"],
                m["score_h"],
                m["score_dist"],
                m["score_rank"],
                m["score_y"],
                m["topk_rank_reid"],
                m["num_valid_a"],
                m["num_valid_b"],
            ])

    print(f"匹配结果已保存到: {save_path}")


# =========================
# 13. 主程序
# =========================
if __name__ == "__main__":
    # -------- 输入 --------
    txt_a = "experient_fig/doublesight_reid/droneA_tracks.txt"
    txt_b = "experient_fig/doublesight_reid/droneB_tracks.txt"

    reid_npz_a = "experient_fig/doublesight_reid/droneA_reid.npz"
    reid_npz_b = "experient_fig/doublesight_reid/droneB_reid.npz"

    txt_format = "custom"
    cls_filter = None

    # -------- 加载 ReID --------
    reid_a = load_reid_npz(reid_npz_a)
    reid_b = load_reid_npz(reid_npz_b)

    # -------- 加载轨迹 --------
    tracks_a = load_track_txt(txt_a, txt_format=txt_format, cls_filter=cls_filter, reid_map=reid_a)
    tracks_b = load_track_txt(txt_b, txt_format=txt_format, cls_filter=cls_filter, reid_map=reid_b)

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

        # 保留面积较大的目标，ReID 对小目标通常不稳定
        size_keep_ratio=0.50,
        min_conf=0.0,

        vp_center_mix=0.75,
        vp_y_offset_ratio=0.15,

        # 先按 ReID 取前5
        reid_topk=5,
        min_reid_score=0.50,

        # top5 内再用几何筛选
        min_refine_score=0.45,

        # 几何门控
        max_rank_gap=0.40,
        max_y_gap=0.30,
        max_h_ratio=3.0,
        max_area_ratio=6.0,

        near_y_thresh=0.55,
        near_max_rank_gap=0.22,
        near_max_y_gap=0.12,
        near_max_h_ratio=1.80,
        near_max_area_ratio=2.50,

        # 几何筛选权重
        refine_pos_weight=0.35,
        refine_area_weight=0.20,
        refine_h_weight=0.20,
        refine_dist_weight=0.25,
    )

    # -------- 建立匹配器 --------
    matcher = FanSeqChainMatcher(cfg)

    # -------- 执行匹配 --------
    matches = matcher.run(tracks_a, tracks_b, img_size_a, img_size_b)

    # -------- 保存结果 --------
    save_matches_csv(matches, "results/exp12_new_new_allreid/reid_top5_then_geometry_match.csv")