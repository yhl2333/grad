







import os
import csv
import math
from itertools import combinations
from dataclasses import dataclass
from collections import defaultdict
from typing import Optional, Iterable

import numpy as np


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
    pred_cx: float = np.nan
    pred_cy_bottom: float = np.nan
    speed_dx: float = np.nan
    speed_dy: float = np.nan
    speed_proj: float = np.nan
    speed_sign: int = 0

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
        feat = feat / norm if norm > 1e-12 else None
        reid_map[key] = feat

    print(f"[Info] 已加载 ReID 特征(仅接口兼容，不参与匹配): {npz_path}, 共 {len(reid_map)} 条")
    return reid_map


def load_kf_txt(kf_txt_path):
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
            kf_map[(frame_id, track_id)] = {"pred_cx": pred_cx, "pred_cy_bottom": pred_cy_bottom}

    print(f"[Info] 已加载 Kalman sidecar(仅接口兼容，不参与匹配): {kf_txt_path}, 共 {len(kf_map)} 条")
    return kf_map


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

                feat = reid_map.get((frame_id, track_id), None) if reid_map is not None else None

                pred_cx = np.nan
                pred_cy_bottom = np.nan
                if kf_map is not None:
                    kf_row = kf_map.get((frame_id, track_id), None)
                    if kf_row is not None:
                        pred_cx = float(kf_row["pred_cx"])
                        pred_cy_bottom = float(kf_row["pred_cy_bottom"])

                det = Detection(
                    frame_id=frame_id, track_id=track_id, cls_id=cls_id,
                    x1=x1, y1=y1, x2=x2, y2=y2, cx=cx, cy_bottom=cy_bottom,
                    conf=conf, reid_feat=feat, pred_cx=pred_cx, pred_cy_bottom=pred_cy_bottom,
                )
                tracks_by_frame[frame_id].append(det)

            except Exception as e:
                raise ValueError(f"解析 {txt_path} 第 {ln} 行失败：{line}\n错误信息：{e}")

    return dict(sorted(tracks_by_frame.items(), key=lambda x: x[0]))


def infer_canvas_size(tracks_by_frame):
    max_x = 0.0
    max_y = 0.0
    for _, dets in tracks_by_frame.items():
        for d in dets:
            max_x = max(max_x, d.x2, d.cx)
            max_y = max(max_y, d.y2, d.cy_bottom)
    w = max(1, int(np.ceil(max_x + 1)))
    h = max(1, int(np.ceil(max_y + 1)))
    return w, h


def normalize_vec2(v):
    arr = np.asarray(v, dtype=np.float32).reshape(2)
    n = float(np.linalg.norm(arr))
    if n < 1e-12:
        return np.array([1.0, 0.0], dtype=np.float32)
    return arr / n


def exp_sim(diff, scale):
    return float(math.exp(-float(diff) * float(scale)))


def safe_log_ratio(a, b, eps=1e-6):
    a = max(float(a), eps)
    b = max(float(b), eps)
    return abs(math.log(a / b))


def wrap_to_2pi(theta):
    return float(theta) % (2.0 * math.pi)


def circular_diff(a, b):
    d = abs(float(a) - float(b))
    return min(d, 2.0 * math.pi - d)


@dataclass
class LocalGraphMatcherConfig:
    min_conf: float = 0.0
    history_frames: int = 3
    min_dir_abs: float = 1.0
    min_speed_proj_abs: float = 1.0
    axis_a: tuple = (1.0, 0.0)
    axis_b: tuple = (0.0, -1.0)
    neighbor_k: int = 4
    allow_unknown_speed: bool = True
    angle_weight: float = 0.4
    dist_weight: float = 0.3
    h_weight: float = 0.15
    area_weight: float = 0.15
    angle_scale: float = 0.9
    dist_scale: float = 2.0
    h_scale: float = 2.2
    area_scale: float = 1.8
    min_match_score: float = 0.60
    print_topk_scores: bool = True
    print_topk_k: int = 5
    print_only_frames: Optional[Iterable[int]] = None
    print_only_track_ids_a: Optional[Iterable[int]] = None
    print_only_above_threshold: bool = False


def build_track_history_index(tracks_by_frame):
    hist = defaultdict(dict)
    for frame_id, dets in tracks_by_frame.items():
        for d in dets:
            hist[int(d.track_id)][int(frame_id)] = d
    return hist


def compute_motion_from_history(det, track_hist_map, history_frames):
    tid = int(det.track_id)
    fid = int(det.frame_id)
    prev_pts = []

    if tid not in track_hist_map:
        return np.nan, np.nan, np.nan, np.nan

    for back in range(1, int(history_frames) + 1):
        prev_fid = fid - back
        prev_det = track_hist_map[tid].get(prev_fid, None)
        if prev_det is not None:
            prev_pts.append([float(prev_det.cx), float(prev_det.cy_bottom)])

    if len(prev_pts) == 0:
        return np.nan, np.nan, np.nan, np.nan

    prev_pts = np.asarray(prev_pts, dtype=np.float32)
    prev_mean_x = float(np.mean(prev_pts[:, 0]))
    prev_mean_y = float(np.mean(prev_pts[:, 1]))
    dx = float(det.cx - prev_mean_x)
    dy = float(det.cy_bottom - prev_mean_y)
    return prev_mean_x, prev_mean_y, dx, dy


def annotate_motion(tracks_by_frame, track_hist_map, axis, cfg: LocalGraphMatcherConfig):
    axis = normalize_vec2(axis)
    for _, dets in tracks_by_frame.items():
        for d in dets:
            _, _, dx, dy = compute_motion_from_history(d, track_hist_map, cfg.history_frames)
            d.speed_dx = float(dx) if not np.isnan(dx) else np.nan
            d.speed_dy = float(dy) if not np.isnan(dy) else np.nan

            if np.isnan(dx) or np.isnan(dy):
                d.speed_proj = np.nan
                d.speed_sign = 0
                continue

            proj = float(dx * axis[0] + dy * axis[1])
            d.speed_proj = proj
            d.speed_sign = 0 if abs(proj) < float(cfg.min_speed_proj_abs) else (1 if proj > 0 else -1)


def build_local_graph_descriptors(dets, axis, cfg: LocalGraphMatcherConfig):
    dets = [d for d in dets if d.conf >= cfg.min_conf]
    if len(dets) == 0:
        return {}, []

    axis = normalize_vec2(axis)
    perp = np.array([axis[1], -axis[0]], dtype=np.float32)
    pts = np.asarray([[d.cx, d.cy_bottom] for d in dets], dtype=np.float32)
    desc_by_tid = {}

    for i, d in enumerate(dets):
        dist_rows = []
        for j, _ in enumerate(dets):
            if i == j:
                continue
            raw_dist = float(np.linalg.norm(pts[j] - pts[i]))
            dist_rows.append((j, raw_dist))

        dist_rows.sort(key=lambda x: x[1])
        nn = dist_rows[:cfg.neighbor_k]

        if len(nn) > 0:
            local_scale = float(np.mean([x[1] for x in nn]))
        else:
            local_scale = max(float(d.h), math.sqrt(max(float(d.area), 1.0)), 1.0)
        local_scale = max(local_scale, 1e-6)

        neighbors = []
        for j, raw_dist in nn:
            rel = pts[j] - pts[i]
            along = float(rel[0] * axis[0] + rel[1] * axis[1])
            ccw = float(rel[0] * perp[0] + rel[1] * perp[1])

            angle = wrap_to_2pi(math.atan2(ccw, along))
            dist_norm = float(raw_dist / local_scale)
            h_ratio = float(max(dets[j].h, 1e-6) / max(d.h, 1e-6))
            area_ratio = float(max(dets[j].area, 1e-6) / max(d.area, 1e-6))

            neighbors.append({
                "track_id": int(dets[j].track_id),
                "angle": float(angle),
                "dist_norm": float(dist_norm),
                "h_ratio": float(h_ratio),
                "area_ratio": float(area_ratio),
            })

        neighbors.sort(key=lambda x: x["angle"])
        while len(neighbors) < cfg.neighbor_k:
            neighbors.append(None)

        desc_by_tid[int(d.track_id)] = {
            "track_id": int(d.track_id),
            "speed_sign": int(d.speed_sign),
            "speed_proj": float(d.speed_proj) if not np.isnan(d.speed_proj) else np.nan,
            "speed_dx": float(d.speed_dx) if not np.isnan(d.speed_dx) else np.nan,
            "speed_dy": float(d.speed_dy) if not np.isnan(d.speed_dy) else np.nan,
            "neighbors": neighbors,
            "self_h": float(d.h),
            "self_area": float(d.area),
        }

    return desc_by_tid, dets


def speed_gate_pass(desc_a, desc_b, cfg: LocalGraphMatcherConfig):
    sa = int(desc_a["speed_sign"])
    sb = int(desc_b["speed_sign"])
    if sa == 0 or sb == 0:
        return bool(cfg.allow_unknown_speed)
    return sa == sb


def neighbor_pair_score(na, nb, cfg: LocalGraphMatcherConfig):
    if na is None and nb is None:
        detail = {"s_angle_pair": 1.0, "s_dist_pair": 1.0, "s_h_pair": 1.0, "s_area_pair": 1.0, "score_pair": 1.0}
        return 1.0, detail
    if na is None or nb is None:
        detail = {"s_angle_pair": 0.2, "s_dist_pair": 0.2, "s_h_pair": 0.2, "s_area_pair": 0.2, "score_pair": 0.2}
        return 0.2, detail

    s_angle = exp_sim(circular_diff(na["angle"], nb["angle"]), cfg.angle_scale)
    s_dist = exp_sim(abs(na["dist_norm"] - nb["dist_norm"]), cfg.dist_scale)
    s_h = exp_sim(safe_log_ratio(na["h_ratio"], nb["h_ratio"]), cfg.h_scale)
    s_area = exp_sim(safe_log_ratio(na["area_ratio"], nb["area_ratio"]), cfg.area_scale)

    score_pair = (
        cfg.angle_weight * s_angle +
        cfg.dist_weight * s_dist +
        cfg.h_weight * s_h +
        cfg.area_weight * s_area
    )
    detail = {
        "s_angle_pair": float(s_angle),
        "s_dist_pair": float(s_dist),
        "s_h_pair": float(s_h),
        "s_area_pair": float(s_area),
        "score_pair": float(score_pair),
    }
    return float(score_pair), detail


def best_ordered_subsequence_match(pair_score_matrix):
    n = len(pair_score_matrix)
    if n == 0:
        return {"sum_best3": 0.0, "penalized_pair_score": 0.0, "adjusted_sum": 0.0, "matched_pairs": [], "penalized_pair": None}

    keep = max(n - 1, 1)
    best_adjusted_sum = -1e18
    best_pairs = []
    best_penalized = None

    idxs = list(range(n))
    for comb_a in combinations(idxs, keep):
        for comb_b in combinations(idxs, keep):
            sum_keep = 0.0
            cur_pairs = []
            for ia, ib in zip(comb_a, comb_b):
                sum_keep += float(pair_score_matrix[ia][ib])
                cur_pairs.append((ia, ib))

            rem_a = [x for x in idxs if x not in comb_a]
            rem_b = [x for x in idxs if x not in comb_b]

            penalized_pair = None
            penalized_score = 0.0
            raw_score = 0.0
            if len(rem_a) == 1 and len(rem_b) == 1:
                pa, pb = rem_a[0], rem_b[0]
                raw_score = float(pair_score_matrix[pa][pb])
                penalized_score = raw_score * 0.5
                penalized_pair = (pa, pb)

            adjusted_sum = sum_keep + penalized_score
            if adjusted_sum > best_adjusted_sum:
                best_adjusted_sum = adjusted_sum
                best_pairs = cur_pairs
                best_penalized = {
                    "pair": penalized_pair,
                    "raw_score": raw_score,
                    "score_after_penalty": penalized_score,
                }

    return {
        "sum_best3": float(sum(pair_score_matrix[i][j] for i, j in best_pairs)) if len(best_pairs) > 0 else 0.0,
        "penalized_pair_score": float(best_penalized["score_after_penalty"]) if best_penalized is not None else 0.0,
        "adjusted_sum": float(best_adjusted_sum if best_adjusted_sum > -1e17 else 0.0),
        "matched_pairs": best_pairs,
        "penalized_pair": best_penalized,
    }


def local_graph_similarity(desc_a, desc_b, cfg: LocalGraphMatcherConfig):
    if not speed_gate_pass(desc_a, desc_b, cfg):
        return 0.0, {
            "rejected": True,
            "reason": "speed_gate",
            "speed_sign_a": int(desc_a["speed_sign"]),
            "speed_sign_b": int(desc_b["speed_sign"]),
        }

    n = min(len(desc_a["neighbors"]), len(desc_b["neighbors"]))
    pair_score_matrix = [[0.0 for _ in range(n)] for _ in range(n)]
    pair_detail_matrix = [[None for _ in range(n)] for _ in range(n)]

    for i in range(n):
        for j in range(n):
            s_pair, detail = neighbor_pair_score(desc_a["neighbors"][i], desc_b["neighbors"][j], cfg)
            pair_score_matrix[i][j] = float(s_pair)
            pair_detail_matrix[i][j] = detail

    best_match = best_ordered_subsequence_match(pair_score_matrix)
    matched_pairs = best_match["matched_pairs"]
    penalized_pair = best_match["penalized_pair"]["pair"] if best_match["penalized_pair"] is not None else None

    used_details = []
    for ia, ib in matched_pairs:
        d = dict(pair_detail_matrix[ia][ib])
        d["pair_a_idx"] = ia
        d["pair_b_idx"] = ib
        d["is_penalized_pair"] = False
        d["score_pair_after_penalty"] = float(d["score_pair"])
        used_details.append(d)

    if penalized_pair is not None:
        pa, pb = penalized_pair
        d = dict(pair_detail_matrix[pa][pb])
        d["pair_a_idx"] = pa
        d["pair_b_idx"] = pb
        d["is_penalized_pair"] = True
        d["score_pair_after_penalty"] = float(best_match["penalized_pair"]["score_after_penalty"])
        used_details.append(d)

    if len(used_details) == 0:
        score_angle = score_dist = score_h = score_area = 0.0
    else:
        score_angle = float(np.mean([x["s_angle_pair"] for x in used_details]))
        score_dist = float(np.mean([x["s_dist_pair"] for x in used_details]))
        score_h = float(np.mean([x["s_h_pair"] for x in used_details]))
        score_area = float(np.mean([x["s_area_pair"] for x in used_details]))

    total = float(best_match["adjusted_sum"] / max(n, 1))

    detail = {
        "rejected": False,
        "speed_sign_a": int(desc_a["speed_sign"]),
        "speed_sign_b": int(desc_b["speed_sign"]),
        "speed_proj_a": float(desc_a["speed_proj"]) if not np.isnan(desc_a["speed_proj"]) else np.nan,
        "speed_proj_b": float(desc_b["speed_proj"]) if not np.isnan(desc_b["speed_proj"]) else np.nan,
        "score_angle": float(score_angle),
        "score_dist": float(score_dist),
        "score_h": float(score_h),
        "score_area": float(score_area),
        "score_total": float(total),
        "pair_details": used_details,
        "neighbors_a": [None if n is None else int(n["track_id"]) for n in desc_a["neighbors"]],
        "neighbors_b": [None if n is None else int(n["track_id"]) for n in desc_b["neighbors"]],
        "matched_pairs_idx": matched_pairs,
        "penalized_pair_idx": penalized_pair,
        "sum_best3": float(best_match["sum_best3"]),
        "penalized_pair_score": float(best_match["penalized_pair_score"]),
    }
    return float(total), detail


def should_print_frame(frame_id: int, cfg: LocalGraphMatcherConfig):
    if cfg.print_only_frames is None:
        return True
    return int(frame_id) in set(int(x) for x in cfg.print_only_frames)


def should_print_track_a(track_id_a: int, cfg: LocalGraphMatcherConfig):
    if cfg.print_only_track_ids_a is None:
        return True
    return int(track_id_a) in set(int(x) for x in cfg.print_only_track_ids_a)


def print_topk_rows(frame_id, track_id_a, rows, cfg: LocalGraphMatcherConfig):
    k = max(1, int(cfg.print_topk_k))
    rows = rows[:k]

    print(f"[TopK][frame={frame_id}] A:{track_id_a} 的局部图前{k}名候选:")
    if len(rows) == 0:
        print("    无可用候选")
        return

    for idx, row in enumerate(rows, start=1):
        detail = row["detail"]
        proj_a_str = "nan" if np.isnan(detail["speed_proj_a"]) else f"{detail['speed_proj_a']:.2f}"
        proj_b_str = "nan" if np.isnan(detail["speed_proj_b"]) else f"{detail['speed_proj_b']:.2f}"

        neighbors_a = detail.get("neighbors_a", [])
        neighbors_b = detail.get("neighbors_b", [])
        neighbors_a_str = ",".join("None" if x is None else str(x) for x in neighbors_a)
        neighbors_b_str = ",".join("None" if x is None else str(x) for x in neighbors_b)

        matched_pairs_idx = detail.get("matched_pairs_idx", [])
        matched_pairs_str = []
        for ia, ib in matched_pairs_idx:
            a_id = "None" if neighbors_a[ia] is None else str(neighbors_a[ia])
            b_id = "None" if neighbors_b[ib] is None else str(neighbors_b[ib])
            matched_pairs_str.append(f"A{ia+1}:{a_id}->B{ib+1}:{b_id}")

        penalized_pair_idx = detail.get("penalized_pair_idx", None)
        penalized_info = "None"
        if penalized_pair_idx is not None:
            pa, pb = penalized_pair_idx
            a_id = "None" if neighbors_a[pa] is None else str(neighbors_a[pa])
            b_id = "None" if neighbors_b[pb] is None else str(neighbors_b[pb])
            penalized_info = f"A{pa+1}:{a_id}->B{pb+1}:{b_id}*0.5"

        print(
            f"    {idx:>2d}. B:{row['track_id_b']:<6d} "
            f"score={row['score_total']:.4f} | "
            f"angle={detail['score_angle']:.4f} | "
            f"dist={detail['score_dist']:.4f} | "
            f"h={detail['score_h']:.4f} | "
            f"area={detail['score_area']:.4f} | "
            f"signA={detail['speed_sign_a']} projA={proj_a_str} | "
            f"signB={detail['speed_sign_b']} projB={proj_b_str}"
        )
        print(f"        A_neighbors=[{neighbors_a_str}]")
        print(f"        B_neighbors=[{neighbors_b_str}]")
        print(f"        best_subseq_pairs=[{' ; '.join(matched_pairs_str) if matched_pairs_str else 'None'}]")
        print(f"        penalized_pair={penalized_info}")


def match_one_frame(frame_id, dets_a, dets_b, axis_a, axis_b, cfg: LocalGraphMatcherConfig):
    desc_a, valid_a = build_local_graph_descriptors(dets_a, axis_a, cfg)
    desc_b, valid_b = build_local_graph_descriptors(dets_b, axis_b, cfg)

    if len(valid_a) == 0 or len(valid_b) == 0:
        return []

    enable_print_this_frame = bool(cfg.print_topk_scores) and should_print_frame(frame_id, cfg)
    if enable_print_this_frame:
        print("\n" + "=" * 140)
        print(f"[DEBUG] frame={frame_id} | A侧目标数={len(valid_a)} | B侧目标数={len(valid_b)}")
        print("=" * 140)

    a_candidate_infos = []

    for da in valid_a:
        desc_da = desc_a[int(da.track_id)]
        cand_rows = []

        for db in valid_b:
            desc_db = desc_b[int(db.track_id)]
            score_total, detail = local_graph_similarity(desc_da, desc_db, cfg)
            if detail.get("rejected", False):
                continue
            cand_rows.append({
                "track_id_b": int(db.track_id),
                "score_total": float(score_total),
                "detail": detail,
                "db": db,
            })

        cand_rows.sort(key=lambda x: x["score_total"], reverse=True)

        if enable_print_this_frame and should_print_track_a(da.track_id, cfg):
            rows_to_show = cand_rows if not cfg.print_only_above_threshold else [r for r in cand_rows if r["score_total"] >= cfg.min_match_score]
            print_topk_rows(frame_id, int(da.track_id), rows_to_show, cfg)

        qualified_rows = [r for r in cand_rows if r["score_total"] >= cfg.min_match_score]
        if len(qualified_rows) == 0:
            continue

        best_score = float(qualified_rows[0]["score_total"])
        second_score = float(qualified_rows[1]["score_total"]) if len(qualified_rows) > 1 else 0.0
        score_gap = best_score - second_score

        a_candidate_infos.append({
            "da": da,
            "qualified_rows": qualified_rows,
            "priority_score": best_score,
            "priority_gap": score_gap,
        })

    # 全局一对一匹配：
    # 1) 只匹配 score >= cfg.min_match_score 的候选
    # 2) 优先级：top1 分数越高越优先；若接近，则 top1-top2 分差越大越优先
    a_candidate_infos.sort(
        key=lambda x: (x["priority_score"], x["priority_gap"], -int(x["da"].track_id)),
        reverse=True,
    )

    used_b = set()
    results = []

    for info in a_candidate_infos:
        da = info["da"]
        selected = None
        for cand in info["qualified_rows"]:
            if cand["track_id_b"] not in used_b:
                selected = cand
                break

        if selected is None:
            continue

        used_b.add(selected["track_id_b"])
        db_best = selected["db"]
        detail = selected["detail"]

        results.append({
            "track_id_a": int(da.track_id),
            "track_id_b": int(db_best.track_id),
            "score_total": float(selected["score_total"]),
            "score_reid": float(selected["score_total"]),
            "score_refine": float(selected["score_total"]),
            "rank_a": np.nan,
            "rank_b": np.nan,
            "y_norm_a": np.nan,
            "y_norm_b": np.nan,
            "h_norm_a": float(da.h),
            "h_norm_b": float(db_best.h),
            "area_norm_a": float(da.area),
            "area_norm_b": float(db_best.area),
            "s_rank": float(detail["score_angle"]),
            "s_y": float(detail["score_dist"]),
            "s_pos": float(selected["score_total"]),
            "s_area": float(detail["score_area"]),
            "s_h": float(detail["score_h"]),
            "s_dist": float(detail["score_dist"]),
            "speed_sign_a": int(detail["speed_sign_a"]),
            "speed_sign_b": int(detail["speed_sign_b"]),
            "speed_proj_a": float(detail["speed_proj_a"]) if not np.isnan(detail["speed_proj_a"]) else np.nan,
            "speed_proj_b": float(detail["speed_proj_b"]) if not np.isnan(detail["speed_proj_b"]) else np.nan,
            "speed_dx_a": float(da.speed_dx) if not np.isnan(da.speed_dx) else np.nan,
            "speed_dy_a": float(da.speed_dy) if not np.isnan(da.speed_dy) else np.nan,
            "speed_dx_b": float(db_best.speed_dx) if not np.isnan(db_best.speed_dx) else np.nan,
            "speed_dy_b": float(db_best.speed_dy) if not np.isnan(db_best.speed_dy) else np.nan,
        })

    if enable_print_this_frame:
        print("=" * 140)

    return results


class LocalGraphMatcher:
    def __init__(self, cfg: LocalGraphMatcherConfig):
        self.cfg = cfg
        self.axis_a = normalize_vec2(cfg.axis_a)
        self.axis_b = normalize_vec2(cfg.axis_b)

    def run(self, tracks_a_by_frame, tracks_b_by_frame, img_size_a=None, img_size_b=None):
        _ = img_size_a
        _ = img_size_b

        hist_a = build_track_history_index(tracks_a_by_frame)
        hist_b = build_track_history_index(tracks_b_by_frame)

        annotate_motion(tracks_a_by_frame, hist_a, self.axis_a, self.cfg)
        annotate_motion(tracks_b_by_frame, hist_b, self.axis_b, self.cfg)

        print(f"[Info] 固定主方向: A视角=向右 axisA=({self.axis_a[0]:.2f},{self.axis_a[1]:.2f})")
        print(f"[Info] 固定主方向: B视角=向上 axisB=({self.axis_b[0]:.2f},{self.axis_b[1]:.2f})")
        print("[Info] 匹配规则: 不用ReID；速度异号先门控；四邻居之间做三对最优相对顺序子序列匹配 + 0.5*剩下一对得分；再做全局一对一匹配，仅保留 score>=0.60，并按 top1 分数与 top1-top2 分差优先。")

        all_frames = sorted(set(tracks_a_by_frame.keys()) | set(tracks_b_by_frame.keys()))
        results = []

        for frame_id in all_frames:
            dets_a = tracks_a_by_frame.get(frame_id, [])
            dets_b = tracks_b_by_frame.get(frame_id, [])

            frame_matches = match_one_frame(frame_id, dets_a, dets_b, self.axis_a, self.axis_b, self.cfg)
            for m in frame_matches:
                out = dict(m)
                out["frame_id"] = int(frame_id)
                results.append(out)

            if frame_id % 100 == 0:
                print(f"[Info] 已处理到 frame {frame_id}, 当前累计匹配 {len(results)} 条")

        return results


def save_matches_csv(matches, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True) if os.path.dirname(save_path) else None
    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "frame_id", "track_id_a", "track_id_b", "score_total", "score_reid", "score_refine",
            "rank_a", "rank_b", "y_norm_a", "y_norm_b", "h_norm_a", "h_norm_b",
            "area_norm_a", "area_norm_b", "s_rank", "s_y", "s_pos", "s_area", "s_h", "s_dist",
            "speed_sign_a", "speed_sign_b", "speed_proj_a", "speed_proj_b",
            "speed_dx_a", "speed_dy_a", "speed_dx_b", "speed_dy_b",
        ])
        for m in matches:
            writer.writerow([
                m["frame_id"], m["track_id_a"], m["track_id_b"], m["score_total"], m["score_reid"], m["score_refine"],
                m["rank_a"], m["rank_b"], m["y_norm_a"], m["y_norm_b"], m["h_norm_a"], m["h_norm_b"],
                m["area_norm_a"], m["area_norm_b"], m["s_rank"], m["s_y"], m["s_pos"], m["s_area"], m["s_h"], m["s_dist"],
                m["speed_sign_a"], m["speed_sign_b"], m["speed_proj_a"], m["speed_proj_b"],
                m["speed_dx_a"], m["speed_dy_a"], m["speed_dx_b"], m["speed_dy_b"],
            ])
    print(f"[Info] 匹配结果已保存到: {save_path}")


if __name__ == "__main__":
    txt_a = "experient_fig/doubleslight_MTDT2/droneA_tracks.txt"
    txt_b = "experient_fig/doubleslight_MTDT2/droneB_tracks.txt"
    reid_npz_a = "experient_fig/doubleslight_MTDT2/droneA_reid.npz"
    reid_npz_b = "experient_fig/doubleslight_MTDT2/droneB_reid.npz"
    kf_txt_a = "experient_fig/doubleslight_MTDT2/droneA_kf.txt"
    kf_txt_b = "experient_fig/doubleslight_MTDT2/droneB_kf.txt"
    txt_format = "custom"
    cls_filter = None

    reid_a = load_reid_npz(reid_npz_a)
    reid_b = load_reid_npz(reid_npz_b)
    kf_map_a = load_kf_txt(kf_txt_a) if kf_txt_a is not None else {}
    kf_map_b = load_kf_txt(kf_txt_b) if kf_txt_b is not None else {}

    tracks_a = load_track_txt(txt_a, txt_format=txt_format, cls_filter=cls_filter, reid_map=reid_a, kf_map=kf_map_a)
    tracks_b = load_track_txt(txt_b, txt_format=txt_format, cls_filter=cls_filter, reid_map=reid_b, kf_map=kf_map_b)

    print(f"A 视角共 {len(tracks_a)} 帧")
    print(f"B 视角共 {len(tracks_b)} 帧")

    img_size_a = infer_canvas_size(tracks_a)
    img_size_b = infer_canvas_size(tracks_b)
    print(f"[Info] A尺寸推断: {img_size_a}")
    print(f"[Info] B尺寸推断: {img_size_b}")

    cfg = LocalGraphMatcherConfig(
        min_conf=0.0,
        history_frames=3,
        min_dir_abs=1.0,
        min_speed_proj_abs=1.0,
        axis_a=(1.0, 0.0),
        axis_b=(0.0, -1.0),
        neighbor_k=4,
        allow_unknown_speed=True,
        angle_weight=0.35,
        dist_weight=0.25,
        h_weight=0.20,
        area_weight=0.20,
        angle_scale=0.9,
        dist_scale=2.0,
        h_scale=2.2,
        area_scale=1.8,
        min_match_score=0.60,
        print_topk_scores=True,
        print_topk_k=5,
        print_only_frames=[8],
        print_only_track_ids_a=None,
        print_only_above_threshold=False,
    )

    import time
    matcher = LocalGraphMatcher(cfg)
    t0 = time.perf_counter()
    matches = matcher.run(tracks_a, tracks_b, img_size_a, img_size_b)
    t1 = time.perf_counter()

    elapsed = t1 - t0
    all_frames = sorted(set(tracks_a.keys()) | set(tracks_b.keys()))
    num_frames = len(all_frames)
    fps = num_frames / elapsed if elapsed > 0 else 0.0

    print(f"[Timing] matching time = {elapsed:.4f} s")
    print(f"[Timing] frames = {num_frames}")
    print(f"[Timing] matching FPS = {fps:.2f}")

    save_matches_csv(matches, "results/exp_local_gragh_fixed_axes_2/cross_match.csv")
