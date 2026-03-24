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
class FanGraphConfig:
    # 历史
    window_size: int = 20
    pair_ttl: int = 5

    # 过滤：只保留更有意义的目标
    min_box_h: float = 18.0
    min_box_area: float = 550.0
    min_conf: float = 0.05
    edge_margin_ratio: float = 0.08       # 左右边缘剔除比例
    min_bottom_ratio: float = 0.35        # 只保留下方一定区域的目标
    max_top_ratio: float = 0.98           # 可留着，通常不用改

    # 中心优先（不是硬过滤，而是用于加分/门控）
    center_soft_ratio: float = 0.60       # 中心区域宽度比例，越小越偏中心
    center_bonus_weight: float = 0.08

    # 参考点（近似“无穷远点”）
    vp_center_mix: float = 0.75           # 0~1，越大越接近图像正中上方
    vp_y_offset_ratio: float = 0.15       # 参考点放到图像上方一点

    # 候选门控
    max_rank_gap: float = 0.35
    max_y_gap: float = 0.28
    max_h_ratio: float = 2.8
    max_area_ratio: float = 6.0

    # 匹配
    min_score: float = 0.55
    huge_cost: float = 1e6


# =========================
# 5. 扇形图特征
# =========================
def filter_detections_for_matching(dets, img_w, img_h, cfg: FanGraphConfig):
    kept = []
    x_left = cfg.edge_margin_ratio * img_w
    x_right = (1.0 - cfg.edge_margin_ratio) * img_w
    y_bottom_min = cfg.min_bottom_ratio * img_h
    y_top_max = cfg.max_top_ratio * img_h

    for d in dets:
        if d.conf < cfg.min_conf:
            continue
        if d.h < cfg.min_box_h:
            continue
        if d.area < cfg.min_box_area:
            continue
        if d.cx < x_left or d.cx > x_right:
            continue
        if d.cy_bottom < y_bottom_min or d.cy_bottom > y_top_max:
            continue
        kept.append(d)

    return kept


def build_fan_graph_features(current_dets, histories, img_w, img_h, cfg: FanGraphConfig):
    """
    用“参考点 -> 底边中心点”构造扇形顺序图
    返回：
        feat_by_id: {track_id: dict}
        ordered_ids: 按角度排序后的 track_id 列表
        vp: (vx, vy)
    """
    if len(current_dets) == 0:
        return {}, [], (img_w * 0.5, -cfg.vp_y_offset_ratio * img_h)

    cxs = [d.cx for d in current_dets]
    hs = [max(d.h, 1.0) for d in current_dets]

    # 参考点 x：以图像中心上方为主，稍微参考当前目标分布
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
    ordered_ids = [d.track_id for d in dets_sorted]

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
    center_half_width = 0.5 * cfg.center_soft_ratio * img_w

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
        centrality = math.exp(-center_bias / max(cfg.center_soft_ratio, 1e-6))

        feat_by_id[d.track_id] = {
            "track_id": d.track_id,
            "theta": float(theta_sorted[i]),
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

    return feat_by_id, ordered_ids, (vx, vy)


def fan_node_similarity(fa, fb):
    """
    节点 + 邻接边（局部图）联合相似度
    """
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


def is_valid_candidate(fa, fb, cfg: FanGraphConfig):
    if abs(fa["rank"] - fb["rank"]) > cfg.max_rank_gap:
        return False
    if abs(fa["y_norm"] - fb["y_norm"]) > cfg.max_y_gap:
        return False

    h_ratio = max(fa["h_norm"], fb["h_norm"]) / (min(fa["h_norm"], fb["h_norm"]) + 1e-6)
    if h_ratio > cfg.max_h_ratio:
        return False

    area_ratio = max(fa["area_norm"], fb["area_norm"]) / (min(fa["area_norm"], fb["area_norm"]) + 1e-6)
    if area_ratio > cfg.max_area_ratio:
        return False

    return True


# =========================
# 6. 匹配器
# =========================
class FanGraphMatcher:
    def __init__(self, cfg: FanGraphConfig):
        self.cfg = cfg
        self.hist_A = defaultdict(lambda: deque(maxlen=self.cfg.window_size))
        self.hist_B = defaultdict(lambda: deque(maxlen=self.cfg.window_size))
        self.pair_memory = {}   # (idA, idB) -> {"last_frame": int, "hits": int}

    def update_histories(self, dets_a, dets_b):
        for d in dets_a:
            self.hist_A[d.track_id].append(d)
        for d in dets_b:
            self.hist_B[d.track_id].append(d)

# 过期删除
    def prune_pair_memory(self, frame_id): 
        new_mem = {}
        for k, v in self.pair_memory.items():
            if frame_id - v["last_frame"] <= self.cfg.pair_ttl:
                new_mem[k] = v
        self.pair_memory = new_mem

    def temporal_bonus(self, id_a, id_b, frame_id):
        item = self.pair_memory.get((id_a, id_b), None)
        if item is None:
            return 0.0

        age = frame_id - item["last_frame"]
        hits = item["hits"]

        if age == 1:
            return min(1.0, 0.35 + 0.07 * hits)
        elif age <= 3:
            return 0.28
        elif age <= self.cfg.pair_ttl:
            return 0.12
        else:
            return 0.0

    def match_one_frame(self, frame_id, dets_a, dets_b, img_w_a, img_h_a, img_w_b, img_h_b):
        self.update_histories(dets_a, dets_b)
        self.prune_pair_memory(frame_id)

        # 先过滤：只保留更有意义的近端/较大/不靠边目标
        valid_a = filter_detections_for_matching(dets_a, img_w_a, img_h_a, self.cfg)
        valid_b = filter_detections_for_matching(dets_b, img_w_b, img_h_b, self.cfg)

        if len(valid_a) == 0 or len(valid_b) == 0:
            return []

        # 构造扇形图特征
        feat_a, ordered_ids_a, vp_a = build_fan_graph_features(valid_a, self.hist_A, img_w_a, img_h_a, self.cfg)
        feat_b, ordered_ids_b, vp_b = build_fan_graph_features(valid_b, self.hist_B, img_w_b, img_h_b, self.cfg)

        na = len(valid_a)
        nb = len(valid_b)

        score_total = np.zeros((na, nb), dtype=np.float32)
        score_graph = np.zeros((na, nb), dtype=np.float32)
        score_temp = np.zeros((na, nb), dtype=np.float32)
        score_center = np.zeros((na, nb), dtype=np.float32)

        cost = np.full((na, nb), self.cfg.huge_cost, dtype=np.float32)

        for i, da in enumerate(valid_a):
            fa = feat_a[da.track_id]
            for j, db in enumerate(valid_b):
                fb = feat_b[db.track_id]

                if not is_valid_candidate(fa, fb, self.cfg):
                    continue

                sg = fan_node_similarity(fa, fb)
                st = self.temporal_bonus(da.track_id, db.track_id, frame_id)
                sc = 0.5 * (fa["centrality"] + fb["centrality"])

                s = (1.0 - self.cfg.center_bonus_weight) * (0.88 * sg + 0.12 * st) \
                    + self.cfg.center_bonus_weight * sc

                score_graph[i, j] = sg
                score_temp[i, j] = st
                score_center[i, j] = sc
                score_total[i, j] = s
                cost[i, j] = 1.0 - s

        row_ind, col_ind = linear_sum_assignment(cost)

        matches = []
        for r, c in zip(row_ind, col_ind):
            if cost[r, c] >= self.cfg.huge_cost * 0.5:
                continue

            s = float(score_total[r, c])
            if s < self.cfg.min_score:
                continue

            id_a = valid_a[r].track_id
            id_b = valid_b[c].track_id

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
                "score_graph": round(float(score_graph[r, c]), 6),
                "score_temp": round(float(score_temp[r, c]), 6),
                "score_center": round(float(score_center[r, c]), 6),
                "num_valid_a": len(valid_a),
                "num_valid_b": len(valid_b),
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
            "score_graph",
            "score_temp",
            "score_center",
            "num_valid_a",
            "num_valid_b",
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
            ])

    print(f"匹配结果已保存到: {save_path}")


# =========================
# 8. 主程序
# =========================
if __name__ == "__main__":
    # -------- 你的输入 --------
    txt_a = "experient_fig/doublesight/droneA_tracks.txt"
    txt_b = "experient_fig/doublesight/droneB_tracks.txt"

    # custom:
    # frame,track_id,cls,x1,y1,x2,y2,cx,cy_bottom,conf
    # mot:
    # frame,id,x,y,w,h,score,class,-1
    txt_format = "custom"

    # 如果只匹配某一类，比如车辆 cls=2 / cls=3，你可自行改
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
    cfg = FanGraphConfig(
        window_size=20,
        pair_ttl=5,

        # 这几个是你最该调的
        min_box_h=18.0,          # 太小的框不要
        min_box_area=550.0,      # 面积太小不要
        edge_margin_ratio=0.08,  # 左右边缘目标不要
        min_bottom_ratio=0.35,   # 太靠上的目标不要

        # 参考点
        vp_center_mix=0.75,
        vp_y_offset_ratio=0.15,

        # 候选门控
        max_rank_gap=0.35,
        max_y_gap=0.28,
        max_h_ratio=2.8,
        max_area_ratio=6.0,

        # 最终阈值
        min_score=0.55,
        center_soft_ratio=0.60,
        center_bonus_weight=0.08,
    )

    # -------- 建立匹配器 --------
    matcher = FanGraphMatcher(cfg)

    # -------- 执行匹配 --------
    matches = matcher.run(tracks_a, tracks_b, img_size_a, img_size_b)

    # -------- 保存结果 --------
    save_matches_csv(matches, "results/exp6/fan_graph_match.csv")