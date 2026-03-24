import os
import cv2
import glob
import csv
import random
from collections import defaultdict


# =========================
# 1. 基础工具
# =========================
def natural_sort_key(path):
    name = os.path.basename(path)
    stem = os.path.splitext(name)[0]
    try:
        return int(stem)
    except ValueError:
        return stem


def make_dir(path):
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def get_image_list(frames_dir):
    exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(frames_dir, ext)))
    files = sorted(files, key=natural_sort_key)
    return files


def get_pair_color(id_a, id_b):
    seed = int(id_a) * 1000003 + int(id_b) * 9176
    rnd = random.Random(seed)
    return (
        rnd.randint(50, 255),
        rnd.randint(50, 255),
        rnd.randint(50, 255)
    )  # BGR


# =========================
# 2. 读取轨迹 txt
# 支持：
# custom: frame,track_id,cls,x1,y1,x2,y2,cx,cy_bottom,conf
# mot:    frame,id,x,y,w,h,score,class,-1
# =========================
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

            tracks_by_frame[frame_id].append({
                "track_id": track_id,
                "cls_id": cls_id,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "cx": cx,
                "cy_bottom": cy_bottom,
                "conf": conf
            })

    return dict(sorted(tracks_by_frame.items(), key=lambda x: x[0]))



# =========================
# 3. 读取匹配 csv
# 兼容：
# frame_id,track_id_a,track_id_b,...
# =========================
def load_match_csv(csv_path, min_score=0.0):
    matches_by_frame = defaultdict(list)

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame_id = int(float(row["frame_id"]))
            track_id_a = int(float(row["track_id_a"]))
            track_id_b = int(float(row["track_id_b"]))

            score_total = float(row["score_total"]) if "score_total" in row else 1.0
            if score_total < min_score:
                continue

            matches_by_frame[frame_id].append({
                "track_id_a": track_id_a,
                "track_id_b": track_id_b,
                "score_total": score_total
            })

    return dict(sorted(matches_by_frame.items(), key=lambda x: x[0]))


# =========================
# 4. 绘图工具
# =========================
def draw_box(img, det, color, text=None, thickness=2):
    x1 = int(round(det["x1"]))
    y1 = int(round(det["y1"]))
    x2 = int(round(det["x2"]))
    y2 = int(round(det["y2"]))

    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

    if text is not None:
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.55
        txt_thick = 2
        (tw, th), _ = cv2.getTextSize(text, font, scale, txt_thick)

        yy1 = max(0, y1 - th - 8)
        yy2 = max(0, y1)
        xx2 = min(img.shape[1] - 1, x1 + tw + 6)

        cv2.rectangle(img, (x1, yy1), (xx2, yy2), color, -1)
        cv2.putText(img, text, (x1 + 3, yy2 - 4), font, scale, (0, 0, 0), txt_thick)


def det_lookup(dets):
    return {d["track_id"]: d for d in dets}


def point_from_det(det, mode="bottom_center"):
    if mode == "center":
        x = int(round((det["x1"] + det["x2"]) / 2.0))
        y = int(round((det["y1"] + det["y2"]) / 2.0))
    else:
        x = int(round(det["cx"]))
        y = int(round(det["cy_bottom"]))
    return (x, y)


def stack_images_lr(img_a, img_b, sep=30):
    h = max(img_a.shape[0], img_b.shape[0])
    wa = img_a.shape[1]
    wb = img_b.shape[1]

    if img_a.shape[0] < h:
        pad = h - img_a.shape[0]
        img_a = cv2.copyMakeBorder(img_a, 0, pad, 0, 0, cv2.BORDER_CONSTANT, value=(20, 20, 20))
    if img_b.shape[0] < h:
        pad = h - img_b.shape[0]
        img_b = cv2.copyMakeBorder(img_b, 0, pad, 0, 0, cv2.BORDER_CONSTANT, value=(20, 20, 20))

    sep_img = 255 * (np.ones((h, sep, 3), dtype=np.uint8))
    canvas = np.concatenate([img_a, sep_img, img_b], axis=1)
    return canvas, wa, sep


def visualize_one_frame_like_sift(
    img_a,
    img_b,
    dets_a,
    dets_b,
    matches,
    frame_id,
    point_mode="bottom_center",
    draw_unmatched=False,
    show_score=True
):
    import numpy as np

    vis_a = img_a.copy()
    vis_b = img_b.copy()

    map_a = det_lookup(dets_a)
    map_b = det_lookup(dets_b)

    if draw_unmatched:
        for d in dets_a:
            draw_box(vis_a, d, (0, 255, 0), text=f"A:{d['track_id']}", thickness=1)
        for d in dets_b:
            draw_box(vis_b, d, (0, 255, 0), text=f"B:{d['track_id']}", thickness=1)

    # 先画左右框
    for m in matches:
        ida = m["track_id_a"]
        idb = m["track_id_b"]
        if ida not in map_a or idb not in map_b:
            continue

        da = map_a[ida]
        db = map_b[idb]
        color = get_pair_color(ida, idb)

        ta = f"A:{ida}"
        tb = f"B:{idb}"
        if show_score:
            ta += f" {m['score_total']:.2f}"
            tb += f" {m['score_total']:.2f}"

        draw_box(vis_a, da, color, text=ta, thickness=3)
        draw_box(vis_b, db, color, text=tb, thickness=3)

        pa = point_from_det(da, mode=point_mode)
        pb = point_from_det(db, mode=point_mode)
        cv2.circle(vis_a, pa, 5, color, -1)
        cv2.circle(vis_b, pb, 5, color, -1)

    cv2.putText(vis_a, f"Drone A  Frame {frame_id}", (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(vis_b, f"Drone B  Frame {frame_id}", (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    canvas, wa, sep = stack_images_lr(vis_a, vis_b, sep=30)
    offset_b = wa + sep

    # 再画跨图连线
    for m in matches:
        ida = m["track_id_a"]
        idb = m["track_id_b"]
        if ida not in map_a or idb not in map_b:
            continue

        da = map_a[ida]
        db = map_b[idb]
        color = get_pair_color(ida, idb)

        pa = point_from_det(da, mode=point_mode)
        pb_local = point_from_det(db, mode=point_mode)
        pb = (pb_local[0] + offset_b, pb_local[1])

        cv2.line(canvas, pa, pb, color, 2)

        mx = (pa[0] + pb[0]) // 2
        my = (pa[1] + pb[1]) // 2
        text = f"{ida}<->{idb}"
        if show_score:
            text += f" {m['score_total']:.2f}"
        cv2.putText(canvas, text, (mx - 45, my - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)

    return canvas


# =========================
# 5. 主流程：批量生成一张张图片
# =========================
def batch_visualize_like_sift(
    frames_dir_a,
    frames_dir_b,
    track_txt_a,
    track_txt_b,
    match_csv,
    out_dir,
    txt_format="custom",
    cls_filter=None,
    min_score=0.0,
    point_mode="bottom_center",
    draw_unmatched=False,
    show_score=True
):
    import numpy as np

    make_dir(out_dir)

    imgs_a = get_image_list(frames_dir_a)
    imgs_b = get_image_list(frames_dir_b)
    if len(imgs_a) == 0 or len(imgs_b) == 0:
        raise FileNotFoundError("A 或 B 的图片目录为空")

    tracks_a = load_track_txt(track_txt_a, txt_format=txt_format, cls_filter=cls_filter)
    tracks_b = load_track_txt(track_txt_b, txt_format=txt_format, cls_filter=cls_filter)
    matches_by_frame = load_match_csv(match_csv, min_score=min_score)

    max_frame = min(len(imgs_a), len(imgs_b))
    print(f"A 图片数: {len(imgs_a)}")
    print(f"B 图片数: {len(imgs_b)}")
    print(f"实际处理帧数: {max_frame}")

    for frame_id in range(1, max_frame + 1):
        img_a = cv2.imread(imgs_a[frame_id - 1])
        img_b = cv2.imread(imgs_b[frame_id - 1])

        if img_a is None or img_b is None:
            print(f"[Warning] 第 {frame_id} 帧读取失败，跳过")
            continue

        dets_a = tracks_a.get(frame_id, [])
        dets_b = tracks_b.get(frame_id, [])
        matches = matches_by_frame.get(frame_id, [])

        canvas = visualize_one_frame_like_sift(
            img_a=img_a,
            img_b=img_b,
            dets_a=dets_a,
            dets_b=dets_b,
            matches=matches,
            frame_id=frame_id,
            point_mode=point_mode,       # "center" 或 "bottom_center"
            draw_unmatched=draw_unmatched,
            show_score=show_score
        )

        save_path = os.path.join(out_dir, f"{frame_id:06d}.jpg")
        cv2.imwrite(save_path, canvas)

        if frame_id % 100 == 0:
            print(f"[Info] 已完成 {frame_id}/{max_frame} 帧")

    print(f"[Done] 可视化结果保存在: {out_dir}")


# =========================
# 6. 单帧调试
# =========================
def visualize_single_frame(
    frame_id,
    frames_dir_a,
    frames_dir_b,
    track_txt_a,
    track_txt_b,
    match_csv,
    save_path,
    txt_format="custom",
    cls_filter=None,
    min_score=0.0,
    point_mode="bottom_center",
    draw_unmatched=True,
    show_score=True
):
    imgs_a = get_image_list(frames_dir_a)
    imgs_b = get_image_list(frames_dir_b)

    if frame_id < 1 or frame_id > min(len(imgs_a), len(imgs_b)):
        raise ValueError(f"frame_id={frame_id} 超出范围")

    tracks_a = load_track_txt(track_txt_a, txt_format=txt_format, cls_filter=cls_filter)
    tracks_b = load_track_txt(track_txt_b, txt_format=txt_format, cls_filter=cls_filter)
    matches_by_frame = load_match_csv(match_csv, min_score=min_score)

    img_a = cv2.imread(imgs_a[frame_id - 1])
    img_b = cv2.imread(imgs_b[frame_id - 1])

    dets_a = tracks_a.get(frame_id, [])
    dets_b = tracks_b.get(frame_id, [])
    matches = matches_by_frame.get(frame_id, [])

    canvas = visualize_one_frame_like_sift(
        img_a=img_a,
        img_b=img_b,
        dets_a=dets_a,
        dets_b=dets_b,
        matches=matches,
        frame_id=frame_id,
        point_mode=point_mode,
        draw_unmatched=draw_unmatched,
        show_score=show_score
    )

    make_dir(os.path.dirname(save_path))
    cv2.imwrite(save_path, canvas)
    print(f"[Done] 单帧可视化已保存: {save_path}")



if __name__ == "__main__":
    import numpy as np

    # ========== 你改这里 ==========
    frames_dir_a = r"experient_fig/doublesight/frame1"
    frames_dir_b = r"experient_fig/doublesight/frame2"

    track_txt_a = r"results/droneA_tracks.txt"
    track_txt_b = r"results/droneB_tracks.txt"

    # 你现在新的匹配文件
    match_csv = r"results/exp2/cross_match_lower_anchor.csv"

    # 轨迹 txt 格式
    txt_format = "custom"   # 或 "mot"

    # 只看车辆 class=3；如果你的 txt 里已经只有车，也可以设 None
    cls_filter = 3

    # 只看高分匹配
    min_score = 0.0

    # 连接点用 "bottom_center" 更适合车辆；想更像 SIFT 可以用 "center"
    point_mode = "bottom_center"

    # -------- 方式1：批量生成全部帧 --------
    out_dir = r"results/vis_like_sift"
    batch_visualize_like_sift(
        frames_dir_a=frames_dir_a,
        frames_dir_b=frames_dir_b,
        track_txt_a=track_txt_a,
        track_txt_b=track_txt_b,
        match_csv=match_csv,
        out_dir=out_dir,
        txt_format=txt_format,
        cls_filter=cls_filter,
        min_score=min_score,
        point_mode=point_mode,
        draw_unmatched=False,
        show_score=True
    )

    # -------- 方式2：只调试某一帧，取消下面注释 --------
    # visualize_single_frame(
    #     frame_id=120,
    #     frames_dir_a=frames_dir_a,
    #     frames_dir_b=frames_dir_b,
    #     track_txt_a=track_txt_a,
    #     track_txt_b=track_txt_b,
    #     match_csv=match_csv,
    #     save_path=r"results/debug_frame_120.jpg",
    #     txt_format=txt_format,
    #     cls_filter=cls_filter,
    #     min_score=min_score,
    #     point_mode=point_mode,
    #     draw_unmatched=True,
    #     show_score=True
    # )