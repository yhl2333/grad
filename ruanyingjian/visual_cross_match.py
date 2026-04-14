import os
import csv
import cv2
import glob
import math
import random
from collections import defaultdict


# =====================================================
# 1. 工具函数
# =====================================================
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


def get_color_by_pair(id_a, id_b):
    """
    给匹配对分配稳定颜色
    """
    seed = int(id_a) * 1000003 + int(id_b) * 9176
    rnd = random.Random(seed)
    return (
        rnd.randint(50, 255),
        rnd.randint(50, 255),
        rnd.randint(50, 255)
    )  # BGR


# =====================================================
# 2. 读取轨迹 txt
# 支持两种格式：
# custom: frame,track_id,cls,x1,y1,x2,y2,cx,cy_bottom,conf
# mot:    frame,id,x,y,w,h,score,class,-1
# =====================================================
def load_track_txt(txt_path, txt_format="custom"):
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

            item = {
                "frame_id": frame_id,
                "track_id": track_id,
                "cls_id": cls_id,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "cx": cx,
                "cy_bottom": cy_bottom,
                "conf": conf
            }
            tracks_by_frame[frame_id].append(item)

    return dict(sorted(tracks_by_frame.items(), key=lambda x: x[0]))


# =====================================================
# 3. 读取跨视角匹配结果
# 格式：
# frame_id,track_id_a,track_id_b,score_total,score_traj,score_rel,score_temp
# =====================================================
def load_cross_match_csv(csv_path, score_thresh=0.0):
    matches_by_frame = defaultdict(list)

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame_id = int(float(row["frame_id"]))
            track_id_a = int(float(row["track_id_a"]))
            track_id_b = int(float(row["track_id_b"]))
            score_total = float(row["score_total"])

            if score_total < score_thresh:
                continue

            item = {
                "track_id_a": track_id_a,
                "track_id_b": track_id_b,
                "score_total": score_total,
                "score_traj": float(row.get("score_traj", 0.0)),
                "score_rel": float(row.get("score_rel", 0.0)),
                "score_temp": float(row.get("score_temp", 0.0)),
            }
            matches_by_frame[frame_id].append(item)

    return dict(sorted(matches_by_frame.items(), key=lambda x: x[0]))


# =====================================================
# 4. 绘制函数
# =====================================================
def draw_detection(img, det, color, label=None, thickness=2):
    x1 = int(round(det["x1"]))
    y1 = int(round(det["y1"]))
    x2 = int(round(det["x2"]))
    y2 = int(round(det["y2"]))

    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

    if label is None:
        label = f"ID:{det['track_id']}"

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    txt_thickness = 2
    (tw, th), _ = cv2.getTextSize(label, font, scale, txt_thickness)

    box_y1 = max(0, y1 - th - 8)
    box_y2 = max(0, y1)
    box_x2 = min(img.shape[1] - 1, x1 + tw + 6)

    cv2.rectangle(img, (x1, box_y1), (box_x2, box_y2), color, -1)
    cv2.putText(img, label, (x1 + 3, box_y2 - 4), font, scale, (0, 0, 0), txt_thickness)


def put_header(img, text, color=(0, 255, 255)):
    org = (15, 30)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.9

    # 黑色描边
    cv2.putText(img, text, org, font, scale, (0, 0, 0), 4)
    # 黄色正文
    cv2.putText(img, text, org, font, scale, color, 2)

def build_track_lookup(dets):
    return {d["track_id"]: d for d in dets}


def visualize_one_frame(
    img_a,
    img_b,
    dets_a,
    dets_b,
    matches,
    frame_id,
    draw_unmatched=True,
    score_show=True,
    separator_width=40
):
    """
    返回拼接后的可视化图
    """
    det_map_a = build_track_lookup(dets_a)
    det_map_b = build_track_lookup(dets_b)

    vis_a = img_a.copy()
    vis_b = img_b.copy()

    matched_a_ids = set()
    matched_b_ids = set()

    # 先画未匹配目标
    if draw_unmatched:
        for d in dets_a:
            draw_detection(vis_a, d, color=(0, 255, 0), label=f"A:{d['track_id']}", thickness=1)
        for d in dets_b:
            draw_detection(vis_b, d, color=(0, 255, 0), label=f"B:{d['track_id']}", thickness=1)

    # 再高亮匹配目标
    for m in matches:
        ida = m["track_id_a"]
        idb = m["track_id_b"]

        if ida not in det_map_a or idb not in det_map_b:
            continue

        matched_a_ids.add(ida)
        matched_b_ids.add(idb)

        color = get_color_by_pair(ida, idb)
        da = det_map_a[ida]
        db = det_map_b[idb]

        label_a = f"A:{ida}"
        label_b = f"B:{idb}"
        if score_show:
            label_a += f" {m['score_total']:.2f}"
            label_b += f" {m['score_total']:.2f}"

        draw_detection(vis_a, da, color=color, label=label_a, thickness=3)
        draw_detection(vis_b, db, color=color, label=label_b, thickness=3)

    put_header(vis_a, f"Drone A  Frame {frame_id}")
    put_header(vis_b, f"Drone B  Frame {frame_id}")

    # 高度不一致时补齐
    h = max(vis_a.shape[0], vis_b.shape[0])
    wa = vis_a.shape[1]
    wb = vis_b.shape[1]

    if vis_a.shape[0] < h:
        pad = h - vis_a.shape[0]
        vis_a = cv2.copyMakeBorder(vis_a, 0, pad, 0, 0, cv2.BORDER_CONSTANT, value=(30, 30, 30))
    if vis_b.shape[0] < h:
        pad = h - vis_b.shape[0]
        vis_b = cv2.copyMakeBorder(vis_b, 0, pad, 0, 0, cv2.BORDER_CONSTANT, value=(30, 30, 30))

    sep = 255 * (np.ones((h, separator_width, 3), dtype=np.uint8))
    canvas = np.concatenate([vis_b, sep, vis_a], axis=1)

    # 在拼接图上画匹配连线
    offset_x_a = wb + separator_width
    for m in matches:
        ida = m["track_id_a"]
        idb = m["track_id_b"]

        if ida not in det_map_a or idb not in det_map_b:
            continue

        da = det_map_a[ida]
        db = det_map_b[idb]
        color = get_color_by_pair(ida, idb)

        # 现在 B 在左边，所以 B 不加偏移
        pt_b = (int(round(db["cx"])), int(round((db["y1"] + db["y2"]) / 2)))
        # A 在右边，所以 A 要加偏移
        pt_a = (offset_x_a + int(round(da["cx"])), int(round((da["y1"] + da["y2"]) / 2)))

        cv2.line(canvas, pt_b, pt_a, color, 2)

        mx = (pt_a[0] + pt_b[0]) // 2
        my = (pt_a[1] + pt_b[1]) // 2
        txt = f"{ida}<->{idb}"
        if score_show:
            txt += f" {m['score_total']:.2f}"
        cv2.putText(canvas, txt, (mx - 40, my - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)

    return canvas


# =====================================================
# 5. 生成可视化
# =====================================================
def get_image_list(frames_dir):
    exts = ["*.jpg", "*.png", "*.jpeg", "*.bmp", "*.webp"]
    paths = []
    for ext in exts:
        paths.extend(glob.glob(os.path.join(frames_dir, ext)))
    paths = sorted(paths, key=natural_sort_key)
    return paths


def visualize_cross_match(
    frames_dir_a,
    frames_dir_b,
    track_txt_a,
    track_txt_b,
    cross_match_csv,
    output_dir,
    txt_format="custom",
    score_thresh=0.0,
    save_video=True,
    video_name="cross_match_vis.mp4",
    fps=25,
    draw_unmatched=True,
    score_show=True
):
    import numpy as np  # 放这里是为了减少上面依赖感

    make_dir(output_dir)
    frames_out_dir = os.path.join(output_dir, "vis_frames")
    make_dir(frames_out_dir)

    img_list_a = get_image_list(frames_dir_a)
    img_list_b = get_image_list(frames_dir_b)

    if len(img_list_a) == 0 or len(img_list_b) == 0:
        raise FileNotFoundError("A 或 B 的图片目录为空，请检查路径")

    tracks_a = load_track_txt(track_txt_a, txt_format=txt_format)
    tracks_b = load_track_txt(track_txt_b, txt_format=txt_format)
    matches_by_frame = load_cross_match_csv(cross_match_csv, score_thresh=score_thresh)

    max_frame = min(len(img_list_a), len(img_list_b))
    print(f"[Info] A 图片数: {len(img_list_a)}")
    print(f"[Info] B 图片数: {len(img_list_b)}")
    print(f"[Info] 实际可视化帧数: {max_frame}")

    video_writer = None

    for frame_id in range(1, max_frame + 1):
        img_a = cv2.imread(img_list_a[frame_id - 1])
        img_b = cv2.imread(img_list_b[frame_id - 1])

        if img_a is None or img_b is None:
            print(f"[Warning] 第 {frame_id} 帧图片读取失败，跳过")
            continue

        dets_a = tracks_a.get(frame_id, [])
        dets_b = tracks_b.get(frame_id, [])
        matches = matches_by_frame.get(frame_id, [])

        canvas = visualize_one_frame(
            img_a=img_a,
            img_b=img_b,
            dets_a=dets_a,
            dets_b=dets_b,
            matches=matches,
            frame_id=frame_id,
            draw_unmatched=draw_unmatched,
            score_show=score_show,
            separator_width=40
        )

        save_path = os.path.join(frames_out_dir, f"{frame_id:06d}.jpg")
        cv2.imwrite(save_path, canvas)

        if save_video:
            if video_writer is None:
                h, w = canvas.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                video_path = os.path.join(output_dir, video_name)
                video_writer = cv2.VideoWriter(video_path, fourcc, fps, (w, h))
            video_writer.write(canvas)

        if frame_id % 100 == 0:
            print(f"[Info] 已完成 {frame_id}/{max_frame} 帧")

    if video_writer is not None:
        video_writer.release()

    print(f"[Done] 可视化图片保存在: {frames_out_dir}")
    if save_video:
        print(f"[Done] 可视化视频保存在: {os.path.join(output_dir, video_name)}")


# =====================================================
# 6. 主函数
# =====================================================
if __name__ == "__main__":
    import numpy as np

    # # -------------- 你需要修改这里 --------------
    # frames_dir_a = r"experient_fig/doublesight/frame1"
    # frames_dir_b = r"experient_fig/doublesight/frame2"

    # track_txt_a = r"experient_fig/doublesight_reidkalm/droneA_tracks.txt"
    # track_txt_b = r"experient_fig/doublesight_reidkalm/droneB_tracks.txt"
    # cross_match_csv = r"results/exp14_test/cross_match.csv"

    # output_dir = r"results/exp14_reidkalam"



    #     # -------------- 你需要修改这里 --------------
    # frames_dir_a = r"datasetTrack/Multi-Drone-Multi-Object-Detection-and-Tracking/test/1/62-1"
    # frames_dir_b = r"datasetTrack/Multi-Drone-Multi-Object-Detection-and-Tracking/test/2/62-2"

    # track_txt_a = r"experient_fig/doubleslight_MTDT/droneA_tracks.txt"
    # track_txt_b = r"experient_fig/doubleslight_MTDT/droneB_tracks.txt"
    # cross_match_csv = r"results/exp_local_graph_fixed_axes_v3/cross_match.csv"

    # output_dir = r"results/exp_local_graph_fixed_axes_v3"


        # -------------- 你需要修改这里 --------------
    frames_dir_a = r"datasetTrack/Multi-Drone-Multi-Object-Detection-and-Tracking/test/1/62-1"
    frames_dir_b = r"datasetTrack/Multi-Drone-Multi-Object-Detection-and-Tracking/test/2/62-2"

    track_txt_a = r"experient_fig/test/droneA_tracks.txt"
    track_txt_b = r"experient_fig/test/droneB_tracks.txt"
    cross_match_csv = r"results/exp_local_gragh_fixed_axes_2/cross_match.csv"

    output_dir = r"results/exp_local_gragh_fixed_axes_2"


    # 如果你的轨迹文件是：
    # frame,track_id,cls,x1,y1,x2,y2,cx,cy_bottom,conf
    txt_format = "custom"

    # 如果想过滤低分匹配，比如只看总分 >= 0.55
    score_thresh = 0.0

    visualize_cross_match(
        frames_dir_a=frames_dir_a,
        frames_dir_b=frames_dir_b,
        track_txt_a=track_txt_a,
        track_txt_b=track_txt_b,
        cross_match_csv=cross_match_csv,
        output_dir=output_dir,
        txt_format=txt_format,
        score_thresh=score_thresh,
        save_video=False,
        video_name="cross_match_vis.mp4",
        fps=25,
        draw_unmatched=True,
        score_show=True
    )