import os
import glob
import csv
import numpy as np
from ultralytics import YOLO


def natural_sort_key(path):
    name = os.path.basename(path)
    stem = os.path.splitext(name)[0]
    return int(stem) if stem.isdigit() else stem


def l2_normalize(feat, eps=1e-12):
    feat = np.asarray(feat, dtype=np.float32).reshape(-1)
    norm = np.linalg.norm(feat)
    if norm < eps:
        return feat
    return feat / norm


def get_active_track_map(model):
    """
    从当前 tracker 中获取:
        track_id -> BOTrack对象
    """
    track_map = {}

    predictor = getattr(model, "predictor", None)
    if predictor is None:
        return track_map

    trackers = getattr(predictor, "trackers", None)
    if not trackers:
        return track_map

    tracker = trackers[0]
    tracked_stracks = getattr(tracker, "tracked_stracks", None)
    if tracked_stracks is None:
        return track_map

    for t in tracked_stracks:
        tid = getattr(t, "track_id", None)
        if tid is None:
            continue
        track_map[int(tid)] = t

    return track_map


def xywh_to_xyxy(xywh):
    cx, cy, w, h = map(float, xywh[:4])
    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0
    return x1, y1, x2, y2


def xywh_to_center_bottom(xywh):
    cx, cy, w, h = map(float, xywh[:4])
    return float(cx), float(cy + h / 2.0)


def get_kf_state_for_export(t):
    """
    从 BOTrack 中读取 Kalman 预测框和速度
    需要你已在 bot_sort.py 中缓存:
      - pred_mean
      - (可选) post_mean

    返回:
      pred_x1, pred_y1, pred_x2, pred_y2,
      pred_cx, pred_cy_bottom,
      vx, vy, vw, vh
    """
    pred_x1 = pred_y1 = pred_x2 = pred_y2 = np.nan
    pred_cx = pred_cy_bottom = np.nan
    vx = vy = vw = vh = np.nan

    if t is None:
        return pred_x1, pred_y1, pred_x2, pred_y2, pred_cx, pred_cy_bottom, vx, vy, vw, vh

    pred_mean = getattr(t, "pred_mean", None)
    post_mean = getattr(t, "post_mean", None)

    # 1) 预测框优先取 pred_mean
    if pred_mean is not None and len(pred_mean) >= 4:
        pred_x1, pred_y1, pred_x2, pred_y2 = xywh_to_xyxy(pred_mean[:4])
        pred_cx, pred_cy_bottom = xywh_to_center_bottom(pred_mean[:4])

    # 2) 速度优先取 pred_mean[4:8]，没有再退回 post_mean[4:8]
    state_for_vel = None
    if pred_mean is not None and len(pred_mean) >= 8:
        state_for_vel = pred_mean
    elif post_mean is not None and len(post_mean) >= 8:
        state_for_vel = post_mean

    if state_for_vel is not None:
        vx = float(state_for_vel[4])
        vy = float(state_for_vel[5])
        vw = float(state_for_vel[6])
        vh = float(state_for_vel[7])

    return pred_x1, pred_y1, pred_x2, pred_y2, pred_cx, pred_cy_bottom, vx, vy, vw, vh


def save_track_txt_from_frames(
    model_path: str,
    frames_dir: str,
    tracker_config: str,
    output_txt: str,
    output_reid_npz: str,
    output_kf_txt: str,
    conf_thres: float = 0.25,
    iou_thres: float = 0.5,
    classes=None
):
    os.makedirs(os.path.dirname(output_txt), exist_ok=True) if os.path.dirname(output_txt) else None
    os.makedirs(os.path.dirname(output_reid_npz), exist_ok=True) if os.path.dirname(output_reid_npz) else None
    os.makedirs(os.path.dirname(output_kf_txt), exist_ok=True) if os.path.dirname(output_kf_txt) else None

    model = YOLO(model_path)

    image_paths = []
    for ext in ("*.jpg", "*.png", "*.jpeg", "*.bmp"):
        image_paths.extend(glob.glob(os.path.join(frames_dir, ext)))
    image_paths = sorted(image_paths, key=natural_sort_key)

    if not image_paths:
        raise FileNotFoundError(f"在 {frames_dir} 中没有找到图片帧")

    # sidecar ReID数据
    reid_frame_ids = []
    reid_track_ids = []
    reid_feats = []

    total_boxes = 0
    saved_feats = 0
    missing_feats = 0
    saved_kf = 0
    missing_kf = 0

    with open(output_txt, "w", encoding="utf-8") as f_main, \
         open(output_kf_txt, "w", newline="", encoding="utf-8") as f_kf:

        kf_writer = csv.writer(f_kf)
        kf_writer.writerow([
            "frame_id", "track_id",
            "pred_x1", "pred_y1", "pred_x2", "pred_y2",
            "pred_cx", "pred_cy_bottom",
            "vx", "vy", "vw", "vh"
        ])

        for frame_id, img_path in enumerate(image_paths, start=1):
            result = model.track(
                source=img_path,
                persist=True,
                tracker=tracker_config,
                conf=conf_thres,
                iou=iou_thres,
                classes=classes,
                verbose=False
            )[0]

            if result.boxes is None or result.boxes.id is None or len(result.boxes) == 0:
                continue

            xyxy = result.boxes.xyxy.cpu().numpy()
            ids = result.boxes.id.int().cpu().tolist()
            confs = result.boxes.conf.cpu().numpy().tolist()
            clss = result.boxes.cls.int().cpu().tolist()

            # 当前帧 tracker 内部 active tracks
            track_map = get_active_track_map(model)

            for box, tid, score, cls_id in zip(xyxy, ids, confs, clss):
                total_boxes += 1

                x1, y1, x2, y2 = map(float, box)
                cx = (x1 + x2) / 2.0
                cy_bottom = y2

                # 1) 原txt保持完全不变
                line = f"{frame_id},{tid},{cls_id},{x1:.2f},{y1:.2f},{x2:.2f},{y2:.2f},{cx:.2f},{cy_bottom:.2f},{score:.4f}\n"
                f_main.write(line)

                # 当前轨迹对象
                t = track_map.get(int(tid), None)

                # 2) 导出ReID特征（保持你原来的逻辑）
                feat = None
                if t is not None:
                    feat = getattr(t, "smooth_feat", None)
                    if feat is None:
                        feat = getattr(t, "curr_feat", None)

                if feat is not None:
                    feat = l2_normalize(feat)
                    reid_frame_ids.append(int(frame_id))
                    reid_track_ids.append(int(tid))
                    reid_feats.append(feat.astype(np.float32))
                    saved_feats += 1
                else:
                    missing_feats += 1

                # 3) 导出Kalman sidecar，不改原txt格式
                pred_x1, pred_y1, pred_x2, pred_y2, pred_cx, pred_cy_bottom, vx, vy, vw, vh = \
                    get_kf_state_for_export(t)

                if not np.isnan(pred_cx):
                    saved_kf += 1
                else:
                    missing_kf += 1

                kf_writer.writerow([
                    int(frame_id),
                    int(tid),
                    round(float(pred_x1), 2) if not np.isnan(pred_x1) else np.nan,
                    round(float(pred_y1), 2) if not np.isnan(pred_y1) else np.nan,
                    round(float(pred_x2), 2) if not np.isnan(pred_x2) else np.nan,
                    round(float(pred_y2), 2) if not np.isnan(pred_y2) else np.nan,
                    round(float(pred_cx), 2) if not np.isnan(pred_cx) else np.nan,
                    round(float(pred_cy_bottom), 2) if not np.isnan(pred_cy_bottom) else np.nan,
                    round(float(vx), 6) if not np.isnan(vx) else np.nan,
                    round(float(vy), 6) if not np.isnan(vy) else np.nan,
                    round(float(vw), 6) if not np.isnan(vw) else np.nan,
                    round(float(vh), 6) if not np.isnan(vh) else np.nan,
                ])

            if frame_id % 100 == 0:
                print(
                    f"已处理 {frame_id}/{len(image_paths)} 帧 | "
                    f"检测框累计: {total_boxes} | "
                    f"ReID累计保存: {saved_feats} | "
                    f"KF累计保存: {saved_kf}"
                )

    print(f"轨迹已保存到: {output_txt}")
    print(f"Kalman sidecar已保存到: {output_kf_txt}")

    # 保存 sidecar npz
    if len(reid_feats) > 0:
        feats_arr = np.stack(reid_feats, axis=0).astype(np.float32)
    else:
        feats_arr = np.empty((0, 0), dtype=np.float32)

    np.savez_compressed(
        output_reid_npz,
        frame_ids=np.asarray(reid_frame_ids, dtype=np.int32),
        track_ids=np.asarray(reid_track_ids, dtype=np.int32),
        feats=feats_arr
    )

    print(f"ReID特征已保存到: {output_reid_npz}")
    print(
        f"[统计] txt检测框数={total_boxes}, "
        f"成功导出特征={saved_feats}, "
        f"缺失特征={missing_feats}, "
        f"成功导出KF={saved_kf}, "
        f"缺失KF={missing_kf}"
    )

    if saved_feats == 0:
        print(
            "[警告] 没有导出任何 ReID 特征，请检查：\n"
            "1) tracker 是否为 BoT-SORT；\n"
            "2) yaml 中是否 with_reid: True；\n"
            "3) bot_sort.py 是否仍会初始化/更新 curr_feat 和 smooth_feat；\n"
            "4) model 设置是否为 auto 或有效 ReID 模型。"
        )

    if saved_kf == 0:
        print(
            "[警告] 没有导出任何 Kalman 预测信息，请检查：\n"
            "1) 你是否已在 BOTrack 中缓存 pred_mean；\n"
            "2) pred_mean / post_mean 字段名是否与你改的代码一致；\n"
            "3) get_kf_state_for_export() 中读取的字段是否匹配你的实现。"
        )


if __name__ == "__main__":
    save_track_txt_from_frames(
        model_path="runs/detect/v6.4_n_p2_shanp5_cl3_SimAM_EUCB/weights/best.pt",
        frames_dir="experient_fig/doublesight/frame1",
        tracker_config="ultralytics/cfg/trackers/botsort.yaml",
        output_txt="experient_fig/doublesight_reidkalm/droneA_tracks.txt",
        output_reid_npz="experient_fig/doublesight_reidkalm/droneA_reid.npz",
        output_kf_txt="experient_fig/doublesight_reidkalm/droneA_kf.txt",
        conf_thres=0.25,
        iou_thres=0.5,
        classes=[3]
    )