# import os
# import glob
# import cv2
# from ultralytics import YOLO


# def natural_sort_key(path):
#     name = os.path.basename(path)
#     stem = os.path.splitext(name)[0]
#     return int(stem) if stem.isdigit() else stem


# def save_track_txt_from_frames(
#     model_path: str,
#     frames_dir: str,
#     tracker_config: str,
#     output_txt: str,
#     conf_thres: float = 0.25,
#     iou_thres: float = 0.5,
#     classes=None
# ):
#     os.makedirs(os.path.dirname(output_txt), exist_ok=True) if os.path.dirname(output_txt) else None

#     model = YOLO(model_path)

#     image_paths = []
#     for ext in ("*.jpg", "*.png", "*.jpeg", "*.bmp"):
#         image_paths.extend(glob.glob(os.path.join(frames_dir, ext)))
#     image_paths = sorted(image_paths, key=natural_sort_key)

#     if not image_paths:
#         raise FileNotFoundError(f"在 {frames_dir} 中没有找到图片帧")

#     with open(output_txt, "w", encoding="utf-8") as f:
#         for frame_id, img_path in enumerate(image_paths, start=1):
#             result = model.track(
#                 source=img_path,
#                 persist=True,             # 保持跟踪状态连续
#                 tracker=tracker_config,   # bytetrack.yaml
#                 conf=conf_thres,
#                 iou=iou_thres,
#                 classes=classes,
#                 verbose=False
#             )[0]

#             if result.boxes is None or result.boxes.id is None or len(result.boxes) == 0:
#                 continue

#             xyxy = result.boxes.xyxy.cpu().numpy()
#             ids = result.boxes.id.int().cpu().tolist()
#             confs = result.boxes.conf.cpu().numpy().tolist()
#             clss = result.boxes.cls.int().cpu().tolist()

#             for box, tid, score, cls_id in zip(xyxy, ids, confs, clss):
#                 x1, y1, x2, y2 = map(float, box)
#                 cx = (x1 + x2) / 2.0
#                 cy_bottom = y2

#                 line = f"{frame_id},{tid},{cls_id},{x1:.2f},{y1:.2f},{x2:.2f},{y2:.2f},{cx:.2f},{cy_bottom:.2f},{score:.4f}\n"
#                 f.write(line)

#             if frame_id % 100 == 0:
#                 print(f"已处理 {frame_id}/{len(image_paths)} 帧")

#     print(f"轨迹已保存到: {output_txt}")


# if __name__ == "__main__":
#     save_track_txt_from_frames(
#         model_path="runs/detect/v6.4_n_p2_shanp5_cl3_SimAM_EUCB/weights/best.pt",
#         frames_dir="experient_fig/doublesight/frame1",
#         tracker_config="ultralytics/cfg/trackers/botsort.yaml",
#         output_txt="results/droneA_tracks.txt",
#         conf_thres=0.25,
#         iou_thres=0.5,
#         classes=[3]   # 例如只跟踪 person；如果不限制就写 None
#     )



import os
import glob
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


def save_track_txt_from_frames(
    model_path: str,
    frames_dir: str,
    tracker_config: str,
    output_txt: str,
    output_reid_npz: str,
    conf_thres: float = 0.25,
    iou_thres: float = 0.5,
    classes=None
):
    os.makedirs(os.path.dirname(output_txt), exist_ok=True) if os.path.dirname(output_txt) else None
    os.makedirs(os.path.dirname(output_reid_npz), exist_ok=True) if os.path.dirname(output_reid_npz) else None

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

    with open(output_txt, "w", encoding="utf-8") as f:
        for frame_id, img_path in enumerate(image_paths, start=1):
            result = model.track(
                source=img_path,
                persist=True,             # 保持跟踪状态连续
                tracker=tracker_config,   # 这里用你改过的 botsort yaml
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

                # 1) 原txt保持不变
                line = f"{frame_id},{tid},{cls_id},{x1:.2f},{y1:.2f},{x2:.2f},{y2:.2f},{cx:.2f},{cy_bottom:.2f},{score:.4f}\n"
                f.write(line)

                # 2) 导出ReID特征
                t = track_map.get(int(tid), None)
                feat = None

                if t is not None:
                    # 优先取 smooth_feat
                    feat = getattr(t, "smooth_feat", None)

                    # 如果这一帧 smooth_feat 还没有，再退回 curr_feat
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

            if frame_id % 100 == 0:
                print(
                    f"已处理 {frame_id}/{len(image_paths)} 帧 | "
                    f"检测框累计: {total_boxes} | "
                    f"ReID累计保存: {saved_feats}"
                )

    print(f"轨迹已保存到: {output_txt}")

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
        f"缺失特征={missing_feats}"
    )

    if saved_feats == 0:
        print(
            "[警告] 没有导出任何 ReID 特征，请检查：\n"
            "1) tracker 是否为 BoT-SORT；\n"
            "2) yaml 中是否 with_reid: True；\n"
            "3) bot_sort.py 是否仍会初始化/更新 curr_feat 和 smooth_feat；\n"
            "4) model 设置是否为 auto 或有效 ReID 模型。"
        )


if __name__ == "__main__":
    save_track_txt_from_frames(
        model_path="runs/detect/v6.4_n_p2_shanp5_cl3_SimAM_EUCB/weights/best.pt",
        frames_dir="experient_fig/doublesight/frame2",
        tracker_config="ultralytics/cfg/trackers/botsort.yaml",
        output_txt="experient_fig/double_reid_test/droneB_tracks.txt",
        output_reid_npz="experient_fig/doublesight_reidkalm/droneB_reid.npz",
        conf_thres=0.25,
        iou_thres=0.5,
        classes=[3]
    )