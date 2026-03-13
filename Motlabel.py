import os
import cv2
from glob import glob
from ultralytics import YOLO


def ensure_parent_dir(file_path: str):
    parent = os.path.dirname(file_path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def list_images(img_dir: str):
    """
    读取 img1 下所有图片，并按 MOT 帧号顺序排序。
    优先按文件名数字排序，例如 000001.jpg -> 1
    """
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    files = []
    for ext in exts:
        files.extend(glob(os.path.join(img_dir, ext)))

    if not files:
        raise FileNotFoundError(f"在目录中没有找到图片: {img_dir}")

    def sort_key(p):
        stem = os.path.splitext(os.path.basename(p))[0]
        try:
            return int(stem)
        except ValueError:
            return stem

    files = sorted(files, key=sort_key)
    return files


def get_frame_id_from_name(img_path: str, fallback_id: int):
    """
    MOT17 通常图片名就是帧号，如 000001.jpg
    若不是纯数字，则回退到 fallback_id
    """
    stem = os.path.splitext(os.path.basename(img_path))[0]
    try:
        return int(stem)
    except ValueError:
        return fallback_id


def xyxy_to_ltwh(x1, y1, x2, y2, one_based=True):
    """
    xyxy -> left, top, width, height
    MOT 常用 left/top 从 1 开始，默认 +1
    """
    left = float(x1) + (1.0 if one_based else 0.0)
    top = float(y1) + (1.0 if one_based else 0.0)
    width = max(0.0, float(x2) - float(x1))
    height = max(0.0, float(y2) - float(y1))
    return left, top, width, height


def format_tracker_line_7col(frame_id, track_id, left, top, width, height, conf):
    """
    TrackEval 本地评测推荐的 7 列 tracker 格式：
    frame, id, left, top, width, height, conf
    """
    return f"{frame_id},{track_id},{left:.2f},{top:.2f},{width:.2f},{height:.2f},{conf:.6f}\n"


def draw_tracks(frame, xyxy, track_ids, confs):
    for box, tid, score in zip(xyxy, track_ids, confs):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"ID:{tid} {score:.2f}",
            (x1, max(0, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2
        )
    return frame


def save_mot17_imgseq_botsort_txt(
    model_path: str,
    img_dir: str,
    output_txt: str,
    tracker: str = "botsort.yaml",
    conf: float = 0.1,
    iou: float = 0.5,
    imgsz: int = 1280,
    device=0,
    person_class_id: int = 0,
    one_based_bbox: bool = True,
    save_vis_video: bool = False,
    vis_video_path: str = None,
    vis_fps: int = 25,
):
    """
    直接读取 MOT17 的 img1 图片序列，用 Ultralytics BoT-SORT 跟踪，
    导出 TrackEval 可读的 txt（只输出 person）。

    参数：
    - model_path: 检测模型路径，例如 weights/best.pt
    - img_dir: MOT17 序列图片目录，例如 MOT17/train/MOT17-05-FRCNN/img1
    - output_txt: 输出 txt 路径，例如 tracker_results/MyBoTSORT/data/MOT17-05-FRCNN.txt
    - tracker: Ultralytics 跟踪器配置，BoT-SORT 用 botsort.yaml
    - person_class_id: 你模型中 person 的类别 ID，COCO 一般为 0
    """
    ensure_parent_dir(output_txt)
    img_paths = list_images(img_dir)

    model = YOLO(model_path)

    writer = None
    if save_vis_video:
        sample = cv2.imread(img_paths[0])
        if sample is None:
            raise ValueError(f"无法读取首帧图片: {img_paths[0]}")
        h, w = sample.shape[:2]

        if vis_video_path is None:
            vis_video_path = os.path.splitext(output_txt)[0] + "_vis.mp4"
        ensure_parent_dir(vis_video_path)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(vis_video_path, fourcc, vis_fps, (w, h))

    saved_lines = 0

    with open(output_txt, "w", encoding="utf-8") as f:
        for idx, img_path in enumerate(img_paths, start=1):
            frame = cv2.imread(img_path)
            if frame is None:
                print(f"[警告] 跳过无法读取的图片: {img_path}")
                continue

            # 直接用图片名作为帧号，MOT17 通常是 000001.jpg -> 1
            frame_id = get_frame_id_from_name(img_path, idx)

            # persist=True: 保持跨帧轨迹连续
            result = model.track(
                frame,
                persist=True,
                tracker=tracker,
                conf=conf,
                iou=iou,
                imgsz=imgsz,
                device=device,
                classes=[person_class_id],   # 只检测/跟踪人
                verbose=False,
            )[0]

            boxes = result.boxes

            draw_xyxy = []
            draw_ids = []
            draw_scores = []

            if boxes is not None and boxes.id is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.cpu().numpy()
                track_ids = boxes.id.int().cpu().tolist()

                if boxes.conf is not None:
                    scores = boxes.conf.cpu().tolist()
                else:
                    scores = [1.0] * len(track_ids)

                if boxes.cls is not None:
                    cls_ids = boxes.cls.int().cpu().tolist()
                else:
                    cls_ids = [person_class_id] * len(track_ids)

                for box, tid, score, cls_id in zip(xyxy, track_ids, scores, cls_ids):
                    # 双保险：即使 classes 已经过滤，这里仍只保留 person
                    if cls_id != person_class_id:
                        continue

                    x1, y1, x2, y2 = box.tolist()
                    left, top, w, h = xyxy_to_ltwh(
                        x1, y1, x2, y2, one_based=one_based_bbox
                    )

                    if w <= 0 or h <= 0:
                        continue

                    line = format_tracker_line_7col(
                        frame_id=frame_id,
                        track_id=int(tid),
                        left=left,
                        top=top,
                        width=w,
                        height=h,
                        conf=float(score),
                    )
                    f.write(line)
                    saved_lines += 1

                    draw_xyxy.append(box)
                    draw_ids.append(int(tid))
                    draw_scores.append(float(score))

            if writer is not None:
                vis = frame.copy()
                if len(draw_xyxy) > 0:
                    vis = draw_tracks(vis, draw_xyxy, draw_ids, draw_scores)
                writer.write(vis)

    if writer is not None:
        writer.release()

    print(f"处理完成")
    print(f"图片目录: {img_dir}")
    print(f"输出 txt: {output_txt}")
    print(f"总图片数: {len(img_paths)}")
    print(f"总写入行数: {saved_lines}")
    if save_vis_video:
        print(f"可视化视频: {vis_video_path}")


if __name__ == "__main__":
    # =========================
    # 你只需要改下面这些路径
    # =========================
    model_path = "pretrained/yolo11s.pt"
    img_dir = r"datasetTrack/MOT17/train/MOT17-13-FRCNN/img1"
    output_txt = r"datasetTrack/MOT17/outmp4/MOT17-13-FRCNN.txt"

    save_mot17_imgseq_botsort_txt(
        model_path=model_path,
        img_dir=img_dir,
        output_txt=output_txt,
        tracker="botsort.yaml",
        conf=0.1,
        iou=0.5,
        imgsz=1280,
        device=0,              # 没有 GPU 就改成 "cpu"
        person_class_id=0,     # 若你的模型中 person 不是 0，这里改掉
        one_based_bbox=True,
        save_vis_video=False,  # 想顺便保存可视化视频就改 True
    )