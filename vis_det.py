from __future__ import annotations

import shutil
import zipfile
from pathlib import Path
import cv2

from ultralytics import YOLO


# =========================
# 路径配置
# =========================
PROJECT_ROOT = Path(__file__).resolve().parent

VAL_ZIP = PROJECT_ROOT / "datasetTrack" / "VisDrone2019-MOT-val.zip"
EXTRACT_ROOT = PROJECT_ROOT / "datasetTrack" / "extracted"
VAL_ROOT = EXTRACT_ROOT / "VisDrone2019-MOT-val"

MODEL_PATH = PROJECT_ROOT / "runs/detect/origin_visdrone_cl9/weights/best.pt"

# 输出目录
OUT_ROOT = PROJECT_ROOT / "runs" / "visdrone_det_vis2"

# 推理参数
IMGSZ = 640        # 你训练时是 640，这里先按 640 看检测结果更稳
CONF = 0.2
IOU = 0.6
DEVICE = 0           # 没GPU可改成 "cpu"

# =========================
# 如果你的模型真的是标准 VisDrone-DET 十类顺序：
# 0 pedestrian
# 1 people
# 2 bicycle
# 3 car
# 4 van
# 5 truck
# 6 tricycle
# 7 awning-tricycle
# 8 bus
# 9 motor
#
# 这里映射到 VisDrone 官方 MOT 常用五类类别号
# 模型类别ID -> VisDrone官方类别ID
# =========================
DET10_TO_VISDRONE_MOT5 = {
    0: 1,  # pedestrian
    3: 4,  # car
    4: 5,  # van
    5: 6,  # truck
    8: 9,  # bus
}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def unzip_if_needed(zip_path: Path, extract_to: Path) -> None:
    if extract_to.exists() and any(extract_to.iterdir()):
        print(f"[Skip unzip] {extract_to} already exists.")
        return

    if not zip_path.exists():
        raise FileNotFoundError(f"Zip file not found: {zip_path}")

    ensure_dir(extract_to)
    print(f"[Unzipping] {zip_path} -> {extract_to}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_to)

    inner_items = list(extract_to.iterdir())
    if len(inner_items) == 1 and inner_items[0].is_dir():
        inner_dir = inner_items[0]
        if (inner_dir / "sequences").exists():
            tmp_dir = extract_to.parent / f"{extract_to.name}_tmp_move"
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir)
            inner_dir.rename(tmp_dir)
            shutil.rmtree(extract_to)
            tmp_dir.rename(extract_to)

    print(f"[Done unzip] {extract_to}")


def resolve_val_root(root: Path) -> Path:
    if (root / "sequences").exists():
        return root

    subdirs = [p for p in root.iterdir() if p.is_dir()]
    for d in subdirs:
        if (d / "sequences").exists():
            return d

    raise FileNotFoundError(f"Cannot find sequences folder under: {root}")


def save_detection_visualization(image_path: Path, result, save_path: Path) -> None:
    """保存画框后的检测图。"""
    plotted = result.plot()  # BGR ndarray
    cv2.imwrite(str(save_path), plotted)


def save_raw_detection_txt(result, save_path: Path) -> None:
    """
    保存原始检测结果：
    每行格式：
    cls_id conf x1 y1 x2 y2
    """
    lines = []

    if result.boxes is not None and len(result.boxes) > 0:
        xyxy = result.boxes.xyxy.cpu().numpy()
        clss = result.boxes.cls.int().cpu().tolist()
        confs = result.boxes.conf.cpu().tolist()

        for box, cls_id, score in zip(xyxy, clss, confs):
            x1, y1, x2, y2 = map(float, box.tolist())
            lines.append(
                f"{int(cls_id)} {float(score):.6f} {x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f}"
            )

    with save_path.open("w", encoding="utf-8") as f:
        if lines:
            f.write("\n".join(lines) + "\n")


def save_mot5_detection_txt(frame_idx: int, result, save_path: Path) -> None:
    """
    保存只保留 MOT 五类的检测结果，写成 VisDrone 风格（单帧版）：
    每行格式：
    frame_idx,-1,left,top,width,height,score,object_category,-1,-1

    注意：这里只是“检测结果”，没有 track id，所以 target_id 用 -1。
    """
    lines = []

    if result.boxes is not None and len(result.boxes) > 0:
        xyxy = result.boxes.xyxy.cpu().numpy()
        clss = result.boxes.cls.int().cpu().tolist()
        confs = result.boxes.conf.cpu().tolist()

        for box, cls_id, score in zip(xyxy, clss, confs):
            cls_id = int(cls_id)
            if cls_id not in DET10_TO_VISDRONE_MOT5:
                continue

            x1, y1, x2, y2 = map(float, box.tolist())
            w = max(0.0, x2 - x1)
            h = max(0.0, y2 - y1)
            if w <= 1.0 or h <= 1.0:
                continue

            vis_cls = DET10_TO_VISDRONE_MOT5[cls_id]
            lines.append(
                f"{frame_idx},-1,{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},{float(score):.6f},{vis_cls},-1,-1"
            )

    with save_path.open("w", encoding="utf-8") as f:
        if lines:
            f.write("\n".join(lines) + "\n")


def run_detection_on_val() -> None:
    unzip_if_needed(VAL_ZIP, VAL_ROOT)
    data_root = resolve_val_root(VAL_ROOT)
    seq_root = data_root / "sequences"

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    model = YOLO(str(MODEL_PATH))

    print("Loaded model:", MODEL_PATH)
    print("model.names:", model.names)
    print("num classes:", len(model.names))
    print("-" * 80)

    seq_dirs = sorted([p for p in seq_root.iterdir() if p.is_dir()])
    if not seq_dirs:
        raise FileNotFoundError(f"No sequence folders found in {seq_root}")

    for seq_dir in seq_dirs:
        seq_name = seq_dir.name
        print(f"[Detecting] {seq_name}")

        out_img_dir = OUT_ROOT / "images" / seq_name
        out_raw_txt_dir = OUT_ROOT / "txt_raw" / seq_name
        out_mot5_txt_dir = OUT_ROOT / "txt_mot5" / seq_name

        ensure_dir(out_img_dir)
        ensure_dir(out_raw_txt_dir)
        ensure_dir(out_mot5_txt_dir)

        img_files = sorted(seq_dir.glob("*.jpg"))
        if not img_files:
            img_files = sorted(seq_dir.glob("*.png"))
        if not img_files:
            raise FileNotFoundError(f"No images found in {seq_dir}")

        for frame_idx, img_path in enumerate(img_files, start=1):
            results = model.predict(
                source=str(img_path),
                conf=CONF,
                iou=IOU,
                imgsz=IMGSZ,
                device=DEVICE,
                save=False,
                verbose=False,
            )

            r = results[0]

            # 1) 保存画框图片
            vis_save_path = out_img_dir / img_path.name
            save_detection_visualization(img_path, r, vis_save_path)

            # 2) 保存原始检测 txt（模型原始类别号）
            raw_txt_path = out_raw_txt_dir / f"{img_path.stem}.txt"
            save_raw_detection_txt(r, raw_txt_path)

            # 3) 保存只保留 MOT 五类的 txt（映射成 VisDrone 官方类别号）
            mot5_txt_path = out_mot5_txt_dir / f"{img_path.stem}.txt"
            save_mot5_detection_txt(frame_idx, r, mot5_txt_path)

        print(f"[OK] {seq_name}")


if __name__ == "__main__":
    run_detection_on_val()