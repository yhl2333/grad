from __future__ import annotations

import os
import shutil
import zipfile
from collections import defaultdict
from pathlib import Path

from PIL import Image


# =========================
# 项目相对路径配置（直接改这里）
# =========================
PROJECT_ROOT = Path(__file__).resolve().parent

TRAIN_ZIP = PROJECT_ROOT / "datasetTrack" / "VisDrone2019-MOT-train.zip"
VAL_ZIP = PROJECT_ROOT / "datasetTrack" / "VisDrone2019-MOT-val.zip"

EXTRACT_ROOT = PROJECT_ROOT / "datasetTrack" / "extracted"
TRAIN_ROOT = EXTRACT_ROOT / "VisDrone2019-MOT-train"
VAL_ROOT = EXTRACT_ROOT / "VisDrone2019-MOT-val"

YOLO_OUT_ROOT = PROJECT_ROOT / "datasetTrack" / "VisDrone2019-MOT-YOLO"

COPY_IMAGES = False  # False=优先软链接，True=复制图片


# 只保留最终 MOT 常用 5 类
# VisDrone官方类别ID -> YOLO类别ID
VISDRONE_TO_YOLO = {
    1: 0,  # pedestrian
    4: 1,  # car
    5: 2,  # van
    6: 3,  # truck
    9: 4,  # bus
}


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def unzip_if_needed(zip_path: Path, extract_to: Path):
    if extract_to.exists() and any(extract_to.iterdir()):
        print(f"[Skip unzip] {extract_to} already exists.")
        return

    if not zip_path.exists():
        raise FileNotFoundError(f"Zip file not found: {zip_path}")

    ensure_dir(extract_to)
    print(f"[Unzipping] {zip_path} -> {extract_to}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_to)

    # 如果解压后多了一层同名目录，做一次兼容
    inner_items = list(extract_to.iterdir())
    if len(inner_items) == 1 and inner_items[0].is_dir():
        inner_dir = inner_items[0]
        # 若该目录中存在 sequences/annotations，移动到 extract_to
        if (inner_dir / "sequences").exists() and (inner_dir / "annotations").exists():
            tmp_dir = extract_to.parent / f"{extract_to.name}_tmp_move"
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir)
            inner_dir.rename(tmp_dir)
            shutil.rmtree(extract_to)
            tmp_dir.rename(extract_to)

    print(f"[Done unzip] {extract_to}")


def safe_link_or_copy(src: Path, dst: Path, copy_images: bool = False):
    if dst.exists():
        return
    if copy_images:
        shutil.copy2(src, dst)
    else:
        try:
            os.symlink(src.resolve(), dst)
        except OSError:
            shutil.copy2(src, dst)


def clamp_box(x: float, y: float, w: float, h: float, img_w: int, img_h: int):
    x1 = max(0.0, x)
    y1 = max(0.0, y)
    x2 = min(float(img_w), x + w)
    y2 = min(float(img_h), y + h)
    bw = x2 - x1
    bh = y2 - y1
    if bw <= 1.0 or bh <= 1.0:
        return None
    return x1, y1, bw, bh


def resolve_dataset_root(root: Path) -> Path:
    """
    兼容两种情况：
    1. root/sequences + root/annotations
    2. root/VisDrone2019-MOT-train(or val)/sequences + annotations
    """
    if (root / "sequences").exists() and (root / "annotations").exists():
        return root

    subdirs = [p for p in root.iterdir() if p.is_dir()]
    for d in subdirs:
        if (d / "sequences").exists() and (d / "annotations").exists():
            return d

    raise FileNotFoundError(f"Cannot find sequences/annotations under: {root}")


def convert_one_split(src_root: Path, dst_root: Path, split: str, copy_images: bool = False):
    src_root = resolve_dataset_root(src_root)

    seq_root = src_root / "sequences"
    ann_root = src_root / "annotations"

    out_img_root = dst_root / "images" / split
    out_lbl_root = dst_root / "labels" / split
    ensure_dir(out_img_root)
    ensure_dir(out_lbl_root)

    seq_dirs = sorted([p for p in seq_root.iterdir() if p.is_dir()])
    if not seq_dirs:
        raise FileNotFoundError(f"No sequence folders found in: {seq_root}")

    for seq_dir in seq_dirs:
        seq_name = seq_dir.name
        ann_path = ann_root / f"{seq_name}.txt"
        if not ann_path.exists():
            raise FileNotFoundError(f"Annotation file not found: {ann_path}")

        out_seq_img = out_img_root / seq_name
        out_seq_lbl = out_lbl_root / seq_name
        ensure_dir(out_seq_img)
        ensure_dir(out_seq_lbl)

        image_files = sorted(seq_dir.glob("*.jpg"))
        if not image_files:
            image_files = sorted(seq_dir.glob("*.png"))
        if not image_files:
            raise FileNotFoundError(f"No images found in sequence: {seq_dir}")

        frameid_to_name = {}
        frameid_to_size = {}

        for idx, img_path in enumerate(image_files, start=1):
            dst_img = out_seq_img / img_path.name
            safe_link_or_copy(img_path, dst_img, copy_images=copy_images)

            with Image.open(img_path) as im:
                img_w, img_h = im.size

            frameid_to_name[idx] = img_path.stem
            frameid_to_size[idx] = (img_w, img_h)

            # 先创建空标签，保证空目标帧也有txt
            (out_seq_lbl / f"{img_path.stem}.txt").touch(exist_ok=True)

        labels_per_frame = defaultdict(list)

        with ann_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split(",")
                if len(parts) < 10:
                    continue

                frame_id = int(float(parts[0]))
                x = float(parts[2])
                y = float(parts[3])
                w = float(parts[4])
                h = float(parts[5])
                score = int(float(parts[6]))       # GT中 1=有效框
                category = int(float(parts[7]))

                if score != 1:
                    continue
                if category not in VISDRONE_TO_YOLO:
                    continue
                if frame_id not in frameid_to_size:
                    continue

                img_w, img_h = frameid_to_size[frame_id]
                clipped = clamp_box(x, y, w, h, img_w, img_h)
                if clipped is None:
                    continue

                x1, y1, bw, bh = clipped
                cx = (x1 + bw / 2.0) / img_w
                cy = (y1 + bh / 2.0) / img_h
                nw = bw / img_w
                nh = bh / img_h

                cls_id = VISDRONE_TO_YOLO[category]
                labels_per_frame[frame_id].append(
                    f"{cls_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}"
                )

        for frame_id, lines in labels_per_frame.items():
            stem = frameid_to_name[frame_id]
            txt_path = out_seq_lbl / f"{stem}.txt"
            with txt_path.open("w", encoding="utf-8") as f:
                f.write("\n".join(lines) + ("\n" if lines else ""))

        print(f"[OK] {split}: {seq_name}")


def main():
    unzip_if_needed(TRAIN_ZIP, TRAIN_ROOT)
    unzip_if_needed(VAL_ZIP, VAL_ROOT)

    convert_one_split(TRAIN_ROOT, YOLO_OUT_ROOT, split="train", copy_images=COPY_IMAGES)
    convert_one_split(VAL_ROOT, YOLO_OUT_ROOT, split="val", copy_images=COPY_IMAGES)

    print("\nDone.")
    print(f"YOLO dataset saved to: {YOLO_OUT_ROOT}")


if __name__ == "__main__":
    main()