# from __future__ import annotations

# import csv
# import gc
# import math
# import os
# from dataclasses import dataclass
# from pathlib import Path
# from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# # Reduce CUDA fragmentation before torch import
# os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt
# from PIL import Image
# import torch
# from ultralytics import YOLO


# # =========================
# # 配置区：直接改这里即可
# # =========================
# PROJECT_ROOT = Path(__file__).resolve().parent
# DATASET_ROOT = PROJECT_ROOT / "datasets" / "VisDrone"
# TRAINED_MODEL = "runs/detect/v6.4_n_p2_shanp5_cl3_SimAM_EUCB/weights/best.pt"
# BASELINE_MODEL = "ultralytics/runs/detect/n_cl3_640/weights/best.pt"
# OUTPUT_DIR = PROJECT_ROOT / "runs" / "scale_compare_visdrone_lowmem"

# # auto / yolo / visdrone_det / visdrone_mot
# DATASET_FORMAT = "auto"
# SPLIT = "val"

# # 低显存默认参数
# DEVICE = "0"          # 显存还是不够就改成 "cpu"
# IMGSZ = 640
# CONF = 0.25
# IOU = 0.70
# MAX_DET = 300
# HALF = True            # CPU 下会自动关闭
# VERBOSE = True

# # 是否只统计这些类别；None 表示不过滤
# # 例如只看 VisDrone-MOT 常用 5 类时可改成 [0, 3, 4, 5, 8]
# KEEP_CLASSES: Optional[List[int]] = None

# # 尺度统计方式：按 sqrt(area) 分箱，更直观
# SCALE_BINS = [0, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 2048]

# # 可选：只分析前 N 张图，调试时有用；None 表示全部
# LIMIT_IMAGES: Optional[int] = None


# @dataclass
# class BoxRecord:
#     cls_id: int
#     x1: float
#     y1: float
#     x2: float
#     y2: float

#     @property
#     def width(self) -> float:
#         return max(0.0, self.x2 - self.x1)

#     @property
#     def height(self) -> float:
#         return max(0.0, self.y2 - self.y1)

#     @property
#     def scale(self) -> float:
#         return math.sqrt(self.width * self.height)


# def ensure_dir(path: Path) -> None:
#     path.mkdir(parents=True, exist_ok=True)


# def release_cuda_memory() -> None:
#     gc.collect()
#     if torch.cuda.is_available():
#         torch.cuda.empty_cache()
#         try:
#             torch.cuda.ipc_collect()
#         except Exception:
#             pass


# def is_image_file(path: Path) -> bool:
#     return path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


# def detect_dataset_format(dataset_root: Path, split: str) -> str:
#     if (dataset_root / "images" / split).exists() and (dataset_root / "labels" / split).exists():
#         return "yolo"
#     if (dataset_root / "images").exists() and (dataset_root / "annotations").exists():
#         return "visdrone_det"
#     if (dataset_root / "sequences").exists() and (dataset_root / "annotations").exists():
#         return "visdrone_mot"

#     # 兼容多包一层目录
#     for sub in dataset_root.iterdir() if dataset_root.exists() else []:
#         if not sub.is_dir():
#             continue
#         if (sub / "images" / split).exists() and (sub / "labels" / split).exists():
#             return "yolo"
#         if (sub / "images").exists() and (sub / "annotations").exists():
#             return "visdrone_det"
#         if (sub / "sequences").exists() and (sub / "annotations").exists():
#             return "visdrone_mot"

#     raise FileNotFoundError(f"Cannot auto-detect dataset format under: {dataset_root}")


# def normalize_dataset_root(dataset_root: Path, fmt: str, split: str) -> Path:
#     if fmt == "yolo" and (dataset_root / "images" / split).exists():
#         return dataset_root
#     if fmt == "visdrone_det" and (dataset_root / "images").exists() and (dataset_root / "annotations").exists():
#         return dataset_root
#     if fmt == "visdrone_mot" and (dataset_root / "sequences").exists() and (dataset_root / "annotations").exists():
#         return dataset_root

#     for sub in dataset_root.iterdir() if dataset_root.exists() else []:
#         if not sub.is_dir():
#             continue
#         if fmt == "yolo" and (sub / "images" / split).exists() and (sub / "labels" / split).exists():
#             return sub
#         if fmt == "visdrone_det" and (sub / "images").exists() and (sub / "annotations").exists():
#             return sub
#         if fmt == "visdrone_mot" and (sub / "sequences").exists() and (sub / "annotations").exists():
#             return sub

#     raise FileNotFoundError(f"Cannot find valid root for format={fmt} under: {dataset_root}")


# def bin_labels(bins: Sequence[float]) -> List[str]:
#     labels = []
#     for i in range(len(bins) - 1):
#         labels.append(f"[{int(bins[i])},{int(bins[i+1])})")
#     return labels


# def scale_to_bin(scale: float, bins: Sequence[float]) -> Optional[int]:
#     for i in range(len(bins) - 1):
#         if bins[i] <= scale < bins[i + 1]:
#             return i
#     return len(bins) - 2 if scale >= bins[-1] else None


# def make_empty_hist(bins: Sequence[float]) -> List[int]:
#     return [0 for _ in range(len(bins) - 1)]


# def add_scale_to_hist(hist: List[int], scale: float, bins: Sequence[float]) -> None:
#     idx = scale_to_bin(scale, bins)
#     if idx is not None:
#         hist[idx] += 1


# def load_yolo_label_file(label_path: Path, img_w: int, img_h: int, keep_classes: Optional[Sequence[int]]) -> List[BoxRecord]:
#     boxes: List[BoxRecord] = []
#     if not label_path.exists():
#         return boxes
#     with label_path.open("r", encoding="utf-8") as f:
#         for line in f:
#             parts = line.strip().split()
#             if len(parts) < 5:
#                 continue
#             cls_id = int(float(parts[0]))
#             if keep_classes is not None and cls_id not in keep_classes:
#                 continue
#             cx = float(parts[1]) * img_w
#             cy = float(parts[2]) * img_h
#             bw = float(parts[3]) * img_w
#             bh = float(parts[4]) * img_h
#             x1 = cx - bw / 2.0
#             y1 = cy - bh / 2.0
#             x2 = cx + bw / 2.0
#             y2 = cy + bh / 2.0
#             boxes.append(BoxRecord(cls_id, x1, y1, x2, y2))
#     return boxes


# def iter_yolo_samples(dataset_root: Path, split: str, keep_classes: Optional[Sequence[int]]) -> Iterable[Tuple[Path, List[BoxRecord]]]:
#     img_root = dataset_root / "images" / split
#     lbl_root = dataset_root / "labels" / split
#     n = 0
#     for img_path in sorted(img_root.rglob("*")):
#         if not img_path.is_file() or not is_image_file(img_path):
#             continue
#         rel = img_path.relative_to(img_root)
#         label_path = (lbl_root / rel).with_suffix(".txt")
#         with Image.open(img_path) as im:
#             img_w, img_h = im.size
#         gt_boxes = load_yolo_label_file(label_path, img_w, img_h, keep_classes)
#         yield img_path, gt_boxes
#         n += 1
#         if LIMIT_IMAGES is not None and n >= LIMIT_IMAGES:
#             break


# def parse_visdrone_det_ann(ann_path: Path, keep_classes: Optional[Sequence[int]]) -> List[BoxRecord]:
#     boxes: List[BoxRecord] = []
#     with ann_path.open("r", encoding="utf-8") as f:
#         for line in f:
#             parts = [p.strip() for p in line.strip().split(",")]
#             if len(parts) < 8:
#                 continue
#             x, y, w, h = map(float, parts[:4])
#             score = int(float(parts[4]))
#             cls_id = int(float(parts[5]))
#             # VisDrone 官方类别从1开始；转成0开始，方便和 YOLO 类别对齐
#             cls_zero_based = cls_id - 1
#             if keep_classes is not None and cls_zero_based not in keep_classes:
#                 continue
#             if score == 0 or w <= 1.0 or h <= 1.0:
#                 continue
#             boxes.append(BoxRecord(cls_zero_based, x, y, x + w, y + h))
#     return boxes


# def iter_visdrone_det_samples(dataset_root: Path, split: str, keep_classes: Optional[Sequence[int]]) -> Iterable[Tuple[Path, List[BoxRecord]]]:
#     split_map = {"train": "train", "val": "val", "test": "test-dev"}
#     sub = split_map.get(split, split)
#     img_root = dataset_root / "images" / sub
#     ann_root = dataset_root / "annotations" / sub
#     n = 0
#     for img_path in sorted(img_root.glob("*")):
#         if not img_path.is_file() or not is_image_file(img_path):
#             continue
#         ann_path = ann_root / f"{img_path.stem}.txt"
#         gt_boxes = parse_visdrone_det_ann(ann_path, keep_classes)
#         yield img_path, gt_boxes
#         n += 1
#         if LIMIT_IMAGES is not None and n >= LIMIT_IMAGES:
#             break


# def parse_visdrone_mot_ann(ann_path: Path, keep_classes: Optional[Sequence[int]]) -> Dict[int, List[BoxRecord]]:
#     frame_to_boxes: Dict[int, List[BoxRecord]] = {}
#     with ann_path.open("r", encoding="utf-8") as f:
#         for line in f:
#             parts = [p.strip() for p in line.strip().split(",")]
#             if len(parts) < 8:
#                 continue
#             frame_id = int(float(parts[0]))
#             x = float(parts[2])
#             y = float(parts[3])
#             w = float(parts[4])
#             h = float(parts[5])
#             score = int(float(parts[6]))
#             cls_id = int(float(parts[7]))
#             cls_zero_based = cls_id - 1
#             if keep_classes is not None and cls_zero_based not in keep_classes:
#                 continue
#             if score != 1 or w <= 1.0 or h <= 1.0:
#                 continue
#             frame_to_boxes.setdefault(frame_id, []).append(BoxRecord(cls_zero_based, x, y, x + w, y + h))
#     return frame_to_boxes


# def iter_visdrone_mot_samples(dataset_root: Path, keep_classes: Optional[Sequence[int]]) -> Iterable[Tuple[Path, List[BoxRecord]]]:
#     seq_root = dataset_root / "sequences"
#     ann_root = dataset_root / "annotations"
#     n = 0
#     for seq_dir in sorted(p for p in seq_root.iterdir() if p.is_dir()):
#         ann_path = ann_root / f"{seq_dir.name}.txt"
#         frame_to_boxes = parse_visdrone_mot_ann(ann_path, keep_classes)
#         frame_idx = 0
#         for img_path in sorted(seq_dir.glob("*.jpg")) + sorted(seq_dir.glob("*.png")):
#             frame_idx += 1
#             gt_boxes = frame_to_boxes.get(frame_idx, [])
#             yield img_path, gt_boxes
#             n += 1
#             if LIMIT_IMAGES is not None and n >= LIMIT_IMAGES:
#                 return


# def sample_iterator(dataset_root: Path, fmt: str, split: str, keep_classes: Optional[Sequence[int]]):
#     if fmt == "yolo":
#         return iter_yolo_samples(dataset_root, split, keep_classes)
#     if fmt == "visdrone_det":
#         return iter_visdrone_det_samples(dataset_root, split, keep_classes)
#     if fmt == "visdrone_mot":
#         return iter_visdrone_mot_samples(dataset_root, keep_classes)
#     raise ValueError(f"Unsupported dataset format: {fmt}")


# def gt_scale_histogram(dataset_root: Path, fmt: str, split: str, keep_classes: Optional[Sequence[int]], bins: Sequence[float]) -> Tuple[List[int], int]:
#     hist = make_empty_hist(bins)
#     image_count = 0
#     for _, gt_boxes in sample_iterator(dataset_root, fmt, split, keep_classes):
#         image_count += 1
#         for box in gt_boxes:
#             add_scale_to_hist(hist, box.scale, bins)
#     return hist, image_count


# def predict_scale_histogram(
#     model_path: str | Path,
#     dataset_root: Path,
#     fmt: str,
#     split: str,
#     keep_classes: Optional[Sequence[int]],
#     bins: Sequence[float],
#     device: str,
#     imgsz: int,
#     conf: float,
#     iou: float,
#     max_det: int,
#     half: bool,
#     verbose: bool,
# ) -> Tuple[List[int], int]:
#     hist = make_empty_hist(bins)
#     image_count = 0

#     if verbose:
#         print(f"\n[Load model] {model_path}")
#     model = YOLO(str(model_path))

#     use_half = half and device != "cpu" and torch.cuda.is_available()

#     try:
#         with torch.inference_mode():
#             for img_path, _ in sample_iterator(dataset_root, fmt, split, keep_classes):
#                 image_count += 1
#                 result_list = model.predict(
#                     source=str(img_path),
#                     imgsz=imgsz,
#                     conf=conf,
#                     iou=iou,
#                     max_det=max_det,
#                     device=device,
#                     half=use_half,
#                     classes=keep_classes,
#                     stream=False,
#                     verbose=False,
#                 )
#                 result = result_list[0]
#                 if result.boxes is not None and len(result.boxes) > 0:
#                     xyxy = result.boxes.xyxy.detach().cpu().numpy()
#                     for x1, y1, x2, y2 in xyxy:
#                         w = max(0.0, float(x2) - float(x1))
#                         h = max(0.0, float(y2) - float(y1))
#                         if w <= 1.0 or h <= 1.0:
#                             continue
#                         scale = math.sqrt(w * h)
#                         add_scale_to_hist(hist, scale, bins)

#                 # 关键：每张图后释放中间显存，牺牲一点速度换稳定
#                 del result_list, result
#                 if image_count % 20 == 0:
#                     release_cuda_memory()
#                 if verbose and image_count % 100 == 0:
#                     print(f"  processed: {image_count} images")
#     finally:
#         del model
#         release_cuda_memory()

#     return hist, image_count


# def save_summary_csv(
#     output_csv: Path,
#     bins: Sequence[float],
#     gt_hist: Sequence[int],
#     trained_hist: Sequence[int],
#     baseline_hist: Sequence[int],
# ) -> None:
#     labels = bin_labels(bins)
#     with output_csv.open("w", newline="", encoding="utf-8") as f:
#         writer = csv.writer(f)
#         writer.writerow(["scale_bin", "gt_count", "trained_pred_count", "yolo11n_pred_count"])
#         for i, label in enumerate(labels):
#             writer.writerow([label, gt_hist[i], trained_hist[i], baseline_hist[i]])


# def plot_all(
#     output_dir: Path,
#     bins: Sequence[float],
#     gt_hist: Sequence[int],
#     trained_hist: Sequence[int],
#     baseline_hist: Sequence[int],
# ) -> None:
#     labels = bin_labels(bins)
#     x = list(range(len(labels)))

#     # 图1：GT 尺度分布
#     plt.figure(figsize=(14, 6))
#     plt.bar(x, gt_hist)
#     plt.xticks(x, labels, rotation=45, ha="right")
#     plt.ylabel("Count")
#     plt.xlabel("sqrt(box area) in pixels")
#     plt.title("VisDrone Ground-Truth Scale Distribution")
#     plt.tight_layout()
#     plt.savefig(output_dir / "01_gt_scale_distribution.png", dpi=200)
#     plt.close()

#     # 图2：两个模型的检测数量随尺度变化
#     plt.figure(figsize=(14, 6))
#     plt.plot(x, gt_hist, marker="^", linestyle="--", label="GT")
#     plt.plot(x, trained_hist, marker="o", label="Trained model")
#     plt.plot(x, baseline_hist, marker="s", label="YOLO11n baseline")
#     plt.xticks(x, labels, rotation=45, ha="right")
#     plt.ylabel("Box count")
#     plt.xlabel("sqrt(box area) in pixels")
#     plt.title("Box Count vs. Object Scale")
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(output_dir / "02_prediction_count_vs_scale.png", dpi=200)
#     plt.close()

#     # 图3：GT + 两个模型放一起看整体趋势
#     width = 0.26
#     plt.figure(figsize=(16, 7))
#     plt.bar([v - width for v in x], gt_hist, width=width, label="GT")
#     plt.bar(x, trained_hist, width=width, label="Trained model")
#     plt.bar([v + width for v in x], baseline_hist, width=width, label="YOLO11n baseline")
#     plt.xticks(x, labels, rotation=45, ha="right")
#     plt.ylabel("Count")
#     plt.xlabel("sqrt(box area) in pixels")
#     plt.title("Scale-wise Count Comparison")
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(output_dir / "03_scale_count_comparison.png", dpi=200)
#     plt.close()

#     # 图4：两个模型相对 GT 的比例（不是召回率，只是数量比）
#     eps = 1e-9
#     trained_ratio = [trained_hist[i] / (gt_hist[i] + eps) for i in range(len(gt_hist))]
#     baseline_ratio = [baseline_hist[i] / (gt_hist[i] + eps) for i in range(len(gt_hist))]
#     plt.figure(figsize=(14, 6))
#     plt.plot(x, trained_ratio, marker="o", label="Trained / GT")
#     plt.plot(x, baseline_ratio, marker="s", label="YOLO11n / GT")
#     plt.xticks(x, labels, rotation=45, ha="right")
#     plt.ylabel("Prediction count / GT count")
#     plt.xlabel("sqrt(box area) in pixels")
#     plt.title("Relative Detection Count by Scale")
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(output_dir / "04_relative_count_vs_gt.png", dpi=200)
#     plt.close()


# def main() -> None:
#     ensure_dir(OUTPUT_DIR)

#     fmt = DATASET_FORMAT
#     if fmt == "auto":
#         fmt = detect_dataset_format(DATASET_ROOT, SPLIT)
#     dataset_root = normalize_dataset_root(DATASET_ROOT, fmt, SPLIT)

#     print(f"Dataset root : {dataset_root}")
#     print(f"Dataset fmt  : {fmt}")
#     print(f"Split        : {SPLIT}")
#     print(f"Device       : {DEVICE}")
#     print(f"Image size   : {IMGSZ}")
#     print(f"Keep classes : {KEEP_CLASSES}")
#     print(f"Output dir   : {OUTPUT_DIR}")

#     gt_hist, image_count = gt_scale_histogram(dataset_root, fmt, SPLIT, KEEP_CLASSES, SCALE_BINS)
#     print(f"[GT done] images={image_count}, boxes={sum(gt_hist)}")

#     trained_hist, trained_images = predict_scale_histogram(
#         model_path=TRAINED_MODEL,
#         dataset_root=dataset_root,
#         fmt=fmt,
#         split=SPLIT,
#         keep_classes=KEEP_CLASSES,
#         bins=SCALE_BINS,
#         device=DEVICE,
#         imgsz=IMGSZ,
#         conf=CONF,
#         iou=IOU,
#         max_det=MAX_DET,
#         half=HALF,
#         verbose=VERBOSE,
#     )
#     print(f"[Trained done] images={trained_images}, predicted_boxes={sum(trained_hist)}")

#     baseline_hist, baseline_images = predict_scale_histogram(
#         model_path=BASELINE_MODEL,
#         dataset_root=dataset_root,
#         fmt=fmt,
#         split=SPLIT,
#         keep_classes=KEEP_CLASSES,
#         bins=SCALE_BINS,
#         device=DEVICE,
#         imgsz=IMGSZ,
#         conf=CONF,
#         iou=IOU,
#         max_det=MAX_DET,
#         half=HALF,
#         verbose=VERBOSE,
#     )
#     print(f"[Baseline done] images={baseline_images}, predicted_boxes={sum(baseline_hist)}")

#     save_summary_csv(
#         output_csv=OUTPUT_DIR / "scale_summary.csv",
#         bins=SCALE_BINS,
#         gt_hist=gt_hist,
#         trained_hist=trained_hist,
#         baseline_hist=baseline_hist,
#     )
#     plot_all(
#         output_dir=OUTPUT_DIR,
#         bins=SCALE_BINS,
#         gt_hist=gt_hist,
#         trained_hist=trained_hist,
#         baseline_hist=baseline_hist,
#     )

#     print("\nDone.")
#     print(f"Saved summary: {OUTPUT_DIR / 'scale_summary.csv'}")
#     print(f"Saved figure : {OUTPUT_DIR / '01_gt_scale_distribution.png'}")
#     print(f"Saved figure : {OUTPUT_DIR / '02_prediction_count_vs_scale.png'}")
#     print(f"Saved figure : {OUTPUT_DIR / '03_scale_count_comparison.png'}")
#     print(f"Saved figure : {OUTPUT_DIR / '04_relative_count_vs_gt.png'}")


# if __name__ == "__main__":
#     main()




from __future__ import annotations

import csv
import gc
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# Reduce CUDA fragmentation before torch import
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import torch
from ultralytics import YOLO


# =========================
# 配置区：直接改这里即可
# =========================
PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_ROOT = PROJECT_ROOT / "datasets" / "VisDrone"
TRAINED_MODEL = "runs/detect/v6.4_n_p2_shanp5_cl3_SimAM_EUCB/weights/best.pt"
BASELINE_MODEL = "ultralytics/runs/detect/n_cl3_640/weights/best.pt"
OUTPUT_DIR = PROJECT_ROOT / "runs" / "scale_compare_visdrone_correct"

# auto / yolo / visdrone_det / visdrone_mot
DATASET_FORMAT = "auto"
SPLIT = "val"

# 低显存默认参数
DEVICE = "0"          # 显存不够改成 "cpu"
IMGSZ = 640
CONF = 0.2
NMS_IOU = 0.70         # 模型预测时 NMS 用
MAX_DET = 300
HALF = True            # CPU 下会自动关闭
VERBOSE = True

# 正确检测判定参数
MATCH_IOU = 0.50       # IoU >= 0.5 视为检测正确
CLASS_AWARE = True     # True: 类别必须一致；False: 只看框是否命中

# 是否只统计这些类别；None 表示不过滤
KEEP_CLASSES: Optional[List[int]] = None

# 尺度统计方式：按 sqrt(area) 分箱，更直观
SCALE_BINS = [0, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 2048]

# 可选：只分析前 N 张图，调试时有用；None 表示全部
LIMIT_IMAGES: Optional[int] = None


@dataclass
class BoxRecord:
    cls_id: int
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float = 1.0

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def scale(self) -> float:
        return math.sqrt(self.width * self.height)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def release_cuda_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def detect_dataset_format(dataset_root: Path, split: str) -> str:
    if (dataset_root / "images" / split).exists() and (dataset_root / "labels" / split).exists():
        return "yolo"
    if (dataset_root / "images").exists() and (dataset_root / "annotations").exists():
        return "visdrone_det"
    if (dataset_root / "sequences").exists() and (dataset_root / "annotations").exists():
        return "visdrone_mot"

    for sub in dataset_root.iterdir() if dataset_root.exists() else []:
        if not sub.is_dir():
            continue
        if (sub / "images" / split).exists() and (sub / "labels" / split).exists():
            return "yolo"
        if (sub / "images").exists() and (sub / "annotations").exists():
            return "visdrone_det"
        if (sub / "sequences").exists() and (sub / "annotations").exists():
            return "visdrone_mot"

    raise FileNotFoundError(f"Cannot auto-detect dataset format under: {dataset_root}")


def normalize_dataset_root(dataset_root: Path, fmt: str, split: str) -> Path:
    if fmt == "yolo" and (dataset_root / "images" / split).exists():
        return dataset_root
    if fmt == "visdrone_det" and (dataset_root / "images").exists() and (dataset_root / "annotations").exists():
        return dataset_root
    if fmt == "visdrone_mot" and (dataset_root / "sequences").exists() and (dataset_root / "annotations").exists():
        return dataset_root

    for sub in dataset_root.iterdir() if dataset_root.exists() else []:
        if not sub.is_dir():
            continue
        if fmt == "yolo" and (sub / "images" / split).exists() and (sub / "labels" / split).exists():
            return sub
        if fmt == "visdrone_det" and (sub / "images").exists() and (sub / "annotations").exists():
            return sub
        if fmt == "visdrone_mot" and (sub / "sequences").exists() and (sub / "annotations").exists():
            return sub

    raise FileNotFoundError(f"Cannot find valid root for format={fmt} under: {dataset_root}")


def bin_labels(bins: Sequence[float]) -> List[str]:
    return [f"[{int(bins[i])},{int(bins[i + 1])})" for i in range(len(bins) - 1)]


def scale_to_bin(scale: float, bins: Sequence[float]) -> Optional[int]:
    for i in range(len(bins) - 1):
        if bins[i] <= scale < bins[i + 1]:
            return i
    return len(bins) - 2 if scale >= bins[-1] else None


def make_empty_hist(bins: Sequence[float]) -> List[int]:
    return [0 for _ in range(len(bins) - 1)]


def add_scale_to_hist(hist: List[int], scale: float, bins: Sequence[float]) -> None:
    idx = scale_to_bin(scale, bins)
    if idx is not None:
        hist[idx] += 1


def load_yolo_label_file(label_path: Path, img_w: int, img_h: int, keep_classes: Optional[Sequence[int]]) -> List[BoxRecord]:
    boxes: List[BoxRecord] = []
    if not label_path.exists():
        return boxes
    with label_path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id = int(float(parts[0]))
            if keep_classes is not None and cls_id not in keep_classes:
                continue
            cx = float(parts[1]) * img_w
            cy = float(parts[2]) * img_h
            bw = float(parts[3]) * img_w
            bh = float(parts[4]) * img_h
            x1 = cx - bw / 2.0
            y1 = cy - bh / 2.0
            x2 = cx + bw / 2.0
            y2 = cy + bh / 2.0
            boxes.append(BoxRecord(cls_id, x1, y1, x2, y2))
    return boxes


def iter_yolo_samples(dataset_root: Path, split: str, keep_classes: Optional[Sequence[int]]) -> Iterable[Tuple[Path, List[BoxRecord]]]:
    img_root = dataset_root / "images" / split
    lbl_root = dataset_root / "labels" / split
    n = 0
    for img_path in sorted(img_root.rglob("*")):
        if not img_path.is_file() or not is_image_file(img_path):
            continue
        rel = img_path.relative_to(img_root)
        label_path = (lbl_root / rel).with_suffix(".txt")
        with Image.open(img_path) as im:
            img_w, img_h = im.size
        gt_boxes = load_yolo_label_file(label_path, img_w, img_h, keep_classes)
        yield img_path, gt_boxes
        n += 1
        if LIMIT_IMAGES is not None and n >= LIMIT_IMAGES:
            break


def parse_visdrone_det_ann(ann_path: Path, keep_classes: Optional[Sequence[int]]) -> List[BoxRecord]:
    boxes: List[BoxRecord] = []
    with ann_path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = [p.strip() for p in line.strip().split(",")]
            if len(parts) < 8:
                continue
            x, y, w, h = map(float, parts[:4])
            score = int(float(parts[4]))
            cls_id = int(float(parts[5]))
            cls_zero_based = cls_id - 1
            if keep_classes is not None and cls_zero_based not in keep_classes:
                continue
            if score == 0 or w <= 1.0 or h <= 1.0:
                continue
            boxes.append(BoxRecord(cls_zero_based, x, y, x + w, y + h))
    return boxes


def iter_visdrone_det_samples(dataset_root: Path, split: str, keep_classes: Optional[Sequence[int]]) -> Iterable[Tuple[Path, List[BoxRecord]]]:
    split_map = {"train": "train", "val": "val", "test": "test-dev"}
    sub = split_map.get(split, split)
    img_root = dataset_root / "images" / sub
    ann_root = dataset_root / "annotations" / sub
    n = 0
    for img_path in sorted(img_root.glob("*")):
        if not img_path.is_file() or not is_image_file(img_path):
            continue
        ann_path = ann_root / f"{img_path.stem}.txt"
        gt_boxes = parse_visdrone_det_ann(ann_path, keep_classes)
        yield img_path, gt_boxes
        n += 1
        if LIMIT_IMAGES is not None and n >= LIMIT_IMAGES:
            break


def parse_visdrone_mot_ann(ann_path: Path, keep_classes: Optional[Sequence[int]]) -> Dict[int, List[BoxRecord]]:
    frame_to_boxes: Dict[int, List[BoxRecord]] = {}
    with ann_path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = [p.strip() for p in line.strip().split(",")]
            if len(parts) < 8:
                continue
            frame_id = int(float(parts[0]))
            x = float(parts[2])
            y = float(parts[3])
            w = float(parts[4])
            h = float(parts[5])
            score = int(float(parts[6]))
            cls_id = int(float(parts[7]))
            cls_zero_based = cls_id - 1
            if keep_classes is not None and cls_zero_based not in keep_classes:
                continue
            if score != 1 or w <= 1.0 or h <= 1.0:
                continue
            frame_to_boxes.setdefault(frame_id, []).append(BoxRecord(cls_zero_based, x, y, x + w, y + h))
    return frame_to_boxes


def iter_visdrone_mot_samples(dataset_root: Path, keep_classes: Optional[Sequence[int]]) -> Iterable[Tuple[Path, List[BoxRecord]]]:
    seq_root = dataset_root / "sequences"
    ann_root = dataset_root / "annotations"
    n = 0
    for seq_dir in sorted(p for p in seq_root.iterdir() if p.is_dir()):
        ann_path = ann_root / f"{seq_dir.name}.txt"
        frame_to_boxes = parse_visdrone_mot_ann(ann_path, keep_classes)
        frame_idx = 0
        img_files = sorted(seq_dir.glob("*.jpg")) + sorted(seq_dir.glob("*.png"))
        img_files = sorted(img_files)
        for img_path in img_files:
            frame_idx += 1
            gt_boxes = frame_to_boxes.get(frame_idx, [])
            yield img_path, gt_boxes
            n += 1
            if LIMIT_IMAGES is not None and n >= LIMIT_IMAGES:
                return


def sample_iterator(dataset_root: Path, fmt: str, split: str, keep_classes: Optional[Sequence[int]]):
    if fmt == "yolo":
        return iter_yolo_samples(dataset_root, split, keep_classes)
    if fmt == "visdrone_det":
        return iter_visdrone_det_samples(dataset_root, split, keep_classes)
    if fmt == "visdrone_mot":
        return iter_visdrone_mot_samples(dataset_root, keep_classes)
    raise ValueError(f"Unsupported dataset format: {fmt}")


def gt_scale_histogram(dataset_root: Path, fmt: str, split: str, keep_classes: Optional[Sequence[int]], bins: Sequence[float]) -> Tuple[List[int], int]:
    hist = make_empty_hist(bins)
    image_count = 0
    for _, gt_boxes in sample_iterator(dataset_root, fmt, split, keep_classes):
        image_count += 1
        for box in gt_boxes:
            add_scale_to_hist(hist, box.scale, bins)
    return hist, image_count


def compute_iou(a: BoxRecord, b: BoxRecord) -> float:
    inter_x1 = max(a.x1, b.x1)
    inter_y1 = max(a.y1, b.y1)
    inter_x2 = min(a.x2, b.x2)
    inter_y2 = min(a.y2, b.y2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    if inter <= 0.0:
        return 0.0
    union = a.width * a.height + b.width * b.height - inter
    return inter / union if union > 0.0 else 0.0


def result_to_pred_boxes(result) -> List[BoxRecord]:
    preds: List[BoxRecord] = []
    if result.boxes is None or len(result.boxes) == 0:
        return preds

    xyxy = result.boxes.xyxy.detach().cpu().numpy()
    clss = result.boxes.cls.detach().cpu().numpy()
    confs = result.boxes.conf.detach().cpu().numpy()
    for (x1, y1, x2, y2), cls_id, conf in zip(xyxy, clss, confs):
        x1 = float(x1)
        y1 = float(y1)
        x2 = float(x2)
        y2 = float(y2)
        if x2 - x1 <= 1.0 or y2 - y1 <= 1.0:
            continue
        preds.append(BoxRecord(int(cls_id), x1, y1, x2, y2, float(conf)))
    preds.sort(key=lambda b: b.conf, reverse=True)
    return preds


def greedy_match_predictions_to_gt(
    preds: Sequence[BoxRecord],
    gts: Sequence[BoxRecord],
    iou_thr: float,
    class_aware: bool,
) -> Tuple[List[int], List[int]]:
    """
    返回两个列表：
    - matched_gt_indices: 被正确检测到的 GT 下标
    - matched_pred_indices: 命中 GT 的预测框下标
    """
    matched_gt: set[int] = set()
    matched_pred: List[int] = []

    for pred_idx, pred in enumerate(preds):
        best_gt_idx = -1
        best_iou = 0.0
        for gt_idx, gt in enumerate(gts):
            if gt_idx in matched_gt:
                continue
            if class_aware and pred.cls_id != gt.cls_id:
                continue
            iou = compute_iou(pred, gt)
            if iou >= iou_thr and iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx
        if best_gt_idx >= 0:
            matched_gt.add(best_gt_idx)
            matched_pred.append(pred_idx)

    return sorted(matched_gt), matched_pred


def correct_detection_scale_histogram(
    model_path: str | Path,
    dataset_root: Path,
    fmt: str,
    split: str,
    keep_classes: Optional[Sequence[int]],
    bins: Sequence[float],
    device: str,
    imgsz: int,
    conf: float,
    nms_iou: float,
    max_det: int,
    half: bool,
    verbose: bool,
    match_iou: float,
    class_aware: bool,
) -> Tuple[List[int], List[int], List[int], int]:
    """
    返回：
    - correct_gt_hist: 正确检测到的 GT 按尺度统计
    - correct_pred_hist: 命中 GT 的预测框按尺度统计
    - pred_hist: 所有预测框按尺度统计
    - image_count
    """
    correct_gt_hist = make_empty_hist(bins)
    correct_pred_hist = make_empty_hist(bins)
    pred_hist = make_empty_hist(bins)
    image_count = 0

    if verbose:
        print(f"\n[Load model] {model_path}")
    model = YOLO(str(model_path))
    use_half = half and device != "cpu" and torch.cuda.is_available()

    try:
        with torch.inference_mode():
            for img_path, gt_boxes in sample_iterator(dataset_root, fmt, split, keep_classes):
                image_count += 1
                result_list = model.predict(
                    source=str(img_path),
                    imgsz=imgsz,
                    conf=conf,
                    iou=nms_iou,
                    max_det=max_det,
                    device=device,
                    half=use_half,
                    classes=keep_classes,
                    stream=False,
                    verbose=False,
                )
                result = result_list[0]
                pred_boxes = result_to_pred_boxes(result)

                for pred in pred_boxes:
                    add_scale_to_hist(pred_hist, pred.scale, bins)

                matched_gt_indices, matched_pred_indices = greedy_match_predictions_to_gt(
                    preds=pred_boxes,
                    gts=gt_boxes,
                    iou_thr=match_iou,
                    class_aware=class_aware,
                )

                for gt_idx in matched_gt_indices:
                    add_scale_to_hist(correct_gt_hist, gt_boxes[gt_idx].scale, bins)
                for pred_idx in matched_pred_indices:
                    add_scale_to_hist(correct_pred_hist, pred_boxes[pred_idx].scale, bins)

                del result_list, result, pred_boxes
                if image_count % 20 == 0:
                    release_cuda_memory()
                if verbose and image_count % 100 == 0:
                    print(f"  processed: {image_count} images")
    finally:
        del model
        release_cuda_memory()

    return correct_gt_hist, correct_pred_hist, pred_hist, image_count


def save_summary_csv(
    output_csv: Path,
    bins: Sequence[float],
    gt_hist: Sequence[int],
    trained_correct_gt_hist: Sequence[int],
    baseline_correct_gt_hist: Sequence[int],
    trained_pred_hist: Sequence[int],
    baseline_pred_hist: Sequence[int],
) -> None:
    labels = bin_labels(bins)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "scale_bin",
            "gt_count",
            "trained_correct_gt_count",
            "baseline_correct_gt_count",
            "trained_pred_count",
            "baseline_pred_count",
            "trained_recall",
            "baseline_recall",
        ])
        for i, label in enumerate(labels):
            gt = gt_hist[i]
            tr = trained_correct_gt_hist[i]
            ba = baseline_correct_gt_hist[i]
            writer.writerow([
                label,
                gt,
                tr,
                ba,
                trained_pred_hist[i],
                baseline_pred_hist[i],
                0.0 if gt == 0 else tr / gt,
                0.0 if gt == 0 else ba / gt,
            ])


def plot_all(
    output_dir: Path,
    bins: Sequence[float],
    gt_hist: Sequence[int],
    trained_correct_gt_hist: Sequence[int],
    baseline_correct_gt_hist: Sequence[int],
    trained_pred_hist: Sequence[int],
    baseline_pred_hist: Sequence[int],
) -> None:
    labels = bin_labels(bins)
    x = list(range(len(labels)))

    # 图1：GT尺度分布
    plt.figure(figsize=(14, 6))
    plt.bar(x, gt_hist)
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel("Count")
    plt.xlabel("sqrt(box area) in pixels")
    plt.title("VisDrone Ground-Truth Scale Distribution")
    plt.tight_layout()
    plt.savefig(output_dir / "01_gt_scale_distribution.png", dpi=200)
    plt.close()

    # 图2：正确检测到的GT数量随尺度变化
    plt.figure(figsize=(14, 6))
    plt.plot(x, gt_hist, marker="^", linestyle="--", label="(a) GT")
    plt.plot(x, trained_correct_gt_hist, marker="o", label="(b) SEAYOLO correct detections")
    plt.plot(x, baseline_correct_gt_hist, marker="s", label="(c) YOLO11n correct detections")
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel("Correctly detected GT count")
    plt.xlabel("sqrt(box area) in pixels")
    plt.title("Correct Detections vs. Object Scale")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "02_correct_detection_count_vs_scale.png", dpi=200)
    plt.close()

    # 图3：按尺度的召回率
    eps = 1e-9
    trained_recall = [trained_correct_gt_hist[i] / (gt_hist[i] + eps) for i in range(len(gt_hist))]
    baseline_recall = [baseline_correct_gt_hist[i] / (gt_hist[i] + eps) for i in range(len(gt_hist))]
    plt.figure(figsize=(14, 6))
    plt.plot(x, trained_recall, marker="s", color="orange", label="Trained recall")
    plt.plot(x, baseline_recall, marker="o", color="blue", label="YOLO11n recall")
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel("Recall")
    plt.xlabel("sqrt(box area) in pixels")
    plt.title("Recall vs. Object Scale")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "03_recall_vs_scale.png", dpi=200)
    plt.close()

    # 图4：所有预测框数量（保留原来这张图）
    plt.figure(figsize=(14, 6))
    plt.plot(x, gt_hist, marker="^", linestyle="--", label="GT")
    plt.plot(x, trained_pred_hist, marker="o", label="Trained predictions")
    plt.plot(x, baseline_pred_hist, marker="s", label="YOLO11n predictions")
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel("Box count")
    plt.xlabel("sqrt(box area) in pixels")
    plt.title("Prediction Count vs. Object Scale")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "04_prediction_count_vs_scale.png", dpi=200)
    plt.close()


def main() -> None:
    ensure_dir(OUTPUT_DIR)

    fmt = DATASET_FORMAT
    if fmt == "auto":
        fmt = detect_dataset_format(DATASET_ROOT, SPLIT)
    dataset_root = normalize_dataset_root(DATASET_ROOT, fmt, SPLIT)

    print(f"Dataset root : {dataset_root}")
    print(f"Dataset fmt  : {fmt}")
    print(f"Split        : {SPLIT}")
    print(f"Device       : {DEVICE}")
    print(f"Image size   : {IMGSZ}")
    print(f"Keep classes : {KEEP_CLASSES}")
    print(f"Output dir   : {OUTPUT_DIR}")
    print(f"Match IoU    : {MATCH_IOU}")
    print(f"Class aware  : {CLASS_AWARE}")

    gt_hist, image_count = gt_scale_histogram(dataset_root, fmt, SPLIT, KEEP_CLASSES, SCALE_BINS)
    print(f"[GT done] images={image_count}, boxes={sum(gt_hist)}")

    trained_correct_gt_hist, _, trained_pred_hist, trained_images = correct_detection_scale_histogram(
        model_path=TRAINED_MODEL,
        dataset_root=dataset_root,
        fmt=fmt,
        split=SPLIT,
        keep_classes=KEEP_CLASSES,
        bins=SCALE_BINS,
        device=DEVICE,
        imgsz=IMGSZ,
        conf=CONF,
        nms_iou=NMS_IOU,
        max_det=MAX_DET,
        half=HALF,
        verbose=VERBOSE,
        match_iou=MATCH_IOU,
        class_aware=CLASS_AWARE,
    )
    print(f"[Trained done] images={trained_images}, correct_gt={sum(trained_correct_gt_hist)}, predictions={sum(trained_pred_hist)}")

    baseline_correct_gt_hist, _, baseline_pred_hist, baseline_images = correct_detection_scale_histogram(
        model_path=BASELINE_MODEL,
        dataset_root=dataset_root,
        fmt=fmt,
        split=SPLIT,
        keep_classes=KEEP_CLASSES,
        bins=SCALE_BINS,
        device=DEVICE,
        imgsz=IMGSZ,
        conf=CONF,
        nms_iou=NMS_IOU,
        max_det=MAX_DET,
        half=HALF,
        verbose=VERBOSE,
        match_iou=MATCH_IOU,
        class_aware=CLASS_AWARE,
    )
    print(f"[Baseline done] images={baseline_images}, correct_gt={sum(baseline_correct_gt_hist)}, predictions={sum(baseline_pred_hist)}")

    save_summary_csv(
        output_csv=OUTPUT_DIR / "scale_summary_correct_detection.csv",
        bins=SCALE_BINS,
        gt_hist=gt_hist,
        trained_correct_gt_hist=trained_correct_gt_hist,
        baseline_correct_gt_hist=baseline_correct_gt_hist,
        trained_pred_hist=trained_pred_hist,
        baseline_pred_hist=baseline_pred_hist,
    )

    plot_all(
        output_dir=OUTPUT_DIR,
        bins=SCALE_BINS,
        gt_hist=gt_hist,
        trained_correct_gt_hist=trained_correct_gt_hist,
        baseline_correct_gt_hist=baseline_correct_gt_hist,
        trained_pred_hist=trained_pred_hist,
        baseline_pred_hist=baseline_pred_hist,
    )

    print("\nDone.")
    print(f"Saved summary: {OUTPUT_DIR / 'scale_summary_correct_detection.csv'}")
    print(f"Saved figure : {OUTPUT_DIR / '01_gt_scale_distribution.png'}")
    print(f"Saved figure : {OUTPUT_DIR / '02_correct_detection_count_vs_scale.png'}")
    print(f"Saved figure : {OUTPUT_DIR / '03_recall_vs_scale.png'}")
    print(f"Saved figure : {OUTPUT_DIR / '04_prediction_count_vs_scale.png'}")


if __name__ == "__main__":
    main()
