# # from __future__ import annotations

# # import re
# # import shutil
# # import zipfile
# # from pathlib import Path

# # import cv2
# # from ultralytics import YOLO


# # # =========================
# # # 路径配置
# # # =========================
# # PROJECT_ROOT = Path(__file__).resolve().parent

# # VAL_ZIP = PROJECT_ROOT / "datasetTrack" / "VisDrone2019-MOT-val.zip"
# # EXTRACT_ROOT = PROJECT_ROOT / "datasetTrack" / "extracted"
# # VAL_ROOT = EXTRACT_ROOT / "VisDrone2019-MOT-val"

# # # 改成你自己的 best.pt 路径
# # MODEL_PATH = PROJECT_ROOT / "runs/detect/origin_visdrone_cl9/weights/best.pt"

# # # 总实验根目录（每次运行会自动创建 exp001 / exp002 / ...）
# # RUNS_ROOT = PROJECT_ROOT / "runs" / "visdrone_mot"

# # # 跟踪器配置
# # TRACKER_YAML = "botsort.yaml"   # 也可改成 "botsort.yaml"
# # IMGSZ = 640
# # CONF = 0.2
# # IOU = 0.6
# # DEVICE = 0  # 没GPU可改成 "cpu"

# # # 可视化保存开关
# # SAVE_VIS_IMAGES = True          # 是否保存带框图片
# # SAVE_ONLY_WITH_TRACKS = False   # True=只保存有跟踪结果的帧；False=所有帧都保存

# # # =========================
# # # 标准 VisDrone-DET 十类顺序下：
# # # 只保留 MOT 常用 5 类
# # # 模型类别ID -> VisDrone官方类别ID
# # # =========================
# # DET10_TO_VISDRONE_MOT5 = {
# #     0: 1,  # pedestrian
# #     3: 4,  # car
# #     4: 5,  # van
# #     5: 6,  # truck
# #     8: 9,  # bus
# # }

# # # 只让模型输出这几个类别
# # KEEP_CLASSES = list(DET10_TO_VISDRONE_MOT5.keys())


# # def ensure_dir(path: Path) -> None:
# #     path.mkdir(parents=True, exist_ok=True)


# # def get_next_exp_dir(base_dir: Path, prefix: str = "exp") -> Path:
# #     """
# #     自动创建递增实验目录：
# #     exp001, exp002, exp003, ...
# #     """
# #     ensure_dir(base_dir)

# #     max_idx = 0
# #     pattern = re.compile(rf"^{re.escape(prefix)}(\d+)$")

# #     for p in base_dir.iterdir():
# #         if not p.is_dir():
# #             continue
# #         m = pattern.match(p.name)
# #         if m:
# #             max_idx = max(max_idx, int(m.group(1)))

# #     new_dir = base_dir / f"{prefix}{max_idx + 1:03d}"
# #     ensure_dir(new_dir)
# #     return new_dir


# # def unzip_if_needed(zip_path: Path, extract_to: Path) -> None:
# #     """如果未解压，则自动解压。"""
# #     if extract_to.exists() and any(extract_to.iterdir()):
# #         print(f"[Skip unzip] {extract_to} already exists.")
# #         return

# #     if not zip_path.exists():
# #         raise FileNotFoundError(f"Zip file not found: {zip_path}")

# #     ensure_dir(extract_to)
# #     print(f"[Unzipping] {zip_path} -> {extract_to}")
# #     with zipfile.ZipFile(zip_path, "r") as zf:
# #         zf.extractall(extract_to)

# #     # 兼容“解压后多套一层目录”的情况
# #     inner_items = list(extract_to.iterdir())
# #     if len(inner_items) == 1 and inner_items[0].is_dir():
# #         inner_dir = inner_items[0]
# #         if (inner_dir / "sequences").exists():
# #             tmp_dir = extract_to.parent / f"{extract_to.name}_tmp_move"
# #             if tmp_dir.exists():
# #                 shutil.rmtree(tmp_dir)
# #             inner_dir.rename(tmp_dir)
# #             shutil.rmtree(extract_to)
# #             tmp_dir.rename(extract_to)

# #     print(f"[Done unzip] {extract_to}")


# # def resolve_val_root(root: Path) -> Path:
# #     """兼容 root/sequences 或 root/VisDrone2019-MOT-val/sequences 两种层级。"""
# #     if (root / "sequences").exists():
# #         return root

# #     subdirs = [p for p in root.iterdir() if p.is_dir()]
# #     for d in subdirs:
# #         if (d / "sequences").exists():
# #             return d

# #     raise FileNotFoundError(f"Cannot find sequences folder under: {root}")


# # def save_result_image(result, save_path: Path) -> None:
# #     """
# #     保存带检测/跟踪框的可视化图片
# #     """
# #     plotted = result.plot()  # 返回 BGR numpy 图像
# #     cv2.imwrite(str(save_path), plotted)


# # def run_tracking_on_val() -> None:
# #     unzip_if_needed(VAL_ZIP, VAL_ROOT)
# #     data_root = resolve_val_root(VAL_ROOT)

# #     seq_root = data_root / "sequences"

# #     if not MODEL_PATH.exists():
# #         raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

# #     # ========= 自动创建本次实验目录 =========
# #     exp_dir = get_next_exp_dir(RUNS_ROOT, prefix="exp")
# #     txt_dir = exp_dir / "txt"
# #     vis_root = exp_dir / "vis_images"

# #     ensure_dir(txt_dir)
# #     if SAVE_VIS_IMAGES:
# #         ensure_dir(vis_root)

# #     print(f"[Experiment Dir] {exp_dir}")

# #     model = YOLO(str(MODEL_PATH))

# #     seq_dirs = sorted([p for p in seq_root.iterdir() if p.is_dir()])
# #     if not seq_dirs:
# #         raise FileNotFoundError(f"No sequence folders found in {seq_root}")

# #     for seq_dir in seq_dirs:
# #         seq_name = seq_dir.name
# #         out_txt = txt_dir / f"{seq_name}.txt"
# #         seq_vis_dir = vis_root / seq_name if SAVE_VIS_IMAGES else None
# #         if seq_vis_dir is not None:
# #             ensure_dir(seq_vis_dir)

# #         lines = []

# #         print(f"[Tracking] {seq_name}")

# #         results = model.track(
# #             source=str(seq_dir),
# #             stream=True,
# #             persist=True,
# #             tracker=TRACKER_YAML,
# #             conf=CONF,
# #             iou=IOU,
# #             imgsz=IMGSZ,
# #             device=DEVICE,
# #             classes=KEEP_CLASSES,   # 只输出 MOT 重点评测类别
# #             save=False,
# #             verbose=False,
# #             workers=4,
# #         )

# #         frame_idx = 0
# #         for r in results:
# #             frame_idx += 1

# #             has_tracks = (
# #                 r.boxes is not None
# #                 and len(r.boxes) > 0
# #                 and r.boxes.id is not None
# #             )

# #             # ===== 保存可视化图片 =====
# #             if SAVE_VIS_IMAGES:
# #                 if (not SAVE_ONLY_WITH_TRACKS) or has_tracks:
# #                     img_name = f"{frame_idx:07d}.jpg"
# #                     save_result_image(r, seq_vis_dir / img_name)

# #             if not has_tracks:
# #                 continue

# #             xyxy = r.boxes.xyxy.cpu().numpy()
# #             tids = r.boxes.id.int().cpu().tolist()
# #             clss = r.boxes.cls.int().cpu().tolist()
# #             confs = r.boxes.conf.cpu().tolist()

# #             for box, tid, cls_id, score in zip(xyxy, tids, clss, confs):
# #                 cls_id = int(cls_id)
# #                 if cls_id not in DET10_TO_VISDRONE_MOT5:
# #                     continue

# #                 x1, y1, x2, y2 = map(float, box.tolist())
# #                 w = max(0.0, x2 - x1)
# #                 h = max(0.0, y2 - y1)
# #                 if w <= 1.0 or h <= 1.0:
# #                     continue

# #                 vis_cls = DET10_TO_VISDRONE_MOT5[cls_id]

# #                 # VisDrone MOT 10列格式：
# #                 # frame_index,target_id,left,top,width,height,score,object_category,truncation,occlusion
# #                 lines.append(
# #                     f"{frame_idx},{int(tid)},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},{float(score):.6f},{vis_cls},-1,-1"
# #                 )

# #         with out_txt.open("w", encoding="utf-8") as f:
# #             if lines:
# #                 f.write("\n".join(lines) + "\n")

# #         print(f"[OK] {seq_name} -> {out_txt}")

# #     print(f"\n[All Done] Results saved in: {exp_dir}")
# #     print(f"  - TXT: {txt_dir}")
# #     if SAVE_VIS_IMAGES:
# #         print(f"  - VIS: {vis_root}")


# # if __name__ == "__main__":
# #     run_tracking_on_val()


# from __future__ import annotations

# import re
# import shutil
# import zipfile
# from pathlib import Path

# from ultralytics import YOLO


# # =========================
# # 路径配置
# # =========================
# PROJECT_ROOT = Path(__file__).resolve().parent

# VAL_ZIP = PROJECT_ROOT / "datasetTrack" / "VisDrone2019-MOT-val.zip"
# EXTRACT_ROOT = PROJECT_ROOT / "datasetTrack" / "extracted"
# VAL_ROOT = EXTRACT_ROOT / "VisDrone2019-MOT-val"

# # 改成你自己的 best.pt 路径
# MODEL_PATH = PROJECT_ROOT / "runs/detect/origin_visdrone_cl9/weights/best.pt"

# # 总实验根目录（每次运行会自动创建 exp001 / exp002 / ...）
# RUNS_ROOT = PROJECT_ROOT / "runs" / "visdrone_mot"

# # 跟踪器配置
# TRACKER_YAML = "botsort.yaml"
# IMGSZ = 640
# CONF = 0.2
# IOU = 0.6
# DEVICE = 0  # 没GPU可改成 "cpu"

# # =========================
# # 标准 VisDrone-DET 十类顺序下：
# # 只保留 MOT 常用 5 类
# # 模型类别ID -> VisDrone官方类别ID
# # =========================
# DET10_TO_VISDRONE_MOT5 = {
#     0: 1,  # pedestrian
#     3: 4,  # car
#     4: 5,  # van
#     5: 6,  # truck
#     8: 9,  # bus
# }

# # 只让模型输出这几个类别
# KEEP_CLASSES = list(DET10_TO_VISDRONE_MOT5.keys())


# def ensure_dir(path: Path) -> None:
#     path.mkdir(parents=True, exist_ok=True)


# def get_next_exp_dir(base_dir: Path, prefix: str = "exp") -> Path:
#     """
#     自动创建递增实验目录：
#     exp001, exp002, exp003, ...
#     """
#     ensure_dir(base_dir)

#     max_idx = 0
#     pattern = re.compile(rf"^{re.escape(prefix)}(\d+)$")

#     for p in base_dir.iterdir():
#         if not p.is_dir():
#             continue
#         m = pattern.match(p.name)
#         if m:
#             max_idx = max(max_idx, int(m.group(1)))

#     new_dir = base_dir / f"{prefix}{max_idx + 1:03d}"
#     ensure_dir(new_dir)
#     return new_dir


# def unzip_if_needed(zip_path: Path, extract_to: Path) -> None:
#     """如果未解压，则自动解压。"""
#     if extract_to.exists() and any(extract_to.iterdir()):
#         print(f"[Skip unzip] {extract_to} already exists.")
#         return

#     if not zip_path.exists():
#         raise FileNotFoundError(f"Zip file not found: {zip_path}")

#     ensure_dir(extract_to)
#     print(f"[Unzipping] {zip_path} -> {extract_to}")
#     with zipfile.ZipFile(zip_path, "r") as zf:
#         zf.extractall(extract_to)

#     # 兼容“解压后多套一层目录”的情况
#     inner_items = list(extract_to.iterdir())
#     if len(inner_items) == 1 and inner_items[0].is_dir():
#         inner_dir = inner_items[0]
#         if (inner_dir / "sequences").exists():
#             tmp_dir = extract_to.parent / f"{extract_to.name}_tmp_move"
#             if tmp_dir.exists():
#                 shutil.rmtree(tmp_dir)
#             inner_dir.rename(tmp_dir)
#             shutil.rmtree(extract_to)
#             tmp_dir.rename(extract_to)

#     print(f"[Done unzip] {extract_to}")


# def resolve_val_root(root: Path) -> Path:
#     """兼容 root/sequences 或 root/VisDrone2019-MOT-val/sequences 两种层级。"""
#     if (root / "sequences").exists():
#         return root

#     subdirs = [p for p in root.iterdir() if p.is_dir()]
#     for d in subdirs:
#         if (d / "sequences").exists():
#             return d

#     raise FileNotFoundError(f"Cannot find sequences folder under: {root}")


# def run_tracking_on_val() -> None:
#     unzip_if_needed(VAL_ZIP, VAL_ROOT)
#     data_root = resolve_val_root(VAL_ROOT)
#     seq_root = data_root / "sequences"

#     if not MODEL_PATH.exists():
#         raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

#     # 只创建一个实验目录，里面直接放 7 个 txt
#     exp_dir = get_next_exp_dir(RUNS_ROOT, prefix="exp")
#     print(f"[Experiment Dir] {exp_dir}")

#     model = YOLO(str(MODEL_PATH))

#     seq_dirs = sorted([p for p in seq_root.iterdir() if p.is_dir()])
#     if not seq_dirs:
#         raise FileNotFoundError(f"No sequence folders found in {seq_root}")

#     for seq_dir in seq_dirs:
#         seq_name = seq_dir.name
#         out_txt = exp_dir / f"{seq_name}.txt"
#         lines = []

#         print(f"[Tracking] {seq_name}")

#         results = model.track(
#             source=str(seq_dir),
#             stream=True,
#             persist=True,
#             tracker=TRACKER_YAML,
#             conf=CONF,
#             iou=IOU,
#             imgsz=IMGSZ,
#             device=DEVICE,
#             classes=KEEP_CLASSES,
#             save=False,
#             verbose=False,
#             workers=0,   # 更稳，少开额外进程
#         )

#         frame_idx = 0
#         for r in results:
#             frame_idx += 1

#             has_tracks = (
#                 r.boxes is not None
#                 and len(r.boxes) > 0
#                 and r.boxes.id is not None
#             )
#             if not has_tracks:
#                 continue

#             xyxy = r.boxes.xyxy.cpu().numpy()
#             tids = r.boxes.id.int().cpu().tolist()
#             clss = r.boxes.cls.int().cpu().tolist()
#             confs = r.boxes.conf.cpu().tolist()

#             for box, tid, cls_id, score in zip(xyxy, tids, clss, confs):
#                 cls_id = int(cls_id)
#                 if cls_id not in DET10_TO_VISDRONE_MOT5:
#                     continue

#                 x1, y1, x2, y2 = map(float, box.tolist())
#                 w = max(0.0, x2 - x1)
#                 h = max(0.0, y2 - y1)
#                 if w <= 1.0 or h <= 1.0:
#                     continue

#                 vis_cls = DET10_TO_VISDRONE_MOT5[cls_id]

#                 # VisDrone MOT 10列格式
#                 lines.append(
#                     f"{frame_idx},{int(tid)},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},{float(score):.6f},{vis_cls},-1,-1"
#                 )

#         with out_txt.open("w", encoding="utf-8") as f:
#             if lines:
#                 f.write("\n".join(lines) + "\n")

#         print(f"[OK] {seq_name} -> {out_txt}")

#     print(f"\n[All Done] Only txt files saved in: {exp_dir}")


# if __name__ == "__main__":
#     run_tracking_on_val()


from __future__ import annotations

import re
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

# 改成你自己的 best.pt 路径
MODEL_PATH = PROJECT_ROOT / "runs/detect/origin_cl9_p2shanp5_sim_EUCB/weights/best.pt"

# 总实验根目录（每次运行会自动创建 exp001 / exp002 / ...）
RUNS_ROOT = PROJECT_ROOT / "runs" / "visdrone_mot"

# 跟踪器配置
TRACKER_YAML = "botsort.yaml"
IMGSZ = 640
CONF = 0.2
IOU = 0.6
DEVICE = 0  # 没GPU可改成 "cpu"

# =========================
# 可视化图片保存配置
# =========================
SAVE_VIS_IMAGES = True          # 是否保存每一帧跟踪结果图
SAVE_ONLY_WITH_TRACKS = False   # True=只保存有轨迹的帧；False=所有帧都保存

# =========================
# 标准 VisDrone-DET 十类顺序下：
# 只保留 MOT 常用 5 类
# 模型类别ID -> VisDrone官方类别ID
# =========================
DET10_TO_VISDRONE_MOT5 = {
    0: 1,  # pedestrian
    3: 4,  # car
    4: 5,  # van
    5: 6,  # truck
    8: 9,  # bus
}

# 只让模型输出这几个类别
KEEP_CLASSES = list(DET10_TO_VISDRONE_MOT5.keys())


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def get_next_exp_dir(base_dir: Path, prefix: str = "exp") -> Path:
    """
    自动创建递增实验目录：
    exp001, exp002, exp003, ...
    """
    ensure_dir(base_dir)

    max_idx = 0
    pattern = re.compile(rf"^{re.escape(prefix)}(\d+)$")

    for p in base_dir.iterdir():
        if not p.is_dir():
            continue
        m = pattern.match(p.name)
        if m:
            max_idx = max(max_idx, int(m.group(1)))

    new_dir = base_dir / f"{prefix}{max_idx + 1:03d}"
    ensure_dir(new_dir)
    return new_dir


def unzip_if_needed(zip_path: Path, extract_to: Path) -> None:
    """如果未解压，则自动解压。"""
    if extract_to.exists() and any(extract_to.iterdir()):
        print(f"[Skip unzip] {extract_to} already exists.")
        return

    if not zip_path.exists():
        raise FileNotFoundError(f"Zip file not found: {zip_path}")

    ensure_dir(extract_to)
    print(f"[Unzipping] {zip_path} -> {extract_to}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_to)

    # 兼容“解压后多套一层目录”的情况
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
    """兼容 root/sequences 或 root/VisDrone2019-MOT-val/sequences 两种层级。"""
    if (root / "sequences").exists():
        return root

    subdirs = [p for p in root.iterdir() if p.is_dir()]
    for d in subdirs:
        if (d / "sequences").exists():
            return d

    raise FileNotFoundError(f"Cannot find sequences folder under: {root}")


def save_result_image(result, save_path: Path) -> None:
    """
    保存带检测/跟踪框的可视化图片
    """
    plotted = result.plot()  # 返回 BGR numpy 图像
    cv2.imwrite(str(save_path), plotted)


def run_tracking_on_val() -> None:
    unzip_if_needed(VAL_ZIP, VAL_ROOT)
    data_root = resolve_val_root(VAL_ROOT)
    seq_root = data_root / "sequences"

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    # 创建实验目录
    exp_dir = get_next_exp_dir(RUNS_ROOT, prefix="exp")
    print(f"[Experiment Dir] {exp_dir}")

    vis_root = exp_dir / "vis_images"
    if SAVE_VIS_IMAGES:
        ensure_dir(vis_root)

    model = YOLO(str(MODEL_PATH))

    seq_dirs = sorted([p for p in seq_root.iterdir() if p.is_dir()])
    if not seq_dirs:
        raise FileNotFoundError(f"No sequence folders found in {seq_root}")

    for seq_dir in seq_dirs:
        seq_name = seq_dir.name
        out_txt = exp_dir / f"{seq_name}.txt"
        lines = []

        seq_vis_dir = vis_root / seq_name if SAVE_VIS_IMAGES else None
        if seq_vis_dir is not None:
            ensure_dir(seq_vis_dir)

        print(f"[Tracking] {seq_name}")

        results = model.track(
            source=str(seq_dir),
            stream=True,
            persist=True,
            tracker=TRACKER_YAML,
            conf=CONF,
            iou=IOU,
            imgsz=IMGSZ,
            device=DEVICE,
            classes=KEEP_CLASSES,
            save=False,
            verbose=False,
            workers=0,   # 更稳，少开额外进程
        )

        frame_idx = 0
        for r in results:
            frame_idx += 1

            has_tracks = (
                r.boxes is not None
                and len(r.boxes) > 0
                and r.boxes.id is not None
            )

            # 保存每一帧跟踪结果图
            if SAVE_VIS_IMAGES:
                if (not SAVE_ONLY_WITH_TRACKS) or has_tracks:
                    img_name = f"{frame_idx:07d}.jpg"
                    save_result_image(r, seq_vis_dir / img_name)

            if not has_tracks:
                continue

            xyxy = r.boxes.xyxy.cpu().numpy()
            tids = r.boxes.id.int().cpu().tolist()
            clss = r.boxes.cls.int().cpu().tolist()
            confs = r.boxes.conf.cpu().tolist()

            for box, tid, cls_id, score in zip(xyxy, tids, clss, confs):
                cls_id = int(cls_id)
                if cls_id not in DET10_TO_VISDRONE_MOT5:
                    continue

                x1, y1, x2, y2 = map(float, box.tolist())
                w = max(0.0, x2 - x1)
                h = max(0.0, y2 - y1)
                if w <= 1.0 or h <= 1.0:
                    continue

                vis_cls = DET10_TO_VISDRONE_MOT5[cls_id]

                # VisDrone MOT 10列格式
                lines.append(
                    f"{frame_idx},{int(tid)},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},{float(score):.6f},{vis_cls},-1,-1"
                )

        with out_txt.open("w", encoding="utf-8") as f:
            if lines:
                f.write("\n".join(lines) + "\n")

        print(f"[OK] {seq_name} -> {out_txt}")
        if SAVE_VIS_IMAGES:
            print(f"[OK] {seq_name} images -> {seq_vis_dir}")

    print(f"\n[All Done] txt files saved in: {exp_dir}")
    if SAVE_VIS_IMAGES:
        print(f"[All Done] vis images saved in: {vis_root}")


if __name__ == "__main__":
    run_tracking_on_val()