from pathlib import Path
import os
import cv2
import numpy as np
from ultralytics import YOLO


# =========================
# 1. 路径与参数配置（直接改这里）
# =========================
MODEL_PATH = r"yolo11n.pt"   # 你的模型路径
IMAGE_PATH = r"experient_fig/bytetrack/bytetrack_workbad/000515.jpg"                                          # 你的单张输入图片
PROJECT_DIR = r"experient_fig/gm"                                 # 总输出目录
RUN_NAME = "single_image_gm"                                      # 本次输出子目录名

# YOLO检测参数
IMGSZ = 640
CONF = 0.2
IOU = 0.6
DEVICE = 0          # 没GPU可改成 "cpu"
SAVE_CONF = True    # txt里是否保存置信度

# GM参数
SIGMA_X_RATIO = 0.25   # x方向高斯扩散占框宽比例
SIGMA_Y_RATIO = 0.25   # y方向高斯扩散占框高比例
GM_MERGE_MODE = "max"  # 多框融合方式: "max" 或 "add"
OVERLAY_ALPHA = 0.45   # 热力图叠加透明度
BG_WEIGHT = 0.35       # 加权图中背景保留亮度

# 是否只保留某些类别（None表示全部保留）
KEEP_CLASSES = None    # 例如 [0, 2, 3]


# =========================
# 2. 工具函数
# =========================
def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def clip_box(box, w, h):
    x1, y1, x2, y2 = box
    x1 = max(0, min(int(round(x1)), w - 1))
    y1 = max(0, min(int(round(y1)), h - 1))
    x2 = max(0, min(int(round(x2)), w - 1))
    y2 = max(0, min(int(round(y2)), h - 1))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def yolo_txt_to_xyxy(txt_path, img_w, img_h, keep_classes=None):
    """
    读取Ultralytics保存的YOLO txt:
    每行格式通常为:
    cls cx cy w h
    或
    cls cx cy w h conf
    其中 cx, cy, w, h 为归一化坐标

    返回:
        boxes: [[x1, y1, x2, y2], ...]
        info_list: [(cls_id, conf), ...]
    """
    boxes = []
    info_list = []

    if not os.path.exists(txt_path):
        print(f"[Warning] 未找到检测txt文件: {txt_path}")
        return boxes, info_list

    with open(txt_path, "r", encoding="utf-8") as f:
        for line_id, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) < 5:
                print(f"[Warning] 第 {line_id} 行格式不正确，已跳过: {line}")
                continue

            try:
                cls_id = int(float(parts[0]))
                cx = float(parts[1])
                cy = float(parts[2])
                bw = float(parts[3])
                bh = float(parts[4])
                conf = float(parts[5]) if len(parts) >= 6 else None
            except ValueError:
                print(f"[Warning] 第 {line_id} 行无法解析，已跳过: {line}")
                continue

            if keep_classes is not None and cls_id not in keep_classes:
                continue

            # 归一化xywh -> 像素xyxy
            box_w = bw * img_w
            box_h = bh * img_h
            center_x = cx * img_w
            center_y = cy * img_h

            x1 = center_x - box_w / 2.0
            y1 = center_y - box_h / 2.0
            x2 = center_x + box_w / 2.0
            y2 = center_y + box_h / 2.0

            box = clip_box([x1, y1, x2, y2], img_w, img_h)
            if box is not None:
                boxes.append(box)
                info_list.append((cls_id, conf))

    return boxes, info_list


def gaussian_map_for_box(box, img_w, img_h, sigma_ratio_x=0.25, sigma_ratio_y=0.25):
    """
    在单个检测框内部生成二维高斯图
    """
    x1, y1, x2, y2 = box
    bw = x2 - x1
    bh = y2 - y1

    if bw <= 1 or bh <= 1:
        return np.zeros((img_h, img_w), dtype=np.float32)

    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    sigma_x = max(bw * sigma_ratio_x, 1.0)
    sigma_y = max(bh * sigma_ratio_y, 1.0)

    xs = np.arange(x1, x2, dtype=np.float32)
    ys = np.arange(y1, y2, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)

    g = np.exp(
        -(
            ((xx - cx) ** 2) / (2 * sigma_x ** 2)
            + ((yy - cy) ** 2) / (2 * sigma_y ** 2)
        )
    ).astype(np.float32)

    gm = np.zeros((img_h, img_w), dtype=np.float32)
    gm[y1:y2, x1:x2] = g
    return gm


def build_global_gm(img_shape, boxes, sigma_ratio_x=0.25, sigma_ratio_y=0.25, mode="max"):
    """
    对多个检测框生成全局 Gaussian Map
    """
    h, w = img_shape[:2]
    gm_all = np.zeros((h, w), dtype=np.float32)

    for box in boxes:
        gm = gaussian_map_for_box(
            box, w, h,
            sigma_ratio_x=sigma_ratio_x,
            sigma_ratio_y=sigma_ratio_y
        )
        if mode == "add":
            gm_all += gm
        else:
            gm_all = np.maximum(gm_all, gm)

    if gm_all.max() > 1e-6:
        gm_all = gm_all / gm_all.max()

    return gm_all


def apply_gm_to_image(image, gm_mask, keep_background=True, bg_weight=0.35):
    """
    将GM权重应用到图像上
    """
    img_f = image.astype(np.float32) / 255.0
    gm_3c = np.repeat(gm_mask[:, :, None], 3, axis=2)

    if keep_background:
        weight = np.where(gm_3c > 0, bg_weight + (1.0 - bg_weight) * gm_3c, bg_weight)
    else:
        weight = gm_3c

    weighted = img_f * weight
    weighted = np.clip(weighted * 255.0, 0, 255).astype(np.uint8)
    return weighted


def make_overlay(image, gm_mask, alpha=0.45):
    """
    生成伪彩色叠加图
    """
    heat = (gm_mask * 255).astype(np.uint8)
    heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(image, 1 - alpha, heat, alpha, 0)
    return overlay


def draw_boxes(image, boxes, info_list=None, color=(0, 255, 0), thickness=2):
    out = image.copy()
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)

        label = f"box{i+1}"
        if info_list is not None and i < len(info_list):
            cls_id, conf = info_list[i]
            if conf is None:
                label = f"cls:{cls_id}"
            else:
                label = f"cls:{cls_id} {conf:.2f}"

        cv2.putText(
            out,
            label,
            (x1, max(0, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )
    return out


# =========================
# 3. 主流程
# =========================
def main():
    image_path = Path(IMAGE_PATH)
    if not image_path.exists():
        raise FileNotFoundError(f"输入图片不存在: {IMAGE_PATH}")

    if not Path(MODEL_PATH).exists():
        raise FileNotFoundError(f"模型不存在: {MODEL_PATH}")

    ensure_dir(PROJECT_DIR)

    # ---------- 第一步：YOLO检测并保存txt ----------
    model = YOLO(MODEL_PATH)
    model.predict(
        source=str(image_path),
        imgsz=IMGSZ,
        conf=CONF,
        iou=IOU,
        device=DEVICE,
        save=True,
        save_txt=True,
        save_conf=SAVE_CONF,
        project=PROJECT_DIR,
        name=RUN_NAME,
        exist_ok=True,
        verbose=False
    )

    run_dir = Path(PROJECT_DIR) / RUN_NAME
    labels_dir = run_dir / "labels"
    txt_path = labels_dir / f"{image_path.stem}.txt"

    # ---------- 第二步：读取原图 ----------
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"无法读取图片: {IMAGE_PATH}")

    h, w = image.shape[:2]

    # ---------- 第三步：读取YOLO输出txt并转为xyxy ----------
    boxes, info_list = yolo_txt_to_xyxy(
        str(txt_path), w, h, keep_classes=KEEP_CLASSES
    )

    if len(boxes) == 0:
        print("[Info] 没有检测到有效框，默认对整张图片生成GM。")
        boxes = [[0, 0, w, h]]
        info_list = [(None, None)]

    # ---------- 第四步：生成GM ----------
    gm_mask = build_global_gm(
        image.shape,
        boxes,
        sigma_ratio_x=SIGMA_X_RATIO,
        sigma_ratio_y=SIGMA_Y_RATIO,
        mode=GM_MERGE_MODE
    )

    gm_mask_u8 = (gm_mask * 255).astype(np.uint8)
    overlay = make_overlay(image, gm_mask, alpha=OVERLAY_ALPHA)
    weighted = apply_gm_to_image(image, gm_mask, keep_background=True, bg_weight=BG_WEIGHT)

    # 画框，便于展示
    input_with_boxes = draw_boxes(image, boxes, info_list=info_list, color=(0, 255, 0), thickness=2)
    overlay_with_boxes = draw_boxes(overlay, boxes, info_list=info_list, color=(255, 255, 255), thickness=2)
    weighted_with_boxes = draw_boxes(weighted, boxes, info_list=info_list, color=(255, 255, 255), thickness=2)

    # ---------- 第五步：保存结果 ----------
    gm_dir = run_dir / "gm_results"
    ensure_dir(gm_dir)

    cv2.imwrite(str(gm_dir / "input_with_boxes.png"), input_with_boxes)
    cv2.imwrite(str(gm_dir / "gm_mask.png"), gm_mask_u8)
    cv2.imwrite(str(gm_dir / "gm_overlay.png"), overlay_with_boxes)
    cv2.imwrite(str(gm_dir / "gm_weighted.png"), weighted_with_boxes)

    print(f"[Done] 检测txt文件: {txt_path}")
    print(f"[Done] GM结果目录: {gm_dir}")
    print("保存文件包括：")
    print("  - input_with_boxes.png   原图+检测框")
    print("  - gm_mask.png            灰度GM图")
    print("  - gm_overlay.png         原图+GM热力叠加图")
    print("  - gm_weighted.png        原图经过GM加权后的结果图")


if __name__ == "__main__":
    main()