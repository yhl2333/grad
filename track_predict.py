from ultralytics import YOLO
from pathlib import Path

import time


# 1. 加载模型（换成你自己的 .pt 也可以）
model = YOLO("runs/detect/v6.4_n_p2_shanp5_cl3_SimAM_EUCB/weights/best.pt")

# ✅ 正确修改方式
model.model.names = {
    0: "people",
    1: "bike",
    2: "tricycle",
    3: "car"
}


# 2. 图片目录
# "./testdet/jpg"
# /disk2/yhl/ultralytics/experient_fig/track/天桥右1.mp4
img_dir = Path("/disk2/yhl/ultralytics/experient_fig/track/天桥右1.mp4")
start = time.time()
# 3. 推理并保存结果
results = model.track(
    source=str(img_dir),
    conf=0.25,
    save=True,
    tracker ="botsort.yaml",
    persist=True,


)

print(model.names)
end = time.time()

print(f"总耗时: {end-start:.2f} s")

print("Done! Results saved to runs/detect/predict/")
