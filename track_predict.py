from ultralytics import YOLO
from pathlib import Path




# 1. 加载模型（换成你自己的 .pt 也可以）
model = YOLO("/disk2/yhl/ultralytics/runs/detect/v6.4_pl3/weights/best.pt")

# ✅ 正确修改方式
model.model.names = {
    0: "people",
    1: "bike",
    2: "tricycle",
    3: "car"
}


# 2. 图片目录
# "./testdet/jpg"
img_dir = Path("/disk2/yhl/ultralytics/experient_fig/track/天桥右1.mp4")

# 3. 推理并保存结果
results = model.track(
    source=str(img_dir),
    conf=0.25,
    save=True,
    tracker ="botsort.yaml",
    persist=True,

)


print("Done! Results saved to runs/detect/predict/")
