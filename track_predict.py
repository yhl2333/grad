from ultralytics import YOLO
from pathlib import Path




# 1. 加载模型（换成你自己的 .pt 也可以）
model = YOLO("/disk2/yhl/ultralytics/ultralytics/runs/detect/p2_n_cl3_640/weights/best.pt")

# 2. 图片目录
# "./testdet/jpg"
img_dir = Path("/disk2/yhl/ultralytics/testdet/video/311158_small.mp4")

# 3. 推理并保存结果
results = model.track(
    source=str(img_dir),
    conf=0.25,
    save=True,
    tracker ="bytetrack.yaml"

)


print("Done! Results saved to runs/detect/predict/")
