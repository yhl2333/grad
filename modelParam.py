from ultralytics import YOLO
from pathlib import Path

import pandas as pd



# 1. 加载模型（换成你自己的 .pt 也可以）
model = YOLO("runs/detect/train82/weights/best.pt")
model.info(detailed=True)



# 文件路径
file_path = "runs/detect/train82/results.csv"   # 改成你的文件路径，也可以是 results.csv

# 读取文件
df = pd.read_csv(file_path)

# 找到 mAP50-95 最大值所在行
best_row = df.loc[df["metrics/mAP50-95(B)"].idxmax()]

# 提取对应指标
precision = best_row["metrics/precision(B)"]
recall = best_row["metrics/recall(B)"]
map50 = best_row["metrics/mAP50(B)"]
map50_95 = best_row["metrics/mAP50-95(B)"]
epoch = best_row["epoch"]

# 输出结果
print(f"最佳 epoch: {epoch}")
print(f"Precision: {precision:.6f}")
print(f"Recall: {recall:.6f}")
print(f"mAP50: {map50:.6f}")
print(f"mAP50-95: {map50_95:.6f}")