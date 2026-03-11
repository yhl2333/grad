import pandas as pd
import matplotlib.pyplot as plt

import os

file1 = r"/disk2/yhl/ultralytics/runs/detect/v6.4_n_p2_shanp5_cl3_SimAM_EUCB/results.csv"
file2 = r"/disk2/yhl/ultralytics/ultralytics/runs/detect/n_cl3_640/results.csv"

label1 = "Improved YOLOv11n"
label2 = "YOLOv11n"
save_dir = r"/disk2/yhl/ultralytics/experient_fig/mpa50"
df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)

df1.columns = df1.columns.str.strip()
df2.columns = df2.columns.str.strip()

# metrics/mAP50(B),metrics/mAP50-95(B),val/box_loss,val/cls_loss,val/dfl_loss

epoch1 = df1["epoch"]
map50_1 = df1["metrics/mAP50(B)"]

epoch2 = df2["epoch"]
map50_2 = df2["metrics/mAP50(B)"]

plt.figure(figsize=(8, 5))

plt.plot(epoch1, map50_1, label=label1, linewidth=2)
plt.plot(epoch2, map50_2, label=label2, linewidth=2)

plt.xlabel("Epoch")
plt.ylabel("mAP@0.5")
plt.title("Comparison of mAP@0.5 During Training")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
png_path = os.path.join(save_dir, "map50_compare.png")
svg_path = os.path.join(save_dir, "map50_compare.svg")

plt.savefig(png_path, dpi=300, bbox_inches="tight")
plt.savefig(svg_path, bbox_inches="tight")
plt.show()

print("图片已保存到：")
print(png_path)
print(svg_path)