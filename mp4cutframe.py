# import cv2
# from pathlib import Path

# video_path = "experient_fig/doublesight/天桥右1.mp4"
# out_dir = Path("experient_fig/doublesight/frame1")
# out_dir.mkdir(exist_ok=True)

# cap = cv2.VideoCapture(video_path)
# idx = 0

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break
#     cv2.imwrite(str(out_dir / f"{idx:06d}.jpg"), frame)
#     idx += 1

# cap.release()



import os
from pathlib import Path

# 改成你的图片文件夹路径
folder = Path(r"experient_fig/doublesight/frame2")

# 先收集所有 jpg 文件
files = sorted(folder.glob("*.jpg"))

# 第一步：先改成临时名字，避免重名冲突
temp_files = []
for f in files:
    old_num = int(f.stem)   # 文件名数字部分
    new_num = old_num - 6
    if new_num < 0:
        print(f"跳过 {f.name}，减6后为负数")
        continue

    temp_name = folder / f"temp_{f.name}"
    f.rename(temp_name)
    temp_files.append((temp_name, new_num, len(f.stem), f.suffix))

# 第二步：改成正式名字
for temp_f, new_num, width, suffix in temp_files:
    new_name = folder / f"{new_num:0{width}d}{suffix}"
    temp_f.rename(new_name)
    print(f"{temp_f.name} -> {new_name.name}")

print("重命名完成")