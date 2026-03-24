# import subprocess
# from pathlib import Path

# avi_dir = Path("runs/detect/track36/微信视频2026-03-10_145412_595.avi")

# for avi_path in avi_dir.glob("*.avi"):
#     mp4_path = avi_path.with_suffix(".mp4")

#     cmd = [
#         "ffmpeg",
#         "-y",
#         "-i", str(avi_path),

#         # ✅ H.264（你环境里真正可用的）
#         "-c:v", "libopenh264",

#         # ✅ 强制标准像素 & 色彩空间
#         "-pix_fmt", "yuv420p",
#         "-colorspace", "bt709",
#         "-color_primaries", "bt709",
#         "-color_trc", "bt709",

#         # ✅ Cursor / 浏览器关键
#         "-movflags", "+faststart",

#         str(mp4_path)
#     ]

#     print(f"Converting: {avi_path.name}")
#     subprocess.run(cmd, check=True)

# print("✅ All AVI files converted to Cursor-playable MP4")


import subprocess
from pathlib import Path

avi_path = Path("runs/detect/track89/天桥右1.avi")
save_path = avi_path.with_suffix(".mp4")

cmd = [
    "ffmpeg",
    "-y",
    "-i", str(avi_path),
    "-c:v", "libopenh264",
    "-pix_fmt", "yuv420p",
    "-colorspace", "bt709",
    "-color_primaries", "bt709",
    "-color_trc", "bt709",
    "-movflags", "+faststart",
    str(save_path)
]

print(f"Converting: {avi_path.name}")
subprocess.run(cmd, check=True)

print("保存路径:", save_path)
print(f"✅ MP4 saved to: {save_path}")