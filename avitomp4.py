import subprocess
from pathlib import Path

avi_dir = Path("runs/detect/track")

for avi_path in avi_dir.glob("*.avi"):
    mp4_path = avi_path.with_suffix(".mp4")

    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(avi_path),

        # ✅ H.264（你环境里真正可用的）
        "-c:v", "libopenh264",

        # ✅ 强制标准像素 & 色彩空间
        "-pix_fmt", "yuv420p",
        "-colorspace", "bt709",
        "-color_primaries", "bt709",
        "-color_trc", "bt709",

        # ✅ Cursor / 浏览器关键
        "-movflags", "+faststart",

        str(mp4_path)
    ]

    print(f"Converting: {avi_path.name}")
    subprocess.run(cmd, check=True)

print("✅ All AVI files converted to Cursor-playable MP4")