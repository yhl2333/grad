import cv2
from pathlib import Path

video_path = "testdet/video/mv_03.mp4"
out_dir = Path("testdet/videocutframe/frame1")
out_dir.mkdir(exist_ok=True)

cap = cv2.VideoCapture(video_path)
idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imwrite(str(out_dir / f"{idx:06d}.jpg"), frame)
    idx += 1

cap.release()
