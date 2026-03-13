import cv2
import os

# 输入图片文件夹
images_path = r"datasetTrack/MOT17/train/MOT17-13-FRCNN/img1"

# 输出视频文件
output_video_path = r"datasetTrack/MOT17/outmp4/MOT17-13-FRCNN.mp4"
os.makedirs(os.path.dirname(output_video_path), exist_ok=True)

# 图像列表排序
images_list = os.listdir(images_path)
images_list.sort()

# 读取第一张图，获取尺寸
first_image = cv2.imread(os.path.join(images_path, images_list[0]))
if first_image is None:
    raise ValueError("第一张图像读取失败，请检查路径")

height, width = first_image.shape[:2]

# 创建视频写入器
video_writer = cv2.VideoWriter(
    output_video_path,
    cv2.VideoWriter_fourcc(*"mp4v"),
    30,
    (width, height)
)

# 检查是否成功打开
if not video_writer.isOpened():
    raise ValueError("VideoWriter 打开失败，请检查输出路径或编码器")

# 写入所有图片
for image_name in images_list:
    image_path = os.path.join(images_path, image_name)
    image = cv2.imread(image_path)

    if image is None:
        print(f"无法读取图像 {image_name}，跳过")
        continue

    # 若尺寸不一致，统一调整
    if image.shape[:2] != (height, width):
        image = cv2.resize(image, (width, height))

    video_writer.write(image)

# 释放资源
video_writer.release()
print("视频保存完成：", output_video_path)