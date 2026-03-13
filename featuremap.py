import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

# runs/detect/v6.4_p2_n_cl3_640/weights/best.pt
# ultralytics/runs/detect/n_cl3_640/weights/best.pt
# /disk2/yhl/ultralytics/runs/detect/v6.4_n_p2_shanp5_cl3_SimAM_EUCB/weights/best.pt
# /disk2/yhl/ultralytics/runs/detect/v6.4_p2_n_cl3_640/weights/best.pt
# runs/detect/v6.4_n_p2_shanp5_cl3_SimAM/weights/best.pt
# runs/detect/v6.4_n_p2_cl3_SimAM/weights/best.pt
# runs/detect/v6.4_p2_n_cl3_640_EUCB/weights/best.pt
# /disk2/yhl/ultralytics/runs/detect/v6.4_n_p2shanp5_cl3/weights/best.pt
# 加载模型
model = YOLO("runs/detect/v6.4_p2_n_cl3_640/weights/best.pt")
net = model.model

# 存储特征图
features = {}

# hook函数
def get_feature(name):
    def hook(module, input, output):
        features[name] = output.detach().cpu()
    return hook


# 注册hook (P3 P4 P5层，根据模型结构可能略有不同)
# net.model[16].register_forward_hook(get_feature("yolov11n_P3"))
# net.model[19].register_forward_hook(get_feature("yolo11n_P4"))
# net.model[22].register_forward_hook(get_feature("yolo11n_P5"))
net.model[19].register_forward_hook(get_feature("yolov11n_P2_1"))
# net.model[22].register_forward_hook(get_feature("yolov11n_P2_p3_1"))
# net.model[25].register_forward_hook(get_feature("yolo11n_P4_10"))

# experient_fig/yolo11n/Snipaste_2026-01-20_14-00-47.png
# experient_fig/bytetrack_workbad/000508.jpg
# experient_fig/yolo11n/0000003_00231_d_0000016.jpg
# 读取图片
img = cv2.imread("experient_fig/yolo11n/0000074_13313_d_0000026.jpg")
img = cv2.resize(img, (640,640))
img = img[:,:,::-1] / 255.0
img = np.transpose(img,(2,0,1))
img = torch.tensor(img).float().unsqueeze(0)

# 前向传播
with torch.no_grad():
    net(img)


# 可视化特征图
for name, fmap in features.items():

    fmap = fmap[0]        # batch=1
    fmap = fmap.mean(0)   # 多通道平均

    plt.figure(figsize=(5,5))
    plt.imshow(fmap, cmap='viridis')
    plt.title(name)
    plt.axis("off")

    plt.savefig(f"experient_fig/featuremap/{name}_feature.png")
    plt.close()

print("Feature maps saved.")