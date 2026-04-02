from ultralytics import YOLO

model_trained = YOLO("ultralytics/runs/detect/n_cl3_640/weights/best.pt").model
model_ref = YOLO("yolo11n.pt").model   # 或 YOLO("yolo11n.yaml").model

sd1 = model_trained.state_dict()
sd2 = model_ref.state_dict()

keys1 = list(sd1.keys())
keys2 = list(sd2.keys())

print("参数数量是否一致：", len(keys1) == len(keys2))

for k1, k2 in zip(keys1, keys2):
    if k1 != k2:
        print("参数名不同：", k1, k2)
        break
    if sd1[k1].shape != sd2[k2].shape:
        print("shape不同：", k1, sd1[k1].shape, sd2[k2].shape)
        break
else:
    print("模型结构和参数shape完全一致")