from ultralytics import YOLO
from pathlib import Path



def print_head_params(model: YOLO):
    """
    打印 YOLOv11 模型中各检测头和总参数量
    """
    net = model.model  # 获取实际 nn.Module
    total_params = 0
    trainable_params = 0

    # 检测 head 的名称一般在 net.model.head 或者 net.model.detect
    # 不同版本可能命名不同，这里尝试 autodetect
    head_module = getattr(net, "head", None) or getattr(net, "detect", None)

    print("=== YOLOv11 参数统计 ===")
    if head_module is not None:
        print(f"检测头数: {len(head_module.cls_head)} (假设 cls_head 和 box_head 对应)")
        for i, (cls_h, box_h) in enumerate(zip(head_module.cls_head, head_module.box_head)):
            cls_params = sum(p.numel() for p in cls_h.parameters())
            box_params = sum(p.numel() for p in box_h.parameters())
            print(f"Head {i}: cls={cls_params}, box={box_params}, total={cls_params+box_params}")

            total_params += cls_params + box_params
            trainable_params += sum(p.numel() for p in list(cls_h.parameters()) + list(box_h.parameters()) if p.requires_grad)

    # 统计 backbone / neck 其他部分参数
    other_params = sum(p.numel() for p in net.parameters()) - total_params
    other_trainable = sum(p.numel() for p in net.parameters() if p.requires_grad) - trainable_params
    print(f"Backbone + Neck 其他参数: {other_params}, 可训练: {other_trainable}")

    total_params += other_params
    trainable_params += other_trainable
    print(f"模型总参数: {total_params}, 可训练参数: {trainable_params}")


# runs/detect/v6.4_n_p2_shanp5_cl3_SimAM_EUCB/weights/best.pt
# /disk2/yhl/ultralytics/ultralytics/runs/detect/n_cl3_640/weights/best.pt
# 1. 加载模型（换成你自己的 .pt 也可以）
model = YOLO("runs/detect/v6.4_n_p2_shanp5_cl3_SimAM_EUCB/weights/best.pt")
# print_head_params(model)

print("before:", model.names)

# ✅ 正确修改方式
model.model.names = {
    0: "people",
    1: "bike",
    2: "tricycle",
    3: "car"
}

print("after:", model.model.names)

# 2. 图片目录
# "./testdet/jpg"
img_dir = Path("experient_fig/yolo11n")

# 3. 推理并保存结果
results = model(
    source=str(img_dir),
    conf=0.25,
    save=True

)


