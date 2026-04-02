from ultralytics import YOLO



from ultralytics import YOLO

def main():
    # 1. 加载 YOLO11 预训练模型
    # 可选：yolo11n.pt / yolo11s.pt / yolo11m.pt / yolo11l.pt
    # model = YOLO(pretrained="./pretrained/yolo11n.pt")
    
    model = YOLO(model = "ultralytics/cfg/models/11/yolo11_EUCB.yaml")
    model.load("./pretrained/yolo11n.pt")
    # print(model.names)

    # 2. 开始训练
    model.train(
        # ===== 数据 =====
        resume = True,
        data="ultralytics/cfg/datasets/NewVisDrone.yaml",   # 数据集配置文件
        epochs=240,             # 训练轮数
        imgsz=640,              # 输入尺寸（VisDrone推荐 ≥ 960）
        batch=2,                # 根据显存调整
        device=0,               # GPU id
        cache='disk',            # 避免内存炸
        workers = 1,
        save=True,
        show = False,


        # ===== 优化器 =====
        # optimizer="AdamW",      # 小目标更友好
        # lr0=0.001,
        # lrf=0.01,
        # weight_decay=5e-4,
        # warmup_epochs=5,

        # # ===== 数据增强 =====
        # mosaic=1,
        # mixup=0.1,
        # scale=0.7,
        # translate=0.2,
        # fliplr=0.5,
  

        # # ===== Loss 权重 =====
        # box=10.0,               # 小目标增强
        # cls=0.5,
        # dfl=1.5,

        # # ===== 训练控制 =====
        # workers=2,              # 服务器上建议 ≤4
        
        # amp=True,               # 混合精度
        # patience=50,            # early stop
        # save_period=10,

        # # ===== 日志 =====
        # project="runs/yolo11",
        # name="yolo11_visdrone"
    )

if __name__ == "__main__":
    main()