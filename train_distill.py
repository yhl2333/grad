import warnings
warnings.filterwarnings('ignore')
import argparse, yaml, copy
from ultralytics.models.yolo.detect.distill import DetectionDistiller

if __name__ == '__main__':
    param_dict = {
        # student
        'model': 'ultralytics/runs/detect/p2_n_cl3_640/weights/best.pt',
        'data':'ultralytics/cfg/datasets/NewVisDrone.yaml',
        'imgsz': 640,
        'epochs': 120,
        'batch': -1,
        'workers': 2,
        'cache': True,
        'optimizer': 'SGD',
        'device': '0',
        'close_mosaic': 20,
        'amp': False, # 如果蒸馏损失为nan，请把amp设置为False
        'project':'runs/distill',
        'name':'exp',
        
        # teacher
        'teacher_weights': 'ultralytics/runs/detect/p2_m_cl3_640/weights/best.pt',
        'teacher_cfg': 'ultralytics/cfg/models/11/yolo11m_new.yaml',
        'kd_loss_type': 'all',
        'kd_loss_decay': 'constant',
        
        'logical_loss_type': 'BCKD',
        'logical_loss_ratio': 0.8,
        
        'teacher_kd_layers': '19,22,25,28',
        'student_kd_layers': '19,22,25,28',
        'feature_loss_type': 'cwd',
        'feature_loss_ratio': 0.5
    }
    
    model = DetectionDistiller(overrides=param_dict)
    model.distill()
