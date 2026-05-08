# utils.py
# 描述: 存放通用的辅助函数

import os
import torch
from ultralytics import YOLO


def parse_ground_truth(txt_path):
    """
    解析标注文件 (来自 evaluate_accuracy.py)

    """
    if not os.path.exists(txt_path):
        print(f"❌ 错误: 标注文件 '{txt_path}' 不存在。")
        return None
    ground_truth_map = {}
    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if ',' in line:
                category, filename = line.split(',', 1)
                ground_truth_map[filename.strip()] = category.strip()
    print(f"✅ 成功加载 {len(ground_truth_map)} 条标注信息。")
    return ground_truth_map


def load_yolo_model(model_path):
    """
    加载YOLOv8模型并确定运行设备
    """
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = YOLO(model_path)
        model.to(device)
        print(f"  -> YOLOv8 模型 '{model_path}' 已加载到 {device}")
        return model, device
    except Exception as e:
        print(f"❌ 加载YOLO模型 '{model_path}' 失败: {e}")
        return None, None