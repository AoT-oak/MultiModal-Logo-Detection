# clip_text_analyzer.py
# 描述: 封装所有与 CLIP 图文匹配 (零样本) 相关的分析功能

import os
import torch
import torch.nn.functional as F
import threading
from PIL import Image

try:
    from transformers import CLIPProcessor, CLIPModel
except ImportError:
    print("❌ 错误: 未找到 'transformers' 库。请运行: pip install transformers")
    exit(1)


def load_clip_model(model_path, device):
    """
    加载CLIP模型和处理器
    (来自 reproduce_paper_model.py)

    """
    try:
        print(f"  -> 正在加载CLIP模型到 {device}...")
        model = CLIPModel.from_pretrained(model_path).to(device)
        processor = CLIPProcessor.from_pretrained(model_path)
        print("  -> CLIP模型加载成功。")
        return model, processor
    except Exception as e:
        print(f"❌ 加载CLIP模型 '{model_path}' 失败: {e}")
        return None, None


def load_text_templates(all_categories):
    """
    为所有类别生成文本提示模板
    (来自 reproduce_paper_model.py)

    """
    templates = {}
    for category in all_categories:
        if category == 'no-logo': continue  # no-logo 单独处理
        templates[category] = [
            f"a photo of the {category} logo",
            f"the {category} trademark",
            f"a logo of {category}",
            f"a photo of {category}",
            f"{category}"
        ]
    print(f"  -> 已为 {len(templates)} 个类别生成文本提示。")
    return templates


def precompute_text_features(model, processor, text_templates, device):
    """
    预计算所有文本提示的特征向量
    (来自 reproduce_paper_model.py)

    """
    text_features_dict = {}
    with torch.no_grad():
        for category, prompts in text_templates.items():
            inputs = processor(text=prompts, return_tensors="pt", padding=True, truncation=True).to(device)
            text_features = model.get_text_features(**inputs)
            text_features_dict[category] = text_features  # 存储 (N, D) 张量
    print(f"  -> 已预计算 {len(text_features_dict)} 个类别的文本特征。")
    return text_features_dict


def _vectorize_single_image(image_pil, model, processor, device, lock):
    """
    为单张PIL图片生成特征向量。
    (改编自 visual_similarity_analyzer.py)

    """
    try:
        image = image_pil.convert("RGB")  # 确保是RGB
        inputs = processor(images=image, return_tensors="pt", padding=True, truncation=True).to(device)

        with lock:  # 保护非线程安全的模型调用
            with torch.no_grad():
                image_features = model.get_image_features(**inputs)

        return image_features  # 返回 (1, D) 张量
    except Exception as e:
        print(f"处理PIL图片时出错: {e}，将跳过。")
        return None


def _get_scores(image_features, text_features_dict):
    """
    (新) 内部辅助函数：计算图像特征与所有文本特征字典的相似度

    """
    category_scores = {}

    # 归一化图像特征 (1, D)
    image_features_norm = F.normalize(image_features, p=2, dim=-1)

    for category, text_feats_list in text_features_dict.items():
        # 归一化文本特征 (N, D)
        text_feats_norm = F.normalize(text_feats_list, p=2, dim=-1)

        # 计算相似度 (1, D) x (D, N) = (1, N)
        similarities = torch.mm(image_features_norm, text_feats_norm.T)

        # 取该类别所有提示词中的最高分作为该类别的得分
        best_score_for_category = torch.max(similarities).item()
        category_scores[category] = best_score_for_category

    return category_scores


def get_all_clip_text_scores(image_pil, model, processor, text_features_dict, device, lock):
    """
    (【新功能】)
    对单个图像(PIL)执行CLIP图文匹配分析，并返回所有类别的分数。

    """
    try:
        image_features = _vectorize_single_image(image_pil, model, processor, device, lock)
        if image_features is None:
            return {}  # 返回空字典

        category_scores = _get_scores(image_features, text_features_dict)
        return category_scores

    except Exception as e:
        print(f"❌ 在CLIP获取所有分数时发生错误: {e}")
        return {}