#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Logo Detector App (Hybrid Architecture)
GitHub-ready Inference Pipeline combining YOLO, MLLM (GLM-4V), and CLIP.

==========================================
版本更迭记录
v1.0.0: 初始混合架构版本 (YOLO + MLLM + CLIP) 单线程顺序执行
v1.1.0: 增强模块耦合，MLLM 模型统一由主程序参数管控
v1.2.0: [性能优化] 引入 ThreadPoolExecutor，实现 MLLM 分类与描述 API 的高并发请求，并与本地 CLIP 推理时间重叠，显著提升运行速度。
==========================================
"""

import os
import json
import logging
import argparse
import threading
import concurrent.futures  # 新增：用于多线程并发
from typing import Dict, Any, List
from PIL import Image

# 禁用并行以防死锁
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"

# ==========================================
# 导入本地功能模块
# ==========================================
try:
    from utils import load_yolo_model
    from mllm_analyzer import run_mllm_classification, encode_image_to_base64_from_path, encode_image_to_base64_from_pil
    from clip_text_analyzer import load_clip_model, load_text_templates, precompute_text_features, \
        get_all_clip_text_scores
    from image_describer import get_image_description
except ImportError as e:
    raise ImportError(f"❌ 导入依赖失败，请检查文件是否齐全: {e}")

# ==========================================
# 日志配置
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("LogoDetector")


class LogoDetectorApp:
    def __init__(self,
                 yolo_model_path: str = 'best.pt',
                 clip_model_path: str = './clip-model-offline',
                 categories_file: str = 'categories.txt',
                 mllm_model_name: str = 'GLM-4V-Flash'):
        """初始化混合推理引擎"""
        logger.info("🚀 正在启动 Logo 检测服务引擎...")

        self.all_categories = self._load_categories(categories_file)
        self.all_categories_str = str(self.all_categories)
        self.mllm_model_name = mllm_model_name

        self.api_key = os.getenv("ZHIPUAI_API_KEY")
        if not self.api_key:
            logger.error("未找到 ZHIPUAI_API_KEY 环境变量！")
            raise ValueError("请设置环境变量: export ZHIPUAI_API_KEY='your_key'")

        logger.info("📦 开始加载视觉模型...")
        self.yolo_model, self.device = load_yolo_model(yolo_model_path)
        self.clip_model, self.clip_processor = load_clip_model(clip_model_path, self.device)

        if not self.yolo_model or not self.clip_model:
            raise RuntimeError("❌ 模型加载失败，请检查模型权重路径！")

        logger.info("🧠 正在预计算 CLIP 文本特征字典...")
        text_templates = load_text_templates(self.all_categories)
        self.text_features_dict = precompute_text_features(self.clip_model, self.clip_processor, text_templates,
                                                           self.device)

        self.model_lock = threading.Lock()
        logger.info(f"✅ 服务引擎初始化完成！全局 MLLM 统管为: {self.mllm_model_name}")
        print("-" * 60)

    def _load_categories(self, file_path: str) -> List[str]:
        """安全读取并清洗类别文件"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"找不到类别文件 {file_path}")

        categories = set()
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip().lower()

                # 确保清理后不是空行，且不是 no-logo
                if line and line != 'no-logo':
                    categories.add(line)

        cleaned_list = sorted(list(categories))
        logger.info(f"📁 成功加载 {len(cleaned_list)} 个商标类别。")
        return cleaned_list

    def predict(self, image_path: str, top_k: int = 3) -> Dict[str, Any]:
        """核心推理函数"""
        if not os.path.exists(image_path):
            return {"status": "error", "message": f"图片 {image_path} 不存在", "predictions": []}

        image_filename = os.path.basename(image_path)
        try:
            original_image = Image.open(image_path).convert("RGB")
        except Exception as e:
            return {"status": "error", "message": f"图片损坏或读取失败: {e}", "predictions": []}

        logger.info(f"🔍 正在处理图片: {image_filename}")

        # --- 阶段 1：YOLO 目标检测与裁剪 ---
        yolo_detected = False
        images_to_process = []
        try:
            with self.model_lock:
                yolo_results = self.yolo_model.predict(source=image_path, conf=0.25, verbose=False, device=self.device)
            if yolo_results and len(yolo_results[0].boxes) > 0:
                yolo_detected = True
                for box in yolo_results[0].boxes:
                    xyxy = box.xyxy[0].cpu().numpy().astype(int)
                    x1, y1, x2, y2 = xyxy
                    # 确保裁剪区域有效（宽和高至少大于2个像素）
                    if x2 - x1 > 2 and y2 - y1 > 2:
                        images_to_process.append(original_image.crop((x1, y1, x2, y2)))
                    else:
                        logger.warning(f"忽略极小或无效的检测框: {xyxy}")
        except Exception as e:
            logger.warning(f"YOLO 检测阶段出现异常: {e}")

        # --- 阶段 2：MLLM 提取候选集 (并发优化) ---
        mllm_candidate_set = set()
        base64_full = encode_image_to_base64_from_path(image_path)

        # 建立最大6个线程的线程池，作用域覆盖到函数末尾，以便接管后台描述任务
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:

            # 1. 最优先：将最耗时的“图片整体描述”丢进后台线程默默执行
            logger.info("⚡ 并发调度：已将图片描述任务丢入后台线程...")
            future_desc = executor.submit(
                get_image_description, base64_full, self.api_key, model_name=self.mllm_model_name
            )

            # 2. 收集所有的 MLLM 分类请求
            mllm_classification_futures = []

            logger.info("⚡ 并发调度：提交全图 MLLM 分类请求...")
            mllm_classification_futures.append(executor.submit(
                run_mllm_classification, base64_full, f"{image_filename}_full", self.all_categories_str,
                self.api_key, model_name=self.mllm_model_name
            ))

            if yolo_detected:
                logger.info(f"⚡ 并发调度：并行提交 {len(images_to_process)} 个局部 Crop 的 MLLM 分类请求...")
                for i, pil_img in enumerate(images_to_process):
                    base64_crop = encode_image_to_base64_from_pil(pil_img)
                    mllm_classification_futures.append(executor.submit(
                        run_mllm_classification, base64_crop, f"{image_filename}_crop{i}", self.all_categories_str,
                        self.api_key, model_name=self.mllm_model_name
                    ))

            # 3. 阻塞等待：等待所有“分类任务”完成，合并结果至候选集
            # 注：必须等候选集集齐，下面的 CLIP 阶段才能进行
            for future in concurrent.futures.as_completed(mllm_classification_futures):
                try:
                    preds = future.result()
                    mllm_candidate_set.update(p.lower() for p in preds if p.lower() in self.all_categories)
                except Exception as e:
                    logger.warning(f"某个 MLLM 分类请求发生异常: {e}")

            if not mllm_candidate_set:
                mllm_candidate_set.add('no-logo')

            # --- 阶段 3：CLIP 精确打分 ---
            # (此时，后台的 future_desc 大概率还在跑网络请求，正好与本地的 CLIP 推理计算在时间上重叠并行！)
            final_clip_scores = {}
            full_scores = get_all_clip_text_scores(original_image, self.clip_model, self.clip_processor,
                                                   self.text_features_dict, self.device, self.model_lock)
            for cat, score in full_scores.items():
                final_clip_scores[cat.lower()] = max(final_clip_scores.get(cat.lower(), 0), score)

            if yolo_detected:
                for pil_img in images_to_process:
                    crop_scores = get_all_clip_text_scores(pil_img, self.clip_model, self.clip_processor,
                                                           self.text_features_dict, self.device, self.model_lock)
                    for cat, score in crop_scores.items():
                        final_clip_scores[cat.lower()] = max(final_clip_scores.get(cat.lower(), 0), score)

            # --- 阶段 4：融合决策与图片描述 ---
            filtered_scores = {cat: final_clip_scores.get(cat, 0.0) for cat in mllm_candidate_set}

            if 'no-logo' in filtered_scores:
                best_logo_score = max((s for c, s in final_clip_scores.items() if c != 'no-logo'), default=0.0)
                filtered_scores['no-logo'] = 1.0 - best_logo_score

            sorted_candidates = sorted(filtered_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
            results = [{"category": cat, "confidence": round(score, 4)} for cat, score in sorted_candidates]

            # 4. 回收后台任务：拿取最早丢入后台的描述结果
            logger.info("📝 正在获取描述任务结果 (如已在后台完成则瞬间通过)...")
            try:
                # 设置一个相对安全的超时，防止极端情况下大模型卡死
                image_description = future_desc.result(timeout=45.0)
            except concurrent.futures.TimeoutError:
                logger.error("获取图片描述任务超时！")
                image_description = "描述生成失败 (请求超时)"
            except Exception as e:
                logger.error(f"获取图片描述任务失败: {e}")
                image_description = "描述生成失败"

            logger.info(f"✅ 检测完成，Top-1 类别: {results[0]['category'] if results else '无'}")

            final_output = {
                "status": "success",
                "image": image_filename,
                "predictions": results,
                "description": image_description
            }

            return final_output


# ==========================================
# 命令行接口 (CLI)
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hybrid Logo Detector App")
    parser.add_argument("--image", type=str, required=True, help="要检测的图片路径")
    parser.add_argument("--yolo", type=str, default="best.pt", help="YOLOv8 模型路径")
    parser.add_argument("--clip", type=str, default="./clip-model-offline", help="CLIP 模型目录")
    parser.add_argument("--categories", type=str, default="categories.txt", help="类别字典文件")
    parser.add_argument("--mllm", type=str, default="GLM-4V-Flash", help="统一使用的 MLLM 模型名称")
    parser.add_argument("--top_k", type=int, default=3, help="返回前 K 个最高分结果")

    args = parser.parse_args()

    try:
        app = LogoDetectorApp(
            yolo_model_path=args.yolo,
            clip_model_path=args.clip,
            categories_file=args.categories,
            mllm_model_name=args.mllm
        )

        result_json = app.predict(args.image, top_k=args.top_k)

        print("\n" + "=" * 40)
        print("🎯 最终输出 (JSON)")
        print("=" * 40)
        print(json.dumps(result_json, indent=4, ensure_ascii=False))
        print("=" * 40 + "\n")

    except Exception as err:
        logger.critical(f"程序运行失败: {err}")