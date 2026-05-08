#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# image_describer.py
# 描述: 负责对输入的图片进行整体的自然语言描述，提取场景和商标信息

import time
import random

try:
    from zhipuai import ZhipuAI
except ImportError:
    pass


def get_image_description(image_base64: str, api_key: str, model_name: str,
                          max_retries: int = 3) -> str:
    """
    调用大视觉模型对图片进行全面描述，并带有网络波动重试机制。
    """
    if not image_base64:
        return "无法获取图片数据进行描述。"

    # 针对免费/性能较弱模型优化的结构化提示词（引入 Few-Shot 示例）
    prompt_text = """请分析这张图片，并严格按照以下两行的格式输出结果，绝不要包含任何额外的解释或废话：

商标：[推测有（填写具体的商标名称） 或 推测无]
内容：[用一句话简单客观地描述图片中的主要物体或场景]

示例 1：
商标：推测有（Nike）
内容：一双黑色的运动鞋放在木地板上。

示例 2：
商标：推测无
内容：一只橘猫正躺在沙发上睡觉。"""

    for attempt in range(max_retries):
        try:
            client = ZhipuAI(api_key=api_key)
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt_text},
                            {"type": "image_url", "image_url": {"url": image_base64}}
                        ]
                    }
                ],
                temperature=0.1,  # 将温度降至 0.1，让模型更严格地遵守示例格式，减少幻觉和长篇大论
                timeout=60.0
            )
            return response.choices[0].message.content.strip()

        except Exception as e:
            if attempt < max_retries - 1:
                # 指数退避策略，加入随机抖动(Jitter)避免请求风暴
                delay = 2 * (2 ** attempt) + random.uniform(0, 1)
                print(
                    f"⚠️ 警告: 图片描述请求失败 (尝试 {attempt + 1}/{max_retries})，将在 {delay:.2f} 秒后重试... 错误: {e}")
                time.sleep(delay)
            else:
                return f"描述生成失败: 经过 {max_retries} 次尝试后依然失败，错误信息: {e}"