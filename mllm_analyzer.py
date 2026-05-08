# mllm_analyzer.py
# 描述: 封装所有与智谱 MLLM API (GLM-4V) 相关的分析功能

import os
import base64
import mimetypes
import io
import ast
import re
import json
import time
import random
from PIL import Image

try:
    from zhipuai import ZhipuAI
    from zhipuai.core._errors import APIStatusError
except ImportError:
    print("❌ 错误: 未找到 'zhipuai' 库。请运行: pip install zhipuai")
    exit(1)


def encode_image_to_base64_from_path(image_path):
    """
    将本地图片文件编码为Base64。
    (来自 evaluate_with_mllm.py)

    """
    mime_type, _ = mimetypes.guess_type(image_path)
    if mime_type is None:
        mime_type = "application/octet-stream"
    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        return f"data:{mime_type};base64,{encoded_string}"
    except Exception as e:
        print(f"❌ 错误: 无法读取或编码图片文件 '{image_path}': {e}")
        return None


def encode_image_to_base64_from_pil(pil_image, format="PNG"):
    """
    将内存中的PIL Image对象编码为Base64。
    (来自 evaluate_with_mllm.py)

    """
    try:
        buffered = io.BytesIO()
        pil_image.save(buffered, format=format)
        encoded_string = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return f"data:image/{format.lower()};base64,{encoded_string}"
    except Exception as e:
        print(f"❌ 错误: 无法编码PIL图像: {e}")
        return None


def run_mllm_classification(image_base64, image_name_for_logging, all_categories_str, api_key, model_name="GLM-4V-Flash",
                           max_retries=3):
    """
    调用MLLM API执行分类任务。
    返回一个包含预测类别名称的集合 (set)。
    (来自 evaluate_with_mllm.py)

    """
    if not image_base64:
        return set()

    last_exception = None
    base_delay = 2

    prompt_text = f"""你是一个专业的商标分类专家。请分析这张图片。
下面是所有可能的商标类别列表：
{all_categories_str}

请从上面的列表中，选出图片内容所对应的所有类别。
你的回答必须是一个Python的列表（list），只包含你从列表中选择的类别名称字符串。
例如：['类别A', '类别B']
如果图片与列表中的任何类别都不匹配，请返回一个空列表：[]。
绝对不要添加任何解释、描述或额外的对话。"""

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
                temperature=0.01,
                top_p=0.99,
                timeout=60.0
            )

            result_content = response.choices[0].message.content.strip()

            try:
                predicted_list = ast.literal_eval(result_content)
                if isinstance(predicted_list, list):
                    return set(str(item).strip() for item in predicted_list)
            except (ValueError, SyntaxError):
                print(
                    f"⚠️ 警告: MLLM为图片 '{image_name_for_logging}' 返回了非标准格式: {result_content}。尝试正则提取...")
                found_categories = re.findall(r"['\"]([^'\"]+)['\"]", result_content)
                if found_categories:
                    return set(cat.strip() for cat in found_categories)

            return set()

        except APIStatusError as e:
            last_exception = e
            try:
                error_data = json.loads(e.response.text)
                error_code = error_data.get("error", {}).get("code")
                if error_code == "1301":
                    print(f"\n提示: MLLM内容安全审核失败 (code: 1301)，已跳过图片 {image_name_for_logging}。")
                    return set()
            except (json.JSONDecodeError, AttributeError):
                pass

            delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
            if attempt < max_retries - 1:
                print(
                    f"警告: MLLM API请求失败 (尝试 {attempt + 1}/{max_retries})。将在 {delay:.2f} 秒后重试... 错误: {e}")
                time.sleep(delay)

        except Exception as e:
            last_exception = e
            delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
            if attempt < max_retries - 1:
                print(
                    f"警告: MLLM处理 '{image_name_for_logging}' 时发生未知错误 (尝试 {attempt + 1}/{max_retries})。将在 {delay:.2f} 秒后重Test... 错误: {e}")
                time.sleep(delay)

    print(f"❌ 错误: MLLM API在 {max_retries} 次尝试后最终失败: {last_exception}")
    return set()