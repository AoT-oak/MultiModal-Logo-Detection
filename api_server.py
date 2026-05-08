#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# api_server.py

#第一步：创建 FastAPI 服务端应用 (api_server.py)
#我们需要写一个新的 Python 脚本，它的任务是启动一个 Web 服务器，接收别人上传的图片，然后把图片交给我们的 LogoDetectorApp 处理
#最后把 JSON 结果返回去

import os
import shutil
import tempfile
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

# 导入我们之前写好的核心推理类
from app_inference import LogoDetectorApp

# 全局变量存储我们的检测器实例
detector = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    生命周期管理器：在服务器启动时自动加载所有的 AI 模型到内存中。
    这样每次来请求时就不需要重新加载模型了，实现极速响应。
    """
    global detector
    print("🚀 正在启动 FastAPI 服务器，并预热 AI 模型库...")
    try:
        # 这里默认加载当前目录下的权重文件，你可以根据实际情况修改路径
        detector = LogoDetectorApp(
            yolo_model_path='best.pt',
            clip_model_path='./clip-model-offline',
            categories_file='categories.txt',
            mllm_model_name='GLM-4V-Flash'
        )
        print("✅ AI 模型加载完毕，API 接口准备就绪！")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")

    yield  # 服务器运行期间停留在这一步

    print("🛑 服务器正在关闭，释放资源...")
    # 这里可以添加清理数据库连接或显存的代码（如果需要）


# 初始化 FastAPI 应用
app = FastAPI(
    title="Hybrid Logo Detector API",
    description="基于 YOLO + GLM-4V + CLIP 的混合商标检测接口",
    version="1.0.0",
    lifespan=lifespan
)


@app.post("/api/v1/detect", summary="上传图片进行商标检测")
async def detect_logo(image: UploadFile = File(...)):
    """
    接收上传的图片，存入临时文件夹，调用检测器，然后返回结果并清理临时文件。
    """
    if detector is None:
        raise HTTPException(status_code=503, detail="服务未就绪，AI 模型未能正确加载。")

    # 1. 验证文件类型
    if not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="请上传有效的图片文件 (如 jpg, png)")

    tmp_path = ""
    try:
        # 2. 安全地将上传的图片保存到临时文件
        suffix = os.path.splitext(image.filename)[1] or '.jpg'
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            shutil.copyfileobj(image.file, tmp_file)
            tmp_path = tmp_file.name

        # 3. 核心步骤：调用我们的模型进行推理！
        print(f"📦 收到新请求，正在处理临时文件: {tmp_path}")
        result = detector.predict(tmp_path, top_k=3)

        # 4. 返回 JSON 结果
        if result.get("status") == "success":
            # 把原始文件名塞回结果里，替换掉难看的临时文件名
            result["image"] = image.filename
            return JSONResponse(content=result)
        else:
            raise HTTPException(status_code=500, detail=result.get("message", "推理过程中发生未知错误"))

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"服务器内部处理错误: {str(e)}")

    finally:
        # 5. 善后工作：无论成功失败，一定要删除硬盘上的临时图片，防止塞满服务器
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)