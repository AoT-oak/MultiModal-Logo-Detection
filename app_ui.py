#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# app_ui.py

import streamlit as st
import requests
from PIL import Image
import io

# 配置页面基础信息
st.set_page_config(
    page_title="MultiModal-Logo-Detection",
    page_icon="🔍",
    layout="wide"
)

# 后端 API 地址 (注意这里的端口是 8001)
API_URL = "http://127.0.0.1:8001/api/v1/detect"

st.title("🔍 MultiModal-Logo-Detection (AoT-oak Edition)")
st.markdown("""
基于 **YOLOv8**  + **GLM-4V**  + **CLIP**  的多模型协同 Workflow。
请上传一张图片，系统将自动分析并提取其中的商标和场景信息。
""")

st.divider()

# 创建左右两列布局
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1. 📸 上传图片")
    uploaded_file = st.file_uploader("支持 JPG, PNG 格式", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # 在前端展示用户上传的图片
        image = Image.open(uploaded_file)
        st.image(image, caption="待检测图片", use_container_width=True)

with col2:
    st.subheader("2. 🤖 分析结果")

    if uploaded_file is None:
        st.info("👈 请先在左侧上传一张图片。")
    else:
        # 添加一个检测按钮
        if st.button("🚀 开始检测", type="primary", use_container_width=True):
            with st.spinner("正在呼叫后端 AI 推理引擎，请稍候..."):
                try:
                    # 准备文件数据并发起 HTTP POST 请求
                    files = {"image": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    response = requests.post(API_URL, files=files)

                    if response.status_code == 200:
                        result = response.json()

                        st.success("检测完成！")

                        # --- 核心数据展示区 ---
                        predictions = result.get("predictions", [])
                        description = result.get("description", "无描述")

                        # 1. 展示 Top-1 结果 (增加 no-logo 友好显示)
                        if predictions:
                            top1 = predictions[0]
                            if top1["category"].lower() == "no-logo":
                                display_name = "商标库未查询到"
                            else:
                                display_name = top1["category"].upper()

                            st.metric(label="🏆 Top-1 推测商标",
                                      value=display_name,
                                      delta=f"置信度: {top1['confidence']:.2f}",
                                      delta_color="normal")

                        # 2. 展示画面描述 (直接处理换行)
                        st.markdown("### 📝 画面语义理解")
                        st.info(description)

                        # 3. 展开查看更多候选 (Top-K) 和原始 JSON
                        with st.expander("📊 查看详细置信度排名与原始 JSON"):
                            for idx, pred in enumerate(predictions):
                                cat_name = "商标库未查询到" if pred['category'].lower() == "no-logo" else pred[
                                    'category']
                                st.write(f"**Top-{idx + 1}:** {cat_name} (得分: {pred['confidence']:.4f})")
                            st.json(result)

                    else:
                        st.error(f"后端返回错误: {response.status_code} - {response.text}")

                except requests.exceptions.ConnectionError:
                    st.error("❌ 无法连接到后端 API！请确保 FastAPI 服务器 (uvicorn) 已在 8001 端口启动。")
                except Exception as e:
                    st.error(f"发生未知错误: {e}")