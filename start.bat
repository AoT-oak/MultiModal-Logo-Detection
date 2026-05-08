@echo off
rem 设置控制台为 UTF-8 编码，防止中文系统下乱码
chcp 65001 >nul

echo ==============================================================================
echo Logo Detector App Startup Script (v1.2.0 - Windows)
echo Description: Checks environment and starts FastAPI ^& Streamlit
echo ==============================================================================
echo.

rem --- 1. Pre-flight check: API Key ---
if "%ZHIPUAI_API_KEY%"=="" (
    echo [X] Error: ZHIPUAI_API_KEY environment variable not found.
    echo Please run the following command first:
    echo set ZHIPUAI_API_KEY=your_api_key
    echo.
    pause
    exit /b 1
)

rem --- 2. Pre-flight check: Models ---
if not exist "best.pt" (
    echo [!] Warning: YOLO weights (best.pt) not found in the root directory.
)

rem --- 3. Start FastAPI (New Window) ---
echo [*] Starting FastAPI backend on port 8001...
start "Logo Detector Backend (FastAPI)" cmd /c "uvicorn api_server:app --host 0.0.0.0 --port 8001"

echo [*] Pre-loading AI models, waiting 10 seconds...
timeout /t 10 /nobreak >nul

rem --- 4. Start Streamlit (Current Window) ---
echo [*] Starting Streamlit UI on port 8501...
streamlit run app_ui.py --server.port 8501 --server.address 0.0.0.0