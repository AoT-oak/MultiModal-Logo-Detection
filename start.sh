#!/bin/bash

# ==============================================================================
# Logo Detector App Startup Script (v1.2.0)
# Description: Checks environment and starts FastAPI & Streamlit concurrently
# ==============================================================================

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}[*] Preparing to start Logo Detector System...${NC}"

# --- 1. Pre-flight check: API Key ---
if [ -z "$ZHIPUAI_API_KEY" ]; then
    echo -e "${RED}[X] Error: ZHIPUAI_API_KEY environment variable not found.${NC}"
    echo -e "${YELLOW}Please run: export ZHIPUAI_API_KEY='your_api_key'${NC}"
    exit 1
fi

# --- 2. Pre-flight check: Models ---
if [ ! -f "best.pt" ]; then
    echo -e "${YELLOW}[!] Warning: YOLO weights (best.pt) not found in the root directory.${NC}"
fi

# --- 3. Clean up ports ---
echo -e "${GREEN}[*] Cleaning up occupied ports (8001, 8501)...${NC}"
fuser -k 8001/tcp > /dev/null 2>&1
fuser -k 8501/tcp > /dev/null 2>&1

# --- 4. Start FastAPI (Background) ---
echo -e "${GREEN}[*] Starting FastAPI backend on port 8001 (Background)...${NC}"
nohup uvicorn api_server:app --host 0.0.0.0 --port 8001 > backend.log 2>&1 &

echo -e "${YELLOW}[*] Pre-loading AI models, waiting 10 seconds...${NC}"
sleep 10

# --- 5. Start Streamlit (Foreground) ---
echo -e "${GREEN}[*] Starting Streamlit UI on port 8501 (Foreground)...${NC}"
streamlit run app_ui.py --server.port 8501 --server.address 0.0.0.0

trap "kill $!" EXIT