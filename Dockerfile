# ============================================================
# 答题卡智能处理系统 (OMR Service) — Docker 镜像
# 轻量版（不含 YOLO，镜像 ~500MB）
# ============================================================
FROM python:3.11-slim-bookworm

LABEL maintainer="lengmodkx"
LABEL description="OMR答题卡识别系统 - Streamlit（无YOLO轻量版）"

# ---- 系统依赖 ----
# libzbar0:  条形码扫描 (pyzbar)
# libjpeg-dev: JPEG 图片处理
# zlib1g-dev: 压缩支持
# libgl1:     OpenCV 兜底 GL 库（headless 不需要但以防万一）
# libgomp1:   OpenCV 并行运行时
RUN apt-get update && apt-get install -y --no-install-recommends \
    libzbar0 \
    libjpeg-dev \
    zlib1g-dev \
    libgl1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ---- Python 依赖 ----
WORKDIR /app

# 使用根目录轻量 requirements（不含 torch/ultralytics）
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# ---- 应用代码 ----
COPY . /app/

# ---- Streamlit 配置 ----
RUN mkdir -p /app/.streamlit
COPY .streamlit/config.toml /app/.streamlit/config.toml

# ---- 运行时 ----
EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

ENTRYPOINT ["streamlit", "run", "omr_demo/app.py"]
CMD ["--server.port=8501", "--server.address=0.0.0.0", \
     "--server.enableCORS=false", "--server.enableXsrfProtection=false"]
