# ============================================================
# OMR Python 服务 — Docker 镜像
# 标准微服务：gRPC + Nacos gRPC 注册 + Redis Stream 消费
# ============================================================
FROM python:3.12-slim-bookworm

LABEL maintainer="lengmodkx"
LABEL description="OMR答题卡识别Python微服务"

# ---- 系统依赖 ----
# libzbar0:  条形码扫描 (pyzbar)
# libjpeg-dev: JPEG 图片处理
# zlib1g-dev: 压缩支持
# libgl1:     OpenCV 兜底 GL 库
# libgomp1:   OpenCV 并行运行时
RUN sed -i 's|http://deb.debian.org/debian|https://mirrors.ustc.edu.cn/debian|g' /etc/apt/sources.list.d/debian.sources \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        libzbar0 \
        libjpeg-dev \
        zlib1g-dev \
        libgl1 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ---- Python 依赖 ----
WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ---- 应用代码 ----
COPY omr_service /app/omr_service
COPY .env.example /app/.env.example

# ---- 运行时 ----
EXPOSE 20884 8080

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')" || exit 1

ENTRYPOINT ["python", "-m", "omr_service.main"]
