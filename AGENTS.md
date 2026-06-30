# OMR Python 服务

## Project Overview

这是一个基于 Python 的 OMR（答题卡识别）微服务，替代原 Streamlit Demo 与 Go 服务方向。

## Technology Stack

- **Python** 3.11+
- **OpenCV** (`opencv-python-headless`) — 图像处理
- **grpcio** — gRPC server，兼容 Dubbo Triple
- **redis-py** — Redis Stream 消息队列
- **requests + tenacity** — 图片下载与重试
- **nacos-sdk-python (v2/v3 gRPC)** — 服务自注册 + 配置中心

## Build and Run Commands

```bash
# 安装依赖
python -m venv .venv
source .venv/Scripts/activate
pip install -r requirements.txt

# 配置环境变量
cp .env.example .env

# 启动服务
python -m omr_service.main

# 运行测试
python -m unittest discover -s tests -p "test_*.py" -v

# Docker 构建
docker compose build
docker compose up -d
```

## 配置来源

优先级：**Nacos 配置中心 > 环境变量 > 默认值**

Nacos 配置：
- dataId: `omr-service.yaml`
- group: `DEFAULT_GROUP`

支持嵌套 YAML，例如：

```yaml
redis:
  host: 47.99.83.217
  port: 6379
  password: xxx
  db: 1
```

会被打平为 `redis.host`, `redis.port` 等键。

## Code Organization

### `omr_service/main.py`

服务入口：加载 Nacos 配置、启动 gRPC server、注册 Nacos、启动 Redis consumer、处理优雅退出。

### `omr_service/nacos_config.py`

Nacos 配置中心客户端：启动时拉取配置，可选后台监听变更。

### `omr_service/nacos_reg.py`

Nacos 服务注册：同时注册应用级 `omr-service` 和接口级 `providers:omr.OmrService::`。

### `omr_service/rpc/omr_service.py`

gRPC 接口实现。

### `omr_service/mq/`

Redis Stream 批量任务消费与结果生产。

### `omr_service/engine/`

OMR 识别引擎。

## Development Conventions

- 所有新增代码注释、日志、文档使用中文。
- 引擎逻辑不改动，仅做薄封装。
- RPC 接口返回 code/message 结构，异常不抛错到 gRPC 层。
- MQ 消息体使用 JSON。
- Nacos 配置变更后不需要重启服务（监听线程会自动刷新）。
