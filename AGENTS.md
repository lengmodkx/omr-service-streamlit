# OMR Python 服务

## Project Overview

这是一个基于 Python 的 OMR（答题卡识别）微服务，替代原 Streamlit Demo 与 Go 服务方向。

## Technology Stack

- **Python** 3.11+
- **FastAPI + uvicorn** — HTTP 接口层（替代原 Dubbo Triple / gRPC）
- **OpenCV** (`opencv-python-headless`) — 图像处理
- **redis-py** — Redis Stream 消息队列
- **requests + tenacity** — 图片下载与重试
- **nacos-sdk-python (v2/v3 gRPC)** — 服务自注册 + 配置中心

## PaddleOCR 环境说明

Windows CPU 上经过验证的稳定组合：

- **Python** 3.11
- **paddlepaddle** 2.6.2
- **paddleocr** 2.7.3
- **numpy** 1.26.4
- **protobuf** 3.20.2

`nacos-sdk-python==3.2.0` 默认生成的 gRPC pb 依赖 protobuf 5.x 的
`runtime_version` API，与 PaddlePaddle 要求的 `protobuf<=3.20.2` 冲突。
安装依赖后必须执行一次 `python scripts/patch_nacos_protobuf.py`，
该脚本会移除 pb 文件中的高版本运行时校验，使 Nacos 注册/配置中心与
PaddleOCR 可以共存。

此外 SDK 3.2.0 的 `AuthClient.get_access_token` 会把用户名/密码拼在 URL
query 中登录，部分网络（WAF/安全中间件）会拦截 query 同时携带
`username`+`password` 的请求导致 `Error [500]: get access token failed`。
`omr_service/nacos_v2_compat.py` 已在运行时把登录请求补丁为表单 POST body
（与 Java 客户端一致），无需额外脚本；升级 SDK 后需确认该补丁仍必要。

## Build and Run Commands

```bash
# 安装依赖（Windows CPU 推荐 Python 3.11 虚拟环境）
python -m venv .venv-py311

# 激活虚拟环境（PowerShell 用 .ps1，Git Bash 用 source）
.venv-py311\Scripts\Activate.ps1        # PowerShell
# source .venv-py311/Scripts/activate   # Git Bash

pip install -r requirements.txt

# 解决 nacos-sdk-python 3.2.0 与 PaddlePaddle 的 protobuf 版本冲突
python scripts/patch_nacos_protobuf.py

# 配置环境变量（PowerShell 用 Copy-Item，Git Bash 用 cp）
Copy-Item .env.example .env
# cp .env.example .env

# 启动服务（main.py 内部调用 uvicorn.run）
python -m omr_service.main
# 或直接用 uvicorn：
# uvicorn omr_service.main:app --host 0.0.0.0 --port 8080

# 运行测试（pytest，tests/ 目录；pytest.ini 已配置 testpaths = tests）
python -m pytest -v

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

## 本地调试隔离：Service Tag 路由

为支持多人在**同一 Nacos 注册中心**下本地调试，OMR 服务支持给实例打 Tag：

- 配置项：`OMR_SERVICE_TAG`（或 Nacos 配置 `service_tag`）。
- 空值表示**基线实例**（测试环境 / 未打标）。
- 非空值表示开发者本地实例，例如 `zhangsan`、`feat-xxx`。

Provider 注册时会把 Tag 写入 Nacos 实例 metadata：

```json
{
  "tag": "zhangsan"
}
```

### 消费端路由方式

1. **API Gateway / Spring Cloud 消费端（推荐）**：
   - 网关层从 HTTP Header `x-service-tag: zhangsan` 读取 Tag。
   - 下游 HTTP 调用时透传该 Header。
   - 消费端从 Nacos 拉取实例列表后按 `metadata.tag` 过滤。

2. **通用消费端 / Python / 自研服务**：
   消费端从 Nacos 拉取实例列表后，按 `metadata.tag` 过滤；命中则在这些实例中负载均衡，未命中则 fallback 到基线实例。

   项目已提供示例客户端：
   ```bash
   curl -H "x-service-tag: zhangsan" \
        -X POST http://omr-service:8080/v1/recognize \
        -H "Content-Type: application/json" \
        -d '{"template_id": 1, "scan_image_url": "http://..."}'
   ```

### 链路传递

典型调用链：网关 / 入口服务 → 服务 A → OMR 服务。

- 入口层从 HTTP Header `x-service-tag: zhangsan` 读取 Tag。
- 向下游 HTTP 调用时，把 Tag 透传到 Header。
- OMR Python 服务在中间件中读取 `x-service-tag` 并记录日志，方便验证请求是否路由到本实例。

### 注意事项

- 测试环境建议常驻至少一个空 Tag 基线实例，否则未打标的请求会找不到实例。
- 当前 `omr:batch:job` Redis Stream 是服务内部任务队列，不经过服务发现，不受 Tag 路由影响。如需隔离批量任务，需在消息体中额外携带 tag。

## Code Organization

### `omr_service/main.py`

服务入口：加载 Nacos 配置、初始化核心组件、启动后台线程（Redis Stream consumer、Nacos 注册/监听）、启动 uvicorn（FastAPI HTTP 入口）、处理优雅退出。

### `omr_service/api/app.py`

FastAPI 应用工厂：注册中间件（request_id）、依赖注入、错误处理和所有路由。

### `omr_service/api/routers/`

FastAPI 路由：health / recognize / templates / tasks / crops。

### `omr_service/api/schemas/`

Pydantic 模型，对应 OpenAPI 请求/响应 schema。

### `omr_service/nacos_config.py`

Nacos 配置中心客户端：启动时拉取配置，可选后台监听变更。

### `omr_service/nacos_reg.py`

Nacos 服务注册：注册应用级 `omr-service` HTTP 实例，写入 metadata（含 Tag）。

### `omr_service/core/`

OMR 服务核心逻辑：`OmrService` 同步识别、`TaskRegistry` 异步任务注册表。

### `omr_service/mq/`

Redis Stream 批量任务消费与结果生产。

### `omr_service/engine/`

OMR 识别引擎。

- `ocr.py`：PaddleOCR 个人信息识别（手写字段 + `student_info_block` 整块 + `barcode_label` 条码标签区小区域多变体识别 + 条码锚定标签条 OCR）；
- `personal_info_block_parser.py`：考生信息区文本解析（印刷体标记/锚点 > 手写标签 > 启发式）+ 花名册相似度裁决（`candidate_names` 随任务下发）；
- `xfyun_ocr.py`：讯飞 OCR 增强通道（可选，默认关闭，按次计费）：OCR 大模型（印刷标签条印证）+ 手写文字识别（手写姓名印证，讯飞更准优先采用）；双引擎识别值以 `name_paddle`/`name_xfyun_doc`/`name_xfyun_hw` 平铺供前端印证展示；配置走 Nacos `xfyun.*` 或环境变量 `OMR_XFYUN_*`。

## Development Conventions

- 所有新增代码注释、日志、文档使用中文。
- 引擎逻辑不改动，仅做薄封装。
- HTTP 接口返回 code/message 结构，异常由 FastAPI 错误处理中间件统一格式化。
- MQ 消息体使用 JSON。
- Nacos 配置变更后不需要重启服务（监听线程会自动刷新）。