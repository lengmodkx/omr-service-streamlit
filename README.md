# OMR Python 服务

答题卡智能识别系统的 Python 微服务实现，替代原 Streamlit Demo 与 Go 服务方向。

## 架构

```
                        Nacos
                    （注册中心 + 配置中心）
                         ▲
exam-admin (Java)        │ heartbeat
     │ Triple/gRPC       │
     ▼                   │
omr-service(Python) :20884
     │
     │ HTTP GET
     ▼
   OSS / 图片 URL

     ◄── Redis Stream ──►
  批量任务下发 / 结果回传
```

## 核心能力

| RPC 方法 | 说明 | 调用方 |
|----------|------|--------|
| `ParseGoldenTemplate` | 解析黄金模板：根据列框配置 + 模板图片，生成气泡网格与标准答案 | admin 发布模板时 |
| `RecognizeByTemplate` | 根据黄金模板识别单张答题卡 | admin 收学生答卷时 |
| `VerifyRecognitionRate` | 验证黄金模板识别成功率 | admin 模板校验时 |
| `ReverifyPaper` | 单张试卷复验 | admin 人工复核时 |

## 快速开始

启动后：
- gRPC server 监听 `:20884`
- HTTP 健康检查 `:8080/health`
- Nacos 服务列表应出现 `omr-service`（协议 `tri`）
- Redis Stream `omr:batch:job` 接收批量任务，`omr:batch:result` 输出结果

### 1. 安装依赖

```bash
python -m venv .venv
source .venv/Scripts/activate  # Windows
# source .venv/bin/activate    # Linux/macOS
pip install -r requirements.txt
```

### 2. 配置

配置来源优先级：**Nacos 配置中心 > 本地环境变量 > 默认值**

#### 方式一：Nacos 配置中心（推荐）

在 Nacos 控制台创建配置：
- `dataId`: `omr-service.yaml`
- `group`: `DEFAULT_GROUP`
- `namespace`: `8c4541fd-870e-414d-bdee-72cab49fe8d2`
- 示例内容：

```yaml
nacos_server: 39.153.154.183:8848
nacos_namespace: 8c4541fd-870e-414d-bdee-72cab49fe8d2
nacos_username: nacos
nacos_password: lemon2judy
redis:
  host: 47.99.83.217
  port: 6379
  password: lemon2judy
  db: 4
omr_worker_count: 4
```

> 也可直接导入 `nacoss-config-example.yaml`。

#### 方式二：本地环境变量

```bash
cp .env.example .env
# 编辑 .env，填入实际 Nacos / Redis 地址
```

### 3. 启动服务

```bash
python -m omr_service.main
```

## Java 端调用示例

```yaml
dubbo:
  application:
    name: ruoyi-exam-admin
    service-discovery:
      migration: FORCE_INTERFACE   # 强制接口级发现
  registry:
    address: nacos://39.153.154.183:8848
  consumer:
    protocol: tri
    timeout: 10000
```

```java
@DubboReference(version = "1.0.0", group = "DEFAULT_GROUP", protocol = "tri")
private OmrService omrService;
```

## 目录结构

```
.
├── omr_service/              # Python 微服务
│   ├── main.py               # 服务入口
│   ├── config.py             # 配置加载（Nacos + env）
│   ├── nacos_config.py       # Nacos 配置中心客户端（gRPC）
│   ├── nacos_reg.py          # Nacos 服务注册（gRPC）
│   ├── nacos_v2_compat.py    # nacos-sdk-python v2 兼容性补丁
│   ├── server.py             # gRPC server 装配
│   ├── health.py             # HTTP 健康检查
│   ├── rpc/                  # protobuf + gRPC 实现
│   ├── mq/                   # Redis Stream 生产/消费
│   ├── engine/               # OMR 识别引擎
│   ├── loader/               # 图片加载 + 模板缓存
│   └── worker/               # 线程池
├── omr_demo/                 # 原 Demo 脚本（已移除 Streamlit UI）
├── testPaper/                # 样例答题卡图片
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```

## Docker 部署

```bash
docker compose build
docker compose up -d
```

## 测试

```bash
source .venv/Scripts/activate
python -m unittest discover -s tests -p "test_*.py" -v
```

## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `OMR_DUBBO_PORT` | `20884` | gRPC 端口 |
| `OMR_HEALTH_PORT` | `8080` | HTTP 健康检查端口 |
| `NACOS_SERVER` | `127.0.0.1:8848` | Nacos 地址 |
| `NACOS_NAMESPACE` | `public` | Nacos 命名空间 |
| `NACOS_USERNAME` | - | Nacos 用户名 |
| `NACOS_PASSWORD` | - | Nacos 密码 |
| `NACOS_CONFIG_DATA_ID` | `omr-service.yaml` | Nacos 配置 dataId |
| `REDIS_HOST` | `127.0.0.1` | Redis 主机 |
| `REDIS_PORT` | `6379` | Redis 端口 |
| `REDIS_PASSWORD` | - | Redis 密码 |
| `REDIS_DB` | `4` | Redis 数据库 |
| `REDIS_JOB_STREAM` | `omr:batch:job` | 批量任务 Stream |
| `REDIS_RESULT_STREAM` | `omr:batch:result` | 结果输出 Stream |
| `OMR_WORKER_COUNT` | CPU 核数 | 并发 worker |

## Redis Stream 批量任务

**任务消息格式（admin → omr-service）**：

```json
{
  "job_id": "uuid",
  "template_id": 1001,
  "image_urls": ["https://oss/xxx/01A.jpg"],
  "result_stream": "omr:batch:result"
}
```

**结果消息格式（omr-service → admin）**：

```json
{
  "job_id": "uuid",
  "template_id": 1001,
  "completed": 1,
  "failed": 0,
  "results": [{"scan_image_url": "...", "answers": [...], "code": 0}]
}
```

Java 端可用 `RedisTemplate.opsForStream()` 或 `StreamListener` 读写。
