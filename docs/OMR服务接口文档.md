# OMR Python 服务接口文档

> 服务名：`omr-service`
> 协议：FastAPI HTTP（基于 uvicorn）
> 入口：`omr_service/main.py` 启动的 FastAPI `app`
> 监听端口：`8080`（可通过 `OMR_HTTP_HOST` / `OMR_HTTP_PORT` 修改）

## 1. 服务发现

### Nacos 注册信息

- **应用级服务名**：`omr-service`
- **分组**：`DEFAULT_GROUP`（默认，可通过 `NACOS_GROUP_NAME` 修改）
- **协议**：HTTP（FastAPI / uvicorn）
- **默认端口**：`8080`
- **健康检查路径**：`/v1/health`

### 消费端配置（HTTP）

Nacos 注册仅作服务发现用途，消费端通过 HTTP 调用 OMR 接口，
不再使用 Dubbo Triple / gRPC 客户端。

```yaml
spring:
  cloud:
    nacos:
      discovery:
        server-addr: 39.153.154.183:8848
```

```java
// 使用 RestTemplate / WebClient / OpenFeign 访问 omr-service
String url = "http://omr-service:8080/v1/recognize";
```

## 2. HTTP 接口概览

| Method | Path | 用途 |
|--------|------|------|
| POST | `/v1/recognize` | 同步识别单张答题卡 |
| POST | `/v1/templates/parse` | 同步解析黄金模板 |
| POST | `/v1/verify_recognition_rate` | 暂返 `501 Not Implemented` |
| POST | `/v1/reverify_paper` | 与 `recognize` 行为等价的复验接口 |
| POST | `/v1/tasks` | 异步任务投递（`202 Accepted`） |
| GET  | `/v1/tasks/{task_id}` | 异步任务状态查询 |
| GET  | `/v1/health` | 存活探针 |
| GET  | `/v1/health/ready` | 就绪探针 |
| GET  | `/v1/omr_crops/{file_path:path}` | 静态裁剪图访问 |
| GET  | `/v1/docs` | Swagger UI |
| GET  | `/v1/openapi.json` | OpenAPI Schema |

## 3. 请求 / 响应 Schema

### 3.1 `POST /v1/templates/parse`（解析黄金模板）

请求 `GoldenTemplateRequest`：

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `template_id` | int64 | 是 | 模板唯一标识，后续识别复用 |
| `template_image_url` | string | 是 | 模板图片 URL（PNG/JPG） |
| `columns` | ColumnConfig[] | 否 | 列框配置列表；多页模板中只有个人信息/主观题的页可以为空 |

`ColumnConfig`：

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `x1` | int32 | 是 | 列框左上角 x |
| `y1` | int32 | 是 | 列框左上角 y |
| `x2` | int32 | 是 | 列框右下角 x |
| `y2` | int32 | 是 | 列框右下角 y |
| `start_q` | int32 | 是 | 该列起始题号 |
| `num_q` | int32 | 是 | 该列题目数量 |
| `num_options` | int32 | 是 | 每题选项数量，例如 4 |
| `option_axis` | string | 否 | 选项排列方向：`"x"` 竖排题；`"y"` 横排题 |
| `reverse_q` | bool | 否 | 题号是否倒序 |

响应 `GoldenTemplateResult`：

| 字段 | 类型 | 说明 |
|------|------|------|
| `code` | int32 | `0` 成功；非 0 见错误码表 |
| `message` | string | 结果说明 |
| `template_id` | int64 | 模板 ID |
| `bubbles` | Bubble[] | 检测到的所有气泡 |
| `answers` | map<int32, string> | 标准答案，key 为题号 |
| `total` | int32 | 气泡总数 |

`Bubble`：

| 字段 | 类型 | 说明 |
|------|------|------|
| `q` | int32 | 所属题号 |
| `opt` | string | 选项，如 `"A"` |
| `x` | int32 | 气泡中心 x |
| `y` | int32 | 气泡中心 y |
| `w` | int32 | 宽度 |
| `h` | int32 | 高度 |

### 3.2 `POST /v1/recognize` / `/v1/reverify_paper`（识别）

请求 `RecognizeRequest`：

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `template_id` | int64 | 是 | 已解析的黄金模板 ID |
| `scan_image_url` | string | 是 | 待识别答题卡图片 URL |
| `question_no` | int32 | 否 | `0` 表示整张识别；非 0 为单题复验（预留） |
| `personal_info_region` | list\|dict | 否 | 个人信息区域配置。`field=student_info_block` 为考生信息区整块识别；`field=barcode_label` 为「条码标签区」（框题管理中单独框选的印刷条码标签条），对小区域做 3 种轻预处理变体（放大、不去噪）高精度识别，产出 `printed_name`（印刷体姓名）与 `exam_no`（条码考号），优先级高于手写字段 |
| `candidate_names` | string[] | 否 | 该考试的考生名单（花名册）。提供后，考生姓名识别结果（条码标签条多预处理变体 OCR + 手写行）会与名单做相似度裁决，纠正少数民族低频人名的单字误识（如“敖其泰”误识为“放其泰”）；两字名要求更高相似度（0.75）防止误配。MQ 批量链路中由 Java 端从座位表自动下发（换行分隔字符串）。 |

## 3.2.1 讯飞 OCR 增强通道（可选，默认关闭）

当 PaddleOCR 对姓名类字段/条码标签条识别无结果时，可按次计费调用讯飞云端引擎兜底：

| 通道 | 讯飞服务 | 触发时机 |
|------|----------|----------|
| 印刷体兜底 | 通用文档识别（OCR 大模型） | `barcode_label` 标签条 Paddle 多变体无姓名候选时 |
| 手写体兜底 | 手写文字识别 | `name`/`姓名` 字段 Paddle 识别为空（或低于置信度阈值）时 |

配置（Nacos `omr-service.yaml` 的 `xfyun.*` 或环境变量 `OMR_XFYUN_*`）：

| 配置项 | 环境变量 | 说明 |
|--------|----------|------|
| `xfyun_enabled` | `OMR_XFYUN_ENABLED` | 总开关，默认 false |
| `xfyun_app_id` | `OMR_XFYUN_APP_ID` | 讯飞应用 APPID |
| `xfyun_doc_api_key` / `xfyun_doc_api_secret` | `OMR_XFYUN_DOC_API_KEY` / `OMR_XFYUN_DOC_API_SECRET` | OCR 大模型凭证 |
| `xfyun_hw_api_key` | `OMR_XFYUN_HW_API_KEY` | 手写识别凭证 |
| `xfyun_timeout_seconds` | `OMR_XFYUN_TIMEOUT_SECONDS` | 单次调用超时，默认 10s |
| `xfyun_verify_ssl` | `OMR_XFYUN_VERIFY_SSL` | 默认 true；部署环境存在 MITM 代理导致 TLS 握手失败时设 false |

双引擎各自的识别值（`name_paddle` / `name_xfyun_doc` / `name_xfyun_hw`）会平铺进 `personal_info`，
前端识别结果页并列展示，供人工核对；不一致时最终姓名优先取讯飞文档大模型（实测单通道最强），
其次 Paddle 标签条，手写通道讯飞优先。有花名册（`candidate_names`）时所有候选统一做相似度裁决。
调用失败仅记 warning 日志，不影响主链路。

响应 `RecognizeResult`：

| 字段 | 类型 | 说明 |
|------|------|------|
| `code` | int32 | `0` 成功；非 0 见错误码表 |
| `message` | string | 结果说明 |
| `template_id` | int64 | 模板 ID |
| `scan_image_url` | string | 被识别的图片 URL |
| `answers` | QuestionAnswer[] | 每题识别结果 |
| `total` | int32 | 总题数 |
| `empty_count` | int32 | 空选数量 |
| `multi_count` | int32 | 多选数量 |
| `card_flag` | string | 异常标记：`abnormal` / `suspicious_blank` / `invalid_image` / `""` |
| `duration_ms` | int32 | 识别耗时（毫秒） |

`QuestionAnswer`：

| 字段 | 类型 | 说明 |
|------|------|------|
| `q` | int32 | 题号 |
| `answer` | string | 识别答案，单选 `"A"`、多选 `"ABC"`、空 `""` |
| `status` | string | `single` / `multi` / `empty` / `uncertain` |
| `correct` | bool | 是否正确（复验/成功率验证场景使用） |

### 3.3 `POST /v1/verify_recognition_rate`（成功率验证）

> 当前实现：直接返回 `501 Not Implemented`，待后续版本提供。

请求 `VerifyRateRequest`：

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `template_id` | int64 | 是 | 模板 ID |
| `image_urls` | string[] | 是 | 已知答案的样本图片 URL 列表 |
| `expected_answers` | map<int32, string> | 是 | 标准答案，key 为题号，value 为 `"A"` 或 `"ABC"` 等 |

响应 `VerifyRateResult`：

| 字段 | 类型 | 说明 |
|------|------|------|
| `code` | int32 | `0` 成功 |
| `message` | string | 结果说明 |
| `success_rate` | float | 成功率 `0.0 ~ 1.0` |
| `total` | int32 | 样本总数 |
| `matched` | int32 | 匹配数 |
| `details` | RecognizeResult[] | 每张样本的详细识别结果 |

### 3.4 `POST /v1/tasks`（异步任务投递）

请求 `TaskSubmitRequest`：

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `template_id` | int64 | 是 | 黄金模板 ID |
| `image_urls` | string[] | 是 | 待识别图片 URL 列表 |
| `callback_url` | string | 否 | 可选结果回调 URL |

响应 `TaskCreatedResponse`（`202 Accepted`）：

| 字段 | 类型 | 说明 |
|------|------|------|
| `task_id` | string | 任务 ID（UUID） |
| `status` | string | 任务初始状态，例如 `pending` |

### 3.5 `GET /v1/tasks/{task_id}`（异步任务查询）

响应 `TaskStatusResponse`：

| 字段 | 类型 | 说明 |
|------|------|------|
| `task_id` | string | 任务 ID |
| `status` | string | `pending` / `running` / `completed` / `failed` |
| `progress` | int32 | 已完成数量 |
| `total` | int32 | 总数量 |
| `results` | RecognizeResult[] | 已完成的识别结果 |
| `error` | string | 失败时的错误信息 |

### 3.6 `GET /v1/health` / `GET /v1/health/ready`

存活：

```json
{ "status": "ok" }
```

就绪：

```json
{ "status": "ready", "components": { "redis": "ok", "template_store": "ok" } }
```

### 3.7 `GET /v1/omr_crops/{file_path:path}`

返回静态裁剪图文件（image/png 等），由 `OMR_CROP_OUTPUT_DIR` 指定的本地目录提供。

## 4. 错误码

业务层 `code` 字段：

| 错误码 | 含义 | 常见原因 |
|--------|------|----------|
| `0` | 成功 | - |
| `4` | 模板未找到 | `template_id` 未调用 `/v1/templates/parse` 解析 |
| `5` | 图片加载失败 | URL 无效、OSS 权限、图片过大/损坏 |
| `6` | 请求参数非法 | `template_id` 为 0、URL 为空等 |
| `99` | 内部错误 | 识别异常，需查看服务端日志 |

HTTP 层（FastAPI 默认行为）：

| 状态码 | 含义 |
|--------|------|
| `200` | 成功 |
| `400` | 请求体校验失败 |
| `404` | 资源不存在（如 task_id 未找到） |
| `422` | 字段类型/取值错误 |
| `500` | 未捕获的内部异常 |
| `501` | 接口未实现（如 `verify_recognition_rate`） |
| `502` | 上游（图片下载、Redis）失败 |

## 5. 调用示例

### 5.1 curl（HTTP）

```bash
# 1. 解析黄金模板
curl -X POST http://localhost:8080/v1/templates/parse \
  -H "Content-Type: application/json" \
  -d '{
    "template_id": 1001,
    "template_image_url": "https://oss/template.jpg",
    "columns": [{
      "x1": 100, "y1": 200, "x2": 300, "y2": 800,
      "start_q": 1, "num_q": 5, "num_options": 4,
      "option_axis": "x"
    }]
  }'

# 2. 识别答题卡
curl -X POST http://localhost:8080/v1/recognize \
  -H "Content-Type: application/json" \
  -d '{
    "template_id": 1001,
    "scan_image_url": "https://oss/scan.jpg"
  }'

# 3. 健康检查
curl http://localhost:8080/v1/health
```

### 5.2 Python（requests）

```python
import requests

BASE = "http://localhost:8080"

resp = requests.post(f"{BASE}/v1/recognize", json={
    "template_id": 1001,
    "scan_image_url": "https://oss/scan.jpg",
}, timeout=30)
resp.raise_for_status()
print(resp.json())
```

### 5.3 Swagger UI

打开浏览器访问 `http://<host>:8080/v1/docs` 查看交互式 API 文档，
或在 `http://<host>:8080/v1/openapi.json` 拉取 OpenAPI Schema。

## 6. Tag 路由隔离（本地调试）

多人共用同一 Nacos 时，为避免请求打到他人本地实例：

1. 本地启动前设置 Tag：
   ```bash
   export OMR_SERVICE_TAG=zhangsan
   python -m omr_service.main
   ```
2. 消费端在 HTTP Header 中携带 Tag，网关层把 Tag 转发到 OMR 服务的请求 header：
   ```
   x-service-tag: zhangsan
   ```
3. Provider 注册时会在 Nacos 实例 metadata 中写入 `tag=zhangsan`。
   消费端（API Gateway / Load Balancer）按 `metadata.tag` 过滤，
   命中则在这些实例中负载均衡，未命中则 fallback 到空 Tag 的基线实例。

## 7. 健康检查

- **存活**：`GET http://<ip>:8080/v1/health` → `{"status":"ok"}`
- **就绪**：`GET http://<ip>:8080/v1/health/ready`
- Nacos 健康检查路径同样配置为 `/v1/health`。

## 8. Redis Stream 批量任务（可选）

除了同步 HTTP 调用，也支持通过 Redis Stream 批量下发任务。

### 任务消息格式（admin → omr-service）

发往 Stream：`omr:batch:job`

```json
{
  "job_id": "uuid",
  "template_id": 1001,
  "image_urls": ["https://oss/01A.jpg", "https://oss/02A.jpg"],
  "result_stream": "omr:batch:result"
}
```

### 结果消息格式（omr-service → admin）

写回 Stream：`omr:batch:result`

```json
{
  "job_id": "uuid",
  "template_id": 1001,
  "completed": 2,
  "failed": 0,
  "results": [
    {"scan_image_url": "...", "answers": [...], "code": 0}
  ]
}
```