"""OMR Python 服务配置"""
import os

from dotenv import load_dotenv

# 加载本地 .env 文件（如果存在）
load_dotenv()
import socket
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


def _get_local_ip() -> str:
    """获取本机可用于 Nacos 注册的 IP。"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


@dataclass
class OmrConfig:
    """服务运行时配置"""

    # gRPC / Dubbo Triple
    dubbo_port: int = 20884

    # HTTP 健康检查
    health_port: int = 9173

    # Nacos 注册中心
    nacos_server: str = "127.0.0.1:8848"
    nacos_namespace: str = "public"
    nacos_username: Optional[str] = None
    nacos_password: Optional[str] = None
    nacos_service_name: str = "omr-service"
    nacos_group_name: str = "DEFAULT_GROUP"
    nacos_heartbeat_interval: int = 5

    # Nacos 配置中心
    nacos_config_data_id: str = "omr-service.yaml"
    nacos_config_group: str = "DEFAULT_GROUP"

    # Redis（替代 RabbitMQ）
    redis_host: str = "127.0.0.1"
    redis_port: int = 6379
    redis_password: Optional[str] = None
    redis_db: int = 1
    redis_timeout: int = 10
    redis_ssl: bool = False
    redis_job_stream: str = "omr:batch:job"
    redis_result_stream: str = "omr:batch:result"
    redis_consumer_group: str = "omr-service"
    redis_consumer_name: str = "consumer-1"

    # 图片加载
    image_timeout: int = 30
    max_image_bytes: int = 50 * 1024 * 1024

    # 主观题裁剪输出
    crop_output_dir: str = "./omr_crops"
    crop_base_url: Optional[str] = None

    # 个人信息 OCR 置信度阈值（低于此值返回空值，避免脏数据）
    ocr_confidence_threshold: float = 0.3

    # 讯飞 OCR 增强通道（默认关闭；按次计费，仅作 Paddle 无结果时的兜底）
    xfyun_enabled: bool = False
    xfyun_app_id: str = ""
    xfyun_doc_api_key: str = ""
    xfyun_doc_api_secret: str = ""
    xfyun_hw_api_key: str = ""
    xfyun_doc_url: str = ""
    xfyun_hw_url: str = ""
    xfyun_timeout_seconds: float = 10.0
    xfyun_verify_ssl: bool = True

    # 工作线程
    worker_count: int = field(default_factory=lambda: os.cpu_count() or 4)

    # 批量任务 / MQ 背压
    omr_max_inflight: int = 16
    omr_batch_size: int = 1
    omr_max_retry: int = 3
    omr_retry_delay_sec: int = 1
    omr_single_task_timeout_sec: int = 60
    # 个人信息 OCR / 主观题裁剪单步超时（秒）：超时跳过该步，不阻塞任务回写
    ocr_timeout_seconds: float = 30.0

    # 服务元数据
    # 留空字符串：与 Java 端 @DubboReference 默认 version="" 对齐。
    # Dubbo 3 接口级服务发现对非空 version 严格匹配（metadata + Nacos 服务名两层都得对），
    # 写 "1.0.0" 会让 consumer 找不到 provider（consumer 默认 version=""，服务名变成 providers:omr.OmrService:1.0.0:）。
    service_version: str = ""
    service_tag_enabled: bool = False
    service_tag: str = field(default_factory=_get_local_ip)
    local_ip: str = field(default_factory=_get_local_ip)

    @property
    def endpoint(self) -> str:
        return f"{self.local_ip}:{self.dubbo_port}"

    @classmethod
    def from_env(cls, nacos_config: Optional[Dict[str, Any]] = None) -> "OmrConfig":
        """从环境变量 + Nacos 配置加载

        优先级：Nacos 配置 > 环境变量 > 默认值
        """
        nacos_config = nacos_config or {}

        def _get(key: str, env_key: str, default, type_fn=None):
            """优先取 Nacos 配置，其次环境变量，最后默认值"""
            value = nacos_config.get(key)
            if value is None:
                value = os.getenv(env_key)
            if value is None:
                value = default
            if type_fn and value is not None:
                try:
                    value = type_fn(value)
                except Exception:
                    value = default
            return value

        # 服务 Tag 开关：默认关闭走基线实例；本地调试隔离可开启（自动使用本机 IP 作为 tag）
        tag_enabled = _get(
            "service_tag_enabled",
            "OMR_SERVICE_TAG_ENABLED",
            False,
            lambda x: str(x).lower() in ("true", "1", "yes"),
        )
        service_tag = "" if not tag_enabled else _get("service_tag", "OMR_SERVICE_TAG", _get_local_ip())

        return cls(
            dubbo_port=_get("dubbo_port", "OMR_DUBBO_PORT", 20884, int),
            health_port=_get("health_port", "OMR_HEALTH_PORT", 9173, int),
            nacos_server=_get("nacos_server", "NACOS_SERVER", "127.0.0.1:8848"),
            nacos_namespace=_get("nacos_namespace", "NACOS_NAMESPACE", "public"),
            nacos_username=_get("nacos_username", "NACOS_USERNAME", None) or None,
            nacos_password=_get("nacos_password", "NACOS_PASSWORD", None) or None,
            nacos_service_name=_get("nacos_service_name", "NACOS_SERVICE_NAME", "omr-service"),
            nacos_group_name=_get("nacos_group_name", "NACOS_GROUP_NAME", "DEFAULT_GROUP"),
            nacos_heartbeat_interval=_get("nacos_heartbeat_interval", "NACOS_HEARTBEAT_INTERVAL", 5, int),
            nacos_config_data_id=_get("nacos_config_data_id", "NACOS_CONFIG_DATA_ID", "omr-service.yaml"),
            nacos_config_group=_get("nacos_config_group", "NACOS_CONFIG_GROUP", "DEFAULT_GROUP"),
            redis_host=_get("redis.host", "REDIS_HOST", "127.0.0.1"),
            redis_port=_get("redis.port", "REDIS_PORT", 6379, int),
            redis_password=_get("redis.password", "REDIS_PASSWORD", None) or None,
            redis_db=_get("redis.db", "REDIS_DB", 1, int),
            redis_timeout=_get("redis.timeout", "REDIS_TIMEOUT", 10, int),
            redis_ssl=_get("redis.ssl", "REDIS_SSL", False, lambda x: str(x).lower() in ("true", "1", "yes")),
            redis_job_stream=_get("redis.job_stream", "REDIS_JOB_STREAM", "omr:batch:job"),
            redis_result_stream=_get("redis.result_stream", "REDIS_RESULT_STREAM", "omr:batch:result"),
            redis_consumer_group=_get("redis.consumer_group", "REDIS_CONSUMER_GROUP", "omr-service"),
            redis_consumer_name=_get("redis.consumer_name", "REDIS_CONSUMER_NAME", "consumer-1"),
            image_timeout=_get("image_timeout", "OMR_IMAGE_TIMEOUT", 30, int),
            max_image_bytes=_get("max_image_bytes", "OMR_MAX_IMAGE_BYTES", 50 * 1024 * 1024, int),
            crop_output_dir=_get("crop_output_dir", "OMR_CROP_OUTPUT_DIR", "./omr_crops"),
            crop_base_url=_get("crop_base_url", "OMR_CROP_BASE_URL", None) or None,
            ocr_confidence_threshold=_get("ocr_confidence_threshold", "OMR_OCR_CONFIDENCE_THRESHOLD", 0.3, float),
            worker_count=_get("worker_count", "OMR_WORKER_COUNT", os.cpu_count() or 4, int),
            omr_max_inflight=_get("omr_max_inflight", "OMR_MAX_INFLIGHT", 16, int),
            omr_batch_size=_get("omr_batch_size", "OMR_BATCH_SIZE", 1, int),
            omr_max_retry=_get("omr_max_retry", "OMR_MAX_RETRY", 3, int),
            omr_retry_delay_sec=_get("omr_retry_delay_sec", "OMR_RETRY_DELAY_SEC", 1, int),
            omr_single_task_timeout_sec=_get("omr_single_task_timeout_sec", "OMR_SINGLE_TASK_TIMEOUT_SEC", 60, int),
            ocr_timeout_seconds=_get("ocr_timeout_seconds", "OMR_OCR_TIMEOUT_SECONDS", 30.0, float),
            xfyun_enabled=_get("xfyun_enabled", "OMR_XFYUN_ENABLED", False,
                               lambda x: str(x).lower() in ("true", "1", "yes")),
            xfyun_app_id=_get("xfyun_app_id", "OMR_XFYUN_APP_ID", ""),
            xfyun_doc_api_key=_get("xfyun_doc_api_key", "OMR_XFYUN_DOC_API_KEY", ""),
            xfyun_doc_api_secret=_get("xfyun_doc_api_secret", "OMR_XFYUN_DOC_API_SECRET", ""),
            xfyun_hw_api_key=_get("xfyun_hw_api_key", "OMR_XFYUN_HW_API_KEY", ""),
            xfyun_doc_url=_get("xfyun_doc_url", "OMR_XFYUN_DOC_URL", ""),
            xfyun_hw_url=_get("xfyun_hw_url", "OMR_XFYUN_HW_URL", ""),
            xfyun_timeout_seconds=_get("xfyun_timeout_seconds", "OMR_XFYUN_TIMEOUT_SECONDS", 10.0, float),
            xfyun_verify_ssl=_get("xfyun_verify_ssl", "OMR_XFYUN_VERIFY_SSL", True,
                                  lambda x: str(x).lower() in ("true", "1", "yes")),
            # service_version 默认改成空字符串：与 Java 端 @DubboReference 默认 version="" 对齐。
            # Dubbo 3 接口级服务发现对非空 version 严格匹配（metadata + Nacos 服务名两层都得对），
            # 默认 "1.0.0" 会让 consumer 找不到 provider（consumer 默认 version=""）。
            service_version=_get("service_version", "OMR_SERVICE_VERSION", ""),
            service_tag_enabled=tag_enabled,
            service_tag=service_tag,
        )


# ========================================================================
# 以下为 FastAPI 改造 P1 引入 (2026-07-28)
# 老的 dataclass OmrConfig 保留以便过渡期使用，新代码统一使用 OmrSettings
# ========================================================================

from typing import Optional  # noqa: E402

import os  # noqa: E402

import warnings  # noqa: E402

from pydantic_settings import BaseSettings, SettingsConfigDict  # noqa: E402


class OmrSettings(BaseSettings):
    """新配置模型: Pydantic Settings.

    加载优先级: Nacos > 环境变量/.env > Pydantic 默认
    兼容期字段 (1 个版本后删除):
    - legacy_dubbo_port: 读取 OMR_LEGACY_DUBBO_PORT, 仅用于 warning 日志
    - 旧 OMR_HEALTH_PORT 读取后写入 warning 日志
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="OMR_",
        case_sensitive=False,
        extra="ignore",
    )

    # 服务
    http_host: str = "0.0.0.0"
    http_port: int = 8080
    # health_port 已合并到 http_port=8080
    log_level: str = "INFO"

    # Nacos
    nacos_enabled: bool = True
    nacos_server: str = "127.0.0.1:8848"
    nacos_namespace: str = "public"
    nacos_username: str = ""
    nacos_password: str = ""
    nacos_group: str = "DEFAULT_GROUP"
    nacos_data_id: str = "omr-service.yaml"
    nacos_service_name: str = "omr-service"
    nacos_ip: str = ""

    # Redis
    redis_enabled: bool = True
    redis_host: str = "127.0.0.1"
    redis_port: int = 6379
    redis_db: int = 1
    redis_password: str = ""
    redis_stream_job: str = "omr:batch:job"
    redis_stream_result: str = "omr:batch:result"
    redis_result_hash_prefix: str = "omr:batch:result:hash"
    redis_consumer_group: str = "omr-service"
    redis_consumer_name: str = "consumer-1"

    # 任务相关
    consumer_enabled: bool = True
    worker_pool_size: int = 4
    sync_timeout_seconds: float = 60.0
    # MQ 单任务处理超时（秒）：PaddleOCR 首次初始化/识别可能较慢，放宽到 120s 避免 future 超时重复投递
    consumer_task_timeout_sec: int = 120
    # 个人信息 OCR / 主观题裁剪单步超时（秒）：超时跳过该步，不阻塞任务回写
    ocr_timeout_seconds: float = 30.0
    # 个人信息 OCR 置信度阈值：低于阈值视为未识别，置空 value（对齐旧 gRPC 分支）
    ocr_confidence_threshold: float = 0.3

    # 讯飞 OCR 增强通道（默认关闭；按次计费，仅作 Paddle 无结果时的兜底）
    xfyun_enabled: bool = False
    xfyun_app_id: str = ""
    xfyun_doc_api_key: str = ""
    xfyun_doc_api_secret: str = ""
    xfyun_hw_api_key: str = ""
    xfyun_doc_url: str = ""
    xfyun_hw_url: str = ""
    xfyun_timeout_seconds: float = 10.0
    xfyun_verify_ssl: bool = True

    # OMR 内部
    template_ttl_seconds: int = 3600
    image_timeout: int = 30
    image_max_bytes: int = 50 * 1024 * 1024
    crop_output_dir: str = "./output"
    crop_base_url: str = ""

    # 兼容期
    legacy_dubbo_port: Optional[int] = None

    @property
    def health_port(self) -> int:
        """兼容期: health 端口已合并到 http_port, 仅用作 warning 日志."""
        return self.http_port


def load_settings() -> OmrSettings:
    """入口: 加载配置 (Nacos > .env > 默认值)."""
    import logging
    _log = logging.getLogger(__name__)

    settings = OmrSettings()
    # 兼容期: 旧 OMR_DUBBO_PORT / OMR_HEALTH_PORT 触发 warning
    if os.getenv("OMR_DUBBO_PORT"):
        warnings.warn(
            "OMR_DUBBO_PORT 已废弃, Dubbo Triple 服务已下线. 请使用 OMR_HTTP_PORT.",
            DeprecationWarning,
            stacklevel=2,
        )
    if os.getenv("OMR_HEALTH_PORT"):
        warnings.warn(
            "OMR_HEALTH_PORT 已合并到 OMR_HTTP_PORT, 仅用于 1 版本兼容期.",
            DeprecationWarning,
            stacklevel=2,
        )

    # 从 Nacos 配置中心拉取配置，覆盖 .env / 默认值
    if settings.nacos_enabled:
        try:
            from omr_service.nacos_config import NacosConfigClient, apply_nacos_config
            client = NacosConfigClient(settings)
            nacos_raw = client.load()
            if nacos_raw:
                apply_nacos_config(settings, nacos_raw)
        except Exception as e:
            _log.warning("Nacos 配置拉取失败，使用本地配置: %s", e)

    return settings
