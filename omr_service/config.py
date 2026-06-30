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
    health_port: int = 8080

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

    # 工作线程
    worker_count: int = field(default_factory=lambda: os.cpu_count() or 4)

    # 服务元数据
    service_version: str = "1.0.0"
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

        return cls(
            dubbo_port=_get("dubbo_port", "OMR_DUBBO_PORT", 20884, int),
            health_port=_get("health_port", "OMR_HEALTH_PORT", 8080, int),
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
            redis_db=_get("redis.db", "REDIS_DB", 4, int),
            redis_timeout=_get("redis.timeout", "REDIS_TIMEOUT", 10, int),
            redis_ssl=_get("redis.ssl", "REDIS_SSL", False, lambda x: str(x).lower() in ("true", "1", "yes")),
            redis_job_stream=_get("redis.job_stream", "REDIS_JOB_STREAM", "omr:batch:job"),
            redis_result_stream=_get("redis.result_stream", "REDIS_RESULT_STREAM", "omr:batch:result"),
            redis_consumer_group=_get("redis.consumer_group", "REDIS_CONSUMER_GROUP", "omr-service"),
            redis_consumer_name=_get("redis.consumer_name", "REDIS_CONSUMER_NAME", "consumer-1"),
            image_timeout=_get("image_timeout", "OMR_IMAGE_TIMEOUT", 30, int),
            max_image_bytes=_get("max_image_bytes", "OMR_MAX_IMAGE_BYTES", 50 * 1024 * 1024, int),
            worker_count=_get("worker_count", "OMR_WORKER_COUNT", os.cpu_count() or 4, int),
            service_version=_get("service_version", "OMR_SERVICE_VERSION", "1.0.0"),
        )
