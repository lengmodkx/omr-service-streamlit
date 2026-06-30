"""OMR Python 服务入口"""
import logging
import os
import signal
import sys
import threading
import time
from typing import Optional

from omr_service.config import OmrConfig
from omr_service.health import HealthServer
from omr_service.loader.image_loader import ImageLoader
from omr_service.loader.template_store import TemplateStore
from omr_service.mq.consumer import MqConsumer
from omr_service.nacos_config import NacosConfigClient
from omr_service.nacos_reg import NacosRegistrator
from omr_service.server import create_server
from omr_service.worker.pool import WorkerPool

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def _load_nacos_config() -> dict:
    """尝试从 Nacos 配置中心拉取配置"""
    # 先用环境变量构造一个最小配置，用于连接 Nacos
    temp_cfg = OmrConfig.from_env()
    if not temp_cfg.nacos_server or temp_cfg.nacos_server == "127.0.0.1:8848":
        logger.info("未配置 Nacos 地址，跳过 Nacos 配置中心")
        return {}
    try:
        client = NacosConfigClient(temp_cfg)
        config = client.load()
        client.close()
        return config
    except Exception as e:
        logger.warning("Nacos 配置中心拉取失败，使用本地环境变量: %s", e)
        return {}


def main():
    # 1. 加载 Nacos 配置，再合并环境变量
    nacos_config = _load_nacos_config()
    cfg = OmrConfig.from_env(nacos_config)
    logger.info(
        "OMR Python 服务启动 (port=%s, health=%s, nacos=%s, namespace=%s, redis=%s:%s, workers=%s)",
        cfg.dubbo_port,
        cfg.health_port,
        cfg.nacos_server,
        cfg.nacos_namespace,
        cfg.redis_host,
        cfg.redis_port,
        cfg.worker_count,
    )

    # 构造依赖
    template_store = TemplateStore(ttl_seconds=3600)
    image_loader = ImageLoader(
        timeout=cfg.image_timeout,
        max_bytes=cfg.max_image_bytes,
    )
    worker_pool = WorkerPool(max_workers=cfg.worker_count)

    # gRPC server
    server = create_server(cfg, template_store, image_loader, worker_pool)
    server.start()
    logger.info("gRPC server 已启动，监听 :%s", cfg.dubbo_port)

    # HTTP 健康检查
    health_server = HealthServer(port=cfg.health_port)
    health_server.start()

    # Nacos 注册
    nacos_reg: Optional[NacosRegistrator] = None
    try:
        nacos_reg = NacosRegistrator(cfg)
        if not nacos_reg.register():
            logger.warning("Nacos 注册失败，服务仍可被直接访问")
    except Exception as e:
        logger.error("Nacos 注册异常: %s", e)

    # Nacos 配置中心监听（可选）
    nacos_config_client: Optional[NacosConfigClient] = None
    try:
        nacos_config_client = NacosConfigClient(cfg)
        nacos_config_client.start_listener()
    except Exception as e:
        logger.warning("Nacos 配置监听启动失败: %s", e)

    # Redis 批量任务消费者
    mq_consumer: Optional[MqConsumer] = None
    try:
        mq_consumer = MqConsumer(cfg, template_store, image_loader, worker_pool)
        mq_consumer.start()
    except Exception as e:
        logger.error("Redis Consumer 启动异常: %s", e)

    # 优雅退出
    stop_event = threading.Event()

    def _shutdown(signum, frame):
        logger.info("收到信号 %s，开始关闭服务...", signum)
        stop_event.set()

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    try:
        while not stop_event.is_set():
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("键盘中断")

    logger.info("正在停止服务...")
    if mq_consumer:
        mq_consumer.stop()
    if nacos_config_client:
        nacos_config_client.close()
    health_server.stop()
    server.stop(5)
    if nacos_reg:
        nacos_reg.close()
    worker_pool.shutdown()
    image_loader.close()
    logger.info("OMR Python 服务已停止")


if __name__ == "__main__":
    main()
