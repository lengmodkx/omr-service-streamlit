"""OMR 服务主入口.

启动流程:
1. 加载配置 (Nacos > env > default)
2. 初始化组件 (TemplateStore, ImageLoader, WorkerPool, OmrService, TaskRegistry)
3. 启动 uvicorn (FastAPI HTTP 入口, 通过 lifespan 管理 Nacos 注册/Consumer 生命周期)
"""
from __future__ import annotations

import logging
import threading
from contextlib import asynccontextmanager
from datetime import datetime, timedelta

import uvicorn

from omr_service.api.app import create_app
from omr_service.config import OmrSettings, load_settings
from omr_service.core.service import OmrService
from omr_service.core.task_registry import TaskRegistry
from omr_service.engine.crop_cleaner import clean_expired_crops

logger = logging.getLogger(__name__)


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def _setup_dependencies(settings: OmrSettings) -> tuple[OmrService, TaskRegistry, dict]:
    """初始化核心组件."""
    from omr_service.engine.ocr import PersonalInfoOcr
    from omr_service.engine.cropper import SubjectiveCropper
    from omr_service.loader.image_loader import ImageLoader
    from omr_service.loader.template_store import TemplateStore
    from omr_service.worker.pool import WorkerPool

    template_store = TemplateStore(ttl_seconds=settings.template_ttl_seconds)
    image_loader = ImageLoader(max_bytes=settings.image_max_bytes)
    worker_pool = WorkerPool(max_workers=settings.worker_pool_size)
    ocr_engine = PersonalInfoOcr()  # 懒加载
    cropper = SubjectiveCropper(output_dir=settings.crop_output_dir, base_url=settings.crop_base_url)

    # 讯飞 OCR 增强通道（xfyun_enabled=false 时为 None）
    from omr_service.engine.xfyun_ocr import build_xfyun_clients
    xfyun_doc, xfyun_hw = build_xfyun_clients(settings)

    service = OmrService(
        template_store=template_store,
        image_loader=image_loader,
        worker_pool=worker_pool,
        ocr_engine=ocr_engine,
        cropper=cropper,
        sync_timeout_seconds=settings.sync_timeout_seconds,
        ocr_timeout_seconds=settings.ocr_timeout_seconds,
        ocr_confidence_threshold=settings.ocr_confidence_threshold,
        xfyun_doc=xfyun_doc,
        xfyun_hw=xfyun_hw,
    )

    # 复用 Redis client
    if settings.redis_enabled:
        try:
            from omr_service.mq.client import get_redis_client
            redis_client = get_redis_client(settings)
        except Exception as e:
            logger.warning("Redis client init failed: %s, task_registry will be None", e)
            redis_client = None
    else:
        redis_client = None

    task_registry = TaskRegistry(
        redis_client=redis_client,
        hash_prefix=settings.redis_result_hash_prefix,
    )

    return service, task_registry, {"template_store": template_store}


def _start_consumer(settings: OmrSettings, service: OmrService, template_store=None):
    """启动 Redis Stream consumer 线程, 返回 MqConsumer 实例."""
    if not settings.consumer_enabled:
        logger.info("OMR_CONSUMER_ENABLED=false, skip consumer")
        return None
    if not settings.redis_enabled:
        logger.warning("redis_enabled=false, cannot start consumer")
        return None

    try:
        from omr_service.mq.consumer import start_consumer_thread
        consumer = start_consumer_thread(
            service=service, settings=settings, template_store=template_store
        )
        logger.info("Redis Stream consumer started")
        return consumer
    except Exception as e:
        logger.warning("Failed to start consumer: %s", e)
        return None


def _start_crop_cleaner(settings: OmrSettings):
    """启动每日 03:30 的本地裁图清理守护线程。

    返回 (stop_event, thread)：shutdown 时既可通知线程退出，也可 join 等待
    当前清理轮次收尾（daemon 线程不 join 时进程退出会直接中断删除过程）。
    """
    if settings.crop_retention_days <= 0:
        logger.info("本地裁图清理已停用，保留期配置为 %s 天", settings.crop_retention_days)
        return None
    stop_event = threading.Event()

    def run():
        while not stop_event.is_set():
            now = datetime.now()
            next_run = now.replace(hour=3, minute=30, second=0, microsecond=0)
            if now >= next_run:
                next_run += timedelta(days=1)
            if stop_event.wait(max(1.0, (next_run - now).total_seconds())):
                break
            try:
                clean_expired_crops(settings.crop_output_dir, settings.crop_retention_days)
            except Exception:
                logger.exception("本地裁图定时清理执行失败")

    thread = threading.Thread(target=run, name="omr-crop-cleaner", daemon=True)
    thread.start()
    logger.info("本地裁图清理线程已启动：保留 %s 天，每日 03:30 执行", settings.crop_retention_days)
    return stop_event, thread


def _start_nacos(settings: OmrSettings) -> None:
    """启动 Nacos 注册 + 配置监听."""
    if not settings.nacos_enabled:
        logger.info("OMR_NACOS_ENABLED=false, skip nacos")
        return
    try:
        from omr_service.nacos_reg import NacosRegistrator
        registrator = NacosRegistrator(settings)
        if registrator.register():
            logger.info("Nacos registered: %s", settings.nacos_service_name)
    except Exception as e:
        logger.warning("Nacos registration failed: %s", e)


def _deregister_nacos() -> None:
    try:
        from omr_service.nacos_reg import deregister_all
        deregister_all()
    except Exception as e:
        logger.warning("Nacos deregister failed: %s", e)


def main() -> None:
    settings = load_settings()
    _configure_logging(settings.log_level)
    logger.info("Starting OMR service on %s:%s", settings.http_host, settings.http_port)

    service, task_registry, deps = _setup_dependencies(settings)
    shared_template_store = deps["template_store"]

    # 使用 lifespan 管理 Nacos 注册与 Consumer 生命周期
    consumer_ref: list = [None]  # 闭包可变引用传递
    crop_cleaner_ref: list = [None]

    @asynccontextmanager
    async def lifespan(app_instance):
        # startup
        consumer_ref[0] = _start_consumer(settings, service, template_store=shared_template_store)
        crop_cleaner_ref[0] = _start_crop_cleaner(settings)
        _start_nacos(settings)
        yield
        # shutdown
        logger.info("Shutting down OMR service")
        _deregister_nacos()
        if crop_cleaner_ref[0] is not None:
            # 先通知清理线程退出，再 join 等当前清理轮次收尾（最多 30s），
            # 避免进程退出时正在删除的轮次被硬中断。
            stop_event, cleaner_thread = crop_cleaner_ref[0]
            stop_event.set()
            cleaner_thread.join(timeout=30)
        if consumer_ref[0] is not None:
            try:
                consumer_ref[0].stop()
            except Exception as e:
                logger.warning("Consumer stop failed: %s", e)

    app = create_app(
        settings=settings,
        service=service,
        task_registry=task_registry,
        lifespan=lifespan,
    )

    uvicorn.run(
        app,
        host=settings.http_host,
        port=settings.http_port,
        workers=1,  # PaddleOCR 显存约束
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
