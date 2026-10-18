"""Redis Stream 批量任务消费者（Phase 3：背压 + 单任务 + 重试）

Phase 4 (FastAPI 改造): 通过 OmrService（dict-based，与协议无关）调用核心识别/解析。
"""
import json
import logging
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import redis

from omr_service.config import OmrConfig
from omr_service.core.service import OmrService
from omr_service.loader.image_loader import ImageLoader
from omr_service.loader.template_store import TemplateStore
from omr_service.mq.client import MqClient
from omr_service.mq.job_handler import BatchJobHandler
from omr_service.mq.producer import MqProducer
from omr_service.worker.pool import WorkerPool

logger = logging.getLogger(__name__)


class MqConsumer:
    """监听 Redis Stream 批量识别任务队列"""

    def __init__(
        self,
        cfg: OmrConfig,
        template_store: TemplateStore,
        image_loader: ImageLoader,
        worker_pool: WorkerPool,
        service: Optional[OmrService] = None,
    ):
        self.cfg = cfg
        self.template_store = template_store
        self.image_loader = image_loader
        self.worker_pool = worker_pool
        self.service = service
        self._client: Optional[MqClient] = None
        self._producer: Optional[MqProducer] = None
        self._handler: Optional[BatchJobHandler] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._inflight_lock = threading.Lock()
        self._inflight = 0

    def start(self):
        """在独立线程中启动消费者"""
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.info("Redis Consumer 已启动，监听 stream: %s", self.cfg.redis_job_stream)

    def stop(self):
        self._stop_event.set()
        # 等待 inflight 任务完成（最多 30 秒）
        deadline = time.time() + 30
        while time.time() < deadline:
            with self._inflight_lock:
                if self._inflight <= 0:
                    break
            time.sleep(0.2)
        with self._inflight_lock:
            if self._inflight > 0:
                logger.warning("停止时仍有 %d 个 inflight 任务未完成", self._inflight)
        # 关闭 worker_pool
        if hasattr(self, "worker_pool") and self.worker_pool is not None:
            try:
                self.worker_pool.close()
            except Exception as e:
                logger.warning("关闭 worker_pool 失败: %s", e)
        if self._client:
            try:
                self._client.close()
            except Exception as e:
                logger.warning("关闭 Redis consumer 失败: %s", e)
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2)

    def _ensure_consumer_group(self, r):
        """确保消费者组存在"""
        try:
            r.xgroup_create(self.cfg.redis_job_stream, self.cfg.redis_consumer_group, id="0", mkstream=True)
            logger.info("Redis 消费者组创建成功: %s", self.cfg.redis_consumer_group)
        except redis.ResponseError as e:
            if "already exists" not in str(e):
                raise

    def _run(self):
        try:
            self._client = MqClient(self.cfg).connect()
            r = self._client.redis
            self._ensure_consumer_group(r)

            self._producer = MqProducer(self.cfg, client=self._client).connect()
            self._handler = BatchJobHandler(
                self.cfg,
                self.template_store,
                self.image_loader,
                self.worker_pool,
                self._producer,
                service=self.service,
            )

            while not self._stop_event.is_set():
                try:
                    # 1. 先尝试处理本消费者遗留的 pending 消息（失败重试）
                    self._process_pending(r)
                    # 2. 读取新消息，受 omr_max_inflight 背压限制
                    self._wait_inflight_slot()
                    batch_size = max(1, self.cfg.omr_batch_size)
                    msgs = r.xreadgroup(
                        groupname=self.cfg.redis_consumer_group,
                        consumername=self.cfg.redis_consumer_name,
                        streams={self.cfg.redis_job_stream: ">"},
                        count=batch_size,
                        block=2000,
                    )
                    if not msgs:
                        continue
                    for stream_name, entries in msgs:
                        self._process_batch(r, entries)
                except redis.ConnectionError as e:
                    logger.error("Redis 连接断开: %s", e)
                    self._stop_event.wait(5)
                except Exception as e:
                    logger.exception("Redis Consumer 异常: %s", e)
                    self._stop_event.wait(1)
        except Exception as e:
            if not self._stop_event.is_set():
                logger.error("Redis Consumer 启动失败: %s", e)

    def _wait_inflight_slot(self):
        """背压：等待 inflight 数量低于阈值"""
        while not self._stop_event.is_set():
            with self._inflight_lock:
                if self._inflight < self.cfg.omr_max_inflight:
                    return
            time.sleep(0.1)

    def _process_pending(self, r):
        """处理 pending 消息（未确认的旧消息）"""
        try:
            pending = r.xpending_range(
                self.cfg.redis_job_stream,
                self.cfg.redis_consumer_group,
                min="-",
                max="+",
                count=1,
                consumername=self.cfg.redis_consumer_name,
            )
            if not pending:
                return
            # 只处理空闲超过 5 秒的消息，避免与当前正在处理的消息冲突
            msg_id = pending[0]["message_id"]
            idle_ms = pending[0].get("time_since_delivered", 0)
            if idle_ms < 5000:
                return
            msgs = r.xreadgroup(
                groupname=self.cfg.redis_consumer_group,
                consumername=self.cfg.redis_consumer_name,
                streams={self.cfg.redis_job_stream: msg_id},
                count=1,
                block=1000,
            )
            if msgs:
                for stream_name, entries in msgs:
                    self._process_batch(r, entries)
        except Exception as e:
            logger.warning("处理 pending 消息失败: %s", e)

    def _process_batch(
        self,
        r,
        entries: List[Tuple[str, Dict[str, str]]],
    ):
        """并发处理一批消息，成功则 ack，失败保留 pending"""
        jobs: List[Tuple[str, Dict[str, Any]]] = []
        for msg_id, fields in entries:
            try:
                payload_str = fields.get("payload", "{}")
                job: Dict[str, Any] = json.loads(payload_str)
                jobs.append((msg_id, job))
            except json.JSONDecodeError as e:
                logger.error("[mq] 任务消息 JSON 解析失败: %s", e)
                self._ack(r, msg_id)

        if not jobs:
            return

        # 控制并发：同时最多 omr_max_inflight 个；通过 done_callback 递减 inflight
        futures = []
        for msg_id, job in jobs:
            self._wait_inflight_slot()
            with self._inflight_lock:
                self._inflight += 1
            try:
                future = self.worker_pool.submit(self._handle_one, msg_id, job)
            except Exception as exc:
                # submit 失败（如线程池已关闭）必须释放 inflight，否则计数泄漏
                logger.error("[mq] 提交任务到线程池失败: %s", exc)
                self._decrement_inflight()
                raise
            future.add_done_callback(lambda _f: self._decrement_inflight())
            futures.append((msg_id, future))

        for msg_id, future in futures:
            try:
                success = future.result(timeout=self.cfg.omr_single_task_timeout_sec)
                if success:
                    self._ack(r, msg_id)
                # 失败不 ack，保留 pending；job_handler 已负责重试/死信
            except Exception as e:
                logger.error("[mq] 任务执行异常或超时: %s", e)
                # 超时也不 ack，等待 pending 重试

    def _decrement_inflight(self):
        with self._inflight_lock:
            self._inflight = max(0, self._inflight - 1)

    def _handle_one(self, msg_id: str, job: Dict[str, Any]) -> bool:
        """在 worker 线程中处理单条任务"""
        try:
            job_type = job.get("job_type", "")
            if job_type == "parse_golden_template":
                logger.info("[mq] 收到黄金模板解析任务: %s", job.get("job_id"))
                result = self._handler.handle_parse_golden_template(job)
            else:
                logger.info("[mq] 收到任务: %s", job.get("task_id"))
                result = self._handler.handle_single_task(job)
            return result.get("success", False) or result.get("retrying", False)
        except Exception as e:
            logger.exception("[mq] 任务处理失败: %s", e)
            return False

    def _ack(self, r, msg_id: str):
        try:
            r.xack(self.cfg.redis_job_stream, self.cfg.redis_consumer_group, msg_id)
            r.xdel(self.cfg.redis_job_stream, msg_id)
        except Exception as e:
            logger.warning("确认消息失败 %s: %s", msg_id, e)


def start_consumer_thread(
    service: OmrService,
    settings: Any,
    template_store: Any = None,
) -> MqConsumer:
    """启动 Redis Stream 消费者线程（FastAPI 启动入口调用）.

    Args:
        service: 已初始化的 OmrService 实例。
        settings: OmrSettings，包含 redis/consumer 相关配置。
        template_store: 共享的 TemplateStore（避免与 HTTP 路径各自维护独立缓存）。
                        None 时回退到新创建。

    Returns:
        MqConsumer 实例，调用方通过 consumer.stop() 优雅关闭。
    """
    if template_store is None:
        template_store = TemplateStore(ttl_seconds=getattr(settings, "template_ttl_seconds", 300))
    image_loader = ImageLoader(max_bytes=getattr(settings, "image_max_bytes", 10 * 1024 * 1024))
    worker_pool = WorkerPool(max_workers=getattr(settings, "worker_pool_size", 4))

    cfg = OmrConfig.from_env()
    cfg.redis_host = settings.redis_host
    cfg.redis_port = settings.redis_port
    cfg.redis_db = settings.redis_db
    cfg.redis_password = settings.redis_password
    cfg.redis_job_stream = settings.redis_stream_job
    cfg.redis_consumer_group = settings.redis_consumer_group
    cfg.redis_consumer_name = settings.redis_consumer_name
    cfg.omr_max_inflight = getattr(settings, "consumer_max_inflight", 8)
    cfg.omr_batch_size = getattr(settings, "consumer_batch_size", 4)
    cfg.omr_single_task_timeout_sec = getattr(settings, "consumer_task_timeout_sec", 120)
    cfg.ocr_timeout_seconds = getattr(settings, "ocr_timeout_seconds", 30.0)
    # 讯飞 OCR 增强通道配置透传
    for attr in ("xfyun_enabled", "xfyun_app_id", "xfyun_doc_api_key", "xfyun_doc_api_secret",
                 "xfyun_hw_api_key", "xfyun_doc_url", "xfyun_hw_url",
                 "xfyun_timeout_seconds", "xfyun_verify_ssl"):
        setattr(cfg, attr, getattr(settings, attr, getattr(cfg, attr)))

    consumer = MqConsumer(
        cfg=cfg,
        template_store=template_store,
        image_loader=image_loader,
        worker_pool=worker_pool,
        service=service,
    )
    consumer.start()
    # 后台线程持久引用，避免 GC
    consumer._worker_pool = worker_pool
    consumer._template_store = template_store
    consumer._image_loader = image_loader
    return consumer
