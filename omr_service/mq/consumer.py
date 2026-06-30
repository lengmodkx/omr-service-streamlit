"""Redis Stream 批量任务消费者"""
import json
import logging
import threading
import time
from typing import Any, Dict, Optional

import redis

from omr_service.config import OmrConfig
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
    ):
        self.cfg = cfg
        self.template_store = template_store
        self.image_loader = image_loader
        self.worker_pool = worker_pool
        self._client: Optional[MqClient] = None
        self._producer: Optional[MqProducer] = None
        self._handler: Optional[BatchJobHandler] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    def start(self):
        """在独立线程中启动消费者"""
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.info("Redis Consumer 已启动，监听 stream: %s", self.cfg.redis_job_stream)

    def stop(self):
        self._stop_event.set()
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
            )

            while not self._stop_event.is_set():
                try:
                    # 阻塞读取，超时 2 秒
                    msgs = r.xreadgroup(
                        groupname=self.cfg.redis_consumer_group,
                        consumername=self.cfg.redis_consumer_name,
                        streams={self.cfg.redis_job_stream: ">"},
                        count=1,
                        block=2000,
                    )
                    if not msgs:
                        continue
                    for stream_name, entries in msgs:
                        for msg_id, fields in entries:
                            self._process_message(r, msg_id, fields)
                except redis.ConnectionError as e:
                    logger.error("Redis 连接断开: %s", e)
                    self._stop_event.wait(5)
                except Exception as e:
                    logger.exception("Redis Consumer 异常: %s", e)
                    self._stop_event.wait(1)
        except Exception as e:
            if not self._stop_event.is_set():
                logger.error("Redis Consumer 启动失败: %s", e)

    def _process_message(self, r, msg_id: str, fields: Dict[str, str]):
        try:
            payload_str = fields.get("payload", "{}")
            job: Dict[str, Any] = json.loads(payload_str)
            logger.info("[mq] 收到任务: %s", job.get("job_id"))
            self._handler.handle(job)
            r.xack(self.cfg.redis_job_stream, self.cfg.redis_consumer_group, msg_id)
            r.xdel(self.cfg.redis_job_stream, msg_id)
        except json.JSONDecodeError as e:
            logger.error("[mq] 任务消息 JSON 解析失败: %s", e)
            r.xack(self.cfg.redis_job_stream, self.cfg.redis_consumer_group, msg_id)
            r.xdel(self.cfg.redis_job_stream, msg_id)
        except Exception as e:
            logger.exception("[mq] 任务处理失败: %s", e)
            # 不 ack，让消息保留在 pending list，后续可重试
