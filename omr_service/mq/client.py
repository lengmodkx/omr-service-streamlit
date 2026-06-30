"""Redis 连接管理"""
import logging
from typing import Optional

import redis

from omr_service.config import OmrConfig

logger = logging.getLogger(__name__)


class MqClient:
    """Redis 连接封装"""

    def __init__(self, cfg: OmrConfig):
        self.cfg = cfg
        self._client: Optional[redis.Redis] = None

    def connect(self) -> "MqClient":
        self._client = redis.Redis(
            host=self.cfg.redis_host,
            port=self.cfg.redis_port,
            password=self.cfg.redis_password,
            db=self.cfg.redis_db,
            decode_responses=True,
            socket_connect_timeout=self.cfg.redis_timeout,
            socket_timeout=self.cfg.redis_timeout,
            ssl=self.cfg.redis_ssl,
        )
        # 测试连接
        self._client.ping()
        logger.info("Redis 连接成功: %s:%s/%s", self.cfg.redis_host, self.cfg.redis_port, self.cfg.redis_db)
        return self

    @property
    def redis(self) -> redis.Redis:
        if self._client is None:
            raise RuntimeError("Redis 未连接，请先调用 connect()")
        return self._client

    def close(self):
        if self._client:
            try:
                self._client.close()
            except Exception as e:
                logger.warning("关闭 Redis 失败: %s", e)
