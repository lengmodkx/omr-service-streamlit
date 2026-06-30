"""Redis Stream 结果生产者"""
import json
import logging
from typing import Any, Optional

from omr_service.config import OmrConfig
from omr_service.mq.client import MqClient

logger = logging.getLogger(__name__)


class MqProducer:
    """发送识别结果到 Redis Stream"""

    def __init__(self, cfg: OmrConfig, client: Optional[MqClient] = None):
        self.cfg = cfg
        self._client = client or MqClient(cfg)
        self._owned_client = client is None

    def connect(self):
        if self._owned_client:
            self._client.connect()
        return self

    def send_result(self, stream: Optional[str] = None, payload: Any = None, message_id: str = "*"):
        """发送 JSON 结果到 Redis Stream"""
        stream = stream or self.cfg.redis_result_stream
        body = json.dumps(payload, ensure_ascii=False, default=str)
        try:
            msg_id = self._client.redis.xadd(stream, {"payload": body}, id=message_id)
            logger.debug("Redis 结果已发送: stream=%s id=%s", stream, msg_id)
            return msg_id
        except Exception as e:
            logger.error("Redis 结果发送失败: %s", e)
            raise

    def close(self):
        if self._owned_client:
            self._client.close()
