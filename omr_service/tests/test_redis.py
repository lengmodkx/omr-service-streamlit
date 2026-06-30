"""Redis MQ 相关单元测试"""
import json
import unittest
from unittest.mock import MagicMock, patch

from omr_service.config import OmrConfig
from omr_service.mq.producer import MqProducer


class TestRedisProducer(unittest.TestCase):
    @patch("omr_service.mq.client.redis.Redis")
    def test_send_result(self, mock_redis_cls):
        cfg = OmrConfig.from_env()
        mock_client = MagicMock()
        mock_redis_cls.return_value = mock_client
        mock_client.xadd.return_value = "1234567890-0"

        client = MagicMock()
        client.redis = mock_client
        producer = MqProducer(cfg, client=client)
        msg_id = producer.send_result(payload={"job_id": "j1"})

        self.assertEqual(msg_id, "1234567890-0")
        mock_client.xadd.assert_called_once()
        call_args = mock_client.xadd.call_args
        # xadd(name, fields, ...)
        self.assertEqual(call_args.args[0], cfg.redis_result_stream)
        payload = json.loads(call_args.args[1]["payload"])
        self.assertEqual(payload["job_id"], "j1")


if __name__ == "__main__":
    unittest.main()
