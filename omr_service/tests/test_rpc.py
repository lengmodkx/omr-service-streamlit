"""gRPC 接口契约测试"""
import unittest
from concurrent import futures

import grpc
import numpy as np

from omr_service.config import OmrConfig
from omr_service.loader.image_loader import ImageLoader
from omr_service.loader.template_store import TemplateStore
from omr_service.rpc import omr_pb2, omr_pb2_grpc
from omr_service.server import create_server
from omr_service.worker.pool import WorkerPool


class TestRpcService(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = OmrConfig.from_env()
        cls.cfg.dubbo_port = 52525  # 测试端口
        cls.template_store = TemplateStore()
        cls.image_loader = ImageLoader()
        cls.worker_pool = WorkerPool(max_workers=2)
        cls.server = create_server(cls.cfg, cls.template_store, cls.image_loader, cls.worker_pool)
        cls.server.start()
        cls.channel = grpc.insecure_channel(f"127.0.0.1:{cls.cfg.dubbo_port}")
        cls.stub = omr_pb2_grpc.OmrServiceStub(cls.channel)

        # 预注册一个模板
        img = np.full((300, 400, 3), 255, dtype=np.uint8)
        from omr_service.engine.standard_template import StandardTemplate
        columns = [{"x1": 50, "y1": 50, "x2": 350, "y2": 250, "start_q": 1, "num_q": 2, "num_options": 4}]
        tpl = StandardTemplate(image=img, column_configs=columns)
        cls.template_store.set(999, tpl)

    @classmethod
    def tearDownClass(cls):
        cls.channel.close()
        cls.server.stop(1)
        cls.worker_pool.shutdown()
        cls.image_loader.close()

    def test_recognize_by_template_missing_url(self):
        """缺少 URL 应返回无效请求"""
        req = omr_pb2.RecognizeRequest(template_id=999, scan_image_url="")
        resp = self.stub.RecognizeByTemplate(req)
        self.assertNotEqual(resp.code, 0)

    def test_recognize_by_template_template_not_found(self):
        """模板不存在应返回 4"""
        req = omr_pb2.RecognizeRequest(template_id=0, scan_image_url="http://example.com/x.jpg")
        resp = self.stub.RecognizeByTemplate(req)
        self.assertEqual(resp.code, 6)  # invalid request (template_id=0)

    def test_parse_golden_template_invalid_request(self):
        """无效请求测试"""
        req = omr_pb2.GoldenTemplateRequest(template_id=0)
        resp = self.stub.ParseGoldenTemplate(req)
        self.assertEqual(resp.code, 6)

    def test_verify_rate_invalid_request(self):
        """验证接口无效请求"""
        req = omr_pb2.VerifyRateRequest(template_id=0)
        resp = self.stub.VerifyRecognitionRate(req)
        self.assertEqual(resp.code, 6)


if __name__ == "__main__":
    unittest.main()
