"""端到端集成测试 — 启动 gRPC server 并调用真实 RPC"""
import unittest
from concurrent import futures

import cv2
import grpc
import numpy as np

from omr_service.config import OmrConfig
from omr_service.loader.image_loader import ImageLoader
from omr_service.loader.template_store import TemplateStore
from omr_service.rpc import omr_pb2, omr_pb2_grpc
from omr_service.server import create_server
from omr_service.worker.pool import WorkerPool


class TestIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = OmrConfig.from_env()
        cls.cfg.dubbo_port = 52526
        cls.template_store = TemplateStore()
        cls.image_loader = ImageLoader()
        cls.worker_pool = WorkerPool(max_workers=2)
        cls.server = create_server(cls.cfg, cls.template_store, cls.image_loader, cls.worker_pool)
        cls.server.start()
        cls.channel = grpc.insecure_channel(f"127.0.0.1:{cls.cfg.dubbo_port}")
        cls.stub = omr_pb2_grpc.OmrServiceStub(cls.channel)

    @classmethod
    def tearDownClass(cls):
        cls.channel.close()
        cls.server.stop(1)
        cls.worker_pool.shutdown()
        cls.image_loader.close()

    def _create_synthetic_template(self, template_id: int):
        """创建合成黄金模板"""
        img = np.full((400, 600, 3), 255, dtype=np.uint8)
        # 画一个暗点作为 Q1:A 填涂标记
        cv2.circle(img, (100, 100), 10, (50, 50, 50), -1)

        columns = [
            omr_pb2.ColumnConfig(
                x1=50, y1=50, x2=550, y2=350,
                start_q=1, num_q=3, num_options=4,
                option_axis="x", reverse_q=False,
            )
        ]
        req = omr_pb2.GoldenTemplateRequest(
            template_id=template_id,
            template_image_url="__synthetic__",
            columns=columns,
        )

        # 绕过 image_loader，直接预加载图片到缓存（image_loader 不支持 __synthetic__ 协议）
        from omr_service.engine.standard_template import StandardTemplate
        tpl = StandardTemplate(image=img, column_configs=[
            {"x1": 50, "y1": 50, "x2": 550, "y2": 350,
             "start_q": 1, "num_q": 3, "num_options": 4,
             "option_axis": "x", "reverse_q": False}
        ])
        self.template_store.set(template_id, tpl)

        # 这里只验证 RPC 返回模板存在
        return req

    def test_recognize_synthetic_image(self):
        """使用合成图测试完整识别链路"""
        template_id = 10001

        # 创建并缓存模板
        self._create_synthetic_template(template_id)

        # 构造一张待识别图（与模板图相同）
        img = np.full((400, 600, 3), 255, dtype=np.uint8)
        cv2.circle(img, (100, 100), 10, (50, 50, 50), -1)

        # 保存为临时文件，让 image_loader 加载
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            path = f.name
            _, buf = cv2.imencode(".jpg", img)
            f.write(buf.tobytes())

        try:
            req = omr_pb2.RecognizeRequest(
                template_id=template_id,
                scan_image_url=path,
            )
            resp = self.stub.RecognizeByTemplate(req)
            self.assertEqual(resp.code, 0)
            self.assertEqual(resp.template_id, template_id)
            self.assertGreater(resp.total, 0)
        finally:
            import os
            os.unlink(path)

    def test_verify_rate(self):
        """测试成功率验证接口"""
        template_id = 10002
        self._create_synthetic_template(template_id)

        img = np.full((400, 600, 3), 255, dtype=np.uint8)
        cv2.circle(img, (100, 100), 10, (50, 50, 50), -1)

        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            path = f.name
            _, buf = cv2.imencode(".jpg", img)
            f.write(buf.tobytes())

        try:
            req = omr_pb2.VerifyRateRequest(
                template_id=template_id,
                image_urls=[path],
                expected_answers={1: "A"},
            )
            resp = self.stub.VerifyRecognitionRate(req)
            self.assertEqual(resp.code, 0)
            self.assertEqual(resp.total, 1)
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()
