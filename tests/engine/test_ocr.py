"""个人信息 OCR 单元测试"""
import unittest

import numpy as np

from omr_service.engine.ocr import PersonalInfoOcr


class FakeOcrEngine:
    """模拟 PaddleOCR 引擎返回结果"""

    def __init__(self, results):
        self.results = results

    def ocr(self, image, cls=True):
        return [self.results]


class TestPersonalInfoOcr(unittest.TestCase):
    def setUp(self):
        self.ocr = PersonalInfoOcr()
        # 重置单例引擎
        self.ocr._ocr_engine = None
        self.ocr._init_in_progress = False
        self.ocr._init_failed = False
        self.image = np.full((100, 200, 3), 255, dtype=np.uint8)

    def test_return_empty_when_engine_unavailable(self):
        """PaddleOCR 初始化失败时应返回空值"""
        # 模拟引擎不可用（初始化失败返回 None）
        regions = [{"field": "name", "x1": 0, "y1": 0, "x2": 50, "y2": 50}]
        results = self.ocr.recognize(self.image, regions)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["field"], "name")
        self.assertEqual(results[0]["value"], "")
        self.assertEqual(results[0]["confidence"], 0.0)

    def test_recognize_single_region(self):
        """正常 OCR 识别单个区域"""
        self.ocr._ocr_engine = FakeOcrEngine([
            (None, ("张三", 0.95)),
        ])
        regions = [{"field": "name", "x1": 0, "y1": 0, "x2": 50, "y2": 50}]
        results = self.ocr.recognize(self.image, regions)
        self.assertEqual(results[0]["value"], "张三")
        self.assertAlmostEqual(results[0]["confidence"], 0.95, places=4)

    def test_recognize_multiple_lines(self):
        """多行 OCR 结果应拼接"""
        self.ocr._ocr_engine = FakeOcrEngine([
            (None, ("张", 0.9)),
            (None, ("三", 0.92)),
        ])
        regions = [{"field": "name", "x1": 0, "y1": 0, "x2": 50, "y2": 50}]
        results = self.ocr.recognize(self.image, regions)
        self.assertEqual(results[0]["value"], "张三")
        self.assertAlmostEqual(results[0]["confidence"], 0.91, places=4)

    def test_invalid_region_returns_empty(self):
        """无效区域（坐标越界）应返回空值"""
        self.ocr._ocr_engine = FakeOcrEngine([])
        regions = [{"field": "name", "x1": 300, "y1": 300, "x2": 400, "y2": 400}]
        results = self.ocr.recognize(self.image, regions)
        self.assertEqual(results[0]["value"], "")

    def test_get_engine_returns_none_when_init_in_progress(self):
        """初始化进行中（可能挂起）时 _get_engine 应秒回 None，不阻塞任务"""
        import sys
        from unittest.mock import MagicMock, patch

        self.ocr._ocr_engine = None
        self.ocr._init_in_progress = True
        fake_paddleocr = MagicMock()
        try:
            with patch.dict(sys.modules, {"paddleocr": fake_paddleocr}):
                self.assertIsNone(self.ocr._get_engine())
            # 关键断言：初始化进行中时不得触发 import/实例化 PaddleOCR
            fake_paddleocr.PaddleOCR.assert_not_called()
        finally:
            self.ocr._init_in_progress = False

    def test_get_engine_skips_retry_after_failure(self):
        """初始化失败后 _init_failed 置位，后续调用不再重复尝试 import"""
        import sys
        from unittest.mock import MagicMock, patch

        self.ocr._ocr_engine = None
        self.ocr._init_failed = True
        fake_paddleocr = MagicMock()
        try:
            with patch.dict(sys.modules, {"paddleocr": fake_paddleocr}):
                self.assertIsNone(self.ocr._get_engine())
            fake_paddleocr.PaddleOCR.assert_not_called()
        finally:
            self.ocr._init_failed = False

    def test_ocr_calls_are_serialized(self):
        """Paddle 推理非线程安全：多线程并发调用 engine.ocr 必须被串行化（否则偶发返回空）"""
        import threading
        import time

        class RaceDetectEngine:
            def __init__(self):
                self.inflight = 0
                self.max_inflight = 0
                self._lock = threading.Lock()

            def ocr(self, image, cls=True):
                with self._lock:
                    self.inflight += 1
                    self.max_inflight = max(self.max_inflight, self.inflight)
                time.sleep(0.02)  # 模拟推理耗时，放大竞态窗口
                with self._lock:
                    self.inflight -= 1
                return [[(None, ("张三", 0.9))]]

        engine = RaceDetectEngine()
        self.ocr._ocr_engine = engine
        region = {"field": "student_info_block", "x1": 0, "y1": 0, "x2": 50, "y2": 50}

        threads = [
            threading.Thread(target=self.ocr.recognize_block, args=(self.image, region))
            for _ in range(8)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(engine.max_inflight, 1, "engine.ocr 存在并发调用，未被串行化")

    def test_recognize_block_retries_with_upscale_when_anchors_missing(self):
        """首遍 OCR 无印刷锚点（班级/考场/座位）时，应放大 2 倍重试并合并文本。"""

        class AnchorAwareEngine:
            def __init__(self):
                self.calls = 0

            def ocr(self, image, cls=True):
                self.calls += 1
                # crop 高 50，放大后高 100：大图返回印刷标签行，小图只返回手写行
                if image.shape[0] > 60:
                    return [[(None, ("东转一中 敖其泰 班级:四班802200231 考场:341班 座位:21", 0.99))]]
                return [[(None, ("姓名放其泰", 0.6))]]

        engine = AnchorAwareEngine()
        self.ocr._ocr_engine = engine
        region = {"field": "student_info_block", "x1": 0, "y1": 0, "x2": 200, "y2": 50}
        result = self.ocr.recognize_block(self.image, region)
        self.assertEqual(engine.calls, 2, "首遍无印刷锚点时应触发一次放大重试")
        self.assertIn("敖其泰", result["raw_text"])
        self.assertIn("姓名放其泰", result["raw_text"])

    def test_recognize_block_no_retry_when_anchor_present(self):
        """首遍 OCR 已含印刷锚点时，不应触发放大重试。"""

        class CountingEngine:
            def __init__(self):
                self.calls = 0

            def ocr(self, image, cls=True):
                self.calls += 1
                return [[(None, ("敖其泰 班级:四班 考场:341班 座位:21", 0.95))]]

        engine = CountingEngine()
        self.ocr._ocr_engine = engine
        region = {"field": "student_info_block", "x1": 0, "y1": 0, "x2": 200, "y2": 50}
        result = self.ocr.recognize_block(self.image, region)
        self.assertEqual(engine.calls, 1)
        self.assertIn("敖其泰", result["raw_text"])

    def test_decode_barcodes_robust_returns_empty_when_pyzbar_missing(self):
        """pyzbar/zbar 不可用时应告警并返回空列表，不影响 OCR 主流程。"""
        import sys
        from unittest.mock import patch

        with patch.dict(sys.modules, {"pyzbar": None, "pyzbar.pyzbar": None}):
            self.assertEqual(self.ocr._decode_barcodes_robust(self.image), [])

    def test_decode_barcodes_robust_tries_preprocess_variants(self):
        """原图解码失败时应继续尝试预处理变体，任一成功即返回。"""
        import sys
        import types
        from unittest.mock import patch

        calls = []

        class FakeBarcode:
            def __init__(self, data):
                self.data = data
                self.rect = None

        def fake_decode(img):
            calls.append(img)
            if len(calls) < 2:
                return []
            return [FakeBarcode(b"802200231")]

        fake_pyzbar = types.ModuleType("pyzbar")
        fake_pyzbar_py = types.ModuleType("pyzbar.pyzbar")
        fake_pyzbar_py.decode = fake_decode
        with patch.dict(sys.modules, {"pyzbar": fake_pyzbar, "pyzbar.pyzbar": fake_pyzbar_py}):
            results = self.ocr._decode_barcodes_robust(self.image)
        self.assertEqual(results, ["802200231"])
        self.assertGreaterEqual(len(calls), 2, "首个变体失败后应继续尝试后续预处理变体")

    def test_recognize_block_barcode_anchored_label_strip_ocr(self):
        """整块 OCR 丢失印刷姓名行时，应以条码位置为锚点裁出标签条放大重识，召回印刷体姓名。"""
        import sys
        import types
        from collections import namedtuple
        from unittest.mock import patch

        Rect = namedtuple("Rect", ["left", "top", "width", "height"])

        class FakeBarcode:
            def __init__(self):
                self.data = b"802200236"
                # 条码位于整图 (100,100) 处，80x20
                self.rect = Rect(100, 100, 80, 20)

        def fake_decode(img):
            return [FakeBarcode()]

        class StripAwareEngine:
            """整块只返回手写误识行；标签条（62x144 放大 3 倍后为 186x432）返回印刷标签行。"""

            def __init__(self):
                self.shapes = []

            def ocr(self, image, cls=True):
                self.shapes.append(image.shape)
                if image.shape == (186, 432):
                    return [[
                        (None, ("鸽一中 乌达木 班级:四班", 0.98)),
                        (None, ("802200236 考场:341班 座位:26", 0.98)),
                    ]]
                return [[(None, ("姓名乌达术", 0.6))]]

        engine = StripAwareEngine()
        self.ocr._ocr_engine = engine
        image = np.full((200, 400, 3), 255, dtype=np.uint8)
        region = {"field": "student_info_block", "x1": 0, "y1": 0, "x2": 400, "y2": 200}

        fake_pyzbar = types.ModuleType("pyzbar")
        fake_pyzbar_py = types.ModuleType("pyzbar.pyzbar")
        fake_pyzbar_py.decode = fake_decode
        with patch.dict(sys.modules, {"pyzbar": fake_pyzbar, "pyzbar.pyzbar": fake_pyzbar_py}):
            result = self.ocr.recognize_block(image, region)

        # 条码解码值与显式标记、印刷标签行都应进入 raw_text
        self.assertIn("条码考号:802200236", result["raw_text"])
        self.assertIn("乌达木", result["raw_text"])
        self.assertIn("印刷体姓名:乌达木", result["raw_text"])
        # 姓名候选应包含标签条变体提取结果
        self.assertIn("乌达木", result["name_candidates"])
        # 整块 + 标签条 3 个预处理变体共 4 次 OCR，不应再触发整块放大重试
        self.assertEqual(len(engine.shapes), 4)
        # 端到端：解析器应得到印刷体姓名“乌达木”而非手写误识“乌达术”
        from omr_service.engine.personal_info_block_parser import parse_personal_info_block
        fields, _ = parse_personal_info_block(result["raw_text"])
        self.assertEqual("乌达木", fields.get("name"))
        self.assertEqual("802200236", fields.get("exam_no"))


if __name__ == "__main__":
    unittest.main()
