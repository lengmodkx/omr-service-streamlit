"""Job handler 单任务处理单元测试"""
import unittest
from unittest.mock import MagicMock

from omr_service.mq.job_handler import (
    BatchJobHandler,
    _normalize_personal_info,
    _normalize_subjective_regions,
)


class TestBatchJobHandler(unittest.TestCase):
    def _make_handler(self, max_retry=3):
        cfg = MagicMock()
        cfg.redis_result_stream = "omr:batch:result"
        cfg.redis_job_stream = "omr:batch:job"
        cfg.omr_max_retry = max_retry
        cfg.ocr_confidence_threshold = 0.3
        cfg.ocr_timeout_seconds = 30.0
        cfg.crop_output_dir = "./omr_crops"
        cfg.crop_base_url = None
        store = MagicMock()
        store.get.return_value = None
        loader = MagicMock()
        pool = MagicMock()
        pool.submit = lambda fn, *args: MagicMock(result=lambda: fn(*args))
        producer = MagicMock()
        return BatchJobHandler(cfg, store, loader, pool, producer)

    def test_failed_result_sent_when_no_retry(self):
        handler = self._make_handler(max_retry=0)
        # 模拟模板存在但识别返回失败，由于 max_retry=0 直接失败
        task = {
            "task_id": "t1",
            "batch_id": "b1",
            "template_id": 1,
            "image_url": "http://x.jpg",
            "retry_count": 0,
        }
        result = handler.handle_single_task(task)
        self.assertFalse(result["success"])
        self.assertFalse(result["retrying"])
        handler.producer.send_result.assert_called_once()
        handler.producer.send_job.assert_not_called()

    def test_no_retry_on_permanent_failure(self):
        """模板不存在等永久性错误不应重试，应直接发送失败结果"""
        handler = self._make_handler(max_retry=3)
        task = {
            "task_id": "t2",
            "batch_id": "b2",
            "template_id": 1,
            "image_url": "http://x.jpg",
            "retry_count": 0,
        }
        result = handler.handle_single_task(task)
        self.assertFalse(result["success"])
        self.assertFalse(result["retrying"])
        handler.producer.send_job.assert_not_called()
        handler.producer.send_result.assert_called_once()


    def test_parse_golden_template_success(self):
        """黄金模板解析任务成功时应发送结果到结果流"""
        handler = self._make_handler(max_retry=0)
        service = MagicMock()
        service.parse_golden_template.return_value = {
            "code": 0,
            "message": "ok",
            "template_id": 1,
            "answers": [
                {"question_no": 1, "selected": ["A"], "is_blank": False, "is_multiple": False},
                {"question_no": 2, "selected": ["B"], "is_blank": False, "is_multiple": False},
            ],
            "bubble_grid": [],
            "personal_info_sample": None,
        }
        handler.service = service
        job = {
            "job_type": "parse_golden_template",
            "job_id": "j1",
            "template_id": 1,
            "pages": [
                {
                    "template_image_url": "http://page0.jpg",
                    "columns": [
                        {
                            "column_id": "c1",
                            "column_index": 0,
                            "question_start": 1,
                            "question_count": 2,
                            "options_per_question": 4,
                            "question_type": "single",
                        }
                    ],
                }
            ],
        }
        result = handler.handle_parse_golden_template(job)
        self.assertTrue(result["success"])
        self.assertFalse(result["retrying"])
        handler.producer.send_result.assert_called_once()
        payload = handler.producer.send_result.call_args.kwargs["payload"]
        self.assertEqual(payload["job_id"], "j1")
        self.assertEqual(payload["status"], 2)
        # answers 在合并后是 dict
        self.assertIn("answers", payload)

    def test_parse_golden_template_permanent_failure_no_retry(self):
        """黄金模板解析遇到永久性错误时不应重试，直接失败"""
        handler = self._make_handler(max_retry=3)
        service = MagicMock()
        service.parse_golden_template.return_value = {
            "code": 5,
            "message": "图片加载失败",
        }
        handler.service = service
        job = {
            "job_type": "parse_golden_template",
            "job_id": "j2",
            "template_id": 1,
            "pages": [{"template_image_url": "http://bad.jpg", "columns": []}],
        }
        result = handler.handle_parse_golden_template(job)
        self.assertFalse(result["success"])
        self.assertFalse(result["retrying"])
        handler.producer.send_job.assert_not_called()
        handler.producer.send_result.assert_called_once()

    def test_parse_golden_template_no_service(self):
        """service 未初始化时直接失败"""
        handler = self._make_handler(max_retry=0)
        handler.service = None
        job = {
            "job_type": "parse_golden_template",
            "job_id": "j3",
            "template_id": 1,
            "pages": [{"template_image_url": "http://page0.jpg", "columns": []}],
        }
        result = handler.handle_parse_golden_template(job)
        self.assertFalse(result["success"])
        handler.producer.send_result.assert_called_once()


class TestXfyunClientReuse(unittest.TestCase):
    """BatchJobHandler 应复用 OmrService 已构建的讯飞客户端，避免同进程重复建 Session。"""

    def test_reuses_service_clients(self):
        cfg = MagicMock()
        service = MagicMock()
        doc, hw = MagicMock(name="doc"), MagicMock(name="hw")
        service.xfyun_doc = doc
        service.xfyun_hw = hw
        handler = BatchJobHandler(
            cfg, MagicMock(), MagicMock(), MagicMock(), MagicMock(), service=service,
        )
        self.assertIs(handler._xfyun_doc, doc)
        self.assertIs(handler._xfyun_hw, hw)


class TestNormalizeRegions(unittest.TestCase):
    """Java camelCase → Python snake_case 字段转换单元测试"""

    def test_personal_info_camel_case_mapping(self):
        regions = [
            {
                "field": "student_name",
                "x1": 10, "y1": 20, "x2": 100, "y2": 40,
                "pageIndex": 2,
            },
        ]
        out = _normalize_personal_info(regions)
        self.assertEqual(out[0]["field"], "student_name")
        self.assertEqual(out[0]["page_index"], 2)
        self.assertEqual(out[0]["x1"], 10)

    def test_personal_info_missing_page_index_defaults_zero(self):
        out = _normalize_personal_info([{"field": "a", "x1": 1, "y1": 2, "x2": 3, "y2": 4}])
        self.assertEqual(out[0]["page_index"], 0)

    def test_personal_info_null_coordinates_safe(self):
        """Java 装箱字段序列化为 null 时不应抛 TypeError"""
        out = _normalize_personal_info([{"field": "a", "x1": None, "y1": None, "x2": None, "y2": None}])
        self.assertEqual(out[0]["x1"], 0)
        self.assertEqual(out[0]["y2"], 0)

    def test_subjective_regions_mapping(self):
        regions = [
            {
                "q": 51,
                "x1": 1, "y1": 2, "x2": 3, "y2": 4,
                "pageIndex": 1,
                "stitchWithNext": True,
            },
        ]
        out = _normalize_subjective_regions(regions)
        self.assertEqual(out[0]["q"], 51)
        self.assertEqual(out[0]["page_index"], 1)
        self.assertTrue(out[0]["stitch_with_next"])

    def test_subjective_stitch_string_false(self):
        """字符串 \"false\" 不应误判为 True"""
        out = _normalize_subjective_regions([{"q": 1, "stitchWithNext": "false"}])
        self.assertFalse(out[0]["stitch_with_next"])


if __name__ == "__main__":
    unittest.main()
