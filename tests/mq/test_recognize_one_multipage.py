"""MQ 批量识别入口的多页回归测试。

`_recognize_one` 必须复用 OmrService.recognize_pages（按页识别后合并），
不能只识别 images[0]，且 total/empty/multi/card_flag 需按合并后的答案统计。
"""
from unittest.mock import MagicMock

import numpy as np

from omr_service.loader.template_store import CachedTemplate
from omr_service.mq.job_handler import BatchJobHandler


def _image(value: int) -> np.ndarray:
    return np.full((20, 20, 3), value, dtype=np.uint8)


def _make_handler(cached, images):
    cfg = MagicMock()
    cfg.redis_result_stream = "omr:batch:result"
    cfg.redis_job_stream = "omr:batch:job"
    cfg.omr_max_retry = 3
    cfg.ocr_confidence_threshold = 0.3
    cfg.ocr_timeout_seconds = 30.0
    cfg.crop_output_dir = "./omr_crops"
    cfg.crop_base_url = None

    store = MagicMock()
    store.get.return_value = cached
    loader = MagicMock()
    loader.load_multi.return_value = images
    service = MagicMock()
    return BatchJobHandler(cfg, store, loader, MagicMock(), MagicMock(), service=service), service


def test_recognize_one_reuses_multipage_recognition():
    """多页答题卡：MQ 入口调用 recognize_pages 并合并两页答案（不再只传 images[0]）。"""
    images = [_image(10), _image(20)]
    cached = CachedTemplate(
        standard_template=MagicMock(),
        personal_info=[],
        subjective_regions=[],
    )
    handler, service = _make_handler(cached, images)
    service.recognize_pages.return_value = (
        {
            1: {"answer": "A", "status": "single", "correct": True},
            2: {"answer": None, "status": "empty", "correct": None},
            3: {"answer": "BC", "status": "multi", "correct": False},
        },
        [{"q": 1, "opt": "A", "x": 0, "y": 0, "w": 1, "h": 1, "page_index": 0}],
        True,
        3,
    )

    data = handler._recognize_one(1, "http://x/page0.jpg,http://x/page1.jpg")

    assert data["code"] == 0
    # 传完整图片列表给多页识别（按页采样），而不是 images[0]
    service.recognize_pages.assert_called_once_with(cached, images)
    assert [a["q"] for a in data["answers"]] == [1, 2, 3]
    assert data["answers"][1]["answer"] == ""
    assert data["answers"][2]["answer"] == "BC"
    assert data["total"] == 3
    assert data["empty_count"] == 1
    assert data["multi_count"] == 1
    # abnormal=True 映射为 card_flag="abnormal"
    assert data["card_flag"] == "abnormal"


def test_recognize_one_returns_empty_card_flag_without_abnormal():
    images = [_image(10)]
    cached = CachedTemplate(
        standard_template=MagicMock(),
        personal_info=[],
        subjective_regions=[],
    )
    handler, service = _make_handler(cached, images)
    service.recognize_pages.return_value = (
        {1: {"answer": "A", "status": "single", "correct": True}},
        [],
        False,
        1,
    )

    data = handler._recognize_one(1, "http://x/page0.jpg")

    assert data["code"] == 0
    assert data["card_flag"] == ""
    assert data["total"] == 1
    assert data["empty_count"] == 0
    assert data["multi_count"] == 0


def test_recognize_one_page_count_mismatch_returns_retryable_error():
    """图片页数与模板页数不一致（load_multi 静默丢页）时必须显式报错并允许重试。

    不能进入 recognize_pages：位置对应错乱会把后页扫描图套到前页版面，
    产出"成功但错答"的结果。
    """
    # 模板两页，但只有 1 张图加载成功（第 2 页下载失败被 load_multi 丢弃）
    images = [_image(10)]
    cached = CachedTemplate(
        standard_template=MagicMock(),
        personal_info=[],
        subjective_regions=[],
        page_images={0: _image(0), 1: _image(1)},
    )
    handler, service = _make_handler(cached, images)

    data = handler._recognize_one(1, "http://x/page0.jpg,http://x/page1.jpg")

    # code 99 走消费端重试路径（4/5 才是永久性错误）
    assert data["code"] == 99
    assert "页数" in data["message"]
    service.recognize_pages.assert_not_called()
