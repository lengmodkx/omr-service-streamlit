"""黄金模板多页合并：重复题号必须报错（不能静默覆盖）。"""
from unittest.mock import MagicMock

from omr_service.mq.job_handler import BatchJobHandler


def _make_handler() -> BatchJobHandler:
    cfg = MagicMock()
    cfg.redis_result_stream = "omr:batch:result"
    cfg.redis_job_stream = "omr:batch:job"
    cfg.omr_max_retry = 3
    cfg.ocr_confidence_threshold = 0.3
    cfg.ocr_timeout_seconds = 30.0
    cfg.crop_output_dir = "./omr_crops"
    cfg.crop_base_url = None
    return BatchJobHandler(cfg, MagicMock(), MagicMock(), MagicMock(), MagicMock())


def _job(pages=None):
    return {
        "job_id": "j1",
        "template_id": 1,
        "pages": pages or [
            {"page_index": 0, "template_image_url": "http://x/page0.jpg", "columns": []},
            {"page_index": 1, "template_image_url": "http://x/page1.jpg", "columns": []},
        ],
    }


def _answers(*question_nos):
    return [
        {"question_no": q, "selected": ["A"], "is_blank": False, "is_multiple": False}
        for q in question_nos
    ]


def _fake_service_by_page(mapping):
    """按请求页码返回假解析结果。"""
    service = MagicMock()

    def _parse(request):
        page = int(request["page_index"])
        return {
            "code": 0,
            "message": "ok",
            "template_id": request["template_id"],
            "answers": _answers(*mapping[page]),
            "bubble_grid": [
                {"q": q, "opt": "A", "x": 1, "y": 2, "w": 3, "h": 4, "page_index": page}
                for q in mapping[page]
            ],
        }

    service.parse_golden_template.side_effect = _parse
    return service


def test_duplicate_question_across_pages_returns_code_99(monkeypatch):
    """第 1 页与第 2 页返回同一题号 → code=99，消息含 "重复题号" 和页码。"""
    handler = _make_handler()
    monkeypatch.setattr(
        handler,
        "service",
        _fake_service_by_page({0: [1, 2], 1: [1, 3]}),
    )

    result = handler._do_parse_golden_template(_job())

    assert result["code"] == 99
    assert "重复题号" in result["message"]
    assert "第1页与第2页" in result["message"]
    assert "Q1" in result["message"]


def test_duplicate_question_within_page_returns_code_99(monkeypatch):
    """同一页内返回重复题号 → 同样 code=99（不允许静默覆盖）。"""
    handler = _make_handler()
    monkeypatch.setattr(
        handler,
        "service",
        _fake_service_by_page({0: [1, 1], 1: [3, 4]}),
    )

    result = handler._do_parse_golden_template(_job())

    assert result["code"] == 99
    assert "重复题号" in result["message"]
    assert "第1页" in result["message"]


def test_multipage_answers_merged_without_conflict(monkeypatch):
    """题号整卷唯一时正常合并两页答案，不误报错。"""
    handler = _make_handler()
    monkeypatch.setattr(
        handler,
        "service",
        _fake_service_by_page({0: [1, 2], 1: [3, 4]}),
    )

    result = handler._do_parse_golden_template(_job())

    assert result["code"] == 0
    assert set(result["answers"]) == {"1", "2", "3", "4"}
    assert result["answers"]["3"] == "A"
    assert result["total"] == 4
    assert [b["page_index"] for b in result["bubbles"]] == [0, 0, 1, 1]
