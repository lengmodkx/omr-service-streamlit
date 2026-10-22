"""多页答题卡按页识别与合并的单元测试。

覆盖 "多页答题卡只识别第一页" 缺陷的核心行为：
- 第 i 张图只采样第 i 页模板（本页气泡 + 本页黄金参考图），不再用整卷气泡采样第一页；
- 各页标准答案按页注入，答案按整卷绝对题号合并；
- 跨页同题号必须报错，不允许静默覆盖。
"""
from unittest.mock import MagicMock

import numpy as np
import pytest

from omr_service.core.exceptions import InvalidRequestError
from omr_service.core.service import OmrService
from omr_service.engine.recognizer import RecognizeResult
from omr_service.engine.standard_template import StandardTemplate
from omr_service.loader.template_store import CachedTemplate


def _make_service() -> OmrService:
    """只测 recognize_pages 的纯识别编排，外部依赖全部 mock。"""
    return OmrService(
        template_store=MagicMock(),
        image_loader=MagicMock(),
        worker_pool=MagicMock(),
        ocr_engine=MagicMock(),
        cropper=MagicMock(),
    )


def _column(page: int, start_q: int, num_q: int = 2, num_options: int = 4) -> dict:
    """构造一列选择题配置（page_index 挂在列上，最终落到气泡上）。"""
    return {
        "x1": 0, "y1": 0, "x2": 100, "y2": 200,
        "start_q": start_q, "num_q": num_q, "num_options": num_options,
        "option_axis": "x", "reverse_q": False, "page_index": page,
    }


def _standard_template(columns) -> StandardTemplate:
    """image=None 时 StandardTemplate 只生成气泡网格，不做 OpenCV 采样。"""
    return StandardTemplate(None, columns)


def _image(value: int) -> np.ndarray:
    return np.full((20, 20, 3), value, dtype=np.uint8)


@pytest.fixture
def fake_recognize(monkeypatch):
    """替换 StandardTemplateRecognizer.recognize。

    记录每次识别收到的 "模板气泡页码集合 + 扫描图对象编号 + 模板参考图对象编号 +
    上下文标准答案"，并返回该页题号的假答案。
    """
    calls = []

    def _fake(self, image, ctx):
        bubbles = list(self._stp.bubbles)
        questions = sorted({b["q"] for b in bubbles})
        calls.append({
            "page_indexes": sorted({int(b.get("page_index", 0)) for b in bubbles}),
            "questions": questions,
            "scan_image_id": id(image),
            "template_image_id": id(getattr(self._stp, "image", None)),
            "ctx_answers": dict(ctx.standard_answers),
        })
        return RecognizeResult(
            answers={
                q: {"answer": "A", "status": "single", "correct": None}
                for q in questions
            },
            total=len(questions),
            empty_count=0,
            multi_count=0,
            card_flag=None,
        )

    from omr_service.engine.recognizers.standard import StandardTemplateRecognizer

    monkeypatch.setattr(StandardTemplateRecognizer, "recognize", _fake)
    return calls


def test_recognize_pages_samples_each_page_with_own_template(fake_recognize):
    """两页答题卡：识别调用 2 次，每次只拿本页气泡与本页黄金参考图，答案按题号合并且不覆盖。"""
    service = _make_service()

    template = _standard_template([_column(0, 1), _column(1, 3)])
    template.answers = {1: "A", 2: "B", 3: "C", 4: "D"}
    golden = {0: _image(0), 1: _image(1)}
    cached = CachedTemplate(standard_template=template, page_images=golden)

    scan0, scan1 = _image(10), _image(20)
    answers_map, bubble_grid, abnormal, total = service.recognize_pages(
        cached, [scan0, scan1]
    )

    # 识别次数 = 页数（不再只识别第一页）
    assert len(fake_recognize) == 2

    # 每次只拿到本页气泡（页码集合、题号集合均为单页）
    assert fake_recognize[0]["page_indexes"] == [0]
    assert fake_recognize[1]["page_indexes"] == [1]
    assert fake_recognize[0]["questions"] == [1, 2]
    assert fake_recognize[1]["questions"] == [3, 4]

    # 用本页扫描图识别，且模板 ECC 参考图是本页黄金图（不能用第 1 页黄金图对齐第 2 页）
    assert fake_recognize[0]["scan_image_id"] == id(scan0)
    assert fake_recognize[1]["scan_image_id"] == id(scan1)
    assert fake_recognize[0]["template_image_id"] == id(golden[0])
    assert fake_recognize[1]["template_image_id"] == id(golden[1])

    # 标准答案按页挂载
    assert fake_recognize[0]["ctx_answers"] == {1: "A", 2: "B"}
    assert fake_recognize[1]["ctx_answers"] == {3: "C", 4: "D"}

    # 合并答案包含两页题号，且没有跨页覆盖（每个题号只来自本页）
    assert set(answers_map) == {1, 2, 3, 4}
    assert answers_map[1] == {"answer": "A", "status": "single", "correct": None}
    assert answers_map[4] == {"answer": "A", "status": "single", "correct": None}
    assert total == 4
    assert abnormal is False

    # bubble_grid 合并两页气泡并保留 page_index（每页 2 题 × 4 选项 = 8 个气泡）
    assert [b["page_index"] for b in bubble_grid] == [0] * 8 + [1] * 8
    for b in bubble_grid:
        assert {"q", "opt", "x", "y", "w", "h", "page_index"} <= set(b)

    # 浅拷贝模板，不能污染共享缓存
    assert template.image is None
    assert len(template.bubbles) == 16
    assert template.answers == {1: "A", 2: "B", 3: "C", 4: "D"}


def test_recognize_pages_raises_on_duplicate_question_across_pages(fake_recognize):
    """两页都从 Q1 开始（跨页同题号）→ 必须抛异常且消息含 "跨页重复"，不允许静默覆盖。"""
    service = _make_service()

    # 第 1 页 Q1-Q2、第 2 页也 Q1-Q2 → 模板题号跨页重复
    template = _standard_template([_column(0, 1), _column(1, 1)])
    cached = CachedTemplate(
        standard_template=template,
        page_images={0: _image(0), 1: _image(1)},
    )

    with pytest.raises(InvalidRequestError) as exc_info:
        service.recognize_pages(cached, [_image(10), _image(20)])

    message = str(exc_info.value)
    assert "跨页重复" in message
    assert "Q1" in message
    assert "Q2" in message


def test_recognize_pages_skips_page_without_bubbles(fake_recognize, caplog):
    """某页无气泡配置（只有个人信息/主观题）时跳过该页并告警，兼容单页模板。"""
    import logging

    service = _make_service()

    # 只有第 1 页（page_index=0）有选择题列，第 2 页（page_index=1）无气泡
    template = _standard_template([_column(0, 1)])
    cached = CachedTemplate(
        standard_template=template,
        page_images={0: _image(0), 1: _image(1)},
    )

    with caplog.at_level(logging.WARNING, logger="omr_service.core.service"):
        answers_map, bubble_grid, abnormal, total = service.recognize_pages(
            cached, [_image(10), _image(20)]
        )

    assert len(fake_recognize) == 1
    assert "第 2 页无气泡配置" in caplog.text
    assert set(answers_map) == {1, 2}
    assert total == 2
    assert [b["page_index"] for b in bubble_grid] == [0] * 8


def test_recognize_returns_multipage_answers_and_bubble_grid(fake_recognize):
    """HTTP /v1/recognize 链路：返回两页答案 + 带 page_index 的 bubble_grid，且答案不新增字段。"""
    service = _make_service()
    # worker_pool 同步执行 _do_recognize，模拟真实识别
    service.worker_pool = MagicMock()
    service.worker_pool.submit.side_effect = (
        lambda fn, *args, **kwargs: MagicMock(
            result=lambda *a, **k: fn(*args)
        )
    )

    template = _standard_template([_column(0, 1), _column(1, 3)])
    cached = CachedTemplate(
        standard_template=template,
        page_images={0: _image(0), 1: _image(1)},
    )
    service.template_store.get.return_value = cached
    service.image_loader.load_multi.return_value = [_image(10), _image(20)]

    result = service.recognize({
        "template_id": 1,
        "scan_image_urls": ["http://x/page0.jpg", "http://x/page1.jpg"],
    })

    assert result["code"] == 0
    # 两页题号都在（此前只识别第一页，第 2 页题号缺失或答串）
    assert {a["question_no"] for a in result["answers"]} == {1, 2, 3, 4}
    # bubble_grid 带 page_index，覆盖两页
    assert [b["page_index"] for b in result["bubble_grid"]] == [0] * 8 + [1] * 8
    assert result["bubbles"] == result["bubble_grid"]
    # 不新增字段（Java 端 OmrResult.QuestionAnswer 按现有字段反序列化）
    assert set(result["answers"][0]) == {
        "question_no", "selected", "status", "is_blank", "is_multiple", "correct",
    }
