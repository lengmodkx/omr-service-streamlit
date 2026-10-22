"""模板缓存多页合并单元测试。

覆盖多页黄金模板解析后缓存合并的两个关键点：
- standard_template.answers（标准答案）必须跨页合并，不能只保留第一页；
- 气泡按 (题号, 选项) 去重时跨页重复只告警（解析阶段允许，识别阶段报错）。
"""
import logging

import numpy as np

from omr_service.engine.standard_template import StandardTemplate
from omr_service.loader.template_store import CachedTemplate, TemplateStore


def _column(page: int, start_q: int, num_q: int = 2, num_options: int = 4) -> dict:
    return {
        "x1": 0, "y1": 0, "x2": 100, "y2": 200,
        "start_q": start_q, "num_q": num_q, "num_options": num_options,
        "option_axis": "x", "reverse_q": False, "page_index": page,
    }


def _template(columns) -> StandardTemplate:
    """image=None 时 StandardTemplate 只生成气泡网格，不做 OpenCV 采样。"""
    return StandardTemplate(None, columns)


def _image(value: int) -> np.ndarray:
    return np.full((20, 20, 3), value, dtype=np.uint8)


def test_merge_keeps_answers_bubbles_and_page_images_of_all_pages():
    """两页模板合并后 answers / bubbles / page_images 都是两页并集。"""
    store = TemplateStore(ttl_seconds=3600)

    tpl0 = _template([_column(0, 1)])
    tpl0.answers = {1: "A", 2: "B"}
    tpl1 = _template([_column(1, 3)])
    tpl1.answers = {3: "C", 4: "D"}

    store.set(
        1,
        CachedTemplate(
            standard_template=tpl0,
            image_url="http://x/page0.jpg",
            page_images={0: _image(0)},
        ),
    )
    store.set(
        1,
        CachedTemplate(
            standard_template=tpl1,
            image_url="http://x/page1.jpg",
            page_images={1: _image(1)},
        ),
    )

    merged = store.get(1)
    assert merged is not None

    # 标准答案 = 两页并集（此前只保留第一页，第 2 页标准答案丢失）
    assert merged.standard_template.answers == {1: "A", 2: "B", 3: "C", 4: "D"}
    # 气泡 = 两页并集，且保留 page_index（每页 2 题 × 4 选项 = 8 个气泡）
    assert len(merged.standard_template.bubbles) == 16
    assert {b["page_index"] for b in merged.standard_template.bubbles} == {0, 1}
    # 各页黄金参考图都在
    assert set(merged.page_images) == {0, 1}
    # 默认参考图仍是第一页（A 面），避免第 2 页覆盖导致 ECC 对齐失败
    assert merged.standard_template.image is None


def test_merge_warns_on_duplicate_question_with_different_answer(caplog):
    """跨页同题号但标准答案不一致 → WARNING（题号跨页重复的早期信号），不抛错。"""
    store = TemplateStore(ttl_seconds=3600)

    tpl0 = _template([_column(0, 1)])
    tpl0.answers = {1: "A"}
    tpl1 = _template([_column(1, 1)])
    tpl1.answers = {1: "B"}

    store.set(1, CachedTemplate(standard_template=tpl0, page_images={0: _image(0)}))
    with caplog.at_level(logging.WARNING, logger="omr_service.loader.template_store"):
        store.set(1, CachedTemplate(standard_template=tpl1, page_images={1: _image(1)}))

    assert "题号跨页重复" in caplog.text
    merged = store.get(1)
    # 解析阶段不抛错，后页答案覆盖（识别阶段才报错）
    assert merged.standard_template.answers == {1: "B"}


def test_merge_warns_on_same_bubble_on_different_pages(caplog):
    """同 (题号, 选项) 出现在不同页 → WARNING（题号跨页重复的早期信号）。"""
    store = TemplateStore(ttl_seconds=3600)

    tpl0 = _template([_column(0, 1)])
    tpl1 = _template([_column(1, 1)])

    store.set(1, CachedTemplate(standard_template=tpl0, page_images={0: _image(0)}))
    with caplog.at_level(logging.WARNING, logger="omr_service.loader.template_store"):
        store.set(1, CachedTemplate(standard_template=tpl1, page_images={1: _image(1)}))

    assert "气泡跨页重复" in caplog.text
    merged = store.get(1)
    assert len(merged.standard_template.bubbles) == 16
