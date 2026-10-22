"""模板缓存多页合并单元测试。

覆盖多页黄金模板解析后缓存合并的关键点：
- standard_template.answers（标准答案）/ bubbles / page_images 跨页合并；
- 同一页重新解析时必须「整页替换」旧气泡与答案，避免旧题号残留（否则识别结果会新旧题号混杂）。
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


def test_merge_replaces_same_page_bubbles_on_reparse():
    """同一页重新解析后，旧气泡必须被整页替换（不能追加）。

    历史缺陷：追加会让缓存里残留旧题号气泡，识别结果出现
    "第1页 Q5..Q12（旧号）+ Q1..Q4（新号）" 混杂。
    """
    store = TemplateStore(ttl_seconds=3600)

    # 旧解析：第 1 页题号从 5 开始
    old_tpl = _template([_column(0, 5)])
    old_tpl.answers = {5: "A", 6: "B"}
    store.set(1, CachedTemplate(standard_template=old_tpl, page_images={0: _image(0)}))

    # 新解析：同一页题号改为从 1 开始（框题页重排后重新解析）
    new_tpl = _template([_column(0, 1)])
    new_tpl.answers = {1: "C", 2: "D"}
    store.set(1, CachedTemplate(standard_template=new_tpl, page_images={0: _image(0)}))

    merged = store.get(1)
    assert merged is not None
    qs = {b["q"] for b in merged.standard_template.bubbles}
    # 只保留新题号，旧题号 5/6 必须消失
    assert qs == {1, 2}
    assert merged.standard_template.answers == {1: "C", 2: "D"}


def test_merge_keeps_other_pages_when_one_page_reparsed():
    """只重新解析第 2 页时，第 1 页气泡与答案必须保留。"""
    store = TemplateStore(ttl_seconds=3600)

    tpl0 = _template([_column(0, 1)])
    tpl0.answers = {1: "A", 2: "B"}
    tpl1 = _template([_column(1, 3)])
    tpl1.answers = {3: "C", 4: "D"}
    store.set(1, CachedTemplate(standard_template=tpl0, page_images={0: _image(0)}))
    store.set(1, CachedTemplate(standard_template=tpl1, page_images={1: _image(1)}))

    # 只重新解析第 2 页（题号保持不变），第 1 页应原样保留
    tpl1b = _template([_column(1, 3)])
    tpl1b.answers = {3: "C", 4: "D"}
    store.set(1, CachedTemplate(standard_template=tpl1b, page_images={1: _image(1)}))

    merged = store.get(1)
    assert merged is not None
    qs = {b["q"] for b in merged.standard_template.bubbles}
    assert qs == {1, 2, 3, 4}
    assert merged.standard_template.answers == {1: "A", 2: "B", 3: "C", 4: "D"}

    assert set(merged.page_images) == {0, 1}
