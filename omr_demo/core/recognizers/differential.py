"""
差分法识别器 — 包装 CardProcessor.recognize_choices()

工作原理:
- CardProcessor.recognize_choices() 返回 {q: str|None} 简单 dict
  - 无答案 → None
  - 多涂 → "A(多涂)"(字符串后缀)
  - 单选 → "A"
- 本适配器把这种"字符串后缀 hack"重整为标准的 {answer, status, correct} 结构
- 多涂字符串去除"(多涂)"后缀,归一化为 answer="A" + status="multi"
"""
import time
from typing import Any

from core.recognizer import Recognizer, RecognizeContext, RecognizeResult


# 多涂后缀(与 CardProcessor.recognize_choices_in_region / recognize_choices_custom 保持一致)
_MULTI_SUFFIX = "(多涂)"


class DifferentialRecognizer:
    """差分法识别器(模板差分 + 空白参考)

    Attributes:
        name: 中文名"差分法"
        id: 唯一 ID "differential"
        requires_blank: True(差分法核心 = 与空白模板对比,需要 blank_refs)
    """

    name = "差分法"
    id = "differential"
    requires_blank = True

    def __init__(self, processor, page: str = "A"):
        """
        Args:
            processor: CardProcessor 实例
            page: "A" 或 "B"(只识别一面)
        """
        self._proc = processor
        self._page = page

    def can_handle(self, ctx: RecognizeContext) -> bool:
        """差分法需要:模板非空 + 空白参考 + 模板有该面的 bubbles"""
        if self._proc is None:
            return False
        template = getattr(self._proc, "template", None)
        if not template:
            return False
        page_config = template.get("pages", {}).get(self._page, {})
        bubbles = page_config.get("bubbles", [])
        if not bubbles:
            return False
        # 差分法不强依赖 blank_refs(无空白参考时降级为相对阈值,见 CardProcessor._bubble_darkness)
        # 所以 can_handle 仅检查模板+bubbles,blank_refs 作为 can_handle 软建议
        return True

    def recognize(self, image: Any, ctx: RecognizeContext) -> RecognizeResult:
        """核心识别:计时 + 调用 processor.recognize_choices + 重整结构

        Args:
            image: numpy.ndarray (BGR)
            ctx: 识别上下文(本适配器用 ctx.threshold)

        Returns:
            RecognizeResult 实例
        """
        start = time.perf_counter()

        raw = self._proc.recognize_choices(
            image, page=self._page, threshold=ctx.threshold
        )

        duration_ms = (time.perf_counter() - start) * 1000.0

        answers = {}
        empty_count = 0
        multi_count = 0

        for q, ans in raw.items():
            if ans is None:
                answers[q] = {"answer": None, "status": "empty", "correct": None}
                empty_count += 1
            elif isinstance(ans, str) and ans.endswith(_MULTI_SUFFIX):
                # 多涂:去除后缀,归一化为 answer + status
                normalized = ans[: -len(_MULTI_SUFFIX)]
                answers[q] = {"answer": normalized, "status": "multi", "correct": None}
                multi_count += 1
            else:
                answers[q] = {"answer": ans, "status": "single", "correct": None}

        return RecognizeResult(
            answers=answers,
            total=len(raw),
            empty_count=empty_count,
            multi_count=multi_count,
            card_flag=None,        # 差分法不做卡片级异常判定(由 Tab3 识别率判定承担)
            debug_lines=[],
            duration_ms=round(duration_ms, 2),
            recognizer_id=self.id,
        )
