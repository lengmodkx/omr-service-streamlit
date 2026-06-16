"""
标准模板对比法识别器 — 包装 StandardTemplate.recognize()

工作原理:
- StandardTemplate 内部已完成全局 ECC 对齐 + 相对阈值判断 + 标准答案对比
- 本适配器只做薄包装:计时 + 字段补齐 + 转换为 RecognizeResult
- 内部识别逻辑 0 改动,保证阶段 1 改造 0 回归
"""
import time
from typing import Any

from core.recognizer import Recognizer, RecognizeContext, RecognizeResult


class StandardTemplateRecognizer:
    """标准模板对比法识别器

    Attributes:
        name: 中文名"标准模板对比法"
        id: 唯一 ID "standard"
        requires_blank: False(用标准模板自身作参考,不需要空白)
    """

    name = "标准模板对比法"
    id = "standard"
    requires_blank = False

    def __init__(self, standard_template):
        """
        Args:
            standard_template: StandardTemplate 实例
        """
        self._stp = standard_template

    def can_handle(self, ctx: RecognizeContext) -> bool:
        """标准模板非空且已生成气泡网格即可处理"""
        if self._stp is None:
            return False
        # StandardTemplate 在构造时已 _calibrate_positions + _auto_detect_answers
        # bubbles 数量 > 0 即表示网格生成成功
        return len(getattr(self._stp, "bubbles", [])) > 0

    def recognize(self, image: Any, ctx: RecognizeContext) -> RecognizeResult:
        """核心识别:计时 + 调用 stp.recognize + 转换结构

        Args:
            image: numpy.ndarray (BGR)
            ctx: 识别上下文(本适配器主要用 standard_answers 控制 debug)

        Returns:
            RecognizeResult 实例
        """
        start = time.perf_counter()

        # debug 模式:有标准答案时打开(对齐旧行为:有 standard_answers 即 debug)
        debug_mode = bool(ctx.standard_answers)
        result_dict = self._stp.recognize(image, debug=debug_mode)

        duration_ms = (time.perf_counter() - start) * 1000.0

        return RecognizeResult(
            answers=result_dict.get("answers", {}),
            total=result_dict.get("total", 0),
            empty_count=result_dict.get("empty_count", 0),
            multi_count=result_dict.get("multi_count", 0),
            card_flag=result_dict.get("card_flag"),
            debug_lines=result_dict.get("debug_lines", []),
            duration_ms=round(duration_ms, 2),
            recognizer_id=self.id,
        )

    # ========== 标准模板特有能力(暂不纳入 Recognizer 协议) ==========

    @property
    def standard_template(self):
        """暴露原始 StandardTemplate 引用,供 Tab2 预览图生成使用(访问 .image/.bubbles)

        未来若 visualize() 需要被多识别器复用,再考虑加到 Recognizer 协议。
        """
        return self._stp
