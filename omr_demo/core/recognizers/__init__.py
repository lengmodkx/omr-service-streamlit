"""
识别器适配器包 — 把现有识别类包装为 Recognizer 协议

各适配器职责:
- golden.py: 包装 GoldenTemplate.recognize()(薄包装,补齐 duration_ms / recognizer_id)
- differential.py: 包装 CardProcessor.recognize_choices()(重整输出结构)

包装而非重写:
- 不修改被包装类的内部逻辑
- 0 回归风险
- 阶段 5 之后可逐步迁移内部实现到协议
"""
from core.recognizers.golden import GoldenTemplateRecognizer
from core.recognizers.differential import DifferentialRecognizer

__all__ = ["GoldenTemplateRecognizer", "DifferentialRecognizer"]
