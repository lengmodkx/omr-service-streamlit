"""
识别器适配器包 — 把现有识别类包装为 Recognizer 协议

各适配器职责:
- standard.py: 包装 StandardTemplate.recognize()(薄包装,补齐 duration_ms / recognizer_id)

包装而非重写:
- 不修改被包装类的内部逻辑
- 0 回归风险

历史:
- 2026-06-04: 移除 DifferentialRecognizer(差分法适配器)
  阶段 7 双识别器交叉验证被废弃,Tab3 改用单识别器 + 人工核对兜底
"""
from omr_service.engine.recognizers.standard import StandardTemplateRecognizer

__all__ = ["StandardTemplateRecognizer"]
