"""
Recognizer 协议 — 统一识别器接口

设计目标:
- 所有识别方式(差分法/黄金模板/YOLO/AI)实现同一接口
- 零 OpenCV 依赖(本文件不 import cv2, 保持微服务可平移)
- runtime_checkable 协议,支持 isinstance(rec, Recognizer) 检查
- RecognizeResult 是 dataclass,便于下游消费者做字段访问

核心字段说明:
- answers: {q: {"answer": str|None, "status": str, "correct": bool|None}}
  - answer: 识别出的答案字符串(单选 "A" / 多选 "ABC" / "(多涂)" / None)
  - status: 题目级状态 - "single" / "multi" / "empty" / "uncertain"
  - correct: 与标准答案对比结果 - True / False / None(无标准答案或无答案)
- card_flag: 卡片级异常标记 - "abnormal" / "suspicious_blank" / "invalid_image" / None
- duration_ms: 单张卡识别耗时,用于性能基线对比
- recognizer_id: 哪个识别器跑的,为阶段 7 交叉验证铺垫

v1.0 范围:
- 定义协议 + 2 个 dataclass + 工厂函数
- 真实识别器实现在 core/recognizers/ 子包
"""
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable, Optional, Any


# ========== 数据结构 ==========

@dataclass
class RecognizeContext:
    """识别上下文 — 识别器所需的所有外部依赖打包

    Attributes:
        template_config: 模板 JSON dict(可空,差分法需要,黄金模板法不需要)
        blank_refs: 空白参考 {A: np.ndarray, B: np.ndarray}(差分法需要)
        column_boxes: 黄金模板列框[{x1, y1, x2, y2}, ...]
        custom_bubbles: 自定义选项框[{q, opt, x, y, w, h}, ...]
        standard_answers: 标准答案 {q: "A" 或 "ABC"}(可选,用于算 correct)
        page: A 面 / B 面
        threshold: 差分法识别阈值(darkness 比例)
    """
    template_config: Optional[dict] = None
    blank_refs: dict = field(default_factory=dict)
    column_boxes: list = field(default_factory=list)
    custom_bubbles: list = field(default_factory=list)
    standard_answers: dict = field(default_factory=dict)
    page: str = "A"
    threshold: float = 0.15


@dataclass
class RecognizeResult:
    """统一识别结果 — 所有识别器都返回这个结构

    字段:
        answers: {q: {answer, status, correct}}
        total: 识别题目总数
        empty_count: 空答数(answer=None 且 status="empty")
        multi_count: 多涂数(status="multi")
        card_flag: 卡片级异常标记
        debug_lines: 调试信息(每行一条,用于诊断面板)
        duration_ms: 识别耗时
        recognizer_id: 哪个识别器跑的
    """
    answers: dict = field(default_factory=dict)
    total: int = 0
    empty_count: int = 0
    multi_count: int = 0
    card_flag: Optional[str] = None
    debug_lines: list = field(default_factory=list)
    duration_ms: float = 0.0
    recognizer_id: str = ""

    def to_legacy_dict(self) -> dict:
        """转换为与 GoldenTemplate.recognize() 旧版 dict 一致的形态

        用于 app.py 下游代码 0 改动。阶段 5 UI 拆分后再移除。
        """
        return {
            "answers": self.answers,
            "total": self.total,
            "empty_count": self.empty_count,
            "multi_count": self.multi_count,
            "card_flag": self.card_flag,
            "debug_lines": list(self.debug_lines),
        }


@dataclass
class CrossValidatedResult(RecognizeResult):
    """交叉验证结果 — 继承 RecognizeResult,新增验证字段

    用于 RecognizerManager.cross_validate() 输出,保持与 RecognizeResult
    的接口兼容性(scoring/preview 等下游代码 0 改动)。

    字段:
        per_q_cv: {q: {per_recognizer: {id: ans}, agreed: bool,
                       consensus: str|None, agreement_type: str}}
        agreement_rate: float            # 全题"agreed=True"占比 (0.0~1.0)
        disputed_questions: list         # 分歧题号列表(供 Tab3 高亮)
        recognizer_results: dict         # 各识别器独立结果 {id: RecognizeResult}
    """
    per_q_cv: dict = field(default_factory=dict)
    agreement_rate: float = 1.0
    disputed_questions: list = field(default_factory=list)
    recognizer_results: dict = field(default_factory=dict)


# ========== 协议 ==========

@runtime_checkable
class Recognizer(Protocol):
    """识别器协议 — 任意识别方式都实现此接口

    实现要求:
    1. 必须定义类属性: name(str)、id(str)、requires_blank(bool)
    2. can_handle(ctx) 根据上下文判断是否可处理
    3. recognize(image, ctx) 返回 RecognizeResult

    推荐做法:
    - 内部计时使用 time.perf_counter() 填 duration_ms
    - 无效图像返回 RecognizeResult(card_flag="invalid_image", total=0)
    - debug_lines 仅在调试场景填充,避免内存膨胀
    """
    name: str
    id: str
    requires_blank: bool

    def can_handle(self, ctx: RecognizeContext) -> bool:
        """根据上下文判断是否可处理

        典型实现:
        - DifferentialRecognizer: ctx.blank_refs.get("A") is not None
        - GoldenTemplateRecognizer: self._gtp is not None 且 bubbles 非空
        - YoloRecognizer: ctx.template_config.get("yolo_model") is not None
        """
        ...

    def recognize(self, image: Any, ctx: RecognizeContext) -> RecognizeResult:
        """核心识别逻辑,返回 RecognizeResult

        Args:
            image: numpy.ndarray (BGR 或灰度)
            ctx: 识别上下文

        Returns:
            RecognizeResult 实例(永不抛异常,异常情况用 card_flag 表达)
        """
        ...


# ========== 工厂函数 ==========

def make_recognizer(recognizer_id: str, **kwargs) -> Recognizer:
    """工厂函数 — 根据 ID 构造识别器实例

    用法:
        rec = make_recognizer("golden", golden_template=gtp)
        rec = make_recognizer("differential", processor=cp, page="A")
        rec = make_recognizer("yolo", model_path="...")        # 未来
        rec = make_recognizer("ai", api_key="...")              # 未来

    Returns:
        Recognizer 协议实例

    Raises:
        ValueError: 未知的 recognizer_id
    """
    if recognizer_id == "golden":
        from core.recognizers.golden import GoldenTemplateRecognizer
        return GoldenTemplateRecognizer(**kwargs)
    if recognizer_id == "differential":
        from core.recognizers.differential import DifferentialRecognizer
        return DifferentialRecognizer(**kwargs)
    raise ValueError(
        f"未知的 recognizer_id: {recognizer_id!r}。"
        f"已实现: golden, differential。未来: yolo, ai。"
    )


def list_recognizer_ids() -> list:
    """列出已注册的识别器 ID(用于 UI 下拉框)"""
    return ["golden", "differential"]
