"""
ScoreCalculator — 集中式答题卡算分模块

设计目标:
- 纯函数,零外部依赖(无 streamlit/pandas/opencv)
- 微服务可平移:可被 FastAPI/gRPC/CLI 直接调用
- 与现有 app.py:810-825 行为严格一致
- status 字段不参与算分,仅用于 UI 显示和 stats 计数

核心原则:
- earned 取决于答案字符串与 gold 字符串的集合/相等比较
- multi 参数 (multi=(len(gold) > 1)) 决定走单选严格匹配还是多选集合匹配
- 1 题 1 分(demo 阶段硬编码,未来可加 per_question_score)
"""
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ScoringConfig:
    """算分配置 — 取代散落的 dict 参数"""
    round_step: float = 1.0        # 取整步长(默认 1 = 整数;0.5 = 半取整)
    round_method: str = "round"    # "round" / "floor" / "ceil"
    multi_partial: bool = False    # 多选少选是否给一半分(demo 阶段永远 False)
    ignore_case: bool = False      # 答案大小写是否不敏感(默认 False,与旧 app.py 行为一致)


# round_method 字符串到 math 函数的映射(避免 if-elif 链)
_ROUND_FUNCS = {
    "round": round,
    "floor": lambda x: int(x // 1) if x >= 0 else -int((-x) // 1) - (1 if x % 1 != 0 else 0),
    "ceil": lambda x: int(x // 1) + (1 if x % 1 != 0 else 0) if x >= 0 else -int((-x) // 1),
}


def round_score(score: Optional[float], cfg: Optional[ScoringConfig] = None) -> Optional[float]:
    """
    按 cfg.round_step 和 cfg.round_method 取整。

    - None 透传不抛异常
    - step=1 时直接 round()
    - step=0.5 时按 method 取整后再乘 step
    """
    if score is None:
        return None
    if cfg is None:
        cfg = ScoringConfig()
    if cfg.round_step == 1:
        return round(score)
    method_fn = _ROUND_FUNCS.get(cfg.round_method, round)
    return method_fn(score / cfg.round_step) * cfg.round_step


def match_answer(
    student: Optional[str],
    gold: Optional[str],
    multi: bool = False,
    cfg: Optional[ScoringConfig] = None,
) -> bool:
    """
    单选严格匹配 / 多选集合匹配。

    - None 或空串 → False
    - multi=False: student == gold (大小写处理由 cfg 决定)
    - multi=True: set(student) == set(gold)
    """
    if cfg is None:
        cfg = ScoringConfig()
    if not student or not gold:
        return False
    if cfg.ignore_case:
        student = student.upper()
        gold = gold.upper()
    if multi:
        return set(student) == set(gold)
    return student == gold


def calc_total_score(
    answers: dict,
    standard: dict,
    cfg: Optional[ScoringConfig] = None,
) -> dict:
    """
    算分主入口。

    Args:
        answers: {q: {"answer": str|None, "status": str}}  识别器输出
        standard: {q: "A" 或 "ABC"}                          标准答案
        cfg: 算分配置(默认 ScoringConfig())

    Returns:
        {
            "total": float,         # 总分(已取整)
            "total_full": float,    # 满分(demo 阶段 = len(standard))
            "per_q": [{"q", "earned", "full", "correct", "status"}],
            "stats": {"single", "empty", "multi", "uncertain", "correct"}
        }

    算分规则:
    - earned = full if match_answer(ans, gold, multi=(len(gold)>1)) else 0
    - full 恒为 1.0(demo 阶段)
    - stats 统计 status 出现次数(用于 UI 显示和准确率分析)
    - standard 中没有的题号 → 忽略
    - answers 中没有的题号 → earned=0 (空答)
    """
    if cfg is None:
        cfg = ScoringConfig()

    per_q = []
    earned_sum = 0.0
    full_sum = 0.0
    counts = {"single": 0, "empty": 0, "multi": 0, "uncertain": 0, "correct": 0}

    for q in sorted(standard.keys()):
        gold = standard[q]
        info = answers.get(q, {"answer": None, "status": "empty"})
        ans = info.get("answer")
        status = info.get("status", "empty")
        full = 1.0

        # multi 由 gold 长度决定(非由 answer 的 status 决定)
        is_multi = len(gold) > 1 if gold else False
        correct = match_answer(ans, gold, is_multi, cfg) if ans else False
        earned = full if correct else 0.0

        per_q.append({
            "q": q,
            "earned": earned,
            "full": full,
            "correct": correct,
            "status": status,
        })
        earned_sum += earned
        full_sum += full

        # status 计入 stats(若状态非预期则忽略)
        if status in counts:
            counts[status] += 1
        if correct:
            counts["correct"] += 1

    return {
        "total": round_score(earned_sum, cfg),
        "total_full": full_sum,
        "per_q": per_q,
        "stats": counts,
    }
