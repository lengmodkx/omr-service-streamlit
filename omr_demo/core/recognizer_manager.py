"""
RecognizerManager — 多个识别器的管理者,提供交叉验证能力

设计目标:
- 持有 N 个 Recognizer 实例,顺序即优先级
- cross_validate() 同一张卡跑所有识别器,逐题对比
- 输出 CrossValidatedResult(继承 RecognizeResult,下游 0 改动)

v1.0 范围:
- 顺序执行(不并行)— 2 个识别器 ~200ms,用户感知不强
- N 个识别器通用(N >= 1,N=1 时降级为单识别器)
- 不持久化,结果仅跟随 session_state
"""
from typing import Any
import time

from core.recognizer import (
    RecognizeContext,
    RecognizeResult,
    CrossValidatedResult,
)


# ========== 4 种 agreement 类型常量(便于测试和 UI 引用) ==========
AGREE_ALL_EMPTY = "all_empty"        # 全空答
AGREE_ALL_MATCH = "all_match"        # 答案一致
AGREE_ONE_UNCERTAIN = "one_uncertain"  # 仅 1 个识别器给出非空答案
AGREE_DISPUTED = "disputed"          # 多个非空答案不一致


class RecognizerManager:
    """识别器管理器

    典型用法:
        manager = RecognizerManager([
            make_recognizer("golden", golden_template=gtp),
            make_recognizer("differential", processor=cp),
        ])
        result = manager.cross_validate(image, ctx)
        if result.disputed_questions:
            # 分歧题,进人工核对面板
            ...
    """

    def __init__(self, recognizers):
        """
        Args:
            recognizers: list[Recognizer] 至少 1 个,顺序即优先级

        Raises:
            ValueError: recognizers 为空
        """
        if not recognizers:
            raise ValueError("recognizers 不能为空,至少需要 1 个识别器")
        self._recognizers = list(recognizers)

    @property
    def recognizers(self):
        """返回识别器列表的副本(避免外部修改)"""
        return list(self._recognizers)

    def add(self, recognizer) -> None:
        """动态注册新识别器(阶段 9 留扩展点)"""
        self._recognizers.append(recognizer)

    def cross_validate(self, image: Any, ctx: RecognizeContext) -> CrossValidatedResult:
        """同一张卡跑所有识别器,对比每题结果

        流程:
        1. 顺序调用每个 recognizer.recognize(image, ctx)
        2. 收集每个识别器的 answers
        3. 对每道题调用 _compare_answers() 判定 agreement_type
        4. 构造 CrossValidatedResult:
           - answers: 全匹配题取共识;分歧题 answer=None + status="uncertain"
           - per_q_cv: 每题详细验证结果
           - agreement_rate: 全题 agreed 占比
           - disputed_questions: 分歧题号列表

        Args:
            image: numpy.ndarray
            ctx: 识别上下文

        Returns:
            CrossValidatedResult 实例
        """
        start = time.perf_counter()

        # 1. 跑所有识别器
        recognizer_results: dict = {}
        for rec in self._recognizers:
            try:
                result = rec.recognize(image, ctx)
                recognizer_results[rec.id] = result
            except Exception as e:
                # 单个识别器异常不应阻塞整个交叉验证
                # 用空结果占位,后续 _compare_answers 视为 None
                recognizer_results[rec.id] = RecognizeResult(
                    answers={},
                    total=0,
                    recognizer_id=rec.id,
                    debug_lines=[f"[cross_validate] {rec.id} 异常: {e}"],
                )

        # 2. 汇总所有题号(取识别器结果的并集)
        all_questions = set()
        for r in recognizer_results.values():
            all_questions.update(r.answers.keys())

        # 3. 逐题交叉验证
        per_q_cv: dict = {}
        merged_answers: dict = {}
        disputed: list = []
        agreed_count = 0
        n_recognizers = len(self._recognizers)

        for q in sorted(all_questions):
            per_recognizer = {}
            for rec_id, r in recognizer_results.items():
                ans_info = r.answers.get(q, {"answer": None, "status": "empty"})
                per_recognizer[rec_id] = ans_info.get("answer")

            # N=1 时,无比较意义,直接采用单识别器结果(标记为 all_match 或 all_empty)
            if n_recognizers == 1:
                only_ans = next(iter(per_recognizer.values()))
                if only_ans is None:
                    comparison = {
                        "agreed": True,
                        "consensus": None,
                        "agreement_type": AGREE_ALL_EMPTY,
                    }
                else:
                    comparison = {
                        "agreed": True,
                        "consensus": only_ans,
                        "agreement_type": AGREE_ALL_MATCH,
                    }
            else:
                comparison = self._compare_answers(per_recognizer)

            per_q_cv[q] = {
                "per_recognizer": per_recognizer,
                "agreed": comparison["agreed"],
                "consensus": comparison["consensus"],
                "agreement_type": comparison["agreement_type"],
            }

            if comparison["agreed"]:
                agreed_count += 1
                # 共识答案 — 写入 merged_answers
                if comparison["consensus"] is not None:
                    status = "multi" if len(comparison["consensus"]) > 1 else "single"
                    merged_answers[q] = {
                        "answer": comparison["consensus"],
                        "status": status,
                        "correct": None,
                    }
                else:
                    # 全空 — 保留 consensus=None
                    merged_answers[q] = {
                        "answer": None,
                        "status": "empty",
                        "correct": None,
                    }
            else:
                # 分歧题 — 标记为 uncertain,进人工核对面板
                disputed.append(q)
                merged_answers[q] = {
                    "answer": None,
                    "status": "uncertain",
                    "correct": None,
                }

        # 4. 汇总统计
        total = len(all_questions)
        agreement_rate = (agreed_count / total) if total > 0 else 1.0

        # 5. 卡片级 flag
        card_flag = None
        if total == 0:
            card_flag = "invalid_image"
        elif len(disputed) > 0:
            # 有任何分歧题即标记
            card_flag = "uncertain"
        # 注:全空/全填涂异常的判定留给原有逻辑(empty_count/total 阈值)

        # 6. duration 累加(展示交叉验证总耗时)
        duration_ms = (time.perf_counter() - start) * 1000.0

        # 7. recognizer_id 标记为"cv"(供 UI 识别这是交叉验证结果)
        return CrossValidatedResult(
            answers=merged_answers,
            total=total,
            empty_count=sum(1 for a in merged_answers.values() if a["status"] == "empty"),
            multi_count=sum(1 for a in merged_answers.values() if a["status"] == "multi"),
            card_flag=card_flag,
            debug_lines=[],
            duration_ms=round(duration_ms, 2),
            recognizer_id="cross_validation",
            per_q_cv=per_q_cv,
            agreement_rate=round(agreement_rate, 4),
            disputed_questions=disputed,
            recognizer_results=recognizer_results,
        )

    @staticmethod
    def _compare_answers(per_recognizer: dict) -> dict:
        """对比 N 个识别器对同一题的答案

        Args:
            per_recognizer: {recognizer_id: answer_str_or_None}

        Returns:
            {
                "agreed": bool,
                "consensus": str | None,        # 共识答案(分歧时为 None)
                "agreement_type": str,           # all_empty / all_match / one_uncertain / disputed
            }

        算法:
            - 全 None → all_empty, agreed=True
            - 仅 1 个非空 → one_uncertain, agreed=False, consensus=那个
            - 多个非空且集合相等 → all_match, agreed=True
            - 多个非空且集合不等 → disputed, agreed=False, consensus=None
            - 多选:用 set 比较,顺序无关
        """
        if not per_recognizer:
            return {"agreed": True, "consensus": None, "agreement_type": AGREE_ALL_EMPTY}

        answers = list(per_recognizer.values())
        non_empty = [a for a in answers if a is not None]
        empty_count = len(answers) - len(non_empty)

        # 全空
        if empty_count == len(answers):
            return {"agreed": True, "consensus": None, "agreement_type": AGREE_ALL_EMPTY}

        # 仅 1 个非空
        if len(non_empty) == 1:
            return {
                "agreed": False,
                "consensus": non_empty[0],
                "agreement_type": AGREE_ONE_UNCERTAIN,
            }

        # 2+ 非空:集合比较
        first = non_empty[0]
        first_set = set(first) if len(first) > 1 else {first}
        all_match = all(
            (set(a) if len(a) > 1 else {a}) == first_set
            for a in non_empty
        )
        if all_match:
            return {
                "agreed": True,
                "consensus": first,
                "agreement_type": AGREE_ALL_MATCH,
            }

        # 多个非空且不一致 → disputed
        return {"agreed": False, "consensus": None, "agreement_type": AGREE_DISPUTED}
