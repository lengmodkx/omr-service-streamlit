"""
RecognizerManager 交叉验证测试套件
设计文档: docs/plans/2026-06-01-stage7-cross-validation-design.md §7

运行方式: python omr_demo/test_recognizer_manager.py
期望输出: 13 PASS, 0 FAIL

测试组织:
- _compare_answers 单元测试 (7 例): 4 种 agreement_type 全覆盖 + 边界
- RecognizerManager 集成测试 (6 例): N=1 / N=2 / disputed 题目 / 异常识别器
"""
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass
sys.path.insert(0, ".")

from unittest.mock import MagicMock
import numpy as np

from core.recognizer import (
    RecognizeContext,
    RecognizeResult,
    CrossValidatedResult,
)
from core.recognizer_manager import (
    RecognizerManager,
    AGREE_ALL_EMPTY,
    AGREE_ALL_MATCH,
    AGREE_ONE_UNCERTAIN,
    AGREE_DISPUTED,
)

PASSED = 0
FAILED = 0


def check(condition, msg):
    global PASSED, FAILED
    if condition:
        PASSED += 1
        print(f"  PASS: {msg}")
    else:
        FAILED += 1
        print(f"  FAIL: {msg}")


# ========== Mock 辅助 ==========

def make_mock_recognizer(rec_id: str, answers_map: dict, fail: bool = False):
    """构造一个 mock 识别器,返回预设的 answers_map

    Args:
        rec_id: 识别器 ID
        answers_map: {q: answer_str_or_None}
        fail: True 则让 recognize() 抛异常
    """
    rec = MagicMock()
    rec.id = rec_id
    rec.name = f"Mock-{rec_id}"
    rec.requires_blank = False

    if fail:
        rec.recognize = MagicMock(side_effect=RuntimeError(f"{rec_id} 模拟异常"))
    else:
        answers = {}
        empty_count = 0
        for q, a in answers_map.items():
            if a is None:
                answers[q] = {"answer": None, "status": "empty", "correct": None}
                empty_count += 1
            elif isinstance(a, str) and len(a) > 1:
                answers[q] = {"answer": a, "status": "multi", "correct": None}
            else:
                answers[q] = {"answer": a, "status": "single", "correct": None}

        result = RecognizeResult(
            answers=answers,
            total=len(answers_map),
            empty_count=empty_count,
            multi_count=sum(1 for a in answers_map.values() if isinstance(a, str) and len(a) > 1),
            recognizer_id=rec_id,
        )
        rec.recognize = MagicMock(return_value=result)

    return rec


# ========== 1. _compare_answers 单元测试 (7 例) ==========

def test_compare_all_empty():
    """全 None → all_empty, agreed=True"""
    cmp = RecognizerManager._compare_answers({"a": None, "b": None})
    check(cmp["agreement_type"] == AGREE_ALL_EMPTY, f"全空 → {AGREE_ALL_EMPTY} (got {cmp['agreement_type']})")
    check(cmp["agreed"] is True, "全空 agreed=True")
    check(cmp["consensus"] is None, "全空 consensus=None")


def test_compare_all_match_single():
    """全 "A" → all_match, agreed=True, consensus="A" """
    cmp = RecognizerManager._compare_answers({"a": "A", "b": "A"})
    check(cmp["agreement_type"] == AGREE_ALL_MATCH, f"全 'A' → {AGREE_ALL_MATCH}")
    check(cmp["agreed"] is True, "全 'A' agreed=True")
    check(cmp["consensus"] == "A", "全 'A' consensus='A'")


def test_compare_all_match_multi():
    """多选 "ABC" / "BCA" / "CAB" → all_match (set 比较)"""
    cmp = RecognizerManager._compare_answers({"a": "ABC", "b": "BCA", "c": "CAB"})
    check(cmp["agreement_type"] == AGREE_ALL_MATCH, f"多选乱序 → {AGREE_ALL_MATCH}")
    check(cmp["agreed"] is True, "多选乱序 agreed=True")
    check(cmp["consensus"] == "ABC", f"consensus 取第一个: 'ABC' (got {cmp['consensus']})")


def test_compare_one_uncertain_empty_diff():
    """None vs "A" → one_uncertain, consensus="A" """
    cmp = RecognizerManager._compare_answers({"a": None, "b": "A"})
    check(cmp["agreement_type"] == AGREE_ONE_UNCERTAIN, f"1 空 1 实 → {AGREE_ONE_UNCERTAIN} (got {cmp['agreement_type']})")
    check(cmp["agreed"] is False, "1 空 1 实 agreed=False")
    check(cmp["consensus"] == "A", f"consensus='A' (got {cmp['consensus']})")


def test_compare_one_uncertain_diff_empty():
    """"A" vs None → one_uncertain (顺序无关)"""
    cmp = RecognizerManager._compare_answers({"a": "A", "b": None})
    check(cmp["agreement_type"] == AGREE_ONE_UNCERTAIN, "1 实 1 空 → one_uncertain (顺序无关)")
    check(cmp["consensus"] == "A", "consensus='A'")


def test_compare_disputed_single():
    """"A" vs "B" → disputed, consensus=None"""
    cmp = RecognizerManager._compare_answers({"a": "A", "b": "B"})
    check(cmp["agreement_type"] == AGREE_DISPUTED, f"单选分歧 → {AGREE_DISPUTED}")
    check(cmp["agreed"] is False, "单选分歧 agreed=False")
    check(cmp["consensus"] is None, "分歧 consensus=None")


def test_compare_disputed_multi():
    """"AB" vs "ABC" → disputed (多选 set 不等)"""
    cmp = RecognizerManager._compare_answers({"a": "AB", "b": "ABC"})
    check(cmp["agreement_type"] == AGREE_DISPUTED, f"多选少选 → {AGREE_DISPUTED}")
    check(cmp["agreed"] is False, "多选少选 agreed=False")
    check(cmp["consensus"] is None, "多选少选 consensus=None")


# ========== 2. RecognizerManager 集成测试 (6 例) ==========

def test_manager_empty_recognizers_raises():
    """空列表 → ValueError"""
    raised = False
    try:
        RecognizerManager([])
    except ValueError as e:
        raised = True
        check("不能为空" in str(e), "ValueError message 提及'不能为空'")
    check(raised, "空 recognizers 抛 ValueError")


def test_manager_single_recognizer():
    """1 个识别器,所有题都应 all_match(降级为单识别器)"""
    rec = make_mock_recognizer("only", {1: "A", 2: "B", 3: None})
    mgr = RecognizerManager([rec])
    result = mgr.cross_validate(np.zeros((100, 100, 3), dtype=np.uint8), RecognizeContext())
    check(isinstance(result, CrossValidatedResult), "返回 CrossValidatedResult")
    check(result.agreement_rate == 1.0, f"N=1 全部 agreed → agreement_rate=1.0 (got {result.agreement_rate})")
    check(result.disputed_questions == [], f"无分歧 (got {result.disputed_questions})")
    check(result.total == 3, f"total=3 (got {result.total})")
    check(result.answers[1]["status"] == "single", "Q1 single")
    check(result.answers[3]["status"] == "empty", "Q3 empty")


def test_manager_two_agree_all():
    """2 个识别器输出相同 → agreement_rate=1.0, 无分歧"""
    golden = make_mock_recognizer("golden", {1: "A", 2: "B", 3: "C"})
    diff = make_mock_recognizer("differential", {1: "A", 2: "B", 3: "C"})
    mgr = RecognizerManager([golden, diff])
    result = mgr.cross_validate(np.zeros((100, 100, 3), dtype=np.uint8), RecognizeContext())
    check(result.agreement_rate == 1.0, f"全一致 → 1.0 (got {result.agreement_rate})")
    check(result.disputed_questions == [], f"无分歧题 (got {result.disputed_questions})")
    check(result.card_flag is None, "无 card_flag")
    check(result.recognizer_id == "cross_validation", "recognizer_id 标记为 cross_validation")


def test_manager_two_dispute_some():
    """2 个识别器部分分歧 → agreement_rate<1.0, disputed_questions 非空"""
    golden = make_mock_recognizer("golden", {1: "A", 2: "B", 3: "C", 4: "D"})
    diff = make_mock_recognizer("differential", {1: "A", 2: "B", 3: "X", 4: "D"})
    mgr = RecognizerManager([golden, diff])
    result = mgr.cross_validate(np.zeros((100, 100, 3), dtype=np.uint8), RecognizeContext())
    check(result.agreement_rate == 0.75, f"3/4 一致 → 0.75 (got {result.agreement_rate})")
    check(result.disputed_questions == [3], f"分歧题: [3] (got {result.disputed_questions})")
    check(result.card_flag == "uncertain", f"有分歧 → card_flag='uncertain' (got {result.card_flag})")
    # 分歧题:answer=None, status="uncertain"
    check(result.answers[3]["answer"] is None, f"分歧题 Q3 answer=None (got {result.answers[3]['answer']})")
    check(result.answers[3]["status"] == "uncertain", "分歧题 Q3 status='uncertain'")
    # 正常题保持共识
    check(result.answers[1]["answer"] == "A", "一致题 Q1 answer='A'")
    check(result.answers[1]["status"] == "single", "一致题 Q1 status='single'")


def test_manager_to_legacy_dict():
    """CrossValidatedResult.to_legacy_dict() 与 RecognizeResult 行为一致"""
    golden = make_mock_recognizer("golden", {1: "A", 2: "B"})
    mgr = RecognizerManager([golden])
    result = mgr.cross_validate(np.zeros((100, 100, 3), dtype=np.uint8), RecognizeContext())
    legacy = result.to_legacy_dict()
    # 必需字段
    for k in ("answers", "total", "empty_count", "multi_count", "card_flag", "debug_lines"):
        check(k in legacy, f"legacy dict 含字段: {k}")
    # 注:CrossValidatedResult 特有字段(per_q_cv / agreement_rate / disputed_questions)
    # 不在 legacy dict 中 — 阶段 5 UI 拆分时通过新字段访问


def test_manager_per_q_cv_format():
    """per_q_cv 字段含 per_recognizer/agreed/agreement_type/consensus"""
    golden = make_mock_recognizer("golden", {1: "A", 2: "B"})
    diff = make_mock_recognizer("differential", {1: "A", 2: "X"})
    mgr = RecognizerManager([golden, diff])
    result = mgr.cross_validate(np.zeros((100, 100, 3), dtype=np.uint8), RecognizeContext())

    # Q1 共识
    q1 = result.per_q_cv[1]
    check(q1["per_recognizer"] == {"golden": "A", "differential": "A"}, "Q1 per_recognizer 全 A")
    check(q1["agreed"] is True, "Q1 agreed=True")
    check(q1["agreement_type"] == AGREE_ALL_MATCH, f"Q1 type={AGREE_ALL_MATCH}")
    check(q1["consensus"] == "A", "Q1 consensus='A'")

    # Q2 分歧
    q2 = result.per_q_cv[2]
    check(q2["per_recognizer"] == {"golden": "B", "differential": "X"}, "Q2 per_recognizer 反映分歧")
    check(q2["agreed"] is False, "Q2 agreed=False")
    check(q2["agreement_type"] == AGREE_DISPUTED, f"Q2 type={AGREE_DISPUTED}")
    check(q2["consensus"] is None, "Q2 consensus=None")


def test_manager_recognizer_exception_does_not_crash():
    """单个识别器抛异常不阻塞整个交叉验证"""
    good = make_mock_recognizer("good", {1: "A", 2: "B"})
    bad = make_mock_recognizer("bad", {}, fail=True)
    mgr = RecognizerManager([good, bad])
    result = mgr.cross_validate(np.zeros((100, 100, 3), dtype=np.uint8), RecognizeContext())
    # 异常识别器的 answers 为空,所有题都"good 给出 X, bad 给出 None" → one_uncertain
    check(result.total == 2, f"total=2 (got {result.total})")
    check(result.agreement_rate == 0.0, f"全 one_uncertain → 0.0 (got {result.agreement_rate})")
    # 异常不应导致崩溃 — result 应仍可访问
    check("bad" in result.recognizer_results, "异常识别器仍在 recognizer_results 中")
    check("bad" in result.recognizer_results["bad"].debug_lines[0] or "bad" in str(result.recognizer_results["bad"].debug_lines),
          f"异常识别器 debug_lines 记录错误 (got {result.recognizer_results['bad'].debug_lines})")


# ========== 运行入口 ==========

if __name__ == "__main__":
    print("=" * 60)
    print("RecognizerManager 交叉验证测试")
    print("=" * 60)

    print("\n[1] _compare_answers 单元测试 (7 例)")
    test_compare_all_empty()
    test_compare_all_match_single()
    test_compare_all_match_multi()
    test_compare_one_uncertain_empty_diff()
    test_compare_one_uncertain_diff_empty()
    test_compare_disputed_single()
    test_compare_disputed_multi()

    print("\n[2] RecognizerManager 集成测试 (6 例)")
    test_manager_empty_recognizers_raises()
    test_manager_single_recognizer()
    test_manager_two_agree_all()
    test_manager_two_dispute_some()
    test_manager_to_legacy_dict()
    test_manager_per_q_cv_format()
    test_manager_recognizer_exception_does_not_crash()

    print("\n" + "=" * 60)
    print(f"总计: {PASSED} passed, {FAILED} failed")
    print("=" * 60)
    sys.exit(0 if FAILED == 0 else 1)
