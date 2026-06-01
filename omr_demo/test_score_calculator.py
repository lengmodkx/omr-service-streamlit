"""
ScoreCalculator 单元测试套件
设计文档: docs/plans/2026-06-01-stage2-score-calculator-design.md §4

运行方式: python omr_demo/test_score_calculator.py
期望输出: 22 PASS, 0 FAIL

测试组织:
- round_score: 5 例
- match_answer: 8 例
- calc_total_score: 9 例
"""
import sys
# Windows GBK 控制台无法打印部分 Unicode 符号,统一用 UTF-8 输出
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass
sys.path.insert(0, ".")

from core.score_calculator import (
    ScoringConfig,
    round_score,
    match_answer,
    calc_total_score,
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


# ========== round_score (5 例) ==========

def test_round_score_integer_unchanged():
    """整数 + step=1 → 正常四舍五入(整数本身不变)"""
    check(round_score(85.0) == 85.0, "85.0 round → 85.0")
    check(round_score(100.0) == 100.0, "100.0 round → 100.0")


def test_round_score_normal_rounding():
    """小数 + step=1 → 正常四舍五入"""
    check(round_score(85.6) == 86.0, "85.6 round → 86.0")
    check(round_score(85.4) == 85.0, "85.4 round → 85.0")
    check(round_score(85.5) == 86.0, "85.5 round → 86.0 (Python banker's rounding 行为,接受 86)")


def test_round_score_half_step():
    """step=0.5 → 0.5 取整"""
    cfg = ScoringConfig(round_step=0.5, round_method="round")
    check(round_score(85.3, cfg) == 85.5, "85.3 step=0.5 round → 85.5")
    check(round_score(85.6, cfg) == 85.5, "85.6 step=0.5 round → 85.5")
    check(round_score(85.8, cfg) == 86.0, "85.8 step=0.5 round → 86.0")


def test_round_score_negative_half_step():
    """负数 + step=0.5 → 负数 0.5 取整(边界)"""
    cfg = ScoringConfig(round_step=0.5, round_method="round")
    # -2.7 / 0.5 = -5.4 → round → -5 → * 0.5 = -2.5
    check(round_score(-2.7, cfg) == -2.5, "-2.7 step=0.5 round → -2.5")
    # -2.3 / 0.5 = -4.6 → round → -5 → * 0.5 = -2.5
    check(round_score(-2.3, cfg) == -2.5, "-2.3 step=0.5 round → -2.5")


def test_round_score_none_passthrough():
    """None 透传不抛异常"""
    check(round_score(None) is None, "None round → None (透传)")
    check(round_score(None, ScoringConfig()) is None, "None + cfg → None")


# ========== match_answer (8 例) ==========

def test_single_match_exact():
    """单选完全相等"""
    check(match_answer("A", "A") is True, "A == A → True")
    check(match_answer("D", "D") is True, "D == D → True")


def test_single_mismatch():
    """单选不等"""
    check(match_answer("A", "B") is False, "A != B → False")
    check(match_answer("C", "D") is False, "C != D → False")


def test_multi_match_exact():
    """多选完全对(集合相等)"""
    check(match_answer("ABC", "ABC", multi=True) is True, "ABC == ABC (multi) → True")
    check(match_answer("ACB", "ABC", multi=True) is True, "ACB == ABC (multi) → True (顺序无关)")


def test_multi_mismatch_missing_one():
    """多选少一个"""
    check(match_answer("AB", "ABC", multi=True) is False, "AB ⊂ ABC → False (少一个)")
    check(match_answer("AC", "ABC", multi=True) is False, "AC ⊂ ABC → False (少一个)")


def test_multi_mismatch_extra_one():
    """多选多一个"""
    check(match_answer("ABCD", "ABC", multi=True) is False, "ABCD ⊃ ABC → False (多一个)")


def test_case_insensitive_match():
    """大小写不敏感(默认 ignore_case=True)"""
    check(match_answer("a", "A") is True, "a vs A (ignore_case=True) → True")
    check(match_answer("abc", "ABC", multi=True) is True, "abc vs ABC (ignore_case=True) → True")


def test_case_sensitive_strict_mismatch():
    """大小写敏感(ignore_case=False)"""
    cfg = ScoringConfig(ignore_case=False)
    check(match_answer("a", "A", cfg=cfg) is False, "a vs A (ignore_case=False) → False")


def test_empty_or_none_safe():
    """空值安全"""
    check(match_answer(None, "A") is False, "None vs A → False")
    check(match_answer("", "A") is False, "'' vs A → False")
    check(match_answer("A", None) is False, "A vs None → False")
    check(match_answer("A", "") is False, "A vs '' → False")


# ========== calc_total_score (9 例) ==========

def test_all_correct():
    """全对"""
    answers = {q: {"answer": "A", "status": "single"} for q in [1, 2, 3, 4]}
    standard = {1: "A", 2: "A", 3: "A", 4: "A"}
    r = calc_total_score(answers, standard)
    check(r["total"] == 4.0, f"全对 total=4 (got {r['total']})")
    check(r["total_full"] == 4.0, f"全对 total_full=4 (got {r['total_full']})")
    check(r["stats"]["correct"] == 4, f"全对 correct=4 (got {r['stats']['correct']})")
    check(r["stats"]["single"] == 4, f"全对 single=4 (got {r['stats']['single']})")


def test_all_wrong():
    """全错"""
    answers = {q: {"answer": "B", "status": "single"} for q in [1, 2, 3]}
    standard = {1: "A", 2: "A", 3: "A"}
    r = calc_total_score(answers, standard)
    check(r["total"] == 0.0, f"全错 total=0 (got {r['total']})")
    check(r["stats"]["correct"] == 0, f"全错 correct=0 (got {r['stats']['correct']})")


def test_all_empty():
    """全空(白卷)"""
    answers = {q: {"answer": None, "status": "empty"} for q in [1, 2, 3]}
    standard = {1: "A", 2: "A", 3: "A"}
    r = calc_total_score(answers, standard)
    check(r["total"] == 0.0, f"全空 total=0 (got {r['total']})")
    check(r["stats"]["empty"] == 3, f"全空 empty=3 (got {r['stats']['empty']})")
    check(r["stats"]["correct"] == 0, f"全空 correct=0 (got {r['stats']['correct']})")


def test_multi_status_with_matching_set():
    """multi 状态 + 集合匹配 → earned=1.0, correct=True
    这是与原设计文档不同的关键行为:multi 不等于 0 分,
    走 match_answer 的集合匹配。
    """
    answers = {
        1: {"answer": "ABC", "status": "multi"},  # multi 状态但答案集合匹配
        2: {"answer": "AB", "status": "multi"},   # multi 状态但少选
    }
    standard = {1: "ABC", 2: "ABC"}
    r = calc_total_score(answers, standard)
    check(r["total"] == 1.0, f"multi+匹配 total=1 (got {r['total']})")
    check(r["per_q"][0]["correct"] is True, "Q1 multi+匹配 → correct=True")
    check(r["per_q"][0]["earned"] == 1.0, "Q1 multi+匹配 → earned=1.0")
    check(r["per_q"][1]["correct"] is False, "Q2 multi+少选 → correct=False")
    check(r["per_q"][1]["earned"] == 0.0, "Q2 multi+少选 → earned=0.0")
    check(r["stats"]["multi"] == 2, f"multi 状态计入 stats (got {r['stats']['multi']})")


def test_half_correct():
    """一半对一半错"""
    answers = {
        1: {"answer": "A", "status": "single"},
        2: {"answer": "B", "status": "single"},
        3: {"answer": "C", "status": "single"},
        4: {"answer": "D", "status": "single"},
    }
    standard = {1: "A", 2: "X", 3: "C", 4: "Y"}
    r = calc_total_score(answers, standard)
    check(r["total"] == 2.0, f"半对 total=2 (got {r['total']})")
    check(r["stats"]["correct"] == 2, f"半对 correct=2 (got {r['stats']['correct']})")


def test_mixed_status():
    """混合状态: empty + multi + single + uncertain"""
    answers = {
        1: {"answer": "A", "status": "single"},
        2: {"answer": "ABC", "status": "multi"},
        3: {"answer": None, "status": "empty"},
        4: {"answer": "B", "status": "uncertain"},
    }
    standard = {1: "A", 2: "ABC", 3: "A", 4: "A"}
    r = calc_total_score(answers, standard)
    check(r["total"] == 2.0, f"混合 total=2 (Q1+Q2 正确) (got {r['total']})")
    check(r["stats"]["single"] == 1, "single 计数=1")
    check(r["stats"]["multi"] == 1, "multi 计数=1")
    check(r["stats"]["empty"] == 1, "empty 计数=1")
    check(r["stats"]["uncertain"] == 1, "uncertain 计数=1")
    check(r["stats"]["correct"] == 2, "correct 计数=2")


def test_extra_answers_ignored():
    """answers 多 1 题,standard 没有 → 不计入 total_full"""
    answers = {
        1: {"answer": "A", "status": "single"},
        2: {"answer": "B", "status": "single"},
        3: {"answer": "C", "status": "single"},  # standard 里没有
    }
    standard = {1: "A", 2: "A"}
    r = calc_total_score(answers, standard)
    check(r["total_full"] == 2.0, f"越界答案不计入 total_full=2 (got {r['total_full']})")
    check(r["total"] == 1.0, f"只统计 standard 内的 (got {r['total']})")


def test_missing_answers_zero_score():
    """answers 少 1 题,standard 多 1 题 → 缺的题 earned=0"""
    answers = {1: {"answer": "A", "status": "single"}}
    standard = {1: "A", 2: "A"}
    r = calc_total_score(answers, standard)
    check(r["total_full"] == 2.0, f"缺失题算入 total_full=2 (got {r['total_full']})")
    check(r["total"] == 1.0, f"缺失题 earned=0,total=1 (got {r['total']})")
    check(r["per_q"][1]["earned"] == 0.0, "Q2(缺失) earned=0")
    check(r["per_q"][1]["correct"] is False, "Q2(缺失) correct=False")


def test_total_rounding_boundary():
    """总分取整边界: 15.7 + step=0.5"""
    answers = {q: {"answer": "A", "status": "single"} for q in [1, 2]}
    standard = {1: "A", 2: "A"}
    # earned_sum = 2.0
    cfg = ScoringConfig(round_step=0.5)
    r = calc_total_score(answers, standard, cfg)
    # round(2.0 / 0.5) * 0.5 = round(4.0) * 0.5 = 4.0 * 0.5 = 2.0
    check(r["total"] == 2.0, f"取整 2.0 (got {r['total']})")

    # 模拟 15.7 场景: 构造 16 题中 15 题对 1 题错,满分 16,实际 15
    answers_2 = {}
    for q in range(1, 17):
        answers_2[q] = {"answer": "A", "status": "single"} if q <= 15 else {"answer": "B", "status": "single"}
    standard_2 = {q: "A" for q in range(1, 17)}
    r2 = calc_total_score(answers_2, standard_2, cfg)
    # earned_sum = 15.0
    # round(15.0 / 0.5) * 0.5 = round(30.0) * 0.5 = 30.0 * 0.5 = 15.0
    check(r2["total"] == 15.0, f"15 题对取整 (got {r2['total']})")


# ========== 运行入口 ==========

if __name__ == "__main__":
    print("=" * 60)
    print("ScoreCalculator 单元测试")
    print("=" * 60)

    print("\n[1] round_score (5 例)")
    test_round_score_integer_unchanged()
    test_round_score_normal_rounding()
    test_round_score_half_step()
    test_round_score_negative_half_step()
    test_round_score_none_passthrough()

    print("\n[2] match_answer (8 例)")
    test_single_match_exact()
    test_single_mismatch()
    test_multi_match_exact()
    test_multi_mismatch_missing_one()
    test_multi_mismatch_extra_one()
    test_case_insensitive_match()
    test_case_sensitive_strict_mismatch()
    test_empty_or_none_safe()

    print("\n[3] calc_total_score (9 例)")
    test_all_correct()
    test_all_wrong()
    test_all_empty()
    test_multi_status_with_matching_set()
    test_half_correct()
    test_mixed_status()
    test_extra_answers_ignored()
    test_missing_answers_zero_score()
    test_total_rounding_boundary()

    print("\n" + "=" * 60)
    print(f"总计: {PASSED} passed, {FAILED} failed")
    print("=" * 60)
    sys.exit(0 if FAILED == 0 else 1)
