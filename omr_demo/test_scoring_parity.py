"""
算分回归对比: 新 ScoreCalculator vs app.py:810-825 旧内联逻辑
目的: 验证 commit 2 改造后, 实际产出与改造前完全一致

运行: python omr_demo/test_scoring_parity.py
期望: ALL PARITY (无差异)
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, ".")

from core.score_calculator import calc_total_score, ScoringConfig


# ========== 旧内联逻辑 (从 app.py:810-825 复制) ==========

def old_calc_score(effective_answers, answer_key):
    """与改造前 app.py:810-825 完全一致的算分逻辑"""
    sc = 0
    tot = len(answer_key)
    for q, std_ans in answer_key.items():
        final_ans = effective_answers.get(q, "")
        if final_ans:
            if len(std_ans) > 1:
                if set(final_ans) == set(std_ans):
                    sc += 1
            elif final_ans == std_ans:
                sc += 1
    return sc, tot


# ========== 测试用例 ==========

CASES = [
    # (名称, effective_answers, answer_key, 期望 sc, 期望 tot)
    ("全对",
     {1: "A", 2: "B", 3: "C"}, {1: "A", 2: "B", 3: "C"}, 3, 3),
    ("全错",
     {1: "A", 2: "B", 3: "C"}, {1: "B", 2: "C", 3: "A"}, 0, 3),
    ("半对",
     {1: "A", 2: "B"}, {1: "A", 2: "C"}, 1, 2),
    ("全空",
     {1: "", 2: "", 3: ""}, {1: "A", 2: "B", 3: "C"}, 0, 3),
    ("部分空",
     {1: "A", 2: "", 3: "C"}, {1: "A", 2: "B", 3: "C"}, 2, 3),
    ("多选全对",
     {1: "ABC", 2: "AB"}, {1: "ABC", 2: "AB"}, 2, 2),
    ("多选少选",
     {1: "AB", 2: "AC"}, {1: "ABC", 2: "ABC"}, 0, 2),
    ("多选多选",
     {1: "ABCD", 2: "ABC"}, {1: "ABC", 2: "ABC"}, 1, 2),
    ("单选被多涂",
     {1: "AB"}, {1: "A"}, 0, 1),
    ("大小写(默认 ignore_case=False,旧行为)",
     {1: "a", 2: "b"}, {1: "A", 2: "B"}, 0, 2),
    ("多选大小写(默认)",
     {1: "abc"}, {1: "ABC"}, 0, 1),
    ("全部题目都答",
     {q: "A" for q in range(1, 11)}, {q: "A" for q in range(1, 11)}, 10, 10),
    ("答一半题",
     {q: "A" for q in range(1, 6)}, {q: "A" for q in range(1, 11)}, 5, 10),
    ("完全白卷(effective_answers 缺 key)",
     {}, {1: "A", 2: "B"}, 0, 2),
    ("effective 多 key(standard 没有)",
     {1: "A", 2: "B", 3: "C"}, {1: "A", 2: "B"}, 2, 2),
]


# ========== 跑对比 ==========

PASSED = 0
FAILED = 0

print("=" * 60)
print("算分回归对比: 新 ScoreCalculator vs app.py 旧逻辑")
print("=" * 60)

for name, eff, gold, exp_sc, exp_tot in CASES:
    # 旧逻辑
    old_sc, old_tot = old_calc_score(eff, gold)

    # 新逻辑: 构造 effective_answers(模拟 app.py 里的组装)
    new_eff = {}
    for q, std_ans in gold.items():
        v = eff.get(q, "")
        new_eff[q] = {"answer": v if v else None, "status": "single" if v else "empty"}
    r = calc_total_score(new_eff, gold, ScoringConfig())
    new_sc = r["total"]
    new_tot = r["total_full"]

    ok = (old_sc == new_sc == exp_sc) and (old_tot == new_tot == exp_tot)
    if ok:
        PASSED += 1
        print(f"  PASS: {name:30s} -> sc={new_sc}/{new_tot} (期望 {exp_sc}/{exp_tot})")
    else:
        FAILED += 1
        print(f"  FAIL: {name:30s} -> old={old_sc}/{old_tot} new={new_sc}/{new_tot} exp={exp_sc}/{exp_tot}")

print()
print("=" * 60)
print(f"总计: {PASSED} passed, {FAILED} failed (期望 0 failed)")
print("=" * 60)
print()
if FAILED == 0:
    print("[OK] 新旧逻辑完全一致, 改造无回归")
    print("     可以安全替换 app.py:810-825 内联循环")
else:
    print("[ERROR] 发现差异! 立即停止,不要合入")

sys.exit(0 if FAILED == 0 else 1)
