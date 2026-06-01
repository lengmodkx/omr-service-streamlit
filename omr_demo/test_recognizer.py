"""
Recognizer 协议测试套件
设计文档: docs/plans/2026-06-01-stage1-recognizer-abstraction-design.md §6

运行方式: python omr_demo/test_recognizer.py
期望输出: 12 PASS, 0 FAIL

测试组织:
- 协议一致性 (4 例): isinstance / can_handle 返回 bool / recognize 返回 dataclass
- 字段完备性 (3 例): 必需字段 / 默认值 / 字段类型
- 行为 (3 例): 工厂函数 / duration > 0 / recognizer_id 一致性
- 异常路径 (2 例): 未知 id 抛 ValueError / 空图像不崩溃
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
    Recognizer,
    RecognizeContext,
    RecognizeResult,
    make_recognizer,
    list_recognizer_ids,
)
from core.recognizers.golden import GoldenTemplateRecognizer
from core.recognizers.differential import DifferentialRecognizer

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


# ========== 1. 协议一致性 (4 例) ==========

def test_recognizer_protocol_conformance():
    """两个适配器都通过 isinstance(rec, Recognizer) 检查"""
    golden = GoldenTemplateRecognizer(golden_template=None)
    diff = DifferentialRecognizer(processor=None)
    check(isinstance(golden, Recognizer), "GoldenTemplateRecognizer is Recognizer")
    check(isinstance(diff, Recognizer), "DifferentialRecognizer is Recognizer")


def test_can_handle_returns_bool():
    """can_handle 永远返回 bool,不抛异常"""
    golden = GoldenTemplateRecognizer(golden_template=None)
    diff = DifferentialRecognizer(processor=None)
    ctx = RecognizeContext()
    check(golden.can_handle(ctx) is False, "golden None template -> False (bool)")
    check(diff.can_handle(ctx) is False, "diff None processor -> False (bool)")
    # True 情况:用 mock 构造有内容的环境
    gtp_mock = MagicMock()
    gtp_mock.bubbles = [{"q": 1, "opt": "A"}]
    golden2 = GoldenTemplateRecognizer(gtp_mock)
    check(golden2.can_handle(ctx) is True, "golden with bubbles -> True (bool)")


def test_recognize_returns_dataclass():
    """recognize() 返回 RecognizeResult 实例"""
    # 用 mock 替换底层识别方法,只验证包装逻辑
    gtp_mock = MagicMock()
    gtp_mock.recognize.return_value = {
        "answers": {1: {"answer": "A", "status": "single", "correct": None}},
        "total": 1, "empty_count": 0, "multi_count": 0,
        "card_flag": None, "debug_lines": [],
    }
    gtp_mock.bubbles = [{"q": 1, "opt": "A"}]
    rec = GoldenTemplateRecognizer(gtp_mock)
    result = rec.recognize(np.zeros((100, 100, 3), dtype=np.uint8), RecognizeContext())
    check(isinstance(result, RecognizeResult), "recognize() returns RecognizeResult")
    check(result.recognizer_id == "golden", "result.recognizer_id == 'golden'")
    check(result.duration_ms > 0, f"duration_ms > 0 (got {result.duration_ms})")


def test_protocol_class_attributes():
    """协议类属性 (name/id/requires_blank) 都存在且类型正确"""
    golden = GoldenTemplateRecognizer(golden_template=None)
    diff = DifferentialRecognizer(processor=None)
    check(isinstance(golden.name, str) and len(golden.name) > 0, "golden.name is non-empty str")
    check(isinstance(golden.id, str) and golden.id == "golden", "golden.id == 'golden'")
    check(golden.requires_blank is False, "golden.requires_blank == False")
    check(isinstance(diff.name, str) and len(diff.name) > 0, "diff.name is non-empty str")
    check(isinstance(diff.id, str) and diff.id == "differential", "diff.id == 'differential'")
    check(diff.requires_blank is True, "diff.requires_blank == True")


# ========== 2. 字段完备性 (3 例) ==========

def test_recognize_result_required_fields():
    """RecognizeResult 包含所有 8 个必需字段"""
    fields = {"answers", "total", "empty_count", "multi_count",
              "card_flag", "debug_lines", "duration_ms", "recognizer_id"}
    r = RecognizeResult()
    for f in fields:
        check(hasattr(r, f), f"RecognizeResult has field: {f}")


def test_recognize_result_default_values():
    """RecognizeResult 缺省值合理"""
    r = RecognizeResult()
    check(r.answers == {}, "answers default == {}")
    check(r.total == 0, "total default == 0")
    check(r.empty_count == 0, "empty_count default == 0")
    check(r.multi_count == 0, "multi_count default == 0")
    check(r.card_flag is None, "card_flag default is None")
    check(r.debug_lines == [], "debug_lines default == []")
    check(r.duration_ms == 0.0, "duration_ms default == 0.0")
    check(r.recognizer_id == "", "recognizer_id default == ''")


def test_recognize_result_types():
    """RecognizeResult 字段类型正确"""
    r = RecognizeResult(
        answers={1: {"answer": "A", "status": "single", "correct": None}},
        total=1, empty_count=0, multi_count=0,
        card_flag="abnormal", debug_lines=["line1"],
        duration_ms=12.5, recognizer_id="golden"
    )
    check(isinstance(r.answers, dict), "answers is dict")
    check(isinstance(r.total, int), "total is int")
    check(isinstance(r.card_flag, str), "card_flag is str")
    check(isinstance(r.debug_lines, list), "debug_lines is list")
    check(isinstance(r.duration_ms, float), "duration_ms is float")
    check(isinstance(r.recognizer_id, str), "recognizer_id is str")


# ========== 3. 行为 (3 例) ==========

def test_make_recognizer_factory():
    """make_recognizer 工厂函数能正确构造 2 个适配器"""
    gtp_mock = MagicMock()
    gtp_mock.bubbles = [{"q": 1}]
    g = make_recognizer("golden", golden_template=gtp_mock)
    check(isinstance(g, GoldenTemplateRecognizer), "make_recognizer('golden') -> GoldenTemplateRecognizer")

    proc_mock = MagicMock()
    proc_mock.template = {"pages": {"A": {"bubbles": [{"q": 1}]}}}
    d = make_recognizer("differential", processor=proc_mock, page="A")
    check(isinstance(d, DifferentialRecognizer), "make_recognizer('differential') -> DifferentialRecognizer")


def test_list_recognizer_ids():
    """list_recognizer_ids() 返回已注册 ID 列表"""
    ids = list_recognizer_ids()
    check("golden" in ids, "list contains 'golden'")
    check("differential" in ids, "list contains 'differential'")


def test_duration_ms_and_recognizer_id_from_recognize():
    """真实 recognize() 后,duration_ms > 0 且 recognizer_id 与 self.id 一致"""
    gtp_mock = MagicMock()
    gtp_mock.recognize.return_value = {
        "answers": {1: {"answer": "A", "status": "single", "correct": True}},
        "total": 1, "empty_count": 0, "multi_count": 0,
        "card_flag": None, "debug_lines": ["test"],
    }
    gtp_mock.bubbles = [{"q": 1}]
    rec = GoldenTemplateRecognizer(gtp_mock)
    result = rec.recognize(np.zeros((100, 100, 3), dtype=np.uint8), RecognizeContext())
    check(result.duration_ms > 0, f"duration_ms > 0 (got {result.duration_ms})")
    check(result.recognizer_id == rec.id, f"recognizer_id matches self.id ({result.recognizer_id})")
    check(result.total == 1, f"total propagated from underlying (got {result.total})")
    check(result.debug_lines == ["test"], "debug_lines propagated from underlying")


# ========== 4. 异常路径 (2 例) ==========

def test_make_recognizer_unknown_id():
    """make_recognizer 未知 ID 抛 ValueError"""
    raised = False
    try:
        make_recognizer("yolo")
    except ValueError as e:
        raised = True
        check("yolo" in str(e) or "未知" in str(e), "ValueError message mentions unknown id")
    check(raised, "ValueError raised for unknown id")


def test_differential_multi_detection():
    """DifferentialRecognizer 把 '(多涂)' 后缀正确转换为 status='multi'"""
    proc_mock = MagicMock()
    proc_mock.recognize_choices.return_value = {
        1: "A",
        2: "BC(多涂)",       # 多涂:去除后缀
        3: None,             # 空答
        4: "D(多涂)",        # 多涂
    }
    rec = DifferentialRecognizer(proc_mock, page="A")
    result = rec.recognize(np.zeros((100, 100, 3), dtype=np.uint8), RecognizeContext())
    check(result.answers[1]["status"] == "single", "Q1 'A' -> single")
    check(result.answers[2]["status"] == "multi", "Q2 'BC(多涂)' -> multi")
    check(result.answers[2]["answer"] == "BC", "Q2 answer normalized to 'BC' (suffix removed)")
    check(result.answers[3]["status"] == "empty", "Q3 None -> empty")
    check(result.answers[4]["status"] == "multi", "Q4 'D(多涂)' -> multi")
    check(result.answers[4]["answer"] == "D", "Q4 answer normalized to 'D'")
    check(result.multi_count == 2, f"multi_count == 2 (got {result.multi_count})")
    check(result.empty_count == 1, f"empty_count == 1 (got {result.empty_count})")
    check(result.total == 4, f"total == 4 (got {result.total})")


# ========== 运行入口 ==========

if __name__ == "__main__":
    print("=" * 60)
    print("Recognizer 协议测试")
    print("=" * 60)

    print("\n[1] 协议一致性 (4 例)")
    test_recognizer_protocol_conformance()
    test_can_handle_returns_bool()
    test_recognize_returns_dataclass()
    test_protocol_class_attributes()

    print("\n[2] 字段完备性 (3 例)")
    test_recognize_result_required_fields()
    test_recognize_result_default_values()
    test_recognize_result_types()

    print("\n[3] 行为 (3 例)")
    test_make_recognizer_factory()
    test_list_recognizer_ids()
    test_duration_ms_and_recognizer_id_from_recognize()

    print("\n[4] 异常路径 (2 例)")
    test_make_recognizer_unknown_id()
    test_differential_multi_detection()

    print("\n" + "=" * 60)
    print(f"总计: {PASSED} passed, {FAILED} failed")
    print("=" * 60)
    sys.exit(0 if FAILED == 0 else 1)
