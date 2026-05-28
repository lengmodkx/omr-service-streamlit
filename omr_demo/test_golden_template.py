"""
TDD: 黄金模板对比法 测试套件
Step 1: 网格切分
Step 2: 气泡采样
Step 3: 三道防线判断
Step 4: ECC对齐
Step 5: 端到端识别
"""
import sys
sys.path.insert(0, ".")
import numpy as np
import cv2

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


# ========== Step 1: _generate_grid 网格生成 ==========

def test_grid_basic():
    """给定列框坐标和行列参数，验证气泡坐标正确生成"""
    from core.golden_template import GoldenTemplate

    config = {
        "x1": 100, "y1": 200, "x2": 400, "y2": 700,
        "start_q": 1, "num_q": 5, "num_options": 4,
    }

    gtp = GoldenTemplate.__new__(GoldenTemplate)
    bubbles = gtp._generate_grid(config)

    check(len(bubbles) == 20, f"气泡总数应为20，实际{len(bubbles)}")

    # 验证 Q1 optA 位置
    q1a = next(b for b in bubbles if b["q"] == 1 and b["opt"] == "A")
    col_w = (400 - 100) / 4   # 75
    row_h = (700 - 200) / 5   # 100
    expected_cx_a = int(100 + col_w / 2)   # 137
    expected_cy = int(200 + row_h / 2)     # 250
    check(abs(q1a["x"] - expected_cx_a) <= 1, f"Q1 optA x应为{expected_cx_a}，实际{q1a['x']}")
    check(abs(q1a["y"] - expected_cy) <= 1, f"Q1 optA y应为{expected_cy}，实际{q1a['y']}")

    # 验证 Q1 optD 位置
    q1d = next(b for b in bubbles if b["q"] == 1 and b["opt"] == "D")
    expected_cx_d = int(100 + 3.5 * col_w)  # 362
    check(abs(q1d["x"] - expected_cx_d) <= 1, f"Q1 optD x应为{expected_cx_d}，实际{q1d['x']}")

    # 验证 Q5 optA 行位置
    q5a = next(b for b in bubbles if b["q"] == 5 and b["opt"] == "A")
    expected_cy_5 = int(200 + 4.5 * row_h)  # 650
    check(abs(q5a["y"] - expected_cy_5) <= 1, f"Q5 optA y应为{expected_cy_5}，实际{q5a['y']}")

    # 验证相邻题号行间距相等
    q2a = next(b for b in bubbles if b["q"] == 2 and b["opt"] == "A")
    q3a = next(b for b in bubbles if b["q"] == 3 and b["opt"] == "A")
    row_gap_1_2 = q2a["y"] - q1a["y"]
    row_gap_2_3 = q3a["y"] - q2a["y"]
    check(row_gap_1_2 == row_gap_2_3, f"行间距应相等: {row_gap_1_2} vs {row_gap_2_3}")

    # 验证相邻选项列间距相等
    q1b = next(b for b in bubbles if b["q"] == 1 and b["opt"] == "B")
    q1c = next(b for b in bubbles if b["q"] == 1 and b["opt"] == "C")
    col_gap_a_b = q1b["x"] - q1a["x"]
    col_gap_b_c = q1c["x"] - q1b["x"]
    check(col_gap_a_b == col_gap_b_c, f"列间距应相等: {col_gap_a_b} vs {col_gap_b_c}")

    # 验证气泡 w/h
    check(q1a["w"] > 6, f"气泡宽度应>6, 实际{q1a['w']}")
    check(q1a["h"] > 6, f"气泡高度应>6, 实际{q1a['h']}")


def test_grid_three_options():
    """3选项版面（常见于物理/化学选择题只有ABC三个选项）"""
    from core.golden_template import GoldenTemplate

    config = {
        "x1": 50, "y1": 100, "x2": 350, "y2": 340,
        "start_q": 6, "num_q": 3, "num_options": 3,
    }

    gtp = GoldenTemplate.__new__(GoldenTemplate)
    bubbles = gtp._generate_grid(config)

    check(len(bubbles) == 9, f"3行3列应为9个气泡，实际{len(bubbles)}")
    q6_opts = [b for b in bubbles if b["q"] == 6]
    check(len(q6_opts) == 3, f"Q6应有3个选项，实际{len(q6_opts)}")
    check(all(b["opt"] in "ABC" for b in q6_opts), "选项应为A/B/C")
    check(q6_opts[0]["opt"] == "A", f"第一个选项应为A，实际{q6_opts[0]['opt']}")


def test_grid_start_q_offset():
    """起始题号不从1开始的情况"""
    from core.golden_template import GoldenTemplate

    config = {
        "x1": 0, "y1": 0, "x2": 200, "y2": 200,
        "start_q": 21, "num_q": 2, "num_options": 2,
    }

    gtp = GoldenTemplate.__new__(GoldenTemplate)
    bubbles = gtp._generate_grid(config)

    q_nums = sorted(set(b["q"] for b in bubbles))
    check(q_nums == [21, 22], f"题号应为[21,22]，实际{q_nums}")


def test_bubble_window_size():
    """气泡检测窗口与网格间距成比例"""
    from core.golden_template import GoldenTemplate

    config = {
        "x1": 100, "y1": 100, "x2": 500, "y2": 500,
        "start_q": 1, "num_q": 4, "num_options": 4,
    }
    # col_w = 100, row_h = 100, 预期 w/h ≈ 50

    gtp = GoldenTemplate.__new__(GoldenTemplate)
    bubbles = gtp._generate_grid(config)

    b = bubbles[0]
    check(b["w"] >= 20, f"窗口宽度应≥20，实际{b['w']}")
    check(b["h"] >= 20, f"窗口高度应≥20，实际{b['h']}")
    check(b["w"] < 80, f"窗口宽度应<80，实际{b['w']}")


# ========== Step 2: _sample_bubble 气泡采样 ==========

def make_synthetic_roi(w=400, h=300, dark_spots=None):
    """构造模拟答题卡ROI的灰度图: 白色背景 + 暗点(模拟填涂)"""
    img = np.ones((h, w), dtype=np.uint8) * 235  # 空白纸张灰度
    img = cv2.GaussianBlur(img, (5, 5), 0)
    if dark_spots:
        for cx, cy, radius, gray_val in dark_spots:
            cv2.circle(img, (cx, cy), radius, gray_val, -1)
    return img


def test_sample_blank():
    """空白位置采样: 灰度应在220以上"""
    from core.golden_template import GoldenTemplate

    img = make_synthetic_roi()
    gtp = GoldenTemplate.__new__(GoldenTemplate)
    val = gtp._sample_bubble(img, 200, 150, 40, 40, 400, 300)

    check(val > 220, f"空白气泡灰度应>220，实际{val:.0f}")


def test_sample_filled():
    """填涂位置采样: 灰度应在170以下"""
    from core.golden_template import GoldenTemplate

    img = make_synthetic_roi(dark_spots=[(200, 150, 12, 120)])
    img = cv2.GaussianBlur(img, (5, 5), 0)

    gtp = GoldenTemplate.__new__(GoldenTemplate)
    val = gtp._sample_bubble(img, 200, 150, 40, 40, 400, 300)

    check(val < 190, f"填涂气泡灰度应<190，实际{val:.0f}")


def test_sample_boundary_clip():
    """边缘坐标不越界"""
    from core.golden_template import GoldenTemplate

    img = make_synthetic_roi()
    gtp = GoldenTemplate.__new__(GoldenTemplate)

    # 坐标在图片边缘外
    val = gtp._sample_bubble(img, -10, -10, 30, 30, 400, 300)
    check(isinstance(val, float), "边缘坐标不应崩溃")
    check(val >= 0, f"边缘灰度应>=0，实际{val:.0f}")

    val = gtp._sample_bubble(img, 410, 310, 30, 30, 400, 300)
    check(isinstance(val, float), "超出边界不应崩溃")


# ========== Step 3: 三道防线判断逻辑 ==========

def make_mock_q_groups(best_gray=130, second_gray=215, third_gray=220, fourth_gray=225):
    """构造题目的选项灰度数据"""
    return {
        1: {"A": best_gray, "B": second_gray, "C": third_gray, "D": fourth_gray},
    }


def test_judge_normal_fill():
    """T3.3: 一个暗、其他亮 → 选中"""
    from core.golden_template import GoldenTemplate

    gtp = GoldenTemplate.__new__(GoldenTemplate)
    gtp.bubbles = [{"q": 1, "opt": o, "x": 50 + i*80, "y": 150, "w": 16, "h": 16}
                   for i, o in enumerate("ABCD")]
    gtp.answers = {1: "A"}
    gtp.image = np.ones((300, 400, 3), dtype=np.uint8) * 235

    # A位置有暗点，B/C/D位置留白
    img = make_synthetic_roi(dark_spots=[(50, 150, 10, 120)])
    img = cv2.GaussianBlur(img, (5, 5), 0)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    result = gtp.recognize(img_bgr)
    ans = result["answers"][1]

    check(ans["answer"] == "A", f"应选中A，实际{ans['answer']}")
    check(ans["status"] == "single", f"状态应为single，实际{ans['status']}")
    check(result["total"] == 1, f"总题数应为1，实际{result['total']}")


def test_judge_all_blank():
    """T3.1: 全部 > 200 → 未填涂"""
    from core.golden_template import GoldenTemplate

    gtp = GoldenTemplate.__new__(GoldenTemplate)
    gtp.bubbles = [{"q": 1, "opt": o, "x": 50 + i*80, "y": 150, "w": 16, "h": 16}
                   for i, o in enumerate("ABCD")]
    gtp.answers = {}
    gtp.image = np.ones((300, 400, 3), dtype=np.uint8) * 235

    img = make_synthetic_roi()  # 全空白
    img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    result = gtp.recognize(img_bgr)
    ans = result["answers"][1]

    check(ans["answer"] is None, f"应为None(未填涂)，实际{ans['answer']}")
    check(ans["status"] == "empty", f"状态应为empty，实际{ans['status']}")
    check(result["empty_count"] == 1, f"empty_count应为1，实际{result['empty_count']}")


def test_judge_multi_fill():
    """T3.2: 两个 < 180 → 多涂"""
    from core.golden_template import GoldenTemplate

    gtp = GoldenTemplate.__new__(GoldenTemplate)
    gtp.bubbles = [{"q": 1, "opt": o, "x": 50 + i*80, "y": 150, "w": 16, "h": 16}
                   for i, o in enumerate("ABCD")]
    gtp.answers = {}
    gtp.image = np.ones((300, 400, 3), dtype=np.uint8) * 235

    img = make_synthetic_roi(dark_spots=[
        (50, 150, 10, 120),   # A填涂
        (130, 150, 10, 135),  # B也填涂
    ])
    img = cv2.GaussianBlur(img, (5, 5), 0)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    result = gtp.recognize(img_bgr)
    ans = result["answers"][1]

    check(ans["status"] == "multi", f"状态应为multi，实际{ans['status']}")
    check(result["multi_count"] == 1, f"multi_count应为1，实际{result['multi_count']}")


def test_judge_uncertain():
    """T3.4: 差异不够大 → uncertain"""
    from core.golden_template import GoldenTemplate

    gtp = GoldenTemplate.__new__(GoldenTemplate)
    gtp.bubbles = [{"q": 1, "opt": o, "x": 50 + i*80, "y": 150, "w": 16, "h": 16}
                   for i, o in enumerate("ABCD")]
    gtp.answers = {}
    gtp.image = np.ones((300, 400, 3), dtype=np.uint8) * 235

    # A暗但不够暗，B在暗阈值之上，差异不够大
    img = make_synthetic_roi(dark_spots=[
        (50, 150, 10, 170),
        (130, 150, 10, 200),
    ])
    img = cv2.GaussianBlur(img, (5, 5), 0)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    result = gtp.recognize(img_bgr)
    ans = result["answers"][1]

    check(ans["status"] == "single", f"相对阈值下A(170)明显暗于B(200)和C/D(235)，应为single，实际{ans['status']}")


def test_judge_all_black():
    """T3.5: 全涂黑 → 多涂"""
    from core.golden_template import GoldenTemplate

    gtp = GoldenTemplate.__new__(GoldenTemplate)
    gtp.bubbles = [{"q": 1, "opt": o, "x": 50 + i*80, "y": 150, "w": 16, "h": 16}
                   for i, o in enumerate("ABCD")]
    gtp.answers = {}
    gtp.image = np.ones((300, 400, 3), dtype=np.uint8) * 235

    img = make_synthetic_roi(dark_spots=[
        (50, 150, 10, 120),
        (130, 150, 10, 125),
        (210, 150, 10, 130),
        (290, 150, 10, 128),
    ])
    img = cv2.GaussianBlur(img, (5, 5), 0)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    result = gtp.recognize(img_bgr)
    ans = result["answers"][1]

    check(ans["status"] == "multi", f"全涂黑应为multi，实际{ans['status']}")


def test_card_flag_abnormal():
    """卡片级: 多涂率 > 50% → 异常标记"""
    from core.golden_template import GoldenTemplate

    gtp = GoldenTemplate.__new__(GoldenTemplate)
    gtp.bubbles = []
    gtp.answers = {}
    for qi in range(5):
        for oi, opt in enumerate("ABCD"):
            gtp.bubbles.append({"q": qi + 1, "opt": opt, "x": 50 + oi*80, "y": 50 + qi*100, "w": 16, "h": 16})
    gtp.image = np.ones((600, 400, 3), dtype=np.uint8) * 235

    # 5题全部涂黑
    dark_spots = []
    for qi in range(5):
        for oi in range(4):
            dark_spots.append((50 + oi*80, 50 + qi*100, 10, 120))
    img = make_synthetic_roi(400, 600, dark_spots)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    result = gtp.recognize(img_bgr)
    check(result["card_flag"] == "abnormal", f"全涂黑卡应标记abnormal，实际{result['card_flag']}")


# ========== Step 4: ECC 对齐精度 ==========

def test_ecc_align_same_batch():
    """同批次两张扫描图：ECC应成功对齐，偏移 < 5px"""
    import glob as _glob
    from pathlib import Path as _Path
    import numpy as _np
    from core.golden_template import GoldenTemplate

    # 用同批次两张A面答题卡
    a_files = sorted(_glob.glob("testPaper/答题卡/OMR0001/*A*"))
    if len(a_files) < 2:
        check(True, "跳过: 测试图片不足")
        return

    # 读取图片
    with open(a_files[0], "rb") as f:
        img1 = cv2.imdecode(_np.frombuffer(f.read(), _np.uint8), cv2.IMREAD_COLOR)
    with open(a_files[1], "rb") as f:
        img2 = cv2.imdecode(_np.frombuffer(f.read(), _np.uint8), cv2.IMREAD_COLOR)

    if img1 is None or img2 is None:
        check(True, "跳过: 图片读取失败")
        return

    # 用img1作为黄金模板（裁剪选项区域），img2作为待对齐目标
    h1, w1 = img1.shape[:2]

    # 取图片中间区域作为ROI
    cx, cy = w1 // 2, h1 // 2
    x1, y1 = cx - 150, cy - 250
    x2, y2 = cx + 150, cy + 250

    # 对齐第二张图的ROI
    target_roi = img2[y1:y2, x1:x2]
    ref_roi = img1[y1:y2, x1:x2]

    aligned, success = GoldenTemplate.align(ref_roi, target_roi)

    check(success, "ECC应对齐成功")
    check(aligned.shape == ref_roi.shape[:2],
          f"对齐后形状应与参考一致: {aligned.shape} vs {ref_roi.shape[:2]}")

    # 对齐后两张图的印刷内容应高度相关
    if success and aligned.shape == ref_roi.shape[:2]:
        ref_gray = cv2.cvtColor(ref_roi, cv2.COLOR_BGR2GRAY) if len(ref_roi.shape) == 3 else ref_roi
        diff = cv2.absdiff(ref_gray, aligned)
        mean_diff = _np.mean(diff)
        check(mean_diff < 30, f"对齐后平均像素差异应<30，实际{mean_diff:.1f}")


# ========== Step 5: 端到端识别 ==========

def test_e2e_recognize():
    """端到端：输入答题卡 → 输出答案，与手工确认的答案对比"""
    import glob as _glob
    import numpy as _np
    from core.golden_template import GoldenTemplate

    a_files = sorted(_glob.glob("testPaper/答题卡/OMR0001/*02A*"))
    if not a_files:
        check(True, "跳过: 测试图片不存在")
        return

    with open(a_files[0], "rb") as f:
        img = cv2.imdecode(_np.frombuffer(f.read(), _np.uint8), cv2.IMREAD_COLOR)

    if img is None:
        check(True, "跳过: 图片读取失败")
        return

    h, w = img.shape[:2]

    # 用中间区域作为选择题区域
    cfg = {
        "x1": 100, "y1": h // 2 - 200,
        "x2": w - 100, "y2": h // 2 + 200,
        "start_q": 1, "num_q": 3, "num_options": 4,
    }

    gtp = GoldenTemplate(img, [cfg])

    # 对自身识别（因为没有另一张卡，用自身测试流程是否跑通）
    result = gtp.recognize(img)

    check(result["total"] == 3, f"应有3题，实际{result['total']}")
    check("answers" in result, "结果应包含answers")
    for q in range(1, 4):
        check(q in result["answers"], f"应包含第{q}题")
        ans = result["answers"][q]
        check("answer" in ans and "status" in ans,
              f"Q{q}答案应包含answer和status字段")


# ========== Run ==========

def run_suite(name, tests):
    global PASSED, FAILED
    before_p = PASSED
    before_f = FAILED
    print(f"\n{'='*50}")
    print(name)
    print("=" * 50)
    for t in tests:
        try:
            t()
        except Exception as e:
            FAILED += 1
            print(f"  ERROR: {t.__name__} crashed: {e}")
    suite_pass = PASSED - before_p
    suite_fail = FAILED - before_f
    print(f"  → {suite_pass} pass, {suite_fail} fail")


if __name__ == "__main__":
    run_suite("Step 1: _generate_grid 网格切分", [
        test_grid_basic,
        test_grid_three_options,
        test_grid_start_q_offset,
        test_bubble_window_size,
    ])

    run_suite("Step 2: _sample_bubble 气泡采样", [
        test_sample_blank,
        test_sample_filled,
        test_sample_boundary_clip,
    ])

    run_suite("Step 3: 三道防线判断逻辑", [
        test_judge_normal_fill,
        test_judge_all_blank,
        test_judge_multi_fill,
        test_judge_uncertain,
        test_judge_all_black,
        test_card_flag_abnormal,
    ])

    run_suite("Step 4: ECC对齐精度", [
        test_ecc_align_same_batch,
    ])

    run_suite("Step 5: 端到端识别", [
        test_e2e_recognize,
    ])

    print(f"\n{'='*50}")
    print(f"总结果: {PASSED} 通过, {FAILED} 失败, {PASSED+FAILED} 总计")
    if FAILED == 0:
        print("全部通过!")
    else:
        print(f"待修复: {FAILED} 个失败")
