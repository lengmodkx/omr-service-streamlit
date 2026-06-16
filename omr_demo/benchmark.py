"""
性能对比：processor.py (原有方法) vs standard_template.py (新方法)
"""
import sys
sys.path.insert(0, ".")
import time
import cv2
import numpy as np
from core.processor import CardProcessor
from core.standard_template import StandardTemplate


def benchmark_processor(img_path, blank_path=None):
    """测试原有 processor.py"""
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        print(f"无法读取: {img_path}")
        return None

    proc = CardProcessor("templates/english.json")
    if blank_path:
        blank = cv2.imread(blank_path, cv2.IMREAD_COLOR)
        proc.set_blank_ref(blank)

    # 预热
    _ = proc.recognize_choices(img)

    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        result = proc.recognize_choices(img)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    avg_time = sum(times) / len(times) * 1000  # ms
    return {"time_ms": avg_time, "result": result}


def benchmark_standard(img_path, column_configs):
    """测试标准模板法"""
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        print(f"无法读取: {img_path}")
        return None

    # 用同一张图作为标准模板（仅用于速度测试，实际应使用正确填涂卡）
    stp = StandardTemplate(img, column_configs)

    # 预热
    _ = stp.recognize(img)

    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        result = stp.recognize(img)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    avg_time = sum(times) / len(times) * 1000  # ms

    # 统计识别率
    answers = result["answers"]
    total = len(answers)
    recognized = sum(1 for a in answers.values() if a["status"] == "single")
    empty = sum(1 for a in answers.values() if a["status"] == "empty")
    multi = sum(1 for a in answers.values() if a["status"] == "multi")
    uncertain = sum(1 for a in answers.values() if a["status"] == "uncertain")

    return {
        "time_ms": avg_time,
        "result": result,
        "stats": {
            "total": total,
            "recognized": recognized,
            "empty": empty,
            "multi": multi,
            "uncertain": uncertain,
            "recognition_rate": recognized / total if total else 0,
        }
    }


def build_configs_from_english_json():
    """从 english.json 反推列框配置（近似）"""
    import json
    with open("templates/english.json") as f:
        t = json.load(f)

    bubbles = t["pages"]["A"]["bubbles"]
    # 按题号分组
    q_groups = {}
    for b in bubbles:
        q_groups.setdefault(b["q"], []).append(b)

    # 按选项数分组题目
    option_groups = {}
    for q, bs in q_groups.items():
        num_opt = len(bs)
        option_groups.setdefault(num_opt, []).append(q)

    # 简单起见，构造一个大框覆盖所有气泡
    xs = [b["x"] for b in bubbles]
    ys = [b["y"] for b in bubbles]
    ws = [b["w"] for b in bubbles]
    hs = [b["h"] for b in bubbles]

    # 计算边界
    x1 = min(xs) - max(ws)
    y1 = min(ys) - max(hs)
    x2 = max(xs) + max(ws)
    y2 = max(ys) + max(hs)

    # 由于 english.json 是复杂布局，这里构造一个简化版：
    # 分三大列，每列覆盖不同题号范围
    configs = [
        {"x1": 220, "y1": 500, "x2": 380, "y2": 1100, "start_q": 1, "num_q": 35, "num_options": 3},
        {"x1": 420, "y1": 500, "x2": 580, "y2": 1100, "start_q": 6, "num_q": 35, "num_options": 3},
        {"x1": 690, "y1": 500, "x2": 950, "y2": 1100, "start_q": 11, "num_q": 35, "num_options": 4},
    ]
    return configs


if __name__ == "__main__":
    import glob

    # 找一张英语答题卡 A 面
    candidates = glob.glob("../testPaper/911156C_*A.jpg")
    if not candidates:
        print("未找到测试图片")
        sys.exit(1)

    test_img = candidates[0]
    print(f"测试图片: {test_img}")
    print("=" * 60)

    # 1. 原有方法
    print("\n[1] 原有 processor.py (固定阈值法)")
    r1 = benchmark_processor(test_img)
    if r1:
        print(f"  平均耗时: {r1['time_ms']:.2f} ms")
        answers = r1["result"]
        total = len(answers)
        recognized = sum(1 for a in answers.values() if a and "(多涂)" not in str(a))
        empty = sum(1 for a in answers.values() if a is None)
        multi = sum(1 for a in answers.values() if a and "(多涂)" in str(a))
        print(f"  总题数: {total}")
        print(f"  已识别: {recognized}  空白: {empty}  多涂: {multi}")

    # 2. 标准模板法
    print("\n[2] 标准模板法 (StandardTemplate)")
    configs = build_configs_from_english_json()
    r2 = benchmark_standard(test_img, configs)
    if r2:
        print(f"  平均耗时: {r2['time_ms']:.2f} ms")
        s = r2["stats"]
        print(f"  总题数: {s['total']}")
        print(f"  已识别(single): {s['recognized']}  空白(empty): {s['empty']}")
        print(f"  多涂(multi): {s['multi']}  模糊(uncertain): {s['uncertain']}")
        print(f"  识别率: {s['recognition_rate']*100:.1f}%")

    # 3. 对比总结
    print("\n" + "=" * 60)
    print("对比总结")
    print("=" * 60)
    if r1 and r2:
        speedup = r1["time_ms"] / r2["time_ms"]
        print(f"  速度比: processor / standard = {speedup:.2f}x")
        print(f"  (标准模板法因包含 ECC 对齐，通常更慢，但识别率更高)")
