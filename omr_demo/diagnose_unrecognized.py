"""
诊断脚本：分析黄金模板未识别题目的具体失败原因
打印每道题的采样值、各阈值判断结果
"""
import sys
sys.path.insert(0, ".")
import numpy as np
import cv2
from core.golden_template import GoldenTemplate


def diagnose(golden_image_path, column_configs):
    """详细诊断每道题的识别过程"""
    img = cv2.imread(golden_image_path, cv2.IMREAD_COLOR)
    if img is None:
        print(f"无法读取: {golden_image_path}")
        return

    gtp = GoldenTemplate(img, column_configs)

    # 重新跑一次 _auto_detect_answers 的详细分析
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    h, w = gray.shape

    # 先校准位置
    for b in gtp.bubbles:
        search_r = max(3, int(min(b["w"], b["h"]) * 0.20))
        x1 = max(0, b["x"] - search_r)
        y1 = max(0, b["y"] - search_r)
        x2 = min(w, b["x"] + search_r)
        y2 = min(h, b["y"] + search_r)
        roi = blurred[y1:y2, x1:x2]
        if roi.size == 0:
            continue
        _, _, min_loc, _ = cv2.minMaxLoc(roi)
        b["x"] = x1 + min_loc[0]
        b["y"] = y1 + min_loc[1]

    # 采样
    q_groups = {}
    for b in gtp.bubbles:
        q = b["q"]
        opt = b["opt"]
        mean_g = gtp._sample_bubble(blurred, b["x"], b["y"], b["w"], b["h"], w, h)
        q_groups.setdefault(q, {})[opt] = mean_g

    print("=" * 80)
    print("逐题诊断分析")
    print("=" * 80)

    unrecognized = []
    for q in sorted(q_groups.keys()):
        opts = q_groups[q]
        sorted_opts = sorted(opts.items(), key=lambda x: x[1])

        best_opt, best_val = sorted_opts[0]
        second_val = sorted_opts[1][1] if len(sorted_opts) > 1 else 255

        all_vals = [v for _, v in sorted_opts]
        other_mean = sum(v for _, v in sorted_opts[1:]) / (len(sorted_opts) - 1) if len(sorted_opts) > 1 else 255

        best_delta = other_mean - best_val
        gap = second_val - best_val
        mean_val = sum(all_vals) / len(all_vals)
        range_val = sorted_opts[-1][1] - sorted_opts[0][1]
        gap_ratio = gap / max(range_val, 10)

        best_vs_brightest = sorted_opts[-1][1] - best_val

        # 各检查结果（与 golden_template.py 主逻辑严格一致）
        empty_check = mean_val > 200 and range_val < 30 and best_val > 210 and best_delta < 8
        dark_count = sum(1 for _, v in sorted_opts if other_mean - v > 24 and v < 175)
        abs_dark = sum(1 for v in all_vals if v < 160)
        multi_check = dark_count >= 2 or abs_dark >= 2
        best_delta_check = best_delta > 9
        gap_check = gap > 2
        gap_ratio_check = gap_ratio > 0.06
        bvb_check = best_vs_brightest > 15
        selection_check = best_delta_check and (gap_check or gap_ratio_check or bvb_check)
        # 浅填涂/阴影兼容
        dim_compat = best_delta > 6 and gap > 0 and (best_val < 210 or mean_val < 215)

        # 判断结果
        if empty_check:
            status = "EMPTY"
        elif multi_check:
            status = "MULTI"
        elif selection_check or dim_compat:
            status = "SINGLE"
        else:
            status = "UNCERTAIN"

        if status != "SINGLE":
            unrecognized.append(q)
            print(f"\n{'─' * 60}")
            print(f"Q{q:>2} 状态: {status}  |  gold_ans: {gtp.answers.get(q)}")
            print(f"  采样值: {', '.join(f'{o}={v:.1f}' for o, v in sorted_opts)}")
            print(f"  best={best_opt}={best_val:.1f}  second={second_val:.1f}  mean={mean_val:.1f}  range={range_val:.1f}")
            print(f"  best_delta={best_delta:.1f}(需>9 或 >6+浅填涂)  gap={gap:.1f}(需>2 或 >0+浅填涂)  gap_ratio={gap_ratio:.2f}(需>0.06)  bvb={best_vs_brightest:.1f}(需>15)")
            print(f"  空题检查: mean>200={mean_val>200} range<30={range_val<30} best>210={best_val>210} best_delta<8={best_delta<8} → {empty_check}")
            print(f"  多涂检查: dark_count={dark_count}(需>=2) abs_dark={abs_dark}(需>=2) → {multi_check}")
            print(f"  选中检查: best_delta>9={best_delta_check} (gap>2={gap_check} or gap_ratio>0.06={gap_ratio_check} or bvb>15={bvb_check}) → {selection_check}")
            print(f"  浅填涂兼容: best_delta>6={best_delta>6} gap>0={gap>0} (best<210={best_val<210} or mean<215={mean_val<215}) → {dim_compat}")

            # 诊断失败原因
            print(f"  🔍 失败原因: ", end="")
            reasons = []
            if empty_check:
                reasons.append("被空题检查拦截（整体偏亮+均匀+最暗项也亮）")
            if multi_check:
                reasons.append("被多涂检查拦截")
            if not selection_check and not dim_compat:
                if not best_delta_check and not (best_delta > 6 and gap > 0):
                    reasons.append(f"best_delta={best_delta:.1f}≤9（且不满足浅填涂兼容），填涂不够明显")
                elif best_delta > 6 and gap > 0:
                    reasons.append(f"浅填涂兼容未满足: best_val={best_val:.1f}≥210 且 mean={mean_val:.1f}≥215")
                else:
                    reasons.append(f"gap={gap:.1f}≤2 且 gap_ratio={gap_ratio:.2f}≤0.06 且 bvb={best_vs_brightest:.1f}≤15（最佳与次佳/最亮差距不够大）")
            print(" | ".join(reasons))

    print(f"\n{'=' * 80}")
    print(f"总结: {len(unrecognized)}/{len(q_groups)} 题未识别: {unrecognized}")
    print("=" * 80)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python diagnose_unrecognized.py <图片路径>")
        print("或修改 main 中的 column_configs 后运行")
        sys.exit(1)

    # 示例配置 - 请根据实际列框修改
    # column_configs = [
    #     {"x1": 100, "y1": 200, "x2": 400, "y2": 700, "start_q": 1, "num_q": 10, "num_options": 4},
    # ]
    # diagnose(sys.argv[1], column_configs)
