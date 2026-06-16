"""
紧急诊断：输出标准模板所有气泡的坐标，检查是否有坐标重合
"""
import sys
sys.path.insert(0, ".")
import cv2
from core.standard_template import StandardTemplate


def diagnose_coords(image_path, column_configs):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        print(f"无法读取: {image_path}")
        return

    stp = StandardTemplate(img, column_configs)
    h, w = img.shape[:2]

    print("=" * 80)
    print("气泡坐标诊断")
    print("=" * 80)

    # 按题号分组
    q_bubbles = {}
    for b in stp.bubbles:
        q_bubbles.setdefault(b["q"], []).append(b)

    issues = []
    for q in sorted(q_bubbles.keys()):
        bubbles = q_bubbles[q]
        coords = [(b["opt"], b["x"], b["y"], b["w"], b["h"]) for b in bubbles]
        x_set = set(x for _, x, _, _, _ in coords)
        y_set = set(y for _, _, y, _, _ in coords)

        # 检查坐标是否超出图片边界
        out_of_bounds = any(x < 0 or x >= w or y < 0 or y >= h for _, x, y, _, _ in coords)
        # 检查同一题的选项是否坐标重合
        x_collision = len(x_set) < len(coords)
        y_collision = len(y_set) < len(coords)

        line = f"Q{q:>2}: " + " | ".join(f"{o}@({x},{y})" for o, x, y, _, _ in coords)
        if out_of_bounds:
            line += "  [超出边界!]"
            issues.append(q)
        elif x_collision or y_collision:
            line += "  [坐标重合!]"
            issues.append(q)
        print(line)

    print("=" * 80)
    if issues:
        print(f"发现 {len(issues)} 题坐标异常: {issues}")
        print("请检查对应列框的 x1,y1,x2,y2 和 num_q,num_options 配置")
    else:
        print("坐标检查通过，无重合或越界")
    print("=" * 80)

    # 额外检查：列框本身
    print("\n列框配置:")
    for i, cfg in enumerate(column_configs):
        x1, y1, x2, y2 = cfg["x1"], cfg["y1"], cfg["x2"], cfg["y2"]
        w_box = x2 - x1
        h_box = y2 - y1
        row_h = h_box / cfg["num_q"]
        col_w = w_box / cfg["num_options"]
        print(f"  列{i+1}: ({x1},{y1})-({x2},{y2}) 宽={w_box}px 高={h_box}px "
              f"start_q={cfg['start_q']} num_q={cfg['num_q']} num_opt={cfg['num_options']} "
              f"row_h={row_h:.1f}px col_w={col_w:.1f}px")
        if h_box < cfg["num_q"]:
            print(f"    ⚠️ 警告：列框高度 {h_box}px < 题目数 {cfg['num_q']}，row_h={row_h:.2f}px < 1，会导致多行坐标重合！")
        if w_box < cfg["num_options"]:
            print(f"    ⚠️ 警告：列框宽度 {w_box}px < 选项数 {cfg['num_options']}，col_w={col_w:.2f}px < 1，会导致多列坐标重合！")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python debug_coords.py <图片路径>")
        print("请先在脚本底部修改 column_configs 为你的实际配置")
        sys.exit(1)

    # ========== 请根据你的实际列框配置修改这里 ==========
    column_configs = [
        # 示例配置，请替换为你在 Streamlit 中使用的实际配置
        # {"x1": 100, "y1": 400, "x2": 300, "y2": 800, "start_q": 1, "num_q": 5, "num_options": 3},
    ]
    # ======================================================

    diagnose_coords(sys.argv[1], column_configs)
