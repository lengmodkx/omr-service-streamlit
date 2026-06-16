"""
可视化诊断：在标准模板上画出网格线、题号、采样窗口
用于确认 column_configs 的框选是否准确
"""
import sys
sys.path.insert(0, ".")
import cv2
import numpy as np
from core.standard_template import StandardTemplate


def visualize(template_image_path, column_configs, output_path="grid_debug.jpg"):
    img = cv2.imread(template_image_path, cv2.IMREAD_COLOR)
    if img is None:
        print(f"无法读取: {template_image_path}")
        return

    stp = StandardTemplate(img, column_configs)
    vis = img.copy()
    h, w = vis.shape[:2]

    # 按题号分组
    q_bubbles = {}
    for b in stp.bubbles:
        q_bubbles.setdefault(b["q"], []).append(b)

    # 画每个气泡的采样窗口（绿色圆圈）和题号（红色）
    for q, bubbles in sorted(q_bubbles.items()):
        # 题号位置：取该题第一个选项的左上方
        first_b = min(bubbles, key=lambda x: x["opt"])
        tx = first_b["x"] - 25
        ty = first_b["y"] - 10
        cv2.putText(vis, f"Q{q}", (tx, ty), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 255), 2)

        for b in bubbles:
            cx, cy = b["x"], b["y"]
            r = max(6, b["w"] // 2)
            # 采样窗口：绿色空心圆
            cv2.circle(vis, (cx, cy), r, (0, 255, 0), 2)
            # 中心点：红色十字
            cv2.line(vis, (cx - 4, cy), (cx + 4, cy), (0, 0, 255), 2)
            cv2.line(vis, (cx, cy - 4), (cx, cy + 4), (0, 0, 255), 2)
            # 选项字母：蓝色小字
            cv2.putText(vis, b["opt"], (cx + r + 2, cy + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

    # 画列框边界（黄色虚线）
    for cfg in column_configs:
        x1, y1, x2, y2 = cfg["x1"], cfg["y1"], cfg["x2"], cfg["y2"]
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 2)
        # 画内部水平分隔线（每道题的分隔）
        num_q = cfg["num_q"]
        row_h = (y2 - y1) / num_q
        for qi in range(1, num_q):
            ly = int(y1 + qi * row_h)
            cv2.line(vis, (x1, ly), (x2, ly), (0, 255, 255), 1)

    cv2.imwrite(output_path, vis)
    print(f"已保存可视化诊断图: {output_path}")
    print("检查要点：")
    print("  1. 绿色圆圈应在填涂区域的中心，不应在题号或空白处")
    print("  2. 黄色虚线框应紧贴该列所有选项的外边界（不要包含题号）")
    print("  3. 内部黄色实线应为每道题的分隔线，应穿过选项之间的间隙")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python visualize_grid.py <标准模板图片路径>")
        print("请先在脚本底部修改 column_configs 为你的实际配置")
        sys.exit(1)

    # ========== 请根据你的实际列框配置修改这里 ==========
    # 示例配置（4列上半部分 + 4列下半部分 + 2列底部）
    # 请根据你在 Streamlit 中生成的实际配置替换
    column_configs = [
        # 上半部分 3选项区域（请替换为实际坐标）
        # {"x1": 100, "y1": 400, "x2": 300, "y2": 800, "start_q": 1, "num_q": 5, "num_options": 3},
    ]
    # ======================================================

    visualize(sys.argv[1], column_configs)
