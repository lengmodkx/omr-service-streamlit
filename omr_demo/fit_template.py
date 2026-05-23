"""
通过少量基准点，自动推算英语答题卡所有选项框坐标
然后可视化验证
"""
import cv2
import numpy as np
import json

img = cv2.imread("../testPaper/911156C_22104651_01A.jpg")
h, w = img.shape[:2]

bubbles = []

def add_bubble(q, opt, x, y):
    bubbles.append({"q": q, "opt": opt, "x": int(x), "y": int(y), "w": 12, "h": 12})

def generate_section(start_q, num_q, options, start_x, start_y, row_h, col_w):
    for i in range(num_q):
        q = start_q + i
        y = start_y + i * row_h
        for j, opt in enumerate(options):
            x = start_x + j * col_w
            add_bubble(q, opt, x, y)

# === A面 选择题区域 - 基于 visual_check.jpg 修正 ===
# X整体右移约12px，Y微调

# 左列（听力1-5，完形16-20，阅读31-35）
generate_section(1,  5, "ABC",     248, 517, 39, 36)
generate_section(16, 5, "ABCD",    248, 727, 39, 36)
generate_section(31, 5, "ABCD",    248, 927, 39, 36)

# 中列（情景交际6-10，完形21-25，阅读36-40）
generate_section(6,  5, "ABC",     447, 517, 39, 36)
generate_section(21, 5, "ABCD",    447, 727, 39, 36)
generate_section(36, 5, "ABCD",    447, 927, 39, 36)

# 右列（情景交际11-15，完形26-30，阅读41-45）
generate_section(11, 5, "ABCDEFG", 717, 517, 39, 35)
generate_section(26, 5, "ABCD",    717, 727, 39, 36)
generate_section(41, 5, "ABCD",    717, 927, 39, 36)

# === 验证可视化 ===
vis = img.copy()
for b in bubbles:
    x, y = b["x"], b["y"]
    cv2.circle(vis, (x, y), 6, (0, 255, 0), 1)
    cv2.putText(vis, b["opt"], (x-3, y+3), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)
    if b["opt"] == "A":
        cv2.putText(vis, str(b["q"]), (x-22, y+3), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)

print(f"共生成 {len(bubbles)} 个选项框")

# 保存模板
template = {
    "name": "english_2026",
    "image_size": {"w": w, "h": h},
    "pages": {
        "A": {
            "barcode": {"x": 730, "y": 220, "w": 280, "h": 80},
            "choice_area": {"x1": 80, "y1": 450, "x2": 1150, "y2": 1060},
            "bubbles": bubbles,
            "subjective": {
                "42": {"x1": 120, "y1": 1100, "x2": 1120, "y2": 1200, "score": 2},
                "43": {"x1": 120, "y1": 1200, "x2": 1120, "y2": 1300, "score": 2},
                "44": {"x1": 120, "y1": 1300, "x2": 1120, "y2": 1400, "score": 2},
                "45": {"x1": 120, "y1": 1400, "x2": 1120, "y2": 1500, "score": 2},
            }
        },
        "B": {
            "subjective": {
                "46-55": {"x1": 120, "y1": 190, "x2": 1120, "y2": 420, "score": 10},
                "56-60": {"x1": 120, "y1": 420, "x2": 1120, "y2": 550, "score": 10},
                "writing": {"x1": 120, "y1": 550, "x2": 1120, "y2": 1650, "score": 15},
            }
        }
    }
}

with open("templates/english.json", "w", encoding="utf-8") as f:
    json.dump(template, f, ensure_ascii=False, indent=2)
print("模板已保存到 templates/english.json")

cv2.imwrite("fit_check.jpg", vis)
print("验证图已保存到 fit_check.jpg")
