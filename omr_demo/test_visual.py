import sys
sys.path.insert(0, ".")
from core.processor import CardProcessor
import cv2
import numpy as np

proc = CardProcessor("templates/english.json")
blank_a = cv2.imread("../testPaper/911156C_22104651_01A.jpg")
proc.set_blank_ref(blank_a, None)

# 读取测试图
test_a = cv2.imread("../testPaper/911156C_22104652_02A.jpg")
h, w = test_a.shape[:2]
gray_test = cv2.cvtColor(test_a, cv2.COLOR_BGR2GRAY)
_, inv_test = cv2.threshold(gray_test, 180, 255, cv2.THRESH_BINARY_INV)

blank_gray = cv2.resize(cv2.cvtColor(blank_a, cv2.COLOR_BGR2GRAY), (w, h))
_, inv_blank = cv2.threshold(blank_gray, 180, 255, cv2.THRESH_BINARY_INV)

vis = test_a.copy()
bubbles = proc.template["pages"]["A"]["bubbles"]

for b in bubbles:
    q = b["q"]
    opt = b["opt"]
    bx, by = proc.scale_coords(b["x"], b["y"], w, h)
    bw = max(10, int(b["w"] * w / proc.ref_w))
    bh = max(10, int(b["h"] * h / proc.ref_h))
    
    half_w, half_h = bw // 2, bh // 2
    x1 = max(0, bx - half_w)
    y1 = max(0, by - half_h)
    x2 = min(w, bx + half_w)
    y2 = min(h, by + half_h)
    
    roi_test = inv_test[y1:y2, x1:x2]
    roi_blank = inv_blank[y1:y2, x1:x2]
    diff = cv2.subtract(roi_test, roi_blank)
    fill_ratio = np.sum(diff > 50) / roi_test.size
    
    # 根据填涂程度着色
    if fill_ratio > 0.15:
        color = (0, 0, 255)  # 红色 = 填涂
        thickness = 2
    elif fill_ratio > 0.05:
        color = (0, 255, 255)  # 黄色 = 可疑
        thickness = 1
    else:
        color = (0, 255, 0)  # 绿色 = 未填
        thickness = 1
    
    cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)
    if q <= 15:
        cv2.putText(vis, f"{q}{opt}", (x1, y1-2), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)

cv2.imwrite("visual_check.jpg", vis)
print("可视化验证图已保存到 visual_check.jpg")

# 用不同阈值测试
for thresh in [0.05, 0.08, 0.10, 0.12, 0.15]:
    result = proc.recognize_choices(test_a, "A", threshold=thresh)
    filled = sum(1 for v in result.values() if v and not str(v).endswith("(多涂)"))
    multi = sum(1 for v in result.values() if v and str(v).endswith("(多涂)"))
    empty = sum(1 for v in result.values() if v is None)
    print(f"阈值 {thresh}: 已填{filled}, 多涂{multi}, 漏涂{empty}")
    print(f"  Q1-Q5: {[result.get(q) for q in range(1,6)]}")
    print(f"  Q6-Q10: {[result.get(q) for q in range(6,11)]}")
