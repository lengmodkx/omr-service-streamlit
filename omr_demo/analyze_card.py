"""
分析答题卡模板 - 模板匹配法探测选项框
"""
import cv2
import numpy as np
import json

def detect_bubbles_template(image_path, roi=None):
    """使用模板匹配探测方框"""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    if roi:
        x1, y1, x2, y2 = roi
        gray = gray[y1:y2, x1:x2]
        offset_x, offset_y = x1, y1
    else:
        offset_x, offset_y = 0, 0
    
    # 二值化（正常：黑字白底）
    _, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    
    # 创建一个模板：一个小方框（模拟选项框）
    # 从图片看，选项框大约是 12x12 像素
    template_size = 14
    template = np.ones((template_size, template_size), dtype=np.uint8) * 255
    cv2.rectangle(template, (0, 0), (template_size-1, template_size-1), 0, 1)
    
    # 模板匹配
    res = cv2.matchTemplate(binary, template, cv2.TM_CCOEFF_NORMED)
    threshold = 0.5
    loc = np.where(res >= threshold)
    
    # 非极大值抑制
    points = list(zip(*loc[::-1]))
    bubbles = []
    for pt in points:
        x, y = pt[0] + template_size//2, pt[1] + template_size//2
        # 检查是否太靠近已有的点
        too_close = False
        for b in bubbles:
            if abs(b["x"] - x) < 10 and abs(b["y"] - y) < 10:
                too_close = True
                break
        if not too_close:
            bubbles.append({"x": x + offset_x, "y": y + offset_y, "w": template_size, "h": template_size})
    
    return bubbles

def cluster_by_rows(bubbles, y_tolerance=15):
    if not bubbles:
        return []
    bubbles_sorted = sorted(bubbles, key=lambda b: b["y"])
    rows = []
    current_row = [bubbles_sorted[0]]
    
    for b in bubbles_sorted[1:]:
        if abs(b["y"] - current_row[0]["y"]) <= y_tolerance:
            current_row.append(b)
        else:
            rows.append(sorted(current_row, key=lambda x: x["x"]))
            current_row = [b]
    rows.append(sorted(current_row, key=lambda x: x["x"]))
    return rows

def visualize(image_path, bubbles, save_path="analyzed.jpg"):
    img = cv2.imread(image_path)
    for i, b in enumerate(bubbles):
        x, y = b["x"], b["y"]
        w, h = b["w"], b["h"]
        cv2.rectangle(img, (x - w//2, y - h//2), (x + w//2, y + h//2), (0, 255, 0), 1)
        cv2.putText(img, str(i), (x - 8, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 0, 0), 1)
    cv2.imwrite(save_path, img)
    print(f"可视化结果已保存: {save_path}, 共标记 {len(bubbles)} 个框")

if __name__ == "__main__":
    img_a = "../testPaper/911156C_22104651_01A.jpg"
    
    # 选择题大区域
    roi_choice = (70, 340, 1160, 1060)
    
    print("正在用模板匹配分析 A面...")
    bubbles = detect_bubbles_template(img_a, roi=roi_choice)
    print(f"探测到 {len(bubbles)} 个选项框")
    
    rows = cluster_by_rows(bubbles, y_tolerance=18)
    print(f"聚类为 {len(rows)} 行")
    
    for i, row in enumerate(rows):
        xs = [b["x"] for b in row]
        print(f"  第{i+1:2d}行: {len(row)}个框, Y≈{row[0]['y']:4d}, X: {min(xs):4d}~{max(xs):4d}")
    
    visualize(img_a, bubbles, "analyzed_a.jpg")
    
    with open("detected_bubbles.json", "w", encoding="utf-8") as f:
        json.dump({"total": len(bubbles), "rows": len(rows), 
                   "detail": [{"row": i+1, "count": len(r), "y": r[0]["y"], 
                               "x_range": [b["x"] for b in r]} for i, r in enumerate(rows)]}, 
                  f, ensure_ascii=False, indent=2)
