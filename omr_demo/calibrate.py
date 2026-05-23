"""
交互式标定工具
用法：python calibrate.py
操作：按顺序点击各区域的角点，自动生成坐标模板
"""
import cv2
import numpy as np
import json

current_img = None
points = []
mode = "none"
img_path = "../testPaper/911156C_22104651_01A.jpg"

# 全局结果
all_bubbles = []

def mouse_callback(event, x, y, flags, param):
    global points, current_img, mode, all_bubbles
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        # 画点
        temp = current_img.copy()
        for i, p in enumerate(points):
            cv2.circle(temp, p, 4, (0, 0, 255), -1)
            cv2.putText(temp, str(i+1), (p[0]+5, p[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.imshow("Calibrate", temp)
        print(f"  点击点 #{len(points)}: ({x}, {y})")

def generate_grid(p1, p2, p3, p4, rows, cols):
    """四边形内插生成网格点"""
    pts = []
    for r in range(rows):
        for c in range(cols):
            u = c / (cols - 1) if cols > 1 else 0
            v = r / (rows - 1) if rows > 1 else 0
            # 双线性插值
            top = (p1[0] + u*(p2[0]-p1[0]), p1[1] + u*(p2[1]-p1[1]))
            bot = (p4[0] + u*(p3[0]-p4[0]), p4[1] + u*(p3[1]-p4[1]))
            x = top[0] + v*(bot[0]-top[0])
            y = top[1] + v*(bot[1]-top[1])
            pts.append((int(x), int(y)))
    return pts

def main():
    global current_img, points, mode
    img = cv2.imread(img_path)
    current_img = img.copy()
    h, w = img.shape[:2]
    
    cv2.namedWindow("Calibrate", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Calibrate", 900, 1200)
    cv2.setMouseCallback("Calibrate", mouse_callback)
    
    print("=" * 60)
    print("英语答题卡标定工具")
    print("=" * 60)
    print("操作说明：")
    print("1. 在图片上点击各区域的4个角点（左上→右上→右下→左下）")
    print("2. 按 'n' 确认当前区域，输入行列数")
    print("3. 按 'r' 撤销上一个点")
    print("4. 按 's' 保存模板")
    print("5. 按 'q' 退出")
    print("=" * 60)
    
    regions = []
    
    while True:
        cv2.imshow("Calibrate", current_img)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        
        elif key == ord('r') and points:
            points.pop()
            temp = img.copy()
            for i, p in enumerate(points):
                cv2.circle(temp, p, 4, (0, 0, 255), -1)
                cv2.putText(temp, str(i+1), (p[0]+5, p[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            current_img = temp
            print(f"  撤销，剩余 {len(points)} 个点")
        
        elif key == ord('n'):
            if len(points) != 4:
                print("错误：需要恰好4个点才能生成网格")
                continue
            
            name = input("区域名称 (如 listening, section1, etc.): ").strip()
            rows = int(input("行数 (题目数): "))
            cols = int(input("列数 (每题选项数): "))
            start_q = int(input("起始题号: "))
            options = input("选项字母 (如 ABC 或 ABCDEFG): ").strip().upper()
            
            grid = generate_grid(points[0], points[1], points[2], points[3], rows, cols)
            
            region = {
                "name": name,
                "start_q": start_q,
                "rows": rows,
                "cols": cols,
                "options": list(options),
                "points": points.copy(),
                "bubbles": []
            }
            
            for i, (x, y) in enumerate(grid):
                q_no = start_q + i // cols
                opt_idx = i % cols
                opt = options[opt_idx] if opt_idx < len(options) else "?"
                region["bubbles"].append({
                    "q": q_no,
                    "opt": opt,
                    "x": x,
                    "y": y
                })
                # 可视化
                cv2.circle(current_img, (x, y), 6, (255, 0, 0), 1)
                cv2.putText(current_img, opt, (x-4, y+3), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1)
            
            regions.append(region)
            points = []
            print(f"  已添加区域 '{name}': {len(grid)} 个框")
        
        elif key == ord('s'):
            template = {
                "name": "english_answer_card",
                "image_size": {"w": w, "h": h},
                "regions": regions
            }
            with open("templates/english_calibrated.json", "w", encoding="utf-8") as f:
                json.dump(template, f, ensure_ascii=False, indent=2)
            print(f"模板已保存到 templates/english_calibrated.json，共 {len(regions)} 个区域，{sum(len(r['bubbles']) for r in regions)} 个框")
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
