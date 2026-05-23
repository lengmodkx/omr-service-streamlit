import sys
sys.path.insert(0, ".")
from core.processor import CardProcessor
import cv2

proc = CardProcessor("templates/english.json")

# 用第一张空白答题卡作为模板参考
blank_a = cv2.imread("../testPaper/911156C_22104651_01A.jpg")
blank_b = cv2.imread("../testPaper/911156C_22104651_01B.jpg")
proc.set_blank_ref(blank_a, blank_b)

# 测试第二张答题卡
test_a = cv2.imread("../testPaper/911156C_22104652_02A.jpg")
test_b = cv2.imread("../testPaper/911156C_22104652_02B.jpg")

result = proc.process_pair(test_a, test_b, "test_02")

print(f"Student: {result['student_id']}")
print(f"Barcode: {result['barcode']}")
print(f"Choices: {result['choice_count']}/45")
print(f"Empty: {result['empty_count']}, Multi: {result['multi_count']}")
print(f"Subjective crops: {len(result['subjective'])}")
print("\nFirst 15 questions:")
for q in range(1, 16):
    ans = result['choices'].get(q)
    print(f"  Q{q}: {ans}")
