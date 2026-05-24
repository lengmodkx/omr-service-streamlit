"""
答题卡核心处理模块 - 支持模板差分法OMR识别
"""
import cv2
import numpy as np
import json
import os
from pathlib import Path
from pyzbar.pyzbar import decode
from typing import Dict, List, Tuple, Optional
import pandas as pd

class CardProcessor:
    def __init__(self, template_path: str = "templates/english.json", blank_ref_path: str = None):
        with open(template_path, "r", encoding="utf-8") as f:
            self.template = json.load(f)
        self.ref_w = self.template["image_size"]["w"]
        self.ref_h = self.template["image_size"]["h"]

        # 空白模板参考（用于差分法）
        self.blank_refs = {}
        if blank_ref_path and os.path.exists(blank_ref_path):
            self.blank_refs["A"] = cv2.imread(blank_ref_path, cv2.IMREAD_GRAYSCALE)

    def set_blank_ref(self, img_a: np.ndarray, img_b: np.ndarray = None):
        """设置空白答题卡参考"""
        if img_a is not None:
            self.blank_refs["A"] = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY) if len(img_a.shape)==3 else img_a
        if img_b is not None:
            self.blank_refs["B"] = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY) if len(img_b.shape)==3 else img_b

    def preprocess(self, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """图像预处理：灰度化、高斯模糊、二值化"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape)==3 else img
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary_inv = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
        return blurred, binary_inv

    def scale_coords(self, x: int, y: int, img_w: int, img_h: int) -> Tuple[int, int]:
        """将参考坐标缩放到当前图片尺寸"""
        sx = int(x * img_w / self.ref_w)
        sy = int(y * img_h / self.ref_h)
        return sx, sy

    def _bubble_darkness(self, blurred: np.ndarray, bx: int, by: int, bw: int, bh: int,
                         img_w: int, img_h: int,
                         blank_blurred: np.ndarray = None) -> Tuple[float, float, float]:
        """计算单个气泡中心的暗度指标，返回 (darkness, mean_gray, std_gray)"""
        half_w = max(4, int(bw * 0.55))
        half_h = max(4, int(bh * 0.55))
        x1 = max(0, bx - half_w)
        y1 = max(0, by - half_h)
        x2 = min(img_w, bx + half_w)
        y2 = min(img_h, by + half_h)

        roi = blurred[y1:y2, x1:x2]
        if roi.size == 0:
            return 0.0, 255.0, 0.0

        mean_gray = float(np.mean(roi))
        std_gray = float(np.std(roi))

        if blank_blurred is not None:
            blank_roi = blank_blurred[y1:y2, x1:x2]
            blank_mean = float(np.mean(blank_roi))
            darkness = max(0, blank_mean - mean_gray) / 255.0
        else:
            darkness = max(0, (230 - mean_gray) / 90)
            darkness = min(1.0, darkness)

        return darkness, mean_gray, std_gray

    def detect_barcode(self, img: np.ndarray, page="A") -> Optional[str]:
        """识别条形码（考号）"""
        bc = self.template["pages"][page].get("barcode")
        if not bc:
            return None
        h, w = img.shape[:2]
        x1, y1 = self.scale_coords(bc["x"], bc["y"], w, h)
        x2, y2 = self.scale_coords(bc["x"] + bc["w"], bc["y"] + bc["h"], w, h)
        roi = img[y1:y2, x1:x2]
        barcodes = decode(roi)
        if barcodes:
            return barcodes[0].data.decode("utf-8")
        return None

    def _calibrate_bubble(self, blurred: np.ndarray, bx: int, by: int,
                          img_w: int, img_h: int, search_r: int = 8) -> Tuple[int, int]:
        """在预期坐标附近搜索填涂区域中心 — 以5x5窗口均值找最暗邻域，避开纯黑印刷线和纯白纸张"""
        x1 = max(0, bx - search_r)
        y1 = max(0, by - search_r)
        x2 = min(img_w, bx + search_r)
        y2 = min(img_h, by + search_r)
        roi = blurred[y1:y2, x1:x2]
        if roi.size == 0:
            return bx, by
        # 只以中等灰度像素为窗口中心候选（排除纯黑印刷和纯白纸张）
        mask = (roi > 100) & (roi < 210)
        if np.count_nonzero(mask) < 5:
            return bx, by
        # 计算每个像素的5x5邻域均值，找最暗邻域的中心
        kernel = np.ones((5, 5), dtype=np.float32) / 25.0
        window_means = cv2.filter2D(roi.astype(np.float32), -1, kernel)
        window_means[~mask] = 255.0
        _, _, min_loc, _ = cv2.minMaxLoc(window_means)
        return x1 + min_loc[0], y1 + min_loc[1]

    def recognize_choices(self, img: np.ndarray, page="A", threshold=0.08, debug=False) -> Dict:
        """
        OMR识别选择题 — 5x5窗口均值校准 + 相对灰度比较
        校准找填涂邻域中心（而非单个最暗像素），再比较窗口灰度
        """
        blurred, _ = self.preprocess(img)
        h, w = img.shape[:2]
        bubbles = self.template["pages"][page].get("bubbles", [])

        # 获取空白模板（差分法）
        blank_gray = self.blank_refs.get(page)
        if blank_gray is not None:
            blank_gray = cv2.resize(blank_gray, (w, h))

        results = {}
        for b in bubbles:
            q = b["q"]
            opt = b["opt"]
            bx, by = self.scale_coords(b["x"], b["y"], w, h)
            bw = max(12, int(b["w"] * w / self.ref_w))
            bh = max(12, int(b["h"] * h / self.ref_h))

            # 5x5窗口均值校准：找填涂邻域中心
            cal_x, cal_y = self._calibrate_bubble(blurred, bx, by, w, h, search_r=14)

            # 计算窗口灰度
            half_w = max(5, int(bw * 0.55))
            half_h = max(5, int(bh * 0.55))
            x1 = max(0, cal_x - half_w)
            y1 = max(0, cal_y - half_h)
            x2 = min(w, cal_x + half_w)
            y2 = min(h, cal_y + half_h)
            roi = blurred[y1:y2, x1:x2]
            if roi.size == 0:
                continue

            mean_g = float(np.mean(roi))
            std_g = float(np.std(roi))

            if q not in results:
                results[q] = {}
            results[q][opt] = {
                "mean": round(mean_g, 1),
                "std": round(std_g, 1),
            }

        # 整理结果 — 相对灰度比较
        final = {}
        for q, opts in results.items():
            if not opts:
                final[q] = None
                continue

            sorted_opts = sorted(opts.items(), key=lambda x: x[1]["mean"])
            best_opt, best_info = sorted_opts[0]
            best_mean = best_info["mean"]

            others_avg = sum(o[1]["mean"] for o in sorted_opts[1:]) / (len(sorted_opts) - 1) if len(sorted_opts) > 1 else 255
            diff_from_avg = others_avg - best_mean

            # 阈值逻辑
            min_diff = max(5, threshold * 100)
            if best_mean < 220 and diff_from_avg > min_diff:
                final[q] = best_opt
            elif best_mean < 180 and diff_from_avg > min_diff * 0.4:
                final[q] = best_opt
            else:
                final[q] = None

        if debug:
            return {"answers": final, "debug": results}
        return final

    def crop_subjective(self, img: np.ndarray, student_id: str, page="A", output_dir="output/subjective") -> List[Dict]:
        """裁剪主观题区域"""
        h, w = img.shape[:2]
        subs = self.template["pages"][page].get("subjective", {})
        crops = []

        for q_name, coords in subs.items():
            x1, y1 = self.scale_coords(coords["x1"], coords["y1"], w, h)
            x2, y2 = self.scale_coords(coords["x2"], coords["y2"], w, h)
            crop = img[y1:y2, x1:x2]

            out_path = os.path.join(output_dir, student_id)
            os.makedirs(out_path, exist_ok=True)
            filename = f"{q_name}.jpg"
            filepath = os.path.join(out_path, filename)
            _, buf = cv2.imencode('.jpg', crop)
            with open(filepath, 'wb') as f:
                f.write(buf)

            crops.append({
                "q": q_name,
                "path": os.path.abspath(filepath),
                "score": coords.get("score", 0),
                "student_id": student_id
            })

        return crops

    def recognize_choices_in_region(self, img: np.ndarray, region: Dict, page="A", threshold=0.15, ref_size: tuple = None, debug=False) -> Dict:
        """
        在指定大区域内，使用模板中的 bubbles 做 OMR 识别
        只处理 bubble 中心落在区域内的选项框
        """
        blurred, _ = self.preprocess(img)
        h, w = img.shape[:2]

        # 获取空白模板
        blank_gray = self.blank_refs.get(page)
        blank_blurred = None
        if blank_gray is not None:
            blank_gray = cv2.resize(blank_gray, (w, h))
            blank_blurred = cv2.GaussianBlur(blank_gray, (5, 5), 0)

        # 将区域坐标缩放到当前图片尺寸
        if ref_size:
            ref_w, ref_h = ref_size
            scale_x = w / ref_w
            scale_y = h / ref_h
            rx1 = int(region["x1"] * scale_x)
            ry1 = int(region["y1"] * scale_y)
            rx2 = int(region["x2"] * scale_x)
            ry2 = int(region["y2"] * scale_y)
            bubble_scale_w = ref_w
            bubble_scale_h = ref_h
        else:
            rx1, ry1, rx2, ry2 = region["x1"], region["y1"], region["x2"], region["y2"]
            bubble_scale_w = self.ref_w
            bubble_scale_h = self.ref_h

        bubbles = self.template["pages"][page].get("bubbles", [])

        results = {}
        for b in bubbles:
            q = b["q"]
            opt = b["opt"]
            bx = int(b["x"] * w / bubble_scale_w)
            by = int(b["y"] * h / bubble_scale_h)
            bw = max(10, int(b["w"] * w / bubble_scale_w))
            bh = max(10, int(b["h"] * h / bubble_scale_h))

            # 只处理中心点落在区域内的 bubble
            if not (rx1 <= bx <= rx2 and ry1 <= by <= ry2):
                continue

            darkness, mean_g, std_g = self._bubble_darkness(blurred, bx, by, bw, bh, w, h, blank_blurred)

            if q not in results:
                results[q] = {"filled": [], "darkness": {}}
            results[q]["darkness"][opt] = round(darkness, 3)
            if darkness > threshold:
                results[q]["filled"].append(opt)

        # 整理结果
        final = {}
        for q, data in results.items():
            filled = data["filled"]
            darkness = data["darkness"]
            if len(filled) == 1:
                final[q] = filled[0]
            elif len(filled) == 0:
                final[q] = None
            else:
                best = max(filled, key=lambda x: darkness[x])
                final[q] = best + "(多涂)"

        if debug:
            return {"answers": final, "debug": results}
        return final

    def recognize_choices_custom(self, img: np.ndarray, custom_bubbles: List[Dict], threshold=0.15, ref_size: tuple = None) -> Dict:
        """
        使用用户自定义的选项框位置进行OMR识别
        """
        blurred, _ = self.preprocess(img)
        h, w = img.shape[:2]

        # 获取空白模板
        blank_gray = self.blank_refs.get("A")
        blank_blurred = None
        if blank_gray is not None:
            blank_gray = cv2.resize(blank_gray, (w, h))
            blank_blurred = cv2.GaussianBlur(blank_gray, (5, 5), 0)

        # 计算缩放比例
        if ref_size:
            ref_w, ref_h = ref_size
            scale_x = w / ref_w
            scale_y = h / ref_h
        else:
            scale_x = 1.0
            scale_y = 1.0

        results = {}
        for b in custom_bubbles:
            q = b["q"]
            opt = b["opt"]
            bx = int(b["x"] * scale_x)
            by = int(b["y"] * scale_y)
            bw = max(10, int(b.get("w", 12) * scale_x))
            bh = max(10, int(b.get("h", 12) * scale_y))

            darkness, mean_g, std_g = self._bubble_darkness(blurred, bx, by, bw, bh, w, h, blank_blurred)

            if q not in results:
                results[q] = {"filled": [], "darkness": {}}
            results[q]["darkness"][opt] = round(darkness, 3)
            if darkness > threshold:
                results[q]["filled"].append(opt)

        # 整理结果
        final = {}
        for q, data in results.items():
            filled = data["filled"]
            darkness = data["darkness"]
            if len(filled) == 1:
                final[q] = filled[0]
            elif len(filled) == 0:
                final[q] = None
            else:
                best = max(filled, key=lambda x: darkness[x])
                final[q] = best + "(多涂)"
        return final

    def detect_barcode_in_region(self, img: np.ndarray, region: Dict, ref_size: tuple = None) -> Optional[str]:
        """在指定区域内检测条形码"""
        h, w = img.shape[:2]
        if ref_size:
            ref_w, ref_h = ref_size
            scale_x = w / ref_w
            scale_y = h / ref_h
            x1 = int(region["x1"] * scale_x)
            y1 = int(region["y1"] * scale_y)
            x2 = int(region["x2"] * scale_x)
            y2 = int(region["y2"] * scale_y)
        else:
            x1, y1, x2, y2 = region["x1"], region["y1"], region["x2"], region["y2"]
        x1 = max(0, min(w, x1))
        y1 = max(0, min(h, y1))
        x2 = max(0, min(w, x2))
        y2 = max(0, min(h, y2))
        if x2 <= x1 or y2 <= y1:
            return None
        roi = img[y1:y2, x1:x2]
        barcodes = decode(roi)
        if barcodes:
            return barcodes[0].data.decode("utf-8")
        return None

    def process_pair(self, img_a: np.ndarray, img_b: np.ndarray, student_id: str = "unknown",
                     output_dir="output",
                     manual_regions_a: List[Dict] = None, ref_size_a: tuple = None,
                     manual_regions_b: List[Dict] = None, ref_size_b: tuple = None,
                     custom_bubbles: List[Dict] = None, custom_ref_size: tuple = None) -> Dict:
        """处理A/B面配对"""
        manual_regions_a = manual_regions_a or []
        manual_regions_b = manual_regions_b or []

        has_mc_a = any(r.get("type") == "选择题" for r in manual_regions_a)
        has_mi_a = any(r.get("type") == "个人信息" for r in manual_regions_a)
        has_ms_a = any(r.get("type") == "非选择题" for r in manual_regions_a)
        has_mc_b = any(r.get("type") == "选择题" for r in manual_regions_b)
        has_mi_b = any(r.get("type") == "个人信息" for r in manual_regions_b)
        has_ms_b = any(r.get("type") == "非选择题" for r in manual_regions_b)

        # Step 1: 个人信息区域
        manual_barcode = None

        if has_mi_a:
            for region in manual_regions_a:
                if region.get("type") == "个人信息":
                    bc = self.detect_barcode_in_region(img_a, region, ref_size_a)
                    if bc:
                        manual_barcode = bc
                        student_id = bc

        if has_mi_b:
            for region in manual_regions_b:
                if region.get("type") == "个人信息":
                    bc = self.detect_barcode_in_region(img_b, region, ref_size_b)
                    if bc:
                        manual_barcode = bc
                        student_id = bc

        # Step 2: 模板处理
        barcode = None
        if not has_mi_a:
            barcode = self.detect_barcode(img_a, "A")

        choices = {}
        if custom_bubbles:
            choices = self.recognize_choices_custom(img_a, custom_bubbles, threshold=0.15, ref_size=custom_ref_size)
        elif not has_mc_a:
            choices = self.recognize_choices(img_a, "A")

        subj_a = []
        if not has_ms_a:
            subj_a = self.crop_subjective(img_a, student_id, "A", os.path.join(output_dir, "subjective"))

        subj_b = []
        if not has_ms_b:
            subj_b = self.crop_subjective(img_b, student_id, "B", os.path.join(output_dir, "subjective"))

        # Step 3: 手动区域
        manual_choices = {}
        manual_crops = []

        def _process_regions(img, regions, ref_size, side):
            for region in regions:
                rtype = region.get("type", "非选择题")

                crops = self.crop_by_regions(
                    img, [region],
                    os.path.join(output_dir, "manual_crop"),
                    f"{student_id}_{side}", ref_size
                )
                for c in crops:
                    c["side"] = side
                    c["type"] = rtype
                    c["region_name"] = region["name"]
                    manual_crops.append(c)

                if rtype == "选择题":
                    rc = self.recognize_choices_in_region(img, region, side, ref_size=ref_size)
                    manual_choices.update(rc)

        if manual_regions_a:
            _process_regions(img_a, manual_regions_a, ref_size_a, "A")
        if manual_regions_b:
            _process_regions(img_b, manual_regions_b, ref_size_b, "B")

        final_choices = choices.copy()
        final_choices.update(manual_choices)

        return {
            "student_id": student_id,
            "barcode": manual_barcode or barcode,
            "choices": final_choices,
            "subjective": subj_a + subj_b,
            "manual_crops": manual_crops,
            "choice_count": sum(1 for v in final_choices.values() if v and not str(v).endswith("(多涂)")),
            "multi_count": sum(1 for v in final_choices.values() if v and str(v).endswith("(多涂)")),
            "empty_count": sum(1 for v in final_choices.values() if v is None),
            "total_questions": len(final_choices),
        }

    @staticmethod
    def crop_by_regions(img: np.ndarray, regions: List[Dict], output_dir: str, base_name: str, ref_size: tuple = None) -> List[Dict]:
        """根据手动标定的区域列表批量裁剪图片"""
        h, w = img.shape[:2]
        results = []
        for region in regions:
            if ref_size:
                ref_w, ref_h = ref_size
                scale_x = w / ref_w
                scale_y = h / ref_h
                x1 = int(region["x1"] * scale_x)
                y1 = int(region["y1"] * scale_y)
                x2 = int(region["x2"] * scale_x)
                y2 = int(region["y2"] * scale_y)
            else:
                x1 = region["x1"]
                y1 = region["y1"]
                x2 = region["x2"]
                y2 = region["y2"]

            x1 = max(0, min(w, x1))
            y1 = max(0, min(h, y1))
            x2 = max(0, min(w, x2))
            y2 = max(0, min(h, y2))
            if x2 <= x1 or y2 <= y1:
                continue
            crop = img[y1:y2, x1:x2]
            out_dir = os.path.join(output_dir, base_name)
            os.makedirs(out_dir, exist_ok=True)
            filepath = os.path.join(out_dir, f"{region['name']}.jpg")
            _, buf = cv2.imencode('.jpg', crop)
            with open(filepath, 'wb') as f:
                f.write(buf)
            results.append({
                "name": region["name"],
                "path": os.path.abspath(filepath),
                "x1": x1, "y1": y1, "x2": x2, "y2": y2
            })
        return results

    def process_folder(self, input_dir: str, output_dir: str = "output") -> pd.DataFrame:
        """批量处理文件夹"""
        input_dir = Path(input_dir)
        all_results = []

        a_files = sorted(input_dir.glob("*A.jpg"))

        for a_file in a_files:
            b_file = a_file.with_name(a_file.stem[:-1] + "B.jpg")
            if not b_file.exists():
                print(f"Warning: B面未找到 {b_file.name}, skip")
                continue

            img_a = cv2.imread(str(a_file))
            img_b = cv2.imread(str(b_file))
            if img_a is None or img_b is None:
                continue

            student_id = a_file.stem.split("_")[-1]
            result = self.process_pair(img_a, img_b, student_id, output_dir)
            result["file_a"] = str(a_file.name)
            result["file_b"] = str(b_file.name)
            all_results.append(result)
            print(f"Done: {a_file.name} -> {result['student_id']}, choices {result['choice_count']}/{result['total_questions']}")

        df_data = []
        for r in all_results:
            row = {"StudentID": r["student_id"], "Barcode": r["barcode"],
                   "ChoiceCount": r["choice_count"], "Multi": r["multi_count"], "Empty": r["empty_count"]}
            for q in sorted(r["choices"].keys()):
                row[f"Q{q}"] = r["choices"].get(q, "")
            df_data.append(row)

        df = pd.DataFrame(df_data)
        return df
