"""
黄金模板对比法 — 核心类
用一张正确填涂的答题卡同时充当定位基准和标准答案
"""
import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional


class GoldenTemplate:
    """正确填涂答题卡的黄金模板"""

    def __init__(self, image: np.ndarray, column_configs: List[Dict]):
        self.image = image
        self.column_configs = column_configs
        self.bubbles = []
        self.answers = {}

        for cfg in column_configs:
            col_bubbles = self._generate_grid(cfg)
            self.bubbles.extend(col_bubbles)

        if image is not None:
            self._calibrate_positions(image)
            self._auto_detect_answers(image)

    @staticmethod
    def _generate_grid(cfg: Dict) -> List[Dict]:
        """根据列框配置均匀切分，返回该列所有气泡坐标"""
        x1, y1 = cfg["x1"], cfg["y1"]
        x2, y2 = cfg["x2"], cfg["y2"]
        start_q = cfg["start_q"]
        num_q = cfg["num_q"]
        num_options = cfg["num_options"]

        col_w = (x2 - x1) / num_options
        row_h = (y2 - y1) / num_q
        bubble_w = max(8, int(col_w * 0.5))
        bubble_h = max(8, int(row_h * 0.5))

        bubbles = []
        for qi in range(num_q):
            qn = start_q + qi
            cy = int(y1 + qi * row_h + row_h / 2)
            for oi in range(num_options):
                opt = chr(ord("A") + oi)
                cx = int(x1 + oi * col_w + col_w / 2)
                bubbles.append({
                    "q": qn,
                    "opt": opt,
                    "x": cx,
                    "y": cy,
                    "w": bubble_w,
                    "h": bubble_h,
                })
        return bubbles

    def _auto_detect_answers(self, image: np.ndarray):
        """对黄金模板自身采样暗度，自动识别标准答案"""
        if image is None:
            return

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        h, w = gray.shape

        q_groups = {}
        for b in self.bubbles:
            q = b["q"]
            opt = b["opt"]
            mean_g = self._sample_bubble(blurred, b["x"], b["y"], b["w"], b["h"], w, h, auto_tune=True)
            q_groups.setdefault(q, {})[opt] = mean_g

        self._debug_samples = []  # 存储每题的采样详情
        for q, opts in q_groups.items():
            sorted_opts = sorted(opts.items(), key=lambda x: x[1])
            if len(sorted_opts) < 2:
                self.answers[q] = sorted_opts[0][0] if sorted_opts else None
                continue

            best_opt, best_val = sorted_opts[0]
            second_val = sorted_opts[1][1]
            all_vals = [v for _, v in sorted_opts]
            mean_val = sum(all_vals) / len(all_vals)

            # 相对阈值：基于本题内部各选项的相对暗度，适应不同扫描亮度
            other_mean = sum(v for _, v in sorted_opts[1:]) / (len(sorted_opts) - 1)
            best_delta = other_mean - best_val    # best 比其他选项均值暗多少
            gap = second_val - best_val
            range_val = sorted_opts[-1][1] - sorted_opts[0][1]
            gap_ratio = gap / max(range_val, 10)
            best_vs_brightest = sorted_opts[-1][1] - best_val

            # 防线1: 全部偏亮且接近且最暗项也不暗 → 未填涂
            # best_val>210 防止浅填涂（如205）被误判为空题
            if mean_val > 200 and range_val < 30 and best_val > 210 and best_delta < 8:
                self.answers[q] = None
            elif best_delta > 9 and (gap > 2 or gap_ratio > 0.06 or best_vs_brightest > 15):
                self.answers[q] = best_opt  # best 明显暗于其他 → 选中
            # 防线3b: 浅填涂/阴影区域兼容，降低 best_delta 和 best_val 门槛
            elif best_delta > 6 and gap > 0 and (best_val < 210 or mean_val < 215):
                self.answers[q] = best_opt
            else:
                self.answers[q] = None  # 模糊

            # 收集调试信息
            self._debug_samples.append({
                "q": q, "answer": self.answers[q],
                "best_opt": best_opt, "best_val": round(best_val, 1),
                "gap": round(gap, 1), "mean_val": round(mean_val, 1),
                "opts": {o: round(v, 1) for o, v in sorted_opts},
            })

    def _calibrate_positions(self, image: np.ndarray):
        """局部搜索校准：在初始网格位置附近搜索最暗点，修正到真实气泡圆心。
        限制 y 方向偏移不超过 3px，防止被上下方的题号、印刷线或相邻行吸引。"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        h, w = gray.shape

        for b in self.bubbles:
            # 水平方向：正常搜索，修正选项列偏移
            search_r_x = max(3, int(min(b["w"], b["h"]) * 0.25))
            # 垂直方向：严格限制，防止被题号或相邻行吸走
            search_r_y = 2
            x1 = max(0, b["x"] - search_r_x)
            y1 = max(0, b["y"] - search_r_y)
            x2 = min(w, b["x"] + search_r_x)
            y2 = min(h, b["y"] + search_r_y)
            roi = blurred[y1:y2, x1:x2]
            if roi.size == 0:
                continue
            _, _, min_loc, _ = cv2.minMaxLoc(roi)
            b["x"] = x1 + min_loc[0]
            # y 方向只允许 ±2px 微调，拒绝大幅偏移
            new_y = y1 + min_loc[1]
            if abs(new_y - b["y"]) <= 3:
                b["y"] = new_y

    @staticmethod
    def _sample_bubble(blurred: np.ndarray, bx: int, by: int,
                       bw: int, bh: int, img_w: int, img_h: int,
                       auto_tune: bool = False) -> float:
        """采样气泡中心暗度。小窗口聚焦圆心（避开印刷轮廓圈），返回均值灰度。"""
        cx, cy = bx, by
        if auto_tune:
            # 极小范围搜索：仅气泡中心15%区域，确保永远不触达轮廓圈
            tune_r = max(3, int(min(bw, bh) * 0.15))
            tx1 = max(0, cx - tune_r)
            ty1 = max(0, cy - tune_r)
            tx2 = min(img_w, cx + tune_r)
            ty2 = min(img_h, cy + tune_r)
            troi = blurred[ty1:ty2, tx1:tx2]
            if troi.size > 0:
                _, _, min_loc, _ = cv2.minMaxLoc(troi)
                cx = tx1 + min_loc[0]
                cy = ty1 + min_loc[1]

        # 小窗口：仅取气泡中心30%，轮廓在边缘（离心10+px），不会进入窗口
        half_w = max(4, int(bw * 0.30))
        half_h = max(4, int(bh * 0.30))
        x1 = max(0, cx - half_w)
        y1 = max(0, cy - half_h)
        x2 = min(img_w, cx + half_w)
        y2 = min(img_h, cy + half_h)
        roi = blurred[y1:y2, x1:x2]
        if roi.size == 0:
            return 255.0
        return float(np.mean(roi))

    @staticmethod
    def align(ref_roi: np.ndarray, target_roi: np.ndarray) -> Tuple[np.ndarray, bool]:
        """ECC像素级对齐，返回 (对齐后的ROI, 是否成功)"""
        if ref_roi is None or ref_roi.size == 0 or target_roi is None or target_roi.size == 0:
            return target_roi, False
        
        ref_gray = cv2.cvtColor(ref_roi, cv2.COLOR_BGR2GRAY) if len(ref_roi.shape) == 3 else ref_roi
        tgt_gray = cv2.cvtColor(target_roi, cv2.COLOR_BGR2GRAY) if len(target_roi.shape) == 3 else target_roi

        if ref_gray.shape != tgt_gray.shape:
            tgt_gray = cv2.resize(tgt_gray, (ref_gray.shape[1], ref_gray.shape[0]))
        
        # 安全检查：图像尺寸必须大于高斯核
        if ref_gray.shape[0] < 5 or ref_gray.shape[1] < 5:
            return tgt_gray, False

        ref_blur = cv2.GaussianBlur(ref_gray, (5, 5), 0)
        tgt_blur = cv2.GaussianBlur(tgt_gray, (5, 5), 0)

        warp_matrix = np.eye(2, 3, dtype=np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-6)

        try:
            _, warp_matrix = cv2.findTransformECC(
                ref_blur, tgt_blur, warp_matrix,
                cv2.MOTION_EUCLIDEAN, criteria
            )
            aligned = cv2.warpAffine(
                tgt_blur, warp_matrix, (ref_blur.shape[1], ref_blur.shape[0]),
                flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP
            )
            return aligned, True
        except cv2.error:
            return tgt_blur, False

    def recognize(self, target_img: np.ndarray, debug: bool = False) -> Dict:
        """主入口：全局ECC对齐 → 采样 → 判断 → 对比黄金答案。
        设置 debug=True 打印每题的详细采样值，用于诊断识别失败原因。"""
        # 安全检查：防止零尺寸或无效图像导致 OpenCV 底层崩溃
        if target_img is None or target_img.size == 0:
            return {
                "answers": {},
                "total": 0,
                "empty_count": 0,
                "multi_count": 0,
                "card_flag": "invalid_image",
                "debug_lines": ["错误：输入图像为空或解码失败"],
            }
        
        gray = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY) if len(target_img.shape) == 3 else target_img
        
        # 额外安全检查：图像尺寸必须大于高斯核
        if gray.shape[0] < 5 or gray.shape[1] < 5:
            return {
                "answers": {},
                "total": 0,
                "empty_count": 0,
                "multi_count": 0,
                "card_flag": "invalid_image",
                "debug_lines": [f"错误：图像尺寸过小 {gray.shape}"],
            }
        
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        h, w = gray.shape

        # 全局ECC对齐：将待识别卡片对齐到黄金模板，消除扫描偏移
        if self.image is not None:
            ref_gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY) if len(self.image.shape) == 3 else self.image
            ref_blur = cv2.GaussianBlur(ref_gray, (5, 5), 0)
            if blurred.shape != ref_blur.shape:
                blurred = cv2.resize(blurred, (ref_blur.shape[1], ref_blur.shape[0]))
            warp_matrix = np.eye(2, 3, dtype=np.float32)
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-6)
            try:
                _, warp_matrix = cv2.findTransformECC(
                    ref_blur, blurred, warp_matrix,
                    cv2.MOTION_EUCLIDEAN, criteria
                )
                blurred = cv2.warpAffine(
                    blurred, warp_matrix, (ref_blur.shape[1], ref_blur.shape[0]),
                    flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP
                )
                h, w = ref_blur.shape
            except cv2.error:
                pass  # 对齐失败则使用原始图像

        q_groups = {}
        for b in self.bubbles:
            q = b["q"]
            opt = b["opt"]
            mean_g = self._sample_bubble(blurred, b["x"], b["y"], b["w"], b["h"], w, h,
                                         auto_tune=True)
            q_groups.setdefault(q, {})[opt] = mean_g

        answers = {}
        total = len(q_groups)
        multi_count = 0
        empty_count = 0
        debug_lines = []  # 收集调试信息

        for q, opts in q_groups.items():
            sorted_opts = sorted(opts.items(), key=lambda x: x[1])
            if not sorted_opts:
                answers[q] = {"answer": None, "status": "empty"}
                empty_count += 1
                continue

            best_opt, best_val = sorted_opts[0]
            second_val = sorted_opts[1][1] if len(sorted_opts) > 1 else 255

            all_vals = [v for _, v in sorted_opts]
            gold_ans = self.answers.get(q)

            # 相对阈值：基于本题内部各选项的相对暗度，适应不同扫描亮度
            other_mean = sum(v for _, v in sorted_opts[1:]) / (len(sorted_opts) - 1) if len(sorted_opts) > 1 else 255
            best_delta = other_mean - best_val    # best 比其他选项均值暗多少
            gap = second_val - best_val
            mean_val = sum(all_vals) / len(all_vals)
            range_val = sorted_opts[-1][1] - sorted_opts[0][1]
            gap_ratio = gap / max(range_val, 10)
            best_vs_brightest = sorted_opts[-1][1] - best_val

            # 多涂检测：相对（明显低于其他均值）或 绝对（多个极暗）双保险
            dark_count = sum(1 for _, v in sorted_opts if other_mean - v > 24 and v < 150)
            abs_dark = sum(1 for v in all_vals if v < 140)  # 处理全体偏暗的极端情况，阈值比relative更严格
            # 收集所有填涂的选项（用于多选题显示），与上方检测口径一致
            dark_opts = sorted([o for o, v in sorted_opts if (other_mean - v > 24 and v < 150) or v < 140])

            # 防线1: 全部偏亮且接近且最暗项也不暗（且无明显填涂信号）→ 未填涂
            # best_val>210 防止浅填涂（如205）被误判为空题
            if mean_val > 200 and range_val < 30 and best_val > 210 and best_delta < 8:
                answers[q] = {"answer": None, "status": "empty"}
                empty_count += 1
            # 防线2: 两个以上暗 → 多选（相对检测 + 绝对检测双保险）
            # gap>30 说明最佳选项明显独占，即使 dark_count>=2 也不判多选
            # abs_dark 仅在整体偏暗时(mean<130)启用，避免正常亮度下误判
            elif (dark_count >= 2 and gap <= 30) or (abs_dark >= 2 and gap <= 20 and mean_val < 130):
                answers[q] = {"answer": "".join(dark_opts) if dark_opts else best_opt, "status": "multi"}
                multi_count += 1
            # 防线3: best明显暗于其他 → 选中
            elif best_delta > 9 and (gap > 2 or gap_ratio > 0.06 or best_vs_brightest > 15):
                answers[q] = {"answer": best_opt, "status": "single"}
            # 防线3b: 浅填涂/阴影区域兼容，降低 best_delta 和 best_val 门槛
            elif best_delta > 6 and gap > 1 and (best_val < 210 or mean_val < 215):
                answers[q] = {"answer": best_opt, "status": "single"}
            # 模糊
            else:
                answers[q] = {"answer": None, "status": "uncertain"}

            # 与黄金答案对比（多选题按字符集比较，顺序无关）
            ans = answers[q]["answer"]
            if ans and gold_ans:
                if answers[q]["status"] == "multi":
                    answers[q]["correct"] = (set(ans) == set(gold_ans))
                else:
                    answers[q]["correct"] = (ans == gold_ans)
            else:
                answers[q]["correct"] = None

            # 调试输出：非 single 状态的题目打印详细采样值
            if debug and answers[q]["status"] != "single":
                line = (f"Q{q:>2}: best={best_opt}={best_val:.0f} gap={gap:.0f} "
                        f"mean={mean_val:.0f} darkCnt={dark_count} status={answers[q]['status']} "
                        f"opts=[{','.join(f'{o}:{v:.0f}' for o,v in sorted_opts)}]")
                print(line)
                debug_lines.append(line)

        # 卡片级检查
        card_flag = None
        if total > 0:
            if multi_count / total > 0.5:
                card_flag = "abnormal"
            elif empty_count / total > 0.8:
                card_flag = "suspicious_blank"

        return {
            "answers": answers,
            "total": total,
            "empty_count": empty_count,
            "multi_count": multi_count,
            "card_flag": card_flag,
            "debug_lines": debug_lines,
        }

    def calibrate_answer(self, q: int, correct_opt: str):
        """人工修正标准答案"""
        self.answers[q] = correct_opt
