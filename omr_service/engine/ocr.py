"""个人信息 OCR 模块

封装 PaddleOCR 初始化与识别接口，输入为完整答题卡图片 + 区域框，
输出为字段标识、文本内容、置信度。
"""

import logging
import re
import threading
from typing import Any, Dict, List, Optional, Tuple



import cv2
import numpy as np

logger = logging.getLogger(__name__)

# 「条码标签区」字段标识（框题管理中单独框选印刷标签条，与前端/Java 约定一致）
BARCODE_LABEL_FIELD = "barcode_label"

# 手写姓名字段标识（讯飞手写识别印证通道作用于这些字段）
NAME_FIELD_IDS = {"name", "姓名", "考生姓名"}

# 姓名标签/字段残留词（含 OCR 常见误读变体）：清洗手写姓名与标签条姓名时都要跳过。
# 姓名标签行常被 OCR 误读（如“考生姓名”→“考生娃名”），仅精确删除标签无法清理干净，
# 故按“以残留词开头”判断后跳过（见 clean_handwritten_name）。
NAME_STOP_WORDS = frozenset({
    "姓名", "名字", "考生", "班级", "斑级", "考场", "老场", "座位", "座号",
    "准考证", "性别", "民族", "学校", "年级",
})


def clean_handwritten_name(text: str) -> str:
    """清洗手写识别结果中的“姓名”标签与符号，提取 2-8 字中文名（含间隔号）。

    标签本身会被 OCR 误读（如“考生姓名”→“考生娃名”、“考生 张三”），因此除精确删除
    标签词外，还要遍历文本中的中文片段，跳过姓名/字段残留词（含以其开头的误读变体），
    取第一个有效片段，避免把标签残留当成姓名（会以 0.99 置信度覆盖正确的 Paddle 结果）。
    """
    t = re.sub(r"考生姓名|姓名|名字", " ", text or "")
    for m in re.finditer(r"[\u4e00-\u9fa5·]{2,8}", t):
        token = m.group(0)
        if any(token.startswith(word) for word in NAME_STOP_WORDS):
            continue
        return token
    return ""

# 常见准考证号/考生号类字段标识，识别后需要兜底提取纯数字
_STUDENT_ID_FIELDS = {
    "student_no",
    "admission_no",
    "exam_no",
    "candidate_no",
    "id_number",
    "准考证号",
    "准考证号码",
    "考号",
    "学号",
    "考生号",
    "报名号",
}


class PersonalInfoOcr:
    """基于 PaddleOCR 的个人信息识别器（懒加载单例，线程安全）"""

    _instance: Optional["PersonalInfoOcr"] = None
    _ocr_engine: Any = None
    _init_lock = threading.Lock()
    # PaddleOCR 推理调用串行锁：Paddle 推理引擎不是线程安全的，
    # 批量任务多 worker 并发调用 engine.ocr() 会偶发异常/返回空，导致个别答题卡个人信息识别为空
    _ocr_call_lock = threading.Lock()
    # 初始化进行中标记：PaddleOCR 初始化（import/模型加载）可能永久挂起，
    # 若在持锁状态下挂起会导致后续所有调用卡在 _init_lock 上（每个任务泄漏一个线程+延迟超时）。
    # 改为锁内只改标记、锁外执行初始化；标记为 True 时后续调用直接按不可用处理（秒回），不等待。
    _init_in_progress: bool = False
    # 初始化失败一次性标记：失败后不再重复尝试 import paddleocr（高频任务下避免噪音+开销）
    _init_failed: bool = False

    def __new__(cls) -> "PersonalInfoOcr":
        if cls._instance is None:
            with cls._init_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def _get_engine(self) -> Any:
        """懒加载 PaddleOCR 引擎，初始化失败/挂起时返回 None 并记录日志（线程安全）

        注意：初始化在锁外执行；若初始化线程挂起（import/模型加载卡死），
        后续调用通过 _init_in_progress 标记直接返回 None，不阻塞任务主流程。
        """
        if self._ocr_engine is not None:
            return self._ocr_engine
        if self._init_failed:
            return None
        with self._init_lock:
            # 双重检查，防止多个线程重复初始化
            if self._ocr_engine is not None:
                return self._ocr_engine
            if self._init_in_progress:
                logger.warning("PaddleOCR 初始化进行中(可能挂起)，本次跳过个人信息 OCR")
                return None
            self._init_in_progress = True
        try:
            from paddleocr import PaddleOCR

            self._ocr_engine = PaddleOCR(
                use_angle_cls=True,
                lang="ch",
                show_log=False,
                enable_mkldnn=False,
            )
            logger.info("PaddleOCR 初始化成功")
        except Exception as e:
            logger.warning("PaddleOCR 初始化失败，个人信息 OCR 将不可用: %s", e)
            self._ocr_engine = None
            self._init_failed = True
        finally:
            # 非挂起路径（正常/异常）复位标记；挂起时保持 True，避免后续重复尝试
            self._init_in_progress = False
        return self._ocr_engine

    def recognize(
        self,
        image: np.ndarray,
        regions: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """识别多个个人信息区域

        Args:
            image: 完整答题卡图片（BGR）
            regions: 区域列表，每项包含 field, x1, y1, x2, y2

        Returns:
            识别结果列表，每项包含 field, value, confidence
        """
        engine = self._get_engine()
        if engine is None:
            return [
                {"field": r.get("field", ""), "value": "", "confidence": 0.0}
                for r in regions
            ]

        results = []
        for region in regions:
            field = region.get("field", "")
            crop = self._crop(image, region)
            if crop.size == 0:
                results.append({"field": field, "value": "", "confidence": 0.0})
                continue
            preprocessed = self._preprocess(crop)
            value, confidence = self._recognize_one(engine, preprocessed)
            # 准考证号等字段：OCR 容易把标签和数字连在一起；同时尝试条码解码兜底
            if field and self._is_student_id_field(field):
                barcode_texts = self._try_decode_barcodes(crop)
                if barcode_texts:
                    combined = (value + "\n" + "\n".join(barcode_texts)).strip()
                    logger.info(
                        "[ocr] 字段 %s 条码解码结果: %s, 与 OCR 合并: '%s'",
                        field, barcode_texts, combined,
                    )
                    value = combined
                extracted = self._extract_id_number(value)
                if extracted and extracted != value:
                    logger.info(
                        "[ocr] 字段 %s 原值 '%s' 包含非数字内容，提取准考证号: %s",
                        field, value, extracted,
                    )
                    value = extracted
            results.append({
                "field": field,
                "value": value,
                "confidence": round(confidence, 4),
            })
        return results

    @staticmethod
    def _try_decode_barcodes(image: np.ndarray) -> List[str]:
        """尝试识别图片中的条形码/二维码，返回解码字符串列表"""
        try:
            from pyzbar.pyzbar import decode

            barcodes = decode(image)
            results = []
            for barcode in barcodes:
                data = barcode.data.decode("utf-8") if barcode.data else ""
                if data:
                    results.append(data)
            return results
        except Exception as e:
            # zbar 原生库缺失（Windows 常见）或解码失败时给出可见告警，便于现场排查
            logger.warning("条码解码不可用/异常，将依赖 OCR 数字兜底: %s", e)
            return []

    @staticmethod
    def _decode_barcodes_robust(image: np.ndarray) -> List[str]:
        """条码解码增强版：对同一张图依次尝试原图、灰度、Otsu 二值化、2 倍放大，任一成功即止。"""
        return [text for text, _ in PersonalInfoOcr._locate_barcodes(image)]

    @staticmethod
    def _locate_barcodes(image: np.ndarray) -> List[tuple]:
        """定位并解码条码，返回 [(文本, rect)] 列表，rect=(x, y, w, h) 相对传入图像。

        扫描件对比度低/条码偏小时，单一路径解码容易失败，多预处理变体可显著提升召回；
        位置信息用于“条码锚定印刷标签条 OCR”。
        """
        try:
            from pyzbar.pyzbar import decode
        except Exception as e:
            logger.warning("pyzbar/zbar 不可用，条码解码跳过（将依赖 OCR 数字兜底）: %s", e)
            return []

        variants = [image]
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            variants.append(gray)
            _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            variants.append(otsu)
            # 整页大图兜底时不再放大，避免内存/耗时爆炸；放大变体的坐标需换算回原图
            if gray.shape[0] * gray.shape[1] < 6_000_000:
                scaled = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
                variants.append((scaled, 0.5))
        except Exception as e:
            logger.warning("条码解码预处理异常，仅使用原图: %s", e)

        for variant in variants:
            # 最后一个元素可能是 (放大图, 坐标缩放比例)
            if isinstance(variant, tuple):
                img_variant, coord_scale = variant
            else:
                img_variant, coord_scale = variant, 1.0
            try:
                barcodes = decode(img_variant)
            except Exception as e:
                logger.warning("条码解码异常: %s", e)
                continue
            results = []
            for barcode in barcodes:
                data = barcode.data.decode("utf-8") if barcode.data else ""
                if not data:
                    continue
                rect = getattr(barcode, "rect", None)
                if rect is not None:
                    box = (
                        int(rect.left * coord_scale), int(rect.top * coord_scale),
                        int(rect.width * coord_scale), int(rect.height * coord_scale),
                    )
                else:
                    box = None
                results.append((data, box))
            if results:
                return results
        return []

    @staticmethod
    def _crop_label_strip(image: np.ndarray, rect: tuple, offset: tuple) -> Optional[np.ndarray]:
        """以条码位置为锚点，裁出其上下印刷标签条（上方姓名行 + 下方考号/考场/座位行）。

        印刷标签字体小，整块考生信息区 OCR 时容易丢行；单独裁出放大识别可显著提高召回。
        """
        rx, ry, rw, rh = rect
        ox, oy = offset
        px, py = ox + rx, oy + ry
        h, w = image.shape[:2]
        # 横向各扩 0.4 倍条码宽（标签文字一般略宽于条码），纵向扩到上/下各一行文字
        x1 = max(0, int(px - 0.4 * rw))
        x2 = min(w, int(px + rw + 0.4 * rw))
        y1 = max(0, int(py - 1.1 * rh))
        y2 = min(h, int(py + rh + 1.0 * rh))
        if x2 <= x1 or y2 <= y1:
            return None
        return image[y1:y2, x1:x2]

    @staticmethod
    def _is_student_id_field(field: str) -> bool:
        """判断字段是否为准考证号/考生号类字段"""
        if not field:
            return False
        key = field.strip().lower().replace(" ", "_")
        return key in _STUDENT_ID_FIELDS or any(
            token in key for token in ("student", "admission", "exam", "candidate", "准考证", "考号", "学号", "考生号", "报名号")
        )

    @staticmethod
    def _extract_id_number(text: str) -> str:
        """从文本中提取最长的 6-20 位数字串，用于准考证号兜底"""
        if not text:
            return ""
        nums = re.findall(r"\d{6,20}", text)
        return max(nums, key=len) if nums else ""

    @staticmethod
    def _pick_exam_no_barcode(
        barcode_hits: List[tuple],
        region: Optional[Dict[str, Any]] = None,
    ) -> str:
        """从条码命中列表中选出考号条码。

        整页兜底解码时页面可能有多个条码/二维码，且 pyzbar 的命中顺序与标签区位置无关，
        直接取第一个会误把其它条码当考号。优先取条码中心落在标签区内的命中
        （rect 均为整页坐标，故仅在整页兜底时传 region），再做数字校验，取第一个能
        提取 6-20 位数字串的命中；全部无法提取数字时保留原始解码值（兼容字母数字混排考号）。
        """
        if not barcode_hits:
            return ""
        candidates = barcode_hits
        if region is not None:
            rx1, ry1 = int(region.get("x1", 0)), int(region.get("y1", 0))
            rx2, ry2 = int(region.get("x2", 0)), int(region.get("y2", 0))
            inside = [
                hit for hit in barcode_hits
                if hit[1] and rx1 <= hit[1][0] + hit[1][2] / 2 <= rx2
                and ry1 <= hit[1][1] + hit[1][3] / 2 <= ry2
            ]
            if inside:
                candidates = inside
        for text, _ in candidates:
            digits = PersonalInfoOcr._extract_id_number(text)
            if digits:
                return digits
        return candidates[0][0] or ""

    @staticmethod
    def _crop(image: np.ndarray, region: Dict[str, Any]) -> np.ndarray:
        """按区域裁剪图片"""
        h, w = image.shape[:2]
        x1 = max(0, min(int(region.get("x1", 0)), w - 1))
        y1 = max(0, min(int(region.get("y1", 0)), h - 1))
        x2 = max(x1 + 1, min(int(region.get("x2", x1 + 1)), w))
        y2 = max(y1 + 1, min(int(region.get("y2", y1 + 1)), h))
        return image[y1:y2, x1:x2]

    @staticmethod
    def _preprocess(image: np.ndarray) -> np.ndarray:
        """OCR 前预处理：灰度、CLAHE 增强、去噪"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        denoised = cv2.fastNlMeansDenoising(enhanced, None, 10, 7, 21)
        return denoised

    @staticmethod
    def _recognize_one(engine: Any, image: np.ndarray) -> tuple[str, float]:
        """单张图片 OCR，返回 (文本, 平均置信度)"""
        try:
            # Paddle 推理非线程安全，串行化调用
            with PersonalInfoOcr._ocr_call_lock:
                result = engine.ocr(image, cls=True)
        except Exception as e:
            logger.warning("OCR 识别异常: %s", e)
            return "", 0.0

        # PaddleOCR 返回结构：[[[box], (text, confidence)], ...]
        if not result or not result[0]:
            return "", 0.0

        texts = []
        confidences = []
        for line in result[0]:
            if line and len(line) == 2:
                _, (text, conf) = line
                texts.append(text or "")
                confidences.append(float(conf) if conf is not None else 0.0)

        full_text = "".join(texts).strip().replace(" ", "")
        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
        return full_text, avg_conf


    def recognize_block(
        self,
        image: np.ndarray,
        region: Dict[str, Any],
    ) -> Dict[str, Any]:
        """识别整块的考生信息区，保留换行与空格，返回原始文本和平均置信度。

        Args:
            image: 完整答题卡图片（BGR）
            region: 区域配置，包含 x1, y1, x2, y2

        Returns:
            {"raw_text": str, "confidence": float}
        """
        engine = self._get_engine()
        crop = self._crop(image, region)
        if crop.size == 0:
            return {"raw_text": "", "confidence": 0.0}

        if engine is None:
            return {"raw_text": "", "confidence": 0.0}

        preprocessed = self._preprocess(crop)
        ocr_result = self._ocr_block_lines(engine, preprocessed)
        if ocr_result is None:
            return {"raw_text": "", "confidence": 0.0}
        texts, confidences = ocr_result

        # 条码定位（准考证号常为条形码）：先考生信息区 crop，失败再整页兜底。
        # 保留条码位置信息（rect 相对被解码图像），用于后续“条码锚定印刷标签条 OCR”。
        barcode_hits = self._locate_barcodes(crop)
        strip_offset = self._crop_offset(image, region)
        if not barcode_hits and image is not None and image.size > 0:
            barcode_hits = self._locate_barcodes(image)
            strip_offset = (0, 0)
            if barcode_hits:
                logger.info("[ocr] 考生信息区内未解码出条码，整页兜底命中: %s",
                            [t for t, _ in barcode_hits])
        for data, _ in barcode_hits:
            if data not in texts:
                texts.append(data)
                confidences.append(1.0)
            # 显式标记行：解析器/Java 端优先采用条码考号，避免手写考号误识（如 80220G229）
            marker = f"条码考号:{data}"
            if marker not in texts:
                texts.append(marker)
                confidences.append(1.0)

        # 条码锚定印刷标签条 OCR：印刷体姓名行（条码上方小字）在整块 OCR 中容易丢行/误识，
        # 只要定位到条码就以其为锚点裁出上下标签条，用 3 种轻预处理变体（放大、不去噪）分别识别，
        # 各变体结果互补（如“阿夏如”只有 CLAHE 变体正确），全部作为姓名候选供花名册裁决；
        # 主变体（纯灰度）的文本行并入 raw_text，并以显式标记行注入主候选姓名。
        name_candidates: List[str] = []
        if barcode_hits:
            for _, rect in barcode_hits[:2]:
                if not rect:
                    continue
                strip = self._crop_label_strip(image, rect, strip_offset)
                if strip is None or strip.size == 0:
                    continue
                for vi, variant in enumerate(self._strip_preprocess_variants(strip)):
                    strip_result = self._ocr_block_lines(engine, variant)
                    if not strip_result:
                        continue
                    strip_lines, strip_confs = strip_result
                    variant_name = self._extract_name_from_strip_lines(strip_lines)
                    if variant_name and variant_name not in name_candidates:
                        name_candidates.append(variant_name)
                    if vi == 0:
                        added = [t for t in strip_lines if t and t not in texts]
                        texts.extend(added)
                        confidences.extend(strip_confs)
                        logger.info("[ocr] 条码锚定标签条 OCR 文本行: %s", added)
                if name_candidates:
                    break
            if name_candidates:
                marker = f"印刷体姓名:{name_candidates[0]}"
                if marker not in texts:
                    texts.append(marker)
                    confidences.append(1.0)
                logger.info("[ocr] 条码标签条姓名候选: %s", name_candidates)

        # 印刷标签行召回增强：首遍 OCR 若三个印刷锚点（班级/考场/座位）都未出现，
        # 说明印刷体小字可能漏识别，对 crop 放大 2 倍重跑一次并合并文本，
        # 以便解析器能命中印刷体姓名/考号（只重试一次，受上层 OCR 超时约束）。
        joined = "\n".join(texts)
        if not any(anchor in joined for anchor in ("班级", "考场", "座位")):
            scaled = cv2.resize(preprocessed, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
            retry_result = self._ocr_block_lines(engine, scaled)
            if retry_result:
                retry_texts, retry_confidences = retry_result
                if retry_texts:
                    for t in retry_texts:
                        if t not in texts:
                            texts.append(t)
                    confidences.extend(retry_confidences)
                    logger.info("[ocr] 考生信息区首遍未识别出印刷标签锚点，已放大重试并合并文本")

        # 按 PaddleOCR 返回的行顺序拼接，保留换行
        raw_text = "\n".join(texts).strip()
        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
        return {"raw_text": raw_text, "confidence": round(avg_conf, 4),
                "name_candidates": name_candidates}

    def recognize_label_strip(
        self,
        image: np.ndarray,
        region: Dict[str, Any],
        candidate_names: Optional[List[str]] = None,
        doc_fallback: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """识别「条码标签区」（框题管理中单独框选的印刷标签条小区域）。

        小区域 + 轻预处理（3 倍放大、不去噪）多变体 OCR，印刷体姓名/条码考号精度
        远高于整块考生信息区识别。有花名册时对姓名候选做相似度裁决。

        Args:
            doc_fallback: 讯飞 OCR 大模型印证通道（按次计费），签名 crop -> List[str] 文本行。
                          与 Paddle 双引擎相互印证：只要配置了就无条件调用（不要求 Paddle
                          先失败），讯飞结果以 xfyun_doc_name 平铺供前端与 Paddle 结果并列核对。

        Returns:
            {"raw_text": str, "name": str, "exam_no": str,
             "name_candidates": list, "confidence": float}
        """
        empty = {"raw_text": "", "name": "", "exam_no": "",
                 "name_candidates": [], "confidence": 0.0}
        crop = self._crop(image, region)
        if crop.size == 0:
            return empty

        # 条码解码（标签区内必有条码，小图解码快且准）；
        # 框选把条码裁残/留白不足时解码会失败，此时对整页兜底再找一次
        barcode_hits = self._locate_barcodes(crop)
        page_fallback = False
        if not barcode_hits and image is not None and image.size > 0:
            barcode_hits = self._locate_barcodes(image)
            page_fallback = True
            if barcode_hits:
                logger.info("[ocr] 标签区内未解码出条码，整页兜底命中: %s",
                            [t for t, _ in barcode_hits])
        # 整页兜底时页面可能有多个条码/二维码，按位置+数字校验挑考号条码
        exam_no = self._pick_exam_no_barcode(
            barcode_hits, region if page_fallback else None
        )

        # 3 种轻预处理变体分别 OCR，姓名候选互补（实测：阿夏如只有 CLAHE 变体正确等）
        all_lines: List[str] = []
        confidences: List[float] = []
        name_candidates: List[str] = []
        engine = self._get_engine()
        if engine is not None:
            for variant in self._strip_preprocess_variants(crop):
                result = self._ocr_block_lines(engine, variant)
                if not result:
                    continue
                lines, confs = result
                confidences.extend(confs)
                for t in lines:
                    if t and t not in all_lines:
                        all_lines.append(t)
                name = self._extract_name_from_strip_lines(lines)
                if name and name not in name_candidates:
                    name_candidates.append(name)

        # 讯飞 OCR 大模型印证（按次计费）：与 Paddle 双引擎相互印证，结果均入候选池
        # paddle_name 记录纯 Paddle 多变体的主候选（讯飞候选加入前），供前端印证展示
        paddle_name = name_candidates[0] if name_candidates else ""
        doc_name = ""
        if doc_fallback is not None:
            try:
                doc_lines = doc_fallback(crop)
            except Exception as e:
                logger.warning("讯飞OCR大模型识别失败: %s", e)
                doc_lines = []
            if doc_lines:
                for t in doc_lines:
                    if t and t not in all_lines:
                        all_lines.append(t)
                doc_name = self._extract_name_from_strip_lines(doc_lines)
                if doc_name:
                    if doc_name not in name_candidates:
                        name_candidates.append(doc_name)
                    logger.info("[ocr] 讯飞OCR大模型姓名候选: %s", doc_name)

        # 条码考号标记行（与 recognize_block 的标记约定一致）
        if exam_no:
            all_lines.append(f"条码考号:{exam_no}")
            confidences.append(1.0)

        # 花名册裁决：候选对名单打分取最优，纠正低频人名单字误识
        # 最终姓名：讯飞文档大模型优先（实测 30 卡 22/30 为最强单通道），Paddle 主候选兜底
        name = doc_name or paddle_name
        if candidate_names:
            from omr_service.engine.personal_info_block_parser import best_roster_match
            matched = best_roster_match(name_candidates, candidate_names)
            if matched:
                if matched != name:
                    logger.info("[ocr] 条码标签区花名册裁决姓名: %s -> %s (候选=%s)",
                                name, matched, name_candidates)
                name = matched
        if name:
            all_lines.append(f"印刷体姓名:{name}")
            confidences.append(1.0)

        raw_text = "\n".join(all_lines).strip()
        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
        return {"raw_text": raw_text, "name": name, "exam_no": exam_no,
                "name_candidates": name_candidates, "confidence": round(avg_conf, 4),
                # 双引擎印证值（前端并列展示，供人工核对）
                "paddle_name": paddle_name, "xfyun_doc_name": doc_name}

    @staticmethod
    def _strip_preprocess_variants(image: np.ndarray) -> List[np.ndarray]:
        """标签条印刷体小字的 3 种轻预处理变体（均放大 3 倍、不去噪）：

        - 纯灰度（主变体，实测准确率最高）
        - CLAHE 增强
        - Otsu 二值化

        注意不做 fastNlMeansDenoising：去噪会模糊小号印刷字细笔画导致丢字
        （如“乌日汗”丢“日”）；CLAHE 对部分扫描件反而有害，故仅作补充变体。
        """
        scaled = cv2.resize(image, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(scaled, cv2.COLOR_BGR2GRAY) if len(scaled.shape) == 3 else scaled
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return [gray, clahe, otsu]

    # 条码标签条内字段标签（含常见 OCR 误读变体），用于行内截断（如“乌尼尔白力夏班级:四班”）
    _STRIP_LINE_CUT_LABELS = ("班级", "斑级", "考场", "老场", "座位", "座号", "姓名")

    # 标签条文本中可能出现的字段词，不能当作姓名（与手写姓名清洗共用同一套残留词）
    _STRIP_STOP_WORDS = NAME_STOP_WORDS

    @staticmethod
    def _extract_name_from_strip_lines(lines: List[str]) -> str:
        """从条码标签条 OCR 行中提取印刷体姓名。

        标签条布局固定：学校行 + 姓名行（或“学校 姓名 班级:X班”合并行）+ 考号/考场/座位行。
        按行序取第一个 2-8 字纯中文（含间隔号）、非学校名、非字段值的词。
        """
        for line in lines:
            cut = line
            for label in PersonalInfoOcr._STRIP_LINE_CUT_LABELS:
                idx = cut.find(label)
                if idx > 0:
                    cut = cut[:idx]
            for token in re.split(r"[\s：:]+", cut):
                token = token.strip()
                if not re.fullmatch(r"[\u4e00-\u9fa5·]{2,8}", token):
                    continue
                if token in PersonalInfoOcr._STRIP_STOP_WORDS:
                    continue
                # 跳过学校名（东乌一中/东转一中/xx中学等）
                if token.endswith(("中", "中学", "学校", "初中", "高中", "附中", "班")):
                    continue
                return token
        return ""

    @staticmethod
    def _crop_offset(image: np.ndarray, region: Dict[str, Any]) -> tuple:
        """计算区域裁剪的偏移量（与 _crop 的边界 clamp 保持一致）"""
        h, w = image.shape[:2]
        x1 = max(0, min(int(region.get("x1", 0)), w - 1))
        y1 = max(0, min(int(region.get("y1", 0)), h - 1))
        return x1, y1

    @staticmethod
    def _ocr_block_lines(engine: Any, image: np.ndarray) -> Optional[tuple]:
        """对整块考生信息区执行一次 OCR，返回 (文本行列表, 置信度列表)；异常时返回 None。"""
        try:
            # Paddle 推理非线程安全，串行化调用
            with PersonalInfoOcr._ocr_call_lock:
                result = engine.ocr(image, cls=True)
        except Exception as e:
            logger.warning("考生信息区 OCR 异常: %s", e)
            return None

        if not result or not result[0]:
            return [], []

        texts = []
        confidences = []
        for line in result[0]:
            if line and len(line) == 2:
                _, (text, conf) = line
                if text:
                    texts.append(text)
                confidences.append(float(conf) if conf is not None else 0.0)
        return texts, confidences


def recognize_personal_info(
    image: np.ndarray,
    regions: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """便捷函数：识别个人信息区域"""
    return PersonalInfoOcr().recognize(image, regions)


def crosscheck_handwritten_names(
    ocr: "PersonalInfoOcr",
    hw_client: Any,
    image: np.ndarray,
    regions: List[Dict[str, Any]],
    indices: List[int],
    results: List[Dict[str, Any]],
    candidate_names: Optional[List[str]],
    log_prefix: str = "ocr",
) -> None:
    """讯飞手写识别印证（core/service.py 与 mq/job_handler.py 共用，避免双份逻辑漂移）。

    手写姓名讯飞更准（实测敖其泰/乌日汗讯飞全对），讯飞非空结果优先采用；
    讯飞识别值以 name_xfyun_hw 字段平铺到结果列表，供前端与 Paddle 结果并列核对。
    """
    from omr_service.engine.personal_info_block_parser import best_roster_match

    if hw_client is None:
        return
    for idx, region in zip(indices, regions):
        if region.get("field") not in NAME_FIELD_IDS:
            continue
        result = results[idx]
        paddle_value = result.get("value") or ""
        try:
            hw_text = hw_client.recognize_text(PersonalInfoOcr._crop(image, region))
            hw_name = clean_handwritten_name(hw_text)
        except Exception as e:
            logger.warning("讯飞手写识别失败: %s", e)
            continue
        if not hw_name:
            continue
        # 平铺讯飞识别值，供前端与 Paddle 结果相互印证
        results.append({"field": "name_xfyun_hw", "value": hw_name, "confidence": 0.99})
        final_name = hw_name
        if candidate_names:
            final_name = best_roster_match([hw_name, paddle_value], candidate_names) or final_name
        if final_name != paddle_value:
            logger.info("[%s] 手写姓名采用讯飞结果: paddle=%s xfyun=%s", log_prefix, paddle_value, final_name)
        result["value"] = final_name
        if final_name:
            result["confidence"] = 0.99


def recognize_barcode_label_strip(
    ocr: "PersonalInfoOcr",
    doc_client: Any,
    image: np.ndarray,
    region: Dict[str, Any],
    candidate_names: Optional[List[str]] = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """识别「条码标签区」并构造平铺字段（core/service.py 与 mq/job_handler.py 共用，
    避免双份逻辑漂移；与 crosscheck_handwritten_names 同一处理思路）。

    标签区：小区域多变体高精度识别印刷体姓名 + 条码考号；
    讯飞 OCR 大模型双引擎印证（按次计费，与 Paddle 结果并列展示供人工核对）。

    Args:
        ocr: Paddle 个人信息识别器。
        doc_client: 讯飞文档大模型客户端（None 表示未启用/无凭证）。
        image: 整页图片。
        region: 条码标签区区域配置（含 x1/y1/x2/y2）。
        candidate_names: 花名册，用于姓名候选相似度裁决。

    Returns:
        (标签区识别结果, 平铺追加到结果列表的字段清单)。
    """
    doc_fallback = None
    if doc_client is not None:
        doc_fallback = lambda crop: doc_client.recognize_lines(crop)
    strip_result = ocr.recognize_label_strip(
        image, region, candidate_names, doc_fallback=doc_fallback
    )
    # 平铺追加（非空才追加，避免覆盖 block 链路结果）：
    # printed_name 由 Java 端优先于手写 name 采用；
    # 考号用 printed_exam_no 独立字段，避免与 block 链路平铺的 exam_no 冲突
    flattened: List[Dict[str, Any]] = []
    if strip_result.get("name"):
        flattened.append({"field": "printed_name", "value": strip_result["name"], "confidence": 1.0})
    if strip_result.get("exam_no"):
        flattened.append({"field": "printed_exam_no", "value": strip_result["exam_no"], "confidence": 1.0})
    # 双引擎印证值（前端并列展示）
    if strip_result.get("paddle_name"):
        flattened.append({"field": "name_paddle", "value": strip_result["paddle_name"], "confidence": 1.0})
    if strip_result.get("xfyun_doc_name"):
        flattened.append({"field": "name_xfyun_doc", "value": strip_result["xfyun_doc_name"], "confidence": 1.0})
    return strip_result, flattened


def recognize_personal_info_block(
    image: np.ndarray,
    region: Dict[str, Any],
) -> Dict[str, Any]:
    """便捷函数：识别考生信息区整体"""
    return PersonalInfoOcr().recognize_block(image, region)
