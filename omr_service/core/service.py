"""OMR 业务核心: 协议无关, 输入输出都是 plain dict.

本模块不依赖 FastAPI / Pydantic / protobuf.

现有实现参考 omr_service/rpc/omr_service.py, 本任务是抽离 + dict 接口.
"""
from __future__ import annotations

import copy
import logging
import threading
import time
from typing import Any, Callable, Dict, List, Optional

from omr_service.core.exceptions import (
    ImageLoadError,
    InternalError,
    InvalidRequestError,
    TemplateNotFoundError,
)
from omr_service.engine.ocr import (
    BARCODE_LABEL_FIELD,
    crosscheck_handwritten_names,
    recognize_barcode_label_strip,
)

logger = logging.getLogger(__name__)


def _pick(d: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    """按优先级取第一个非 None 的字段值（兼容 snake_case / camelCase / Java 同步链路命名）。"""
    for k in keys:
        v = d.get(k)
        if v is not None:
            return v
    return default


def _as_int(value: Any, default: int = 0) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


def normalize_columns(columns: Any) -> List[Dict[str, Any]]:
    """归一化选择题列配置为 StandardTemplate 期望的 snake_case 字段.

    兼容三种来源:
    - MQ 链路 _build_parse_request 已归一化的 start_q/num_q/num_options
    - Java 同步 HTTP 链路 (buildFastApiColumns) 的 question_start/question_count/options_per_question
    - camelCase (startQ/numQ/numOptions/optionAxis/reverseQ/pageIndex)
    坐标支持 x1/y1/x2/y2 或 x/y/width/height 两种写法。
    缺坐标或 num_q<=0 的列直接跳过（视为该页无选择题列），不再抛 KeyError。
    """
    out: List[Dict[str, Any]] = []
    for c in columns or []:
        if not isinstance(c, dict):
            continue
        x1 = _pick(c, "x1", "x")
        y1 = _pick(c, "y1", "y")
        x2 = _pick(c, "x2")
        y2 = _pick(c, "y2")
        if x2 is None and x1 is not None and _pick(c, "width") is not None:
            x2 = _as_int(x1) + _as_int(_pick(c, "width"))
        if y2 is None and y1 is not None and _pick(c, "height") is not None:
            y2 = _as_int(y1) + _as_int(_pick(c, "height"))
        num_q = _as_int(_pick(c, "num_q", "numQ", "question_count", "questionCount"))
        if x1 is None or y1 is None or x2 is None or y2 is None or num_q <= 0:
            logger.warning("跳过无效选择题列配置: %s", c)
            continue
        out.append({
            "x1": _as_int(x1),
            "y1": _as_int(y1),
            "x2": _as_int(x2),
            "y2": _as_int(y2),
            "start_q": _as_int(_pick(c, "start_q", "startQ", "question_start", "questionStart"), 1),
            "num_q": num_q,
            "num_options": _as_int(_pick(c, "num_options", "numOptions", "options_per_question", "optionsPerQuestion"), 4),
            "option_axis": _pick(c, "option_axis", "optionAxis", default="x") or "x",
            "reverse_q": _as_bool(_pick(c, "reverse_q", "reverseQ")),
            "page_index": _as_int(_pick(c, "page_index", "pageIndex")),
        })
    return out


def normalize_personal_info(regions: Any) -> List[Dict[str, Any]]:
    """归一化个人信息区域配置（camelCase pageIndex → page_index）。"""
    out: List[Dict[str, Any]] = []
    for r in regions or []:
        if not isinstance(r, dict):
            continue
        out.append({
            "field": r.get("field", ""),
            "x1": _as_int(r.get("x1")),
            "y1": _as_int(r.get("y1")),
            "x2": _as_int(r.get("x2")),
            "y2": _as_int(r.get("y2")),
            "page_index": _as_int(_pick(r, "page_index", "pageIndex")),
        })
    return out


def normalize_subjective_regions(regions: Any) -> List[Dict[str, Any]]:
    """归一化主观题区域配置（pageIndex/stitchWithNext → snake_case）。"""
    out: List[Dict[str, Any]] = []
    for r in regions or []:
        if not isinstance(r, dict):
            continue
        out.append({
            "q": _as_int(r.get("q")),
            "x1": _as_int(r.get("x1")),
            "y1": _as_int(r.get("y1")),
            "x2": _as_int(r.get("x2")),
            "y2": _as_int(r.get("y2")),
            "page_index": _as_int(_pick(r, "page_index", "pageIndex")),
            "stitch_with_next": _as_bool(_pick(r, "stitch_with_next", "stitchWithNext")),
        })
    return out


def run_with_timeout(
    fn: Callable[[], Any],
    timeout_seconds: float,
    desc: str,
    default: Any,
) -> Any:
    """在独立 daemon 线程中执行 fn，超时返回 default，不阻塞任务回写.

    背景: PaddleOCR 首次初始化（模型加载/下载）可能耗时数分钟甚至挂起，
    若直接在 worker 线程调用，会导致黄金模板解析/批量识别任务卡住、
    前端轮询超时、页面无任何识别结果。超时后线程继续在后台运行
    （daemon，不可强杀），但任务主体已先行完成回写；
    fn 内抛出的 Python 异常在此上抛给调用方处理。
    """
    box: Dict[str, Any] = {}

    def runner() -> None:
        try:
            box["value"] = fn()
        except Exception as e:  # noqa: BLE001 - 由调用方统一处理
            box["error"] = e

    t = threading.Thread(target=runner, daemon=True, name=f"omr_{desc}")
    t.start()
    t.join(timeout_seconds)
    if t.is_alive():
        logger.warning(
            "%s 超时(>%.0fs)，跳过该步骤，任务继续；若持续超时请检查 PaddleOCR "
            "环境（建议 Python 3.11 + paddlepaddle 2.6.2）或增大 ocr_timeout_seconds",
            desc, timeout_seconds,
        )
        return default
    if "error" in box:
        raise box["error"]
    return box.get("value", default)


class OmrService:
    """协议无关的 OMR 业务服务."""

    def __init__(
        self,
        template_store,
        image_loader,
        worker_pool,
        ocr_engine,
        cropper,
        *,
        sync_timeout_seconds: float = 60.0,
        ocr_timeout_seconds: float = 30.0,
        ocr_confidence_threshold: float = 0.3,
        xfyun_doc=None,
        xfyun_hw=None,
    ):
        self.template_store = template_store
        self.image_loader = image_loader
        self.worker_pool = worker_pool
        self.ocr_engine = ocr_engine
        self.cropper = cropper
        self.sync_timeout_seconds = sync_timeout_seconds
        self.ocr_timeout_seconds = ocr_timeout_seconds
        # 个人信息 OCR 置信度阈值：低于阈值视为未识别，置空 value（对齐旧 gRPC 分支行为）
        self.ocr_confidence_threshold = ocr_confidence_threshold
        # 考生信息区字段标识（与 Java 端 OmrPayloadBuilder.STUDENT_INFO_BLOCK_FIELD 对齐）
        self.student_info_block_field = "student_info_block"
        # 讯飞 OCR 增强通道（可选，None 表示未启用）：doc=OCR大模型(印刷), hw=手写识别
        self.xfyun_doc = xfyun_doc
        self.xfyun_hw = xfyun_hw

    def recognize(self, request: dict[str, Any]) -> dict[str, Any]:
        """同步识别. 返回 RecognizeResult dict."""
        template_id = request.get("template_id")
        scan_urls = request.get("scan_image_urls")
        if not template_id or not scan_urls:
            raise InvalidRequestError("template_id or scan_image_urls", "missing")

        t0 = time.monotonic()
        template_id = self._coerce_template_id(template_id)
        images = self._load_images(scan_urls)
        cached = self.template_store.get(template_id)
        if cached is None:
            raise TemplateNotFoundError(template_id)

        future = self.worker_pool.submit(self._do_recognize, cached, images, request)
        try:
            answers, abnormal = future.result(timeout=self.sync_timeout_seconds)
        except TimeoutError as e:
            raise InternalError(f"识别超时 ({self.sync_timeout_seconds}s)") from e

        # 同时输出 bubbles (按气泡坐标) 供前端可视化。
        # 多页模板必须按 page_index 合并各页气泡（带 page_index），
        # 不能直接输出合并后的全量气泡，否则前端无法区分页面。
        bubble_grid = self.build_multipage_bubble_grid(cached)

        result = {
            "code": 0,
            "message": "ok",
            "template_id": template_id,
            "answers": answers,
            "bubbles": bubble_grid,
            "bubble_grid": bubble_grid,
            "abnormal": abnormal,
            "empty_count": sum(1 for a in answers if a.get("is_blank")),
            "multiple_count": sum(1 for a in answers if a.get("is_multiple")),
            "elapsed_ms": int((time.monotonic() - t0) * 1000),
            "personal_info": [],
            "subjective_crops": [],
        }

        # OCR 个人信息（含考生信息区整块识别 + 结构化解析）
        if request.get("personal_info_region"):
            regions = normalize_personal_info(request["personal_info_region"])
            if isinstance(request["personal_info_region"], dict):
                regions = normalize_personal_info([request["personal_info_region"]])
            # 考生名单（花名册），用于姓名候选相似度裁决，可选
            candidate_names = request.get("candidate_names") or None
            result["personal_info"] = run_with_timeout(
                lambda: self._recognize_personal_info(images, regions, candidate_names),
                self.ocr_timeout_seconds,
                "个人信息OCR",
                [],
            )

        # 主观题裁剪
        if request.get("subjective_regions"):
            regions = normalize_subjective_regions(request["subjective_regions"])
            result["subjective_crops"] = run_with_timeout(
                lambda: self._crop_subjective(
                    images, regions, f"task_{template_id}", cached.page_images
                ),
                self.ocr_timeout_seconds,
                "主观题裁剪",
                [],
            )

        return result

    def _crop_subjective(self, images: list, regions: List[Dict[str, Any]], namespace: str,
                         reference_images: Optional[Dict[int, Any]] = None) -> List[Dict[str, Any]]:
        """主观题裁剪：单图场景（Java 按页拆请求）把 page_index 归一化为 0 选图，结果还原原始页码.

        背景：Java 端按页拆分黄金模板解析/识别请求，每页只传一张图片，但 region 的
        page_index 仍是原始页码（第 2 页为 1）。直接传 [image] + page_index=1 会被
        cropper 判定页码越界而静默丢弃（FastAPI 重写时丢失该逻辑导致第二页裁不出来）。
        """
        if not regions:
            return []
        single_image = len(images) == 1
        original_page_by_q: Dict[int, int] = {}
        norm_regions: List[Dict[str, Any]] = []
        for r in regions:
            region_copy = dict(r)
            q = _as_int(region_copy.get("q"))
            original_page_by_q.setdefault(q, _as_int(region_copy.get("page_index")))
            if single_image:
                region_copy["page_index"] = 0
            norm_regions.append(region_copy)

        crops = self.cropper.crop_subjective_regions(
            images, norm_regions, namespace, reference_images=reference_images
        )
        if single_image:
            for c in crops:
                q = _as_int(c.get("q"))
                c["page_index"] = original_page_by_q.get(q, _as_int(c.get("page_index")))
        return crops

    def _recognize_personal_info(self, images, regions, candidate_names=None):
        """按 page_index 分组图片，批量识别个人信息，并单独解析整块考生信息区。

        与 job_handler.py 行为一致：普通字段单字段 OCR，整块区域用 recognize_block + parse_personal_info_block。
        单图场景（Java 按页拆请求）page_index 归一化为 0 选图（对齐旧 gRPC 分支）。
        考生信息区解析出的子字段平铺追加到结果列表（Java 端按 field 平铺读取）。
        candidate_names: 该考试的考生名单（花名册），用于姓名候选相似度裁决。
        """
        from omr_service.engine.personal_info_block_parser import (
            parse_personal_info_block,
            upsert_marked_name,
        )

        single_image = len(images) == 1
        page_groups: Dict[int, list] = {}
        for idx, region in enumerate(regions):
            page_index = _as_int(region.get("page_index"))
            if single_image:
                page_index = 0
            page_groups.setdefault(page_index, []).append((idx, region))

        results: List[Optional[Dict[str, Any]]] = [None] * len(regions)

        for page_index, indexed_regions in page_groups.items():
            if page_index < 0 or page_index >= len(images):
                for idx, region in indexed_regions:
                    results[idx] = {"field": region.get("field", ""), "value": "", "confidence": 0.0}
                continue

            image = images[page_index]
            normal_regions, normal_indices, block_regions, block_indices = [], [], [], []
            label_regions, label_indices = [], []
            for idx, region in indexed_regions:
                field = region.get("field")
                if field == self.student_info_block_field:
                    block_regions.append(region)
                    block_indices.append(idx)
                elif field == BARCODE_LABEL_FIELD:
                    label_regions.append(region)
                    label_indices.append(idx)
                else:
                    normal_regions.append(region)
                    normal_indices.append(idx)

            if normal_regions:
                page_results = self.ocr_engine.recognize(image, normal_regions)
                for idx, page_result in zip(normal_indices, page_results):
                    if page_result is None:
                        results[idx] = {"field": normal_regions[normal_indices.index(idx)].get("field", ""), "value": "", "confidence": 0.0}
                    else:
                        # 低置信度结果置空（对齐旧 gRPC 分支 ocr_confidence_threshold 过滤）
                        if page_result.get("confidence", 0.0) < self.ocr_confidence_threshold:
                            page_result["value"] = ""
                        results[idx] = page_result
                # 讯飞手写识别印证（按次计费，共享实现见 crosscheck_handwritten_names）
                crosscheck_handwritten_names(
                    self.ocr_engine, self.xfyun_hw, image,
                    normal_regions, normal_indices, results, candidate_names, "ocr",
                )

            for idx, region in zip(label_indices, label_regions):
                # 条码标签区识别 + 平铺字段构造（与 MQ 链路共用 recognize_barcode_label_strip）
                strip_result, flattened = recognize_barcode_label_strip(
                    self.ocr_engine, self.xfyun_doc, image, region, candidate_names,
                )
                results[idx] = {
                    "field": BARCODE_LABEL_FIELD,
                    "value": strip_result.get("raw_text", ""),
                    "confidence": strip_result.get("confidence", 0.0),
                }
                results.extend(flattened)

            for idx, region in zip(block_indices, block_regions):
                raw_result = self.ocr_engine.recognize_block(image, region)
                raw_text = raw_result.get("raw_text", "")
                try:
                    fields, conf = parse_personal_info_block(
                        raw_text,
                        extra_name_candidates=raw_result.get("name_candidates"),
                        candidate_names=candidate_names,
                    )
                except Exception as e:
                    logger.warning("考生信息解析异常: %s | raw_text=%s", e, raw_text)
                    fields, conf = {"raw_text": raw_text}, 0.0
                # 花名册裁决后的最终姓名回写标记行，保证 Java 端二次解析结论一致
                if fields.get("name"):
                    raw_text = upsert_marked_name(raw_text, fields["name"])
                    fields["raw_text"] = raw_text
                block_entry: Dict[str, Any] = {
                    "field": self.student_info_block_field,
                    "value": raw_text,
                    "confidence": conf,
                }
                parsed_fields = {k: v for k, v in fields.items() if k != "raw_text"}
                block_entry["parsed_fields"] = parsed_fields
                results[idx] = block_entry
                # 解析出的子字段平铺追加到结果列表（对齐旧 gRPC 分支，Java 端按 field 平铺读取）
                for k, v in parsed_fields.items():
                    results.append({
                        "field": k,
                        "value": v,
                        "confidence": conf if v else 0.0,
                    })

        for idx, region in enumerate(regions):
            if results[idx] is None:
                results[idx] = {"field": region.get("field", ""), "value": "", "confidence": 0.0}
        return results

    def parse_golden_template(self, request: dict[str, Any]) -> dict[str, Any]:
        """同步模板解析. 返回 GoldenTemplateResult dict 并缓存模板上下文供后续识别."""
        template_id = self._coerce_template_id(request.get("template_id"))
        template_url = request.get("template_image_url")
        # 归一化三类区域配置（兼容 MQ snake_case / Java 同步链路 question_start / camelCase）
        columns = normalize_columns(request.get("columns"))
        personal_info_regions = normalize_personal_info(request.get("personal_info_region"))
        subjective_regions = normalize_subjective_regions(request.get("subjective_regions"))
        if not template_id or not template_url:
            raise InvalidRequestError("template_id or template_image_url", "missing")

        # 推断本次请求对应的页码：优先取请求显式 page_index（MQ 链路传入），
        # 否则取所有配置中的最小 page_index（对齐旧 gRPC 分支），默认 0
        req_page_index = request.get("page_index", request.get("pageIndex"))
        if req_page_index is None:
            page_indexes = {c.get("page_index", 0) for c in columns}
            page_indexes.update(r.get("page_index", 0) for r in personal_info_regions)
            page_indexes.update(r.get("page_index", 0) for r in subjective_regions)
            current_page = min(page_indexes) if page_indexes else 0
        else:
            current_page = _as_int(req_page_index)

        t0 = time.monotonic()
        image = self.image_loader.load(template_url)
        if image is None:
            raise ImageLoadError(url=template_url, reason="模板图片加载失败")
        logger.info("模板图片加载成功: %s shape=%s", template_url, image.shape)
        if columns:
            answers, bubble_grid = self._do_parse(image, columns)
        else:
            answers, bubble_grid = [], []

        logger.info("模板解析完成: answers=%d bubbles=%d", len(answers), len(bubble_grid))
        for a in answers:
            logger.debug("  答案: q=%s ans=%s", a.get("question_no"), a.get("selected"))

        # 缓存模板上下文供后续 recognize 调用(关键! 之前从未缓存 → 全部识别失败)
        try:
            from omr_service.loader.template_store import CachedTemplate
            from omr_service.engine.standard_template import StandardTemplate

            # 构造 StandardTemplate 实例用于缓存
            std_template = StandardTemplate(image, columns) if columns else None
            cached = CachedTemplate(
                standard_template=std_template,
                personal_info=personal_info_regions,
                subjective_regions=subjective_regions,
                image_url=template_url,
                # 用真实页码作为键，避免多页解析时后页参考图覆盖前页
                page_images={current_page: image} if image is not None else {},
            )
            self.template_store.set(int(template_id), cached)
            logger.info("模板已缓存: template_id=%s page=%s bubbles=%d", template_id, current_page, len(bubble_grid))
        except Exception as e:
            logger.error("模板缓存失败: %s", e)

        # 个人信息示例 OCR(整块识别 + 行解析)
        personal_info_sample = []
        if personal_info_regions:
            try:
                personal_info_sample = run_with_timeout(
                    lambda: self._recognize_personal_info(
                        [image], personal_info_regions
                    ),
                    self.ocr_timeout_seconds,
                    "个人信息OCR",
                    [],
                )
            except Exception as e:
                logger.warning("个人信息ORC失败: %s", e)
            logger.info("个人信息OCR完成: %d 个字段", len(personal_info_sample))

        # 主观题区域裁剪预览（单图场景 page_index 自动归一化，结果还原原始页码）
        subjective_crops = []
        if subjective_regions:
            try:
                namespace = f"template_{template_id}"
                subjective_crops = run_with_timeout(
                    lambda: self._crop_subjective([image], subjective_regions, namespace),
                    self.ocr_timeout_seconds,
                    "主观题裁剪",
                    [],
                )
            except Exception as e:
                logger.warning("主观题裁剪失败: %s", e)
            logger.info("主观题裁剪完成: %d 个区域", len(subjective_crops))

        result = {
            "code": 0,
            "message": "ok",
            "template_id": template_id,
            "answers": answers,
            "bubble_grid": bubble_grid,
            "personal_info_sample": personal_info_sample,
            "subjective_crops": subjective_crops,
            # 与 Java 端 OmrResult 对齐：这些 primitive 字段缺失会导致 Jackson 反序列化 NPE
            "abnormal": False,
            "empty_count": sum(1 for a in answers if a.get("is_blank")),
            "multiple_count": sum(1 for a in answers if a.get("is_multiple")),
            "elapsed_ms": int((time.monotonic() - t0) * 1000),
        }
        return result

    def verify_recognition_rate(self, request: dict[str, Any]) -> dict[str, Any]:
        """暂未实现."""
        raise InternalError("verify_recognition_rate 暂未通过 HTTP 暴露")

    def reverify_paper(self, request: dict[str, Any]) -> dict[str, Any]:
        """与 recognize 行为等价."""
        return self.recognize(request)

    # ---------- 内部辅助 ----------

    def _load_images(self, urls: list[str]) -> list:
        try:
            return self.image_loader.load_multi(','.join(urls))
        except FileNotFoundError as e:
            raise ImageLoadError(url=getattr(e, "url", "?"), reason=str(e)) from e
        except Exception as e:
            raise ImageLoadError(url="?", reason=str(e)) from e

    @staticmethod
    def _coerce_template_id(raw) -> Optional[int]:
        """兼容 str/int 两种 template_id 写法."""
        if raw is None or raw == "":
            return None
        try:
            return int(raw)
        except (TypeError, ValueError):
            return None

    # ---------- 多页识别（HTTP /v1/recognize 与 MQ 批量识别共用） ----------

    @staticmethod
    def _bubbles_by_page(template) -> Dict[int, List[Dict[str, Any]]]:
        """按 page_index 归组模板气泡，返回按页码升序的 {page: bubbles}。"""
        grouped: Dict[int, List[Dict[str, Any]]] = {}
        for b in getattr(template, "bubbles", None) or []:
            grouped.setdefault(_as_int(b.get("page_index", 0)), []).append(b)
        return {page: grouped[page] for page in sorted(grouped)}

    @staticmethod
    def _bubble_entry(bubble: Dict[str, Any], page: int) -> Dict[str, Any]:
        """构造对外输出的单气泡坐标（保留 page_index，供前端按页可视化）。"""
        return {
            "q": bubble.get("q"),
            "opt": bubble.get("opt"),
            "x": bubble.get("x"),
            "y": bubble.get("y"),
            "w": bubble.get("w"),
            "h": bubble.get("h"),
            "page_index": page,
        }

    def build_multipage_bubble_grid(self, cached) -> List[Dict[str, Any]]:
        """构造多页气泡坐标：按页码顺序合并各页气泡并保留 page_index。

        与 recognize_pages 内部的 bubble_grid 使用同一套气泡结构（同源），
        供 recognize 返回给前端做多页可视化。
        """
        template = getattr(cached, "standard_template", None)
        if template is None:
            return []
        grouped = self._bubbles_by_page(template)
        page_keys = sorted(getattr(cached, "page_images", None) or {}) or [0]
        return [
            self._bubble_entry(b, page)
            for page in page_keys
            for b in grouped.get(page, [])
        ]

    @staticmethod
    def _build_page_template(template, page: int, page_bubbles: List[Dict[str, Any]], page_image):
        """构造单页专用模板（浅拷贝，避免污染共享缓存）。

        必须替换 .image：StandardTemplate.recognize 用 self.image 做全局 ECC 参考图
        （engine/standard_template.py 中 cv2.findTransformECC(ref_blur, blurred, ...)），
        否则第 2 页会拿第 1 页黄金图做对齐，导致识别错位。
        标准答案也按该页题号裁剪，避免跨页题号互相污染。
        """
        page_template = copy.copy(template)
        page_template.bubbles = page_bubbles
        page_template.image = page_image
        template_answers = getattr(template, "answers", None) or {}
        page_questions = {b.get("q") for b in page_bubbles}
        page_template.answers = (
            {q: template_answers[q] for q in page_questions if q in template_answers}
            if template_answers else {}
        )
        return page_template

    def recognize_pages(self, cached, images) -> tuple[dict, list, str, int]:
        """按 page_index 逐页识别并合并；返回 (answers_map, bubble_grid, card_flag, total)。

        背景：引擎题号是整卷绝对题号（standard_template.py: qn = start_q + qi），
        page_index 只挂在气泡上；模板缓存把多页气泡合并进同一个 standard_template，
        page_images 按页码保存各页黄金参考图。若直接用合并后的全部气泡在第 1 页图上
        采样，第 2 页题号会答串、第 1 页结果被污染（"多页答题卡只识别第一页"缺陷）。
        因此这里按页拆开：每页只给本页气泡 + 本页黄金参考图，识别后按题号合并。

        返回：
        - answers_map: {题号: {"answer": str|None, "status": str, "correct": bool|None}}
        - bubble_grid: 各页气泡坐标合并结果（保留 page_index）
        - card_flag: 卡片级异常标记（多页取最强：invalid_image > abnormal > suspicious_blank），
          无异常为空串；保留细粒度标记供 MQ 结果原样透传给 Java 端
        - total: 各页识别题目数之和

        跨页同题号视为模板配置错误，直接抛 InvalidRequestError，绝不静默覆盖。
        """
        from omr_service.engine.recognizer import RecognizeContext
        from omr_service.engine.recognizers.standard import StandardTemplateRecognizer

        template = getattr(cached, "standard_template", None)
        if template is None or not hasattr(template, "recognize"):
            logger.error("CachedTemplate 中无 StandardTemplate: %s", type(cached))
            return {}, [], "", 0

        page_images = getattr(cached, "page_images", None) or {}
        # 页码顺序：以缓存中的页码升序为准；无页码缓存时按图片顺序（兼容单页模板）
        page_keys = sorted(page_images.keys()) or [0]
        grouped = self._bubbles_by_page(template)

        answers_map: Dict[Any, Dict[str, Any]] = {}
        bubble_grid: List[Dict[str, Any]] = []
        question_page: Dict[Any, int] = {}
        duplicate_questions: List[Any] = []
        page_flags: List[str] = []
        total = 0

        image_list = images if isinstance(images, (list, tuple)) else [images]
        for idx, image in enumerate(image_list):
            # 第 i 张图对应 page_keys[i]，图片数超出页码数时回退为 i
            page = page_keys[idx] if idx < len(page_keys) else idx
            page_bubbles = grouped.get(page, [])
            if not page_bubbles:
                # 兼容"某页只有个人信息/主观题"的模板：该页无选择题气泡，跳过但不算失败
                logger.warning("第 %d 页无气泡配置，跳过识别", page + 1)
                continue

            # 本页黄金参考图（numpy 数组不能用 `or` 判空，必须显式 None 判断）
            page_image = page_images.get(page)
            if page_image is None:
                page_image = getattr(template, "image", None)
            page_template = self._build_page_template(template, page, page_bubbles, page_image)
            recognizer = StandardTemplateRecognizer(standard_template=page_template)
            # 标准答案按该页题号注入，保证 correct 只与该页标准答案比对
            ctx = RecognizeContext(standard_answers=page_template.answers, column_boxes=[])
            result = recognizer.recognize(image, ctx)

            for q, info in (result.answers or {}).items():
                # 同一题号出现在不同页 → 模板题号跨页重复，记录后统一报错
                if q in answers_map and question_page.get(q) != page:
                    duplicate_questions.append(q)
                question_page.setdefault(q, page)
                answers_map[q] = info

            bubble_grid.extend(self._bubble_entry(b, page) for b in page_bubbles)
            # 卡片级标记逐页收集：invalid_image（空图/退化小图）该页不贡献任何答案，
            # 若仍按"正常"返回会让缺题被当成识别成功的干净结果
            if result.card_flag:
                page_flags.append(str(result.card_flag))
            total += _as_int(getattr(result, "total", 0))

        if duplicate_questions:
            labels = ",".join(f"Q{q}" for q in sorted(set(duplicate_questions), key=str))
            logger.error("模板题号跨页重复：%s", labels)
            raise InvalidRequestError(
                f"模板题号跨页重复：{labels}，请把题号改为整卷唯一后重新解析模板"
            )

        # 多页取最强标记（invalid_image > abnormal > suspicious_blank），保留细粒度供下游区分
        card_flag = ""
        for candidate in ("invalid_image", "abnormal", "suspicious_blank"):
            if candidate in page_flags:
                card_flag = candidate
                break

        logger.info(
            "多页识别完成: pages=%d questions=%d card_flag=%s",
            len(image_list), len(answers_map), card_flag or "-",
        )
        return answers_map, bubble_grid, card_flag, total

    def _do_recognize(self, cached, images, request):
        """真实识别流程：按页码逐页识别后合并（多页答题卡修复，HTTP / MQ 共用）。

        cached: CachedTemplate 实例
        返回 (answers, abnormal)。answers 是 list of dict，字段与 Java 端
        OmrResult.QuestionAnswer 对齐（不新增字段，避免反序列化问题）。
        abnormal 由各页 card_flag 汇总（任一页异常即为 True）。
        """
        answers_map, _bubble_grid, card_flag, _total = self.recognize_pages(cached, images)
        abnormal = bool(card_flag)

        # answers: {q: {answer, status, correct}} -> list[dict]
        answers = []
        for q, info in answers_map.items():
            answers.append({
                "question_no": q,
                "selected": [info["answer"]] if info.get("answer") else [],
                "status": info.get("status", "empty"),
                "is_blank": info.get("status") == "empty",
                "is_multiple": info.get("status") == "multi",
                "correct": info.get("correct"),
            })
        logger.info("识别完成: q=%d abnormal=%s answers=%d",
                    len(answers), abnormal, sum(1 for a in answers if a["selected"]))
        return answers, abnormal

    def _do_parse(self, image, columns):
        """模板解析: 用 StandardTemplate 生成气泡网格并自动识别标准答案."""
        from omr_service.engine.standard_template import StandardTemplate

        st = StandardTemplate(image, columns)
        logger.info("StandardTemplate 生成: bubbles=%d answers=%d",
                     len(st.bubbles), len(st.answers))

        # ---- 诊断输出: 打印前5题每个选项的原始灰度值 + 最终判定 ----
        if hasattr(st, "_debug_samples") and st._debug_samples:
            logger.info("=== 气泡灰度采样诊断 (前%d题) ===", min(5, len(st._debug_samples)))
            for ds in st._debug_samples[:5]:
                logger.info("  Q%-3d detected=%-5s best=%-2s=%-5.0f gap=%-4.0f mean=%-5.0f opts=%s",
                    ds["q"], ds["answer"], ds["best_opt"], ds["best_val"],
                    ds["gap"], ds["mean_val"], ds["opts"])
            logger.info("=== 诊断结束 ===")
        # answers: {question_no: answer_str_or_None} → 转为 list of dict
        # 字段与 Java 端 OmrResult.QuestionAnswer 对齐（is_blank/is_multiple 为 primitive，
        # 响应缺失会导致 Jackson 反序列化 NPE）；多涂答案为多字母拼接（如 "AB"），长度>1 即 multi
        answers = []
        for q, ans in st.answers.items():
            if ans is not None:
                answers.append({
                    "question_no": q, "selected": [ans],
                    "status": "filled", "correct": True,
                    "is_blank": False, "is_multiple": len(str(ans)) > 1,
                })
            else:
                answers.append({
                    "question_no": q, "selected": [],
                    "status": "empty", "correct": False,
                    "is_blank": True, "is_multiple": False,
                })
        logger.info("_do_parse 结果: answers=%d (filled=%d)",
                     len(answers),
                     sum(1 for a in answers if a["status"] == "filled"))
        # bubbles: 带坐标信息的气泡列表
        # page_index 必须透传：模板解析响应与识别响应共用同一气泡结构，
        # 缺省回退 0 会让多页模板解析结果丢失页码（Java 端按 page_index 按页渲染）。
        bubble_grid = [
            {
                "q": b["q"], "opt": b["opt"],
                "x": b["x"], "y": b["y"], "w": b["w"], "h": b["h"],
                "page_index": _as_int(b.get("page_index", 0)),
            }
            for b in st.bubbles
        ]
        return answers, bubble_grid
