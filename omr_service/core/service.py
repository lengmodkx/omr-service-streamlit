"""OMR 业务核心: 协议无关, 输入输出都是 plain dict.

本模块不依赖 FastAPI / Pydantic / protobuf.

现有实现参考 omr_service/rpc/omr_service.py, 本任务是抽离 + dict 接口.
"""
from __future__ import annotations

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

        # 同时输出 bubbles (按气泡坐标) 供前端可视化
        bubble_grid = []
        stp = getattr(cached, "standard_template", None)
        if stp is not None and hasattr(stp, "bubbles"):
            for b in stp.bubbles:
                bubble_grid.append({
                    "q": b.get("q"), "opt": b.get("opt"),
                    "x": b.get("x"), "y": b.get("y"),
                    "w": b.get("w"), "h": b.get("h"),
                })

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
                lambda: self._crop_subjective(images, regions, f"task_{template_id}"),
                self.ocr_timeout_seconds,
                "主观题裁剪",
                [],
            )

        return result

    def _crop_subjective(self, images: list, regions: List[Dict[str, Any]], namespace: str) -> List[Dict[str, Any]]:
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

        crops = self.cropper.crop_subjective_regions(images, norm_regions, namespace)
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
            for idx, region in indexed_regions:
                if region.get("field") == self.student_info_block_field:
                    block_regions.append(region)
                    block_indices.append(idx)
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

    def _do_recognize(self, cached, images, request):
        """真实识别流程 (从 rpc/omr_service.py 迁移).

        cached: CachedTemplate 实例
        返回 (answers, abnormal). answers 是 list of dict.
        """
        from omr_service.engine.recognizers.standard import StandardTemplateRecognizer
        from omr_service.engine.recognizer import RecognizeContext

        # 从 CachedTemplate 中取出 StandardTemplate
        template = getattr(cached, "standard_template", None)
        if template is None or not hasattr(template, "recognize"):
            logger.error("CachedTemplate 中无 StandardTemplate: %s", type(cached))
            return [], False

        # 把模板里缓存的 standard_answers 注入到 context
        ctx = RecognizeContext(
            standard_answers=getattr(template, "answers", {}) or {},
            column_boxes=[],
        )
        recognizer = StandardTemplateRecognizer(standard_template=template)
        image = images[0] if isinstance(images, list) else images
        result = recognizer.recognize(image, ctx)

        # answers: {q: {answer, status, correct}} -> list[dict]
        answers = []
        abnormal = result.card_flag in ("abnormal", "suspicious_blank")
        for q, info in result.answers.items():
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
        bubble_grid = [
            {
                "q": b["q"], "opt": b["opt"],
                "x": b["x"], "y": b["y"], "w": b["w"], "h": b["h"],
            }
            for b in st.bubbles
        ]
        return answers, bubble_grid
