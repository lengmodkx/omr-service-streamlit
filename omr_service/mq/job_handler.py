"""MQ 批量任务处理（Phase 3：单任务 + 重试 + 死信）

Phase 4 (FastAPI 改造): 通过 OmrService（dict-based，与协议无关）调用核心识别/解析。
"""
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from omr_service.config import OmrConfig
from omr_service.core.exceptions import ImageLoadError
from omr_service.core.service import OmrService, run_with_timeout
from omr_service.engine.cropper import SubjectiveCropper
from omr_service.engine.ocr import (
    BARCODE_LABEL_FIELD,
    NAME_FIELD_IDS,
    PersonalInfoOcr,
    clean_handwritten_name,
)
from omr_service.engine.personal_info_block_parser import (
    best_roster_match,
    parse_personal_info_block,
    upsert_marked_name,
)
from omr_service.loader.image_loader import ImageLoader
from omr_service.loader.template_store import CachedTemplate, TemplateStore
from omr_service.mq.producer import MqProducer
from omr_service.worker.pool import WorkerPool

# 考生信息区（整体框选，一次识别姓名/准考证号/考场/座号等）字段标识
STUDENT_INFO_BLOCK_FIELD = "student_info_block"

logger = logging.getLogger(__name__)


def _as_int(value, default: int = 0) -> int:
    """int() 安全包装：None/空字符串/非数字回退默认值，避免 int(None) 抛 TypeError."""
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_bool(value, default: bool = False) -> bool:
    """布尔安全转换：字符串 "false"/"0" 不误判为 True."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


def _normalize_personal_info(regions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Java 端 camelCase 字段(pageIndex) → Python snake_case(page_index)，对齐旧分支 proto 转换.

    旧分支通过 _personal_info_from_proto 显式转换；FastAPI 分支的 _build_parse_request
    此前原样透传，导致 page_index 永远取默认值 0（多页模板时错页）。
    """
    out: List[Dict[str, Any]] = []
    for r in regions or []:
        out.append({
            "field": r.get("field", ""),
            "x1": _as_int(r.get("x1")),
            "y1": _as_int(r.get("y1")),
            "x2": _as_int(r.get("x2")),
            "y2": _as_int(r.get("y2")),
            "page_index": _as_int(r.get("pageIndex") if r.get("pageIndex") is not None else r.get("page_index")),
        })
    return out


def _normalize_subjective_regions(regions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Java 端 camelCase 字段(pageIndex/stitchWithNext) → Python snake_case.

    对齐旧分支 _subjective_region_from_proto 的转换；缺失时 page_index=0、stitch_with_next=False。
    """
    out: List[Dict[str, Any]] = []
    for r in regions or []:
        out.append({
            "q": _as_int(r.get("q")),
            "x1": _as_int(r.get("x1")),
            "y1": _as_int(r.get("y1")),
            "x2": _as_int(r.get("x2")),
            "y2": _as_int(r.get("y2")),
            "page_index": _as_int(r.get("pageIndex") if r.get("pageIndex") is not None else r.get("page_index")),
            "stitch_with_next": _as_bool(r.get("stitchWithNext") if r.get("stitchWithNext") is not None else r.get("stitch_with_next")),
        })
    return out


class BatchJobHandler:
    """处理批量识别任务"""

    def __init__(
        self,
        cfg: OmrConfig,
        template_store: TemplateStore,
        image_loader: ImageLoader,
        worker_pool: WorkerPool,
        producer: MqProducer,
        service: Optional[OmrService] = None,
    ):
        self.cfg = cfg
        self.template_store = template_store
        self.image_loader = image_loader
        self.worker_pool = worker_pool
        self.producer = producer
        self.service = service
        # 讯飞 OCR 增强通道（xfyun_enabled=false 时为 None）
        from omr_service.engine.xfyun_ocr import build_xfyun_clients
        self._xfyun_doc, self._xfyun_hw = build_xfyun_clients(cfg)
        self._ocr = PersonalInfoOcr()
        self._cropper = SubjectiveCropper(
            output_dir=cfg.crop_output_dir,
            base_url=cfg.crop_base_url,
        )

    def _get_or_load_template(self, template_id: int) -> Optional[CachedTemplate]:
        tpl = self.template_store.get(template_id)
        if tpl is not None:
            return tpl
        logger.warning("模板未找到: template_id=%s", template_id)
        return None

    def _recognize_one(
        self,
        template_id: int,
        image_url: str,
        task_id: Optional[str] = None,
        candidate_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """识别单张（可能多页）答题卡，返回可序列化的 dict"""
        cached = self._get_or_load_template(template_id)
        if cached is None:
            return {
                "scan_image_url": image_url,
                "code": 4,
                "message": "模板未找到",
                "answers": [],
            }

        images = self.image_loader.load_multi(image_url)
        logger.info("[mq] 答题卡图片加载完成: url=%s pages=%d", image_url, len(images))
        if not images:
            return {
                "scan_image_url": image_url,
                "code": 5,
                "message": "图片加载失败",
                "answers": [],
            }

        try:
            from omr_service.engine.recognizer import RecognizeContext
            from omr_service.engine.recognizers import StandardTemplateRecognizer

            recognizer = StandardTemplateRecognizer(standard_template=cached.standard_template)
            ctx = RecognizeContext()
            result = recognizer.recognize(images[0], ctx)

            personal_info: List[Dict[str, Any]] = []
            if cached.personal_info:
                personal_info = run_with_timeout(
                    lambda: self._recognize_personal_info(
                        images, cached.personal_info, candidate_names
                    ),
                    self.cfg.ocr_timeout_seconds,
                    "个人信息OCR",
                    [],
                )

            subjective_crops: List[Dict[str, Any]] = []
            if cached.subjective_regions:
                namespace = f"task_{task_id or int(time.time() * 1000)}"
                subjective_crops = run_with_timeout(
                    lambda: self._cropper.crop_subjective_regions(
                        images, cached.subjective_regions, namespace=namespace
                    ),
                    self.cfg.ocr_timeout_seconds,
                    "主观题裁剪",
                    [],
                )

            return {
                "scan_image_url": image_url,
                "code": 0,
                "message": "ok",
                "template_id": template_id,
                "answers": [
                    {
                        "q": q,
                        "answer": info.get("answer") or "",
                        "status": info.get("status", "empty"),
                        "correct": info.get("correct") or False,
                    }
                    for q, info in result.answers.items()
                ],
                "total": result.total,
                "empty_count": result.empty_count,
                "multi_count": result.multi_count,
                "card_flag": result.card_flag or "",
                "duration_ms": int(result.duration_ms),
                "personal_info": personal_info,
                "subjective_crops": subjective_crops,
            }
        except Exception as e:
            logger.exception("[mq] 单张识别失败: %s", image_url)
            return {"scan_image_url": image_url, "code": 99, "message": str(e), "answers": []}

    def _recognize_personal_info(
        self,
        images: List[np.ndarray],
        regions: List[Dict[str, Any]],
        candidate_names: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """按 page_index 分组图片，批量识别普通个人信息，并单独解析考生信息区整体框。

        candidate_names: 该考试的考生名单（花名册），随任务 payload 下发，用于姓名候选相似度裁决。
        """
        page_groups: Dict[int, List[Tuple[int, Dict[str, Any]]]] = {}
        for idx, region in enumerate(regions):
            page_index = int(region.get("page_index", 0))
            page_groups.setdefault(page_index, []).append((idx, region))

        results: List[Optional[Dict[str, Any]]] = [None] * len(regions)

        for page_index, indexed_regions in page_groups.items():
            if page_index < 0 or page_index >= len(images):
                for idx, region in indexed_regions:
                    if region.get("field") == STUDENT_INFO_BLOCK_FIELD:
                        results[idx] = {"field": STUDENT_INFO_BLOCK_FIELD, "value": "", "confidence": 0.0}
                    else:
                        results[idx] = {
                            "field": region.get("field", ""),
                            "value": "",
                            "confidence": 0.0,
                        }
                continue

            image = images[page_index]
            normal_regions = []
            normal_indices = []
            block_regions = []
            block_indices = []
            label_regions = []
            label_indices = []
            for idx, region in indexed_regions:
                field = region.get("field")
                if field == STUDENT_INFO_BLOCK_FIELD:
                    block_regions.append(region)
                    block_indices.append(idx)
                elif field == BARCODE_LABEL_FIELD:
                    label_regions.append(region)
                    label_indices.append(idx)
                else:
                    normal_regions.append(region)
                    normal_indices.append(idx)

            if normal_regions:
                page_results = self._ocr.recognize(image, normal_regions)
                for idx, page_result in zip(normal_indices, page_results):
                    result = page_result if page_result else {"field": normal_regions[normal_indices.index(idx)].get("field", ""), "value": "", "confidence": 0.0}
                    if result.get("confidence", 0.0) < self.cfg.ocr_confidence_threshold:
                        result["value"] = ""
                    results[idx] = result
                # 讯飞手写识别印证（按次计费）：与 Paddle 双引擎相互印证；
                # 手写姓名讯飞更准（实测敖其泰/乌日汗讯飞全对），非空时优先采用讯飞结果
                if self._xfyun_hw is not None:
                    for idx, region in zip(normal_indices, normal_regions):
                        if region.get("field") not in NAME_FIELD_IDS:
                            continue
                        result = results[idx]
                        paddle_value = result.get("value") or ""
                        try:
                            hw_text = self._xfyun_hw.recognize_text(PersonalInfoOcr._crop(image, region))
                            hw_name = clean_handwritten_name(hw_text)
                        except Exception as e:
                            logger.warning("讯飞手写识别失败: %s", e)
                            continue
                        if not hw_name:
                            continue
                        # 平铺讯飞识别值，供前端与 Paddle 结果相互印证
                        results.append({"field": "name_xfyun_hw", "value": hw_name, "confidence": 0.99})
                        final_name = hw_name or paddle_value
                        if candidate_names:
                            final_name = best_roster_match(
                                [hw_name, paddle_value], candidate_names) or final_name
                        if final_name != paddle_value:
                            logger.info("[mq] 手写姓名采用讯飞结果: paddle=%s xfyun=%s", paddle_value, final_name)
                        result["value"] = final_name
                        if final_name:
                            result["confidence"] = 0.99

            for idx, region in zip(label_indices, label_regions):
                # 条码标签区：小区域多变体高精度识别印刷体姓名 + 条码考号；
                # 讯飞 OCR 大模型双引擎印证（按次计费，与 Paddle 结果并列展示供人工核对）
                doc_fallback = None
                if self._xfyun_doc is not None:
                    doc_fallback = lambda crop: self._xfyun_doc.recognize_lines(crop)
                strip_result = self._ocr.recognize_label_strip(
                    image, region, candidate_names, doc_fallback=doc_fallback
                )
                results[idx] = {
                    "field": BARCODE_LABEL_FIELD,
                    "value": strip_result.get("raw_text", ""),
                    "confidence": strip_result.get("confidence", 0.0),
                }
                # 平铺追加（非空才追加，避免覆盖 block 链路结果）：
                # printed_name 由 Java 端优先于手写 name 采用
                if strip_result.get("name"):
                    results.append({"field": "printed_name", "value": strip_result["name"], "confidence": 1.0})
                if strip_result.get("exam_no"):
                    results.append({"field": "exam_no", "value": strip_result["exam_no"], "confidence": 1.0})
                # 双引擎印证值（前端并列展示）
                if strip_result.get("paddle_name"):
                    results.append({"field": "name_paddle", "value": strip_result["paddle_name"], "confidence": 1.0})
                if strip_result.get("xfyun_doc_name"):
                    results.append({"field": "name_xfyun_doc", "value": strip_result["xfyun_doc_name"], "confidence": 1.0})

            for idx, region in zip(block_indices, block_regions):
                raw_result = self._ocr.recognize_block(image, region)
                raw_text = raw_result.get("raw_text", "")
                try:
                    fields, conf = parse_personal_info_block(
                        raw_text,
                        extra_name_candidates=raw_result.get("name_candidates"),
                        candidate_names=candidate_names,
                    )
                except Exception as e:
                    logger.warning("考生信息解析异常，保留原始文本: %s | raw_text=%s", e, raw_text)
                    fields, conf = {"raw_text": raw_text}, 0.0
                # 花名册裁决后的最终姓名回写标记行，保证 Java 端二次解析结论一致
                if fields.get("name"):
                    raw_text = upsert_marked_name(raw_text, fields["name"])
                    fields["raw_text"] = raw_text
                results[idx] = {
                    "field": STUDENT_INFO_BLOCK_FIELD,
                    "value": raw_text,
                    "confidence": conf,
                }
                # 解析出的子字段平铺追加到结果列表（对齐旧 gRPC 分支，Java 端按 field 平铺读取）
                for k, v in fields.items():
                    if k == "raw_text":
                        continue
                    results.append({
                        "field": k,
                        "value": v,
                        "confidence": conf if v else 0.0,
                    })

        for idx, region in enumerate(regions):
            if results[idx] is None:
                results[idx] = {"field": region.get("field", ""), "value": "", "confidence": 0.0}
        return results

    def handle_single_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """处理单条任务，失败时自动重试，达到最大重试后标记 FAILED。

        Args:
            task: 包含 task_id, batch_id, paper_id, template_id, image_url,
                  retry_count, max_retry 的字典。

        Returns:
            {"success": bool, "task_id": str, "retrying": bool}
        """
        task_id = task.get("task_id", "")
        batch_id = task.get("batch_id", "")
        paper_id = task.get("paper_id")
        template_id = int(task.get("template_id", 0))
        image_url = task.get("image_url", "")
        retry_count = int(task.get("retry_count", 0))
        max_retry = int(task.get("max_retry", self.cfg.omr_max_retry))
        # 考生名单（花名册）：Java 端随任务下发（换行分隔），用于姓名候选相似度裁决
        candidate_names = [
            n.strip() for n in str(task.get("candidate_names") or "").split("\n") if n.strip()
        ] or None

        try:
            data = self._recognize_one(
                template_id, image_url, task_id=task_id, candidate_names=candidate_names
            )
            code = data.get("code", -1)
            if code == 0:
                payload = self._build_success_payload(task, data)
                self.producer.send_result(payload=payload)
                logger.info("[mq] 任务成功 task_id=%s", task_id)
                return {"success": True, "task_id": task_id, "retrying": False}

            # 永久性错误（模板不存在、图片加载失败）直接失败，不再重试
            if code in (4, 5):
                logger.warning(
                    "[mq] 任务遇到永久性错误，直接失败 task_id=%s code=%s message=%s",
                    task_id, code, data.get("message", ""),
                )
                payload = self._build_failure_payload(task, data.get("message", "识别失败"))
                try:
                    self.producer.send_result(payload=payload)
                except Exception as se:
                    logger.error("[mq] 失败结果发送失败 task_id=%s", task_id, exc_info=se)
                return {"success": False, "task_id": task_id, "retrying": False}

            raise RuntimeError(data.get("message", "识别失败"))
        except Exception as e:
            logger.exception("[mq] 任务处理失败 task_id=%s retry=%s", task_id, retry_count)
            retry_count += 1
            if retry_count <= max_retry:
                retry_task = dict(task)
                retry_task["retry_count"] = retry_count
                try:
                    self.producer.send_job(payload=retry_task)
                    logger.info("[mq] 任务重试入队 task_id=%s retry=%s", task_id, retry_count)
                    return {"success": False, "task_id": task_id, "retrying": True}
                except Exception as re:
                    logger.error("[mq] 重试入队失败 task_id=%s", task_id, exc_info=re)
                    retry_count = max_retry + 1

            if retry_count > max_retry:
                payload = self._build_failure_payload(task, str(e))
                try:
                    self.producer.send_result(payload=payload)
                except Exception as se:
                    logger.error("[mq] 失败结果发送失败 task_id=%s", task_id, exc_info=se)
            return {"success": False, "task_id": task_id, "retrying": False}

    def handle_parse_golden_template(self, job: Dict[str, Any]) -> Dict[str, Any]:
        """处理整张试卷模板的黄金模板解析任务（可能多页）。

        Args:
            job: 包含 job_id, template_id, pages, retry_count, max_retry 的字典。

        Returns:
            {"success": bool, "job_id": str, "retrying": bool}
        """
        job_id = job.get("job_id", "")
        template_id = int(job.get("template_id", 0))
        retry_count = int(job.get("retry_count", 0))
        max_retry = int(job.get("max_retry", self.cfg.omr_max_retry))

        try:
            data = self._do_parse_golden_template(job)
            code = data.get("code", -1)
            if code == 0:
                payload = self._build_parse_success_payload(job_id, template_id, data)
                self.producer.send_result(payload=payload)
                logger.info("[mq] 黄金模板解析任务成功 job_id=%s", job_id)
                return {"success": True, "job_id": job_id, "retrying": False}

            # 永久性错误（请求无效、图片加载失败）直接失败，不再重试
            if code in (4, 5, 6):
                logger.warning(
                    "[mq] 黄金模板解析遇到永久性错误，直接失败 job_id=%s code=%s message=%s",
                    job_id, code, data.get("message", ""),
                )
                payload = self._build_parse_failure_payload(
                    job_id, template_id, data.get("message", "黄金模板解析失败")
                )
                self.producer.send_result(payload=payload)
                return {"success": False, "job_id": job_id, "retrying": False}

            raise RuntimeError(data.get("message", "黄金模板解析失败"))
        except Exception as e:
            logger.exception("[mq] 黄金模板解析任务失败 job_id=%s retry=%s", job_id, retry_count)
            retry_count += 1
            if retry_count <= max_retry:
                retry_job = dict(job)
                retry_job["retry_count"] = retry_count
                try:
                    self.producer.send_job(payload=retry_job)
                    logger.info("[mq] 黄金模板解析任务重试入队 job_id=%s retry=%s", job_id, retry_count)
                    return {"success": False, "job_id": job_id, "retrying": True}
                except Exception as re:
                    logger.error("[mq] 黄金模板解析重试入队失败 job_id=%s", job_id, exc_info=re)
                    retry_count = max_retry + 1

            if retry_count > max_retry:
                payload = self._build_parse_failure_payload(job_id, template_id, str(e))
                try:
                    self.producer.send_result(payload=payload)
                except Exception as se:
                    logger.error("[mq] 黄金模板解析失败结果发送失败 job_id=%s", job_id, exc_info=se)
            return {"success": False, "job_id": job_id, "retrying": False}

    def _do_parse_golden_template(self, job: Dict[str, Any]) -> Dict[str, Any]:
        """逐页调用 OmrService.parse_golden_template 并合并结果。"""
        template_id = int(job.get("template_id", 0))
        pages: List[Dict[str, Any]] = job.get("pages", []) or []
        if not pages:
            return {"code": 6, "message": "pages 不能为空"}

        if self.service is None:
            return {"code": 99, "message": "service 未初始化，无法解析黄金模板"}

        merged_answers: Dict[str, str] = {}
        merged_bubbles: List[Dict[str, Any]] = []
        merged_personal_info: List[Dict[str, Any]] = []
        merged_subjective_crops: List[Dict[str, Any]] = []
        total = 0
        page_errors: List[str] = []

        for page in pages:
            page_index = page.get("page_index", page.get("pageIndex", 0))
            request = self._build_parse_request(template_id, page)
            try:
                resp_dict = self.service.parse_golden_template(request)
            except ImageLoadError as exc:
                # 图片加载失败是永久性错误，重试无意义，直接失败
                logger.error("[mq] 黄金模板解析图片加载失败 page=%s: %s", page_index, exc)
                return {"code": 5, "message": f"第{page_index + 1}页: {exc}"}
            except Exception as exc:
                logger.exception("[mq] 黄金模板解析异常 page=%s", page_index)
                page_errors.append(f"第{page_index + 1}页: {exc}")
                continue

            code = resp_dict.get("code", -1)
            if code != 0:
                msg = f"第{page_index + 1}页: {resp_dict.get('message', '解析失败')}"
                # 永久性错误（模板不存在/图片加载失败/请求无效）直接中止，不进入重试
                if code in (4, 5, 6):
                    return {"code": code, "message": msg}
                page_errors.append(msg)
                continue

            page_answers = resp_dict.get("answers", []) or []
            # 新 OmrService 返回 list；旧协议可能返回 dict。统一合并为 dict。
            if isinstance(page_answers, dict):
                merged_answers.update(page_answers)
            else:
                for ans in page_answers:
                    q_no = ans.get("question_no") if isinstance(ans, dict) else None
                    if q_no is None:
                        continue
                    selected = ans.get("selected", []) if isinstance(ans, dict) else []
                    merged_answers[str(q_no)] = ",".join(selected) if selected else ""
            bubble_grid = resp_dict.get("bubble_grid") or []
            for b in bubble_grid:
                merged_bubbles.append(b)
            personal_info_sample = resp_dict.get("personal_info_sample")
            if personal_info_sample is not None:
                if isinstance(personal_info_sample, list):
                    merged_personal_info.extend(personal_info_sample)
                elif isinstance(personal_info_sample, dict):
                    merged_personal_info.append(personal_info_sample)
            subjective_crops = resp_dict.get("subjective_crops") or []
            merged_subjective_crops.extend(subjective_crops)
            total += len(page_answers) if isinstance(page_answers, list) else len(page_answers)

        if page_errors and total == 0:
            return {"code": 99, "message": "; ".join(page_errors)}

        return {
            "code": 0,
            "message": "ok",
            "template_id": template_id,
            "answers": merged_answers,
            "bubbles": merged_bubbles,
            "personal_info": merged_personal_info,
            "subjective_crops": merged_subjective_crops,
            "total": total,
            "total_images": len(pages),
            "success_count": len(pages),
            "failed_count": 0,
        }

    @staticmethod
    def _build_parse_request(template_id: int, page: Dict[str, Any]) -> Dict[str, Any]:
        """将 MQ 任务的 page 信息转换为 OmrService.parse_golden_template 的输入 dict.

        Java 端字段为 camelCase (startQ/numQ/numOptions), StandardTemplate 期望 snake_case
        (start_q/num_q/num_options). 此处做映射转换.
        """
        columns_raw = page.get("columns", []) or []
        columns: List[Dict[str, Any]] = []
        for c in columns_raw:
            cols = {
                "x1": int(c.get("x1", 0)),
                "y1": int(c.get("y1", 0)),
                "x2": int(c.get("x2", 0)),
                "y2": int(c.get("y2", 0)),
                "start_q": int(c.get("startQ", c.get("start_q", 1))),
                "num_q": int(c.get("numQ", c.get("num_q", 0))),
                "num_options": int(c.get("numOptions", c.get("num_options", 4))),
                "option_axis": c.get("optionAxis", c.get("option_axis", "x")),
                "reverse_q": c.get("reverseQ", c.get("reverse_q", False)),
                "page_index": int(c.get("pageIndex", c.get("page_index", 0))),
            }
            logger.debug("[mq] column: x1=%s y1=%s x2=%s y2=%s startQ=%s numQ=%s numOpt=%s",
                         cols["x1"], cols["y1"], cols["x2"], cols["y2"],
                         cols["start_q"], cols["num_q"], cols["num_options"])
            columns.append(cols)

        request: Dict[str, Any] = {
            "template_id": template_id,
            "template_image_url": page.get("template_image_url") or page.get("templateImageUrl"),
            "columns": columns,
            # 传入页码供模板缓存 page_images 使用真实页码作为键
            "page_index": _as_int(page.get("page_index") if page.get("page_index") is not None else page.get("pageIndex")),
        }
        # personalInfo / subjectiveRegions 来自 Java 端
        pi = page.get("personalInfo")
        sr = page.get("subjectiveRegions")
        logger.info("[mq] 模板解析请求: template_id=%s image=%s columns=%d personalInfo=%s subjectiveRegions=%s",
                    template_id, request["template_image_url"], len(columns),
                    "yes" if pi else "no", "yes" if sr else "no")
        if pi:
            request["personal_info_region"] = _normalize_personal_info(pi)
        if sr:
            request["subjective_regions"] = _normalize_subjective_regions(sr)
        return request

    def _build_parse_success_payload(
        self,
        job_id: str,
        template_id: int,
        data: Dict[str, Any],
    ) -> Dict[str, Any]:
        return {
            "job_id": job_id,
            "job_type": "parse_golden_template",
            "template_id": template_id,
            "status": 2,
            "answers": data.get("answers", {}),
            "bubbles": data.get("bubbles", []),
            "personal_info": data.get("personal_info", []),
            "subjective_crops": data.get("subjective_crops", []),
            "total": data.get("total", 0),
            "total_images": data.get("total_images", 1),
            "success_count": data.get("success_count", 1),
            "failed_count": data.get("failed_count", 0),
        }

    def _build_parse_failure_payload(
        self,
        job_id: str,
        template_id: int,
        error_msg: str,
    ) -> Dict[str, Any]:
        return {
            "job_id": job_id,
            "job_type": "parse_golden_template",
            "template_id": template_id,
            "status": 3,
            "error_msg": error_msg,
            "answers": {},
            "bubbles": [],
            "personal_info": [],
            "subjective_crops": [],
            "total": 0,
            "total_images": 0,
            "success_count": 0,
            "failed_count": 1,
        }

    def handle(self, job: Dict[str, Any]) -> Dict[str, Any]:
        """兼容旧版聚合消息：拆分为单任务逐个处理。"""
        job_id = job.get("job_id", "")
        template_id = int(job.get("template_id", 0))
        image_url = job.get("image_url")
        image_urls: List[str] = job.get("image_urls", []) or []
        if image_url:
            image_urls = [image_url]

        logger.info("[mq] 开始批量任务 job_id=%s template_id=%s images=%d", job_id, template_id, len(image_urls))
        completed = 0
        failed = 0
        for idx, url in enumerate(image_urls):
            task = {
                "task_id": job.get("task_id") or f"{job_id}_{idx}_{int(time.time() * 1000)}",
                "batch_id": job.get("batch_id", ""),
                "paper_id": job.get("paper_id"),
                "template_id": template_id,
                "image_url": url,
                "retry_count": 0,
                "max_retry": self.cfg.omr_max_retry,
            }
            res = self.handle_single_task(task)
            if res.get("success"):
                completed += 1
            elif not res.get("retrying"):
                failed += 1
        logger.info("[mq] 批量任务完成 job_id=%s completed=%d failed=%d", job_id, completed, failed)
        return {"job_id": job_id, "completed": completed, "failed": failed}

    def _build_success_payload(self, task: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "task_id": task.get("task_id", ""),
            "batch_id": task.get("batch_id", ""),
            "paper_id": task.get("paper_id"),
            "template_id": task.get("template_id", 0),
            "status": 2,
            "scan_image_url": data.get("scan_image_url"),
            "answers": data.get("answers", []),
            "personal_info": data.get("personal_info", []),
            "subjective_crops": data.get("subjective_crops", []),
            "card_flag": data.get("card_flag", ""),
            "duration_ms": data.get("duration_ms", 0),
        }

    def _build_failure_payload(self, task: Dict[str, Any], error_msg: str) -> Dict[str, Any]:
        return {
            "task_id": task.get("task_id", ""),
            "batch_id": task.get("batch_id", ""),
            "paper_id": task.get("paper_id"),
            "template_id": task.get("template_id", 0),
            "status": 3,
            "scan_image_url": task.get("image_url", ""),
            "answers": [],
            "personal_info": [],
            "subjective_crops": [],
            "card_flag": "",
            "duration_ms": 0,
            "error_msg": error_msg,
        }
