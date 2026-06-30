"""gRPC OmrService 实现"""
import logging
import time
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from omr_service.config import OmrConfig
from omr_service.engine.recognizer import RecognizeContext
from omr_service.engine.recognizers import StandardTemplateRecognizer
from omr_service.engine.standard_template import StandardTemplate
from omr_service.loader.image_loader import ImageLoader
from omr_service.loader.template_store import TemplateStore
from omr_service.rpc import omr_pb2
from omr_service.rpc import omr_pb2_grpc
from omr_service.worker.pool import WorkerPool

logger = logging.getLogger(__name__)

CODE_OK = 0
CODE_TEMPLATE_NOT_FOUND = 4
CODE_IMAGE_LOAD_FAIL = 5
CODE_INVALID_REQUEST = 6
CODE_INTERNAL_ERROR = 99


def _column_config_from_proto(cfg: omr_pb2.ColumnConfig) -> Dict[str, Any]:
    return {
        "x1": cfg.x1,
        "y1": cfg.y1,
        "x2": cfg.x2,
        "y2": cfg.y2,
        "start_q": cfg.start_q,
        "num_q": cfg.num_q,
        "num_options": cfg.num_options,
        "option_axis": cfg.option_axis or "x",
        "reverse_q": cfg.reverse_q,
    }


def _bubble_to_proto(b: Dict[str, Any]) -> omr_pb2.Bubble:
    return omr_pb2.Bubble(
        q=b["q"],
        opt=b["opt"],
        x=b["x"],
        y=b["y"],
        w=b["w"],
        h=b["h"],
    )


def _result_to_proto(
    template_id: int,
    image_url: str,
    result: Any,
    code: int = CODE_OK,
    message: str = "ok",
) -> omr_pb2.RecognizeResult:
    """将 StandardTemplateRecognizer 结果转换为 protobuf"""
    answers_pb = []
    for q, info in result.answers.items():
        answers_pb.append(
            omr_pb2.QuestionAnswer(
                q=q,
                answer=info.get("answer") or "",
                status=info.get("status", "empty"),
                correct=info.get("correct") or False,
            )
        )
    return omr_pb2.RecognizeResult(
        code=code,
        message=message,
        template_id=template_id,
        scan_image_url=image_url,
        answers=answers_pb,
        total=result.total,
        empty_count=result.empty_count,
        multi_count=result.multi_count,
        card_flag=result.card_flag or "",
        duration_ms=int(result.duration_ms),
    )


class OmrServiceServicer(omr_pb2_grpc.OmrServiceServicer):
    """OMR gRPC 服务实现"""

    def __init__(
        self,
        cfg: OmrConfig,
        template_store: TemplateStore,
        image_loader: ImageLoader,
        worker_pool: WorkerPool,
    ):
        self.cfg = cfg
        self.template_store = template_store
        self.image_loader = image_loader
        self.worker_pool = worker_pool

    def _get_template(self, template_id: int) -> Optional[StandardTemplate]:
        return self.template_store.get(template_id)

    def _load_image(self, url: str) -> Optional[np.ndarray]:
        return self.image_loader.load(url)

    def ParseGoldenTemplate(self, request, context):
        """解析黄金模板"""
        logger.info("[rpc] ParseGoldenTemplate template_id=%s url=%s", request.template_id, request.template_image_url)

        if request.template_id == 0 or not request.template_image_url:
            return omr_pb2.GoldenTemplateResult(
                code=CODE_INVALID_REQUEST,
                message="template_id / template_image_url 必填",
            )

        img = self._load_image(request.template_image_url)
        if img is None:
            return omr_pb2.GoldenTemplateResult(
                code=CODE_IMAGE_LOAD_FAIL,
                message="模板图片加载失败",
            )

        columns = [_column_config_from_proto(c) for c in request.columns]
        if not columns:
            return omr_pb2.GoldenTemplateResult(
                code=CODE_INVALID_REQUEST,
                message="columns 不能为空",
            )

        try:
            tpl = StandardTemplate(image=img, column_configs=columns)
            self.template_store.set(request.template_id, tpl)

            bubbles_pb = [_bubble_to_proto(b) for b in tpl.bubbles]
            return omr_pb2.GoldenTemplateResult(
                code=CODE_OK,
                message=f"黄金模板解析成功，共 {len(tpl.bubbles)} 个气泡",
                template_id=request.template_id,
                bubbles=bubbles_pb,
                answers=tpl.answers,
                total=len(tpl.bubbles),
            )
        except Exception as e:
            logger.exception("[rpc] ParseGoldenTemplate 失败")
            return omr_pb2.GoldenTemplateResult(
                code=CODE_INTERNAL_ERROR,
                message=f"解析失败: {e}",
            )

    def RecognizeByTemplate(self, request, context):
        """根据模板识别单张答题卡"""
        logger.info("[rpc] RecognizeByTemplate template_id=%s url=%s", request.template_id, request.scan_image_url)
        return self._recognize(request.template_id, request.scan_image_url)

    def ReverifyPaper(self, request, context):
        """单张试卷复验"""
        logger.info("[rpc] ReverifyPaper template_id=%s url=%s", request.template_id, request.scan_image_url)
        return self._recognize(request.template_id, request.scan_image_url)

    def _recognize(self, template_id: int, image_url: str) -> omr_pb2.RecognizeResult:
        if template_id == 0 or not image_url:
            return omr_pb2.RecognizeResult(
                code=CODE_INVALID_REQUEST,
                message="template_id / scan_image_url 必填",
            )

        tpl = self._get_template(template_id)
        if tpl is None:
            return omr_pb2.RecognizeResult(
                code=CODE_TEMPLATE_NOT_FOUND,
                message="模板未找到，请先调用 ParseGoldenTemplate",
            )

        img = self._load_image(image_url)
        if img is None:
            return omr_pb2.RecognizeResult(
                code=CODE_IMAGE_LOAD_FAIL,
                message="答题卡图片加载失败",
                template_id=template_id,
                scan_image_url=image_url,
            )

        try:
            recognizer = StandardTemplateRecognizer(standard_template=tpl)
            ctx = RecognizeContext()
            result = recognizer.recognize(img, ctx)
            return _result_to_proto(template_id, image_url, result)
        except Exception as e:
            logger.exception("[rpc] 识别失败")
            return omr_pb2.RecognizeResult(
                code=CODE_INTERNAL_ERROR,
                message=f"识别失败: {e}",
                template_id=template_id,
                scan_image_url=image_url,
            )

    def VerifyRecognitionRate(self, request, context):
        """验证黄金模板识别成功率"""
        logger.info(
            "[rpc] VerifyRecognitionRate template_id=%s images=%d",
            request.template_id,
            len(request.image_urls),
        )

        template_id = request.template_id
        if template_id == 0:
            return omr_pb2.VerifyRateResult(
                code=CODE_INVALID_REQUEST,
                message="template_id 必填",
            )

        tpl = self._get_template(template_id)
        if tpl is None:
            return omr_pb2.VerifyRateResult(
                code=CODE_TEMPLATE_NOT_FOUND,
                message="模板未找到，请先调用 ParseGoldenTemplate",
            )

        expected = dict(request.expected_answers)
        if not expected:
            return omr_pb2.VerifyRateResult(
                code=CODE_INVALID_REQUEST,
                message="expected_answers 不能为空",
            )

        image_urls = list(request.image_urls)
        if not image_urls:
            return omr_pb2.VerifyRateResult(
                code=CODE_INVALID_REQUEST,
                message="image_urls 不能为空",
            )

        def _task(url: str) -> omr_pb2.RecognizeResult:
            return self._recognize(template_id, url)

        # 并发识别
        futures = [self.worker_pool.submit(_task, url) for url in image_urls]
        details: List[omr_pb2.RecognizeResult] = [f.result() for f in futures]

        matched = 0
        for detail in details:
            if detail.code != CODE_OK:
                continue
            actual = {a.q: a.answer for a in detail.answers}
            # 只比较 expected 里的题号
            ok = True
            for q, exp_ans in expected.items():
                if actual.get(q, "") != (exp_ans or ""):
                    ok = False
                    break
            if ok:
                matched += 1

        total = len(details)
        success_rate = matched / total if total > 0 else 0.0

        return omr_pb2.VerifyRateResult(
            code=CODE_OK,
            message="成功率验证完成",
            success_rate=success_rate,
            total=total,
            matched=matched,
            details=details,
        )
