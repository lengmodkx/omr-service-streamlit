"""POST /v1/ocr/handwriting、/v1/ocr/formula - 通用 OCR 对外路由.

供 AI 判卷 Java 端调用：整图 OCR，返回行级文本与置信度。
公式 OCR 一期没有专用公式模型：返回 PaddleOCR 文本，latex 恒为 null。
"""
from __future__ import annotations

import base64
import binascii
import logging

import cv2
import numpy as np
from fastapi import APIRouter, Request

from omr_service.api.deps import get_service
from omr_service.api.schemas.ocr import OcrRequest, OcrResponse
from omr_service.core.exceptions import ImageLoadError, InvalidRequestError
from omr_service.core.service import run_with_timeout
from omr_service.engine.ocr import PersonalInfoOcr

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/ocr", tags=["ocr"])

# 单次 OCR 推理超时（秒）：Paddle 推理可能挂起，超时返回空文本而非抛 500
_OCR_TIMEOUT_SECONDS = 30.0

_EMPTY_RESULT = {"text": "", "lines": [], "engine": "paddle"}


def _strip_data_uri_prefix(value: str) -> str:
    """去掉 data URI 前缀（如 data:image/png;base64,），仅保留纯 base64。"""
    if value.startswith("data:") and "," in value:
        return value.split(",", 1)[1]
    return value


def _load_image(request: Request, body: OcrRequest) -> np.ndarray:
    """按 image_url / image_base64 二选一加载图片；失败抛出明确业务错误（不抛 500）。

    - image_url：复用现有 ImageLoader（超时/大小限制/重试安全限制），失败抛 ImageLoadError(502)；
    - image_base64：base64 解码失败 / 内容非图片抛 InvalidRequestError(400)。
    """
    if body.image_url:
        service = get_service(request)
        image = service.image_loader.load(body.image_url)
        if image is None:
            raise ImageLoadError(body.image_url, "下载失败或超时")
        return image

    payload = _strip_data_uri_prefix((body.image_base64 or "").strip())
    try:
        raw = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as e:
        raise InvalidRequestError("image_base64", "base64 解码失败") from e
    if not raw:
        raise InvalidRequestError("image_base64", "内容为空")

    arr = np.frombuffer(raw, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise InvalidRequestError("image_base64", "无法解码为图片")
    return image


def _run_ocr(image: np.ndarray) -> dict:
    """复用现有 PaddleOCR 引擎整图识别；超时/引擎不可用/识别为空均返回空文本（不抛 500）。"""
    if image is None or image.size == 0:
        return _EMPTY_RESULT
    try:
        result = run_with_timeout(
            lambda: PersonalInfoOcr().recognize_page_lines(image),
            timeout_seconds=_OCR_TIMEOUT_SECONDS,
            desc="通用 OCR 推理",
            default=_EMPTY_RESULT,
        )
    except Exception as e:  # noqa: BLE001 - 兜底，保证 OCR 链路永不抛 500
        logger.warning("通用 OCR 推理异常: %s", e)
        return _EMPTY_RESULT
    return result or _EMPTY_RESULT


@router.post("/handwriting", response_model=OcrResponse)
def handwriting(request: Request, body: OcrRequest) -> dict:
    """手写文字 OCR：复用现有 PaddleOCR 引擎整图识别，返回行级文本与置信度。

    无图/识别为空返回 200 空文本；参数非法返回 400；图片下载失败返回 502（详见接口文档）。
    """
    image = _load_image(request, body)
    return _run_ocr(image)


@router.post("/formula", response_model=OcrResponse)
def formula(request: Request, body: OcrRequest) -> dict:
    """公式 OCR（一期）：没有专用公式模型，返回 PaddleOCR 文本行；latex 恒为 null。"""
    image = _load_image(request, body)
    result = _run_ocr(image)
    # 一期无专用公式模型：latex 恒为 null（接口文档已注明「公式 OCR 一期仅文本」）
    result["latex"] = None
    return result
