"""通用 OCR 路由请求/响应 schema（/v1/ocr/handwriting、/v1/ocr/formula）。"""
from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


class OcrRequest(BaseModel):
    """OCR 请求：image_url 与 image_base64 必须且只能提供其一。"""
    image_url: str | None = Field(
        default=None,
        description="图片 URL（http/https 或本地路径），下载复用现有图片加载器安全限制",
    )
    image_base64: str | None = Field(
        default=None,
        description="图片 base64（可选 data URI 前缀，如 data:image/png;base64,...）",
    )

    @model_validator(mode="after")
    def _check_exactly_one_source(self) -> "OcrRequest":
        """二选一校验：两者都缺或都传均为非法请求。"""
        has_url = bool((self.image_url or "").strip())
        has_b64 = bool((self.image_base64 or "").strip())
        if has_url == has_b64:
            raise ValueError("image_url 与 image_base64 必须且只能提供其一")
        return self


class OcrLine(BaseModel):
    """单行识别结果。"""
    text: str = ""
    confidence: float = 0.0


class OcrResponse(BaseModel):
    """OCR 识别响应。

    公式 OCR 一期没有专用公式模型，latex 恒为 null（见接口文档「公式 OCR 一期仅文本」）；
    手写路由同样输出该字段，保持两个路由响应结构统一。
    """
    text: str = ""
    lines: list[OcrLine] = Field(default_factory=list)
    engine: str = "paddle"
    latex: str | None = None
