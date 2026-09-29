# -*- coding: utf-8 -*-
"""讯飞 OCR 引擎封装（可选增强通道，默认关闭）

- XfyunDocOcr：通用文档识别（OCR 大模型），印刷体标签条识别，HMAC-SHA256 URL 鉴权。
- XfyunHandwritingOcr：手写文字识别，手写姓名行识别，X-CheckSum（MD5）鉴权。

成本控制：两个接口均按调用次数计费，默认只作为 PaddleOCR 的兜底通道
（Paddle 无结果时才调用），由调用方控制触发时机。

凭证通过 Nacos（omr-service.yaml 的 xfyun.*）或环境变量（OMR_XFYUN_*）注入，不写入仓库。
"""
import base64
import hashlib
import hmac
import json
import logging
import ssl
import time
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode
from wsgiref.handlers import format_date_time

import cv2
import numpy as np
import requests
from requests.adapters import HTTPAdapter

logger = logging.getLogger(__name__)

# 通用文档识别（OCR 大模型）默认接口地址
DOC_OCR_URL = "https://cbm01.cn-huabei-1.xf-yun.com/v1/private/se75ocrbm"
# 手写文字识别默认接口地址
HW_OCR_URL = "https://webapi.xfyun.cn/v1/service/v1/ocr/handwriting"


class _LegacyTLSAdapter(HTTPAdapter):
    """兼容弱签名证书链的 TLS 适配器（仅在 xfyun_verify_ssl=false 时启用）。

    适用于部署环境存在 MITM 代理、其 TLS 证书签名被 OpenSSL 3 拒绝的场景。
    注意：启用后关闭证书校验，仅限可信内网/代理环境使用。
    """

    def init_poolmanager(self, *args, **kwargs):
        ctx = ssl.create_default_context()
        ctx.set_ciphers("DEFAULT:@SECLEVEL=0")
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        kwargs["ssl_context"] = ctx
        return super().init_poolmanager(*args, **kwargs)


def _make_session(verify_ssl: bool) -> requests.Session:
    session = requests.Session()
    if not verify_ssl:
        session.mount("https://", _LegacyTLSAdapter())
        logger.warning("讯飞 OCR：xfyun_verify_ssl=false，已关闭 TLS 证书校验（仅限可信代理环境）")
    return session


def _encode_jpg(image: np.ndarray) -> bytes:
    return cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tobytes()


class XfyunDocOcr:
    """通用文档识别（OCR 大模型）客户端：印刷体标签条识别。"""

    def __init__(self, app_id: str, api_key: str, api_secret: str,
                 url: str = DOC_OCR_URL, timeout: float = 10.0, verify_ssl: bool = True):
        self.app_id = app_id
        self.api_key = api_key
        self.api_secret = api_secret
        self.url = url
        self.timeout = timeout
        self._session = _make_session(verify_ssl)

    @property
    def available(self) -> bool:
        return bool(self.app_id and self.api_key and self.api_secret)

    def _auth_url(self) -> str:
        """HMAC-SHA256 URL 鉴权（host/date/request-line 签名）。"""
        from urllib.parse import urlparse

        parsed = urlparse(self.url)
        host = parsed.netloc
        path = parsed.path
        date = format_date_time(time.time())  # RFC1123 GMT
        signature_origin = f"host: {host}\ndate: {date}\nPOST {path} HTTP/1.1"
        signature_sha = hmac.new(self.api_secret.encode(), signature_origin.encode(),
                                 digestmod=hashlib.sha256).digest()
        signature = base64.b64encode(signature_sha).decode()
        authorization_origin = (
            f'api_key="{self.api_key}", algorithm="hmac-sha256", '
            f'headers="host date request-line", signature="{signature}"'
        )
        authorization = base64.b64encode(authorization_origin.encode()).decode()
        return self.url + "?" + urlencode({"host": host, "date": date, "authorization": authorization})

    def recognize_lines(self, image: np.ndarray) -> List[str]:
        """识别图片，返回文本行列表。失败抛出 RuntimeError（调用方兜底）。"""
        body = {
            "header": {"app_id": self.app_id, "status": 2},
            "parameter": {"ocr": {
                "result_option": "normal",
                "result_format": "json",
                "output_type": "one_shot",
                "exif_option": "0",
                "result": {"encoding": "utf8", "compress": "raw", "format": "plain"},
            }},
            "payload": {"image": {
                "encoding": "jpg",
                "image": base64.b64encode(_encode_jpg(image)).decode(),
                "status": 2,
            }},
        }
        resp = self._session.post(self._auth_url(), json=body, timeout=self.timeout)
        data = resp.json()
        header = data.get("header", {})
        if header.get("code") != 0:
            raise RuntimeError(
                f"讯飞OCR大模型错误: HTTP {resp.status_code} code={header.get('code')} "
                f"message={header.get('message')} sid={header.get('sid')}"
            )
        text = base64.b64decode(data["payload"]["result"]["text"]).decode("utf-8")
        return _extract_text_lines(text)


def _extract_text_lines(text: str) -> List[str]:
    """把 OCR 大模型输出整理为文本行列表。

    result_format=json 时 text 是 JSON 字符串（结构随版本可能变化），
    递归收集所有含中文的字符串值；非 JSON 时按行切分。
    """
    text = (text or "").strip()
    if not text:
        return []
    if text.startswith("{") or text.startswith("["):
        try:
            parsed = json.loads(text)
        except Exception:
            parsed = None
        if parsed is not None:
            lines: List[str] = []

            def walk(node: Any) -> None:
                if isinstance(node, str):
                    if any("一" <= ch <= "鿿" for ch in node):
                        lines.append(node)
                elif isinstance(node, dict):
                    for v in node.values():
                        walk(v)
                elif isinstance(node, list):
                    for item in node:
                        walk(item)

            walk(parsed)
            if lines:
                return lines
    return [line.strip() for line in text.splitlines() if line.strip()]


class XfyunHandwritingOcr:
    """手写文字识别客户端：手写姓名行识别（X-CheckSum MD5 鉴权）。"""

    def __init__(self, app_id: str, api_key: str,
                 url: str = HW_OCR_URL, timeout: float = 10.0, verify_ssl: bool = True):
        self.app_id = app_id
        self.api_key = api_key
        self.url = url
        self.timeout = timeout
        self._session = _make_session(verify_ssl)

    @property
    def available(self) -> bool:
        return bool(self.app_id and self.api_key)

    def recognize_text(self, image: np.ndarray) -> str:
        """识别图片中的手写文字，返回拼接文本。失败抛出 RuntimeError（调用方兜底）。"""
        cur_time = str(int(time.time()))
        # language 可选值仅 en / cn|en（写 cn 会报 10107）
        param = base64.b64encode(
            json.dumps({"language": "cn|en", "location": "false"}).encode()
        ).decode()
        check_sum = hashlib.md5((self.api_key + cur_time + param).encode()).hexdigest()
        headers = {
            "X-Appid": self.app_id,
            "X-CurTime": cur_time,
            "X-Param": param,
            "X-CheckSum": check_sum,
            "Content-Type": "application/x-www-form-urlencoded; charset=utf-8",
        }
        resp = self._session.post(
            self.url, headers=headers,
            data={"image": base64.b64encode(_encode_jpg(image)).decode()},
            timeout=self.timeout,
        )
        data = resp.json()
        if str(data.get("code")) != "0":
            raise RuntimeError(
                f"讯飞手写识别错误: code={data.get('code')} desc={data.get('desc')} sid={data.get('sid')}"
            )
        texts: List[str] = []

        def walk(node: Any) -> None:
            if isinstance(node, dict):
                content = node.get("content")
                if isinstance(content, str):
                    texts.append(content)
                for v in node.values():
                    walk(v)
            elif isinstance(node, list):
                for item in node:
                    walk(item)

        walk(data.get("data", {}))
        return " ".join(texts)


def build_xfyun_clients(cfg: Any) -> tuple:
    """按配置构建讯飞客户端（xfyun_enabled=false 或凭证缺失时返回 (None, None)）。

    cfg 兼容 OmrConfig（MQ 链路）与 OmrSettings（HTTP 链路），两者字段同名。
    """
    if not getattr(cfg, "xfyun_enabled", False):
        return None, None
    timeout = float(getattr(cfg, "xfyun_timeout_seconds", 10.0))
    verify_ssl = bool(getattr(cfg, "xfyun_verify_ssl", True))
    app_id = getattr(cfg, "xfyun_app_id", "") or ""
    doc = XfyunDocOcr(
        app_id=app_id,
        api_key=getattr(cfg, "xfyun_doc_api_key", "") or "",
        api_secret=getattr(cfg, "xfyun_doc_api_secret", "") or "",
        url=getattr(cfg, "xfyun_doc_url", "") or DOC_OCR_URL,
        timeout=timeout,
        verify_ssl=verify_ssl,
    )
    hw = XfyunHandwritingOcr(
        app_id=app_id,
        api_key=getattr(cfg, "xfyun_hw_api_key", "") or "",
        url=getattr(cfg, "xfyun_hw_url", "") or HW_OCR_URL,
        timeout=timeout,
        verify_ssl=verify_ssl,
    )
    doc = doc if doc.available else None
    hw = hw if hw.available else None
    if doc is None and hw is None:
        logger.warning("讯飞 OCR 已启用但未配置有效凭证，增强通道不生效")
    else:
        logger.info("讯飞 OCR 增强通道已启用: doc=%s, handwriting=%s",
                    doc is not None, hw is not None)
    return doc, hw
