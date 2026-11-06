"""图片加载器 — 支持 HTTP 下载、本地路径、内存缓存"""
import io
import logging
import os
from typing import List, Optional

import cv2
import numpy as np
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


class ImageLoader:
    """加载答题卡图片"""

    def __init__(self, timeout: int = 30, max_bytes: int = 50 * 1024 * 1024):
        self.timeout = timeout
        self.max_bytes = max_bytes
        self._session = requests.Session()

    def load(self, url: str) -> Optional[np.ndarray]:
        """加载图片，返回 BGR np.ndarray；失败返回 None。

        支持 url 字段里塞了多个 URL（用逗号分隔）的情况，取第一个能成功加载的。
        """
        if not url:
            return None
        urls = [u.strip() for u in url.split(",") if u.strip()]
        if not urls:
            return None

        for one_url in urls:
            # 本地文件
            if os.path.exists(one_url):
                try:
                    img = self._load_local(one_url)
                    if img is not None:
                        return img
                except Exception as exc:
                    logger.warning("本地图片读取异常 %s: %s", one_url, exc)
                continue

            # HTTP(S) 下载
            if one_url.startswith(("http://", "https://")):
                try:
                    img = self._download_one(one_url)
                    if img is not None:
                        return img
                except Exception as exc:
                    logger.warning("图片下载失败 %s: %s", one_url, exc)
            else:
                logger.debug("非 HTTP/本地路径，已跳过: %s", one_url)

        logger.warning("所有 URL 均加载失败: %s", url)
        return None

    def load_multi(self, url: str) -> List[np.ndarray]:
        """加载多页答题卡图片，返回 BGR 图片列表。

        逗号分隔的 URL 按顺序加载；本地文件不存在或 HTTP 下载失败时跳过并记录警告。
        """
        if not url:
            return []
        urls = [u.strip() for u in url.split(",") if u.strip()]
        if not urls:
            return []

        images: List[np.ndarray] = []
        for one_url in urls:
            img: Optional[np.ndarray] = None
            if os.path.exists(one_url):
                try:
                    img = self._load_local(one_url)
                except Exception as exc:
                    logger.warning("本地图片读取异常 %s: %s", one_url, exc)
            elif one_url.startswith(("http://", "https://")):
                try:
                    img = self._download_one(one_url)
                except Exception as exc:
                    logger.warning("图片下载失败 %s: %s", one_url, exc)
                    continue
            else:
                logger.debug("非 HTTP/本地路径，已跳过: %s", one_url)
                continue
            if img is not None:
                images.append(img)
            else:
                logger.warning("图片加载失败，已跳过: %s", one_url)
        return images

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=5),
        reraise=True,
    )
    def _download_one(self, url: str) -> Optional[np.ndarray]:
        """下载单张 HTTP 图片"""
        with self._session.get(url, timeout=self.timeout, stream=True) as resp:
            resp.raise_for_status()
            content_length = resp.headers.get("Content-Length")
            if content_length and int(content_length) > self.max_bytes:
                raise ValueError(f"图片大小超过限制: {content_length} bytes")

            data = b""
            for chunk in resp.iter_content(chunk_size=8192):
                data += chunk
                if len(data) > self.max_bytes:
                    raise ValueError("图片下载超过最大字节限制")

        if not data:
            return None

        arr = np.frombuffer(data, dtype=np.uint8)
        return self._decode(arr)

    @staticmethod
    def _decode(arr: np.ndarray) -> Optional[np.ndarray]:
        """解码图片并应用 EXIF 方向。

        手机拍照（微信图片）常带 EXIF Orientation（6/8），OpenCV 的 imdecode 不读 EXIF，
        会把 A3 横向答题卡解成竖向，导致模板匹配失败。这里先按 EXIF 摆正，再转 BGR。
        """
        try:
            from PIL import Image, ImageOps

            with Image.open(io.BytesIO(arr.tobytes())) as image:
                orientation = image.getexif().get(274)
                if orientation in (2, 3, 4, 5, 6, 7, 8):
                    image = ImageOps.exif_transpose(image)
                rgb = np.array(image.convert("RGB"))
                return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        except Exception as exc:
            logger.warning("按 EXIF 解码失败，回退 OpenCV 解码: %s", exc)

        return cv2.imdecode(arr, cv2.IMREAD_COLOR)

    @staticmethod
    def _load_local(path: str) -> Optional[np.ndarray]:
        """读取本地图片，支持中文路径（cv2.imread 对非 ASCII 路径不稳定）。"""
        with open(path, "rb") as f:
            data = f.read()
        if not data:
            return None
        arr = np.frombuffer(data, dtype=np.uint8)
        return ImageLoader._decode(arr)

    def close(self):
        self._session.close()
