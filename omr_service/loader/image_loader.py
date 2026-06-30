"""图片加载器 — 支持 HTTP 下载、本地路径、内存缓存"""
import io
import os
from typing import Optional

import cv2
import numpy as np
import requests
from tenacity import retry, stop_after_attempt, wait_exponential


class ImageLoader:
    """加载答题卡图片"""

    def __init__(self, timeout: int = 30, max_bytes: int = 50 * 1024 * 1024):
        self.timeout = timeout
        self.max_bytes = max_bytes
        self._session = requests.Session()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=5),
        reraise=True,
    )
    def load(self, url: str) -> Optional[np.ndarray]:
        """加载图片，返回 BGR np.ndarray；失败返回 None。"""
        if not url:
            return None

        # 本地文件
        if os.path.exists(url):
            img = cv2.imread(url, cv2.IMREAD_COLOR)
            return img

        # HTTP(S) 下载
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
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return img

    def close(self):
        self._session.close()
