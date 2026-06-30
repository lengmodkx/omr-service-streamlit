"""模板缓存 — 内存存储解析后的 StandardTemplate"""
import threading
import time
from typing import Dict, Optional

from omr_service.engine.standard_template import StandardTemplate


class TemplateStore:
    """线程安全的模板缓存，支持 TTL。"""

    def __init__(self, ttl_seconds: int = 3600):
        self._ttl = ttl_seconds
        self._store: Dict[int, dict] = {}
        self._lock = threading.RLock()

    def get(self, template_id: int) -> Optional[StandardTemplate]:
        with self._lock:
            item = self._store.get(template_id)
            if item is None:
                return None
            if time.time() - item["ts"] > self._ttl:
                del self._store[template_id]
                return None
            return item["template"]

    def set(self, template_id: int, template: StandardTemplate) -> None:
        with self._lock:
            self._store[template_id] = {"template": template, "ts": time.time()}

    def delete(self, template_id: int) -> None:
        with self._lock:
            self._store.pop(template_id, None)

    def clear(self):
        with self._lock:
            self._store.clear()
