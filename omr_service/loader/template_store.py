"""模板缓存 — 内存存储解析后的模板上下文"""
import copy
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from omr_service.engine.standard_template import StandardTemplate

logger = logging.getLogger(__name__)


def _bubble_page(bubble: Dict[str, Any]) -> int:
    """取气泡所属页码（脏数据回退 0）。"""
    try:
        return int(bubble.get("page_index", 0) or 0)
    except (TypeError, ValueError):
        return 0


@dataclass
class CachedTemplate:
    """缓存的模板上下文"""

    standard_template: StandardTemplate
    # 个人信息区域配置（全局，跨页）
    personal_info: List[Dict[str, Any]] = field(default_factory=list)
    # 主观题区域配置（全局，跨页）
    subjective_regions: List[Dict[str, Any]] = field(default_factory=list)
    # 黄金模板图片 URL（逗号分隔多页）
    image_url: str = ""
    # 每页黄金模板参考图 {page_index: image}
    page_images: Dict[int, np.ndarray] = field(default_factory=dict)


class TemplateStore:
    """线程安全的模板缓存，支持 TTL。"""

    def __init__(self, ttl_seconds: int = 3600):
        self._ttl = ttl_seconds
        self._store: Dict[int, dict] = {}
        self._lock = threading.RLock()

    def get(self, template_id: int) -> Optional[CachedTemplate]:
        with self._lock:
            item = self._store.get(template_id)
            if item is None:
                return None
            if time.time() - item["ts"] > self._ttl:
                del self._store[template_id]
                return None
            return item["template"]

    def set(self, template_id: int, template: CachedTemplate) -> None:
        """设置模板缓存；如果已存在则合并，保证多页配置不丢失。"""
        with self._lock:
            existing = self._store.get(template_id)
            if existing is None:
                self._store[template_id] = {"template": template, "ts": time.time()}
                return

            merged = self._merge(existing["template"], template)
            self._store[template_id] = {"template": merged, "ts": time.time()}

    @staticmethod
    def _merge(old: CachedTemplate, new: CachedTemplate) -> CachedTemplate:
        """合并两个缓存：保留旧配置，追加新页配置。"""
        # 个人信息：按 (field, page_index) 去重合并
        old_personal = {(p.get("field"), p.get("page_index", 0)): p for p in old.personal_info}
        for p in new.personal_info:
            key = (p.get("field"), p.get("page_index", 0))
            old_personal[key] = p
        merged_personal = list(old_personal.values())

        # 主观题区域：按 (q, page_index) 去重合并
        old_subjective = {(r.get("q"), r.get("page_index", 0)): r for r in old.subjective_regions}
        for r in new.subjective_regions:
            key = (r.get("q"), r.get("page_index", 0))
            old_subjective[key] = r
        merged_subjective = list(old_subjective.values())

        # 气泡：追加并保留 page_index 标记；同 (题号, 选项) 出现在不同页时告警
        # （题号跨页重复的早期信号，解析阶段允许，识别阶段会直接报错）
        merged_bubbles = list(getattr(old.standard_template, "bubbles", []) or [])
        bubble_page = {(b.get("q"), b.get("opt")): _bubble_page(b) for b in merged_bubbles}
        for b in getattr(new.standard_template, "bubbles", []) or []:
            key = (b.get("q"), b.get("opt"))
            page = _bubble_page(b)
            prev_page = bubble_page.get(key)
            if prev_page is not None and prev_page != page:
                logger.warning(
                    "模板气泡跨页重复: 题号=%s 选项=%s 第%d页与第%d页",
                    b.get("q"), b.get("opt"), prev_page + 1, page + 1,
                )
            bubble_page.setdefault(key, page)
            merged_bubbles.append(b)

        # 标准答案：跨页合并（键为整卷绝对题号）。
        # 之前只替换 bubbles 不合并 answers，导致第 2 页标准答案丢失、correct 全为 None。
        merged_answers = dict(getattr(old.standard_template, "answers", None) or {})
        for q, ans in (getattr(new.standard_template, "answers", None) or {}).items():
            if q in merged_answers and merged_answers[q] != ans:
                # 同题号不同答案 = 题号跨页重复的早期信号（识别阶段会直接报错）
                logger.warning(
                    "模板题号跨页重复且标准答案不一致: 题号=%s 旧答案=%s 新答案=%s",
                    q, merged_answers[q], ans,
                )
            merged_answers[q] = ans

        # 保留旧的 StandardTemplate 作为默认参考图（通常是 A 面），
        # 避免后页（B 面）调用覆盖默认参考图导致 A 面选择题 ECC 对齐失败。
        # 多页选择题支持后续再按 page_index 选择对应参考图。
        # deep copy 避免修改共享缓存
        merged_template = copy.deepcopy(old.standard_template)
        merged_template.bubbles = merged_bubbles
        merged_template.answers = merged_answers

        # 合并多页参考图
        merged_page_images = dict(old.page_images or {})
        merged_page_images.update(new.page_images or {})

        # 合并 image_url（逗号分隔多页）
        merged_urls = []
        for url in [old.image_url, new.image_url]:
            if url:
                merged_urls.extend([u.strip() for u in url.split(",") if u.strip()])
        # 去重并保持顺序
        seen = set()
        unique_urls = []
        for u in merged_urls:
            if u not in seen:
                seen.add(u)
                unique_urls.append(u)
        merged_image_url = ",".join(unique_urls)

        return CachedTemplate(
            standard_template=merged_template,
            personal_info=merged_personal,
            subjective_regions=merged_subjective,
            image_url=merged_image_url,
            page_images=merged_page_images,
        )

    def delete(self, template_id: int) -> None:
        with self._lock:
            self._store.pop(template_id, None)

    def clear(self):
        with self._lock:
            self._store.clear()
