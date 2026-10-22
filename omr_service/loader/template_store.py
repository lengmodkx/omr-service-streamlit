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

        # 气泡：按「页码」整体替换（同一页重新解析后必须覆盖旧气泡，不能追加）。
        # 历史问题：追加会导致反复保存/预热后缓存里残留旧题号气泡，
        # 识别结果出现"第1页 Q5..Q12 + Q1..Q4"这种新旧题号混杂的答案。
        old_bubbles = list(getattr(old.standard_template, "bubbles", []) or []) \
            if old.standard_template is not None else []
        new_bubbles = list(getattr(new.standard_template, "bubbles", []) or []) \
            if new.standard_template is not None else []
        new_pages = {_bubble_page(b) for b in new_bubbles}
        kept_old_bubbles = [b for b in old_bubbles if _bubble_page(b) not in new_pages]
        merged_bubbles = kept_old_bubbles + new_bubbles

        # 标准答案：保留未被重新解析页面的旧答案，再用新解析结果覆盖（题号为引擎内部题号）
        old_answers = dict(getattr(old.standard_template, "answers", None) or {}) \
            if old.standard_template is not None else {}
        new_answers = dict(getattr(new.standard_template, "answers", None) or {}) \
            if new.standard_template is not None else {}
        kept_qs = {b["q"] for b in kept_old_bubbles}
        merged_answers = {q: a for q, a in old_answers.items() if q in kept_qs}
        merged_answers.update(new_answers)

        # 保留旧的 StandardTemplate 作为默认参考图（通常是 A 面），
        # 避免后页（B 面）调用覆盖默认参考图导致 A 面选择题 ECC 对齐失败。
        # 多页选择题支持后续再按 page_index 选择对应参考图。
        # deep copy 避免修改共享缓存；旧模板为空（首次只有单页配置）时用新模板作基底
        base_template = old.standard_template if old.standard_template is not None else new.standard_template
        if base_template is None:
            merged_template = None
        else:
            merged_template = copy.deepcopy(base_template)
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
