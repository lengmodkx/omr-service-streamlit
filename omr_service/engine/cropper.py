"""主观题裁剪模块

支持：
- 按区域列表从单页图片裁剪
- 跨页区域垂直拼接后裁剪
- 保存到本地目录并返回可访问 URL
"""

import logging
import os
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class SubjectiveCropper:
    """主观题图片裁剪器"""

    def __init__(
        self,
        output_dir: str = "./omr_crops",
        base_url: Optional[str] = None,
    ):
        self.output_dir = Path(output_dir).resolve()
        self.base_url = base_url.rstrip("/") if base_url else None
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def crop_subjective_regions(
        self,
        images: List[np.ndarray],
        regions: List[Dict[str, Any]],
        namespace: str,
        reference_images: Optional[Dict[int, np.ndarray]] = None,
    ) -> List[Dict[str, Any]]:
        """裁剪主观题区域（含跨页拼接）

        Args:
            images: 多页答题卡图片列表，下标即 page_index
            regions: 主观题区域配置列表，每项包含 q, x1, y1, x2, y2, page_index, stitch_with_next
            namespace: 用于生成目录的唯一标识，如 task_id 或 template_id
            reference_images: 每页模板参考图 {page_index: image}；模板坐标按参考图尺寸记录，
                扫描图尺寸不同（手机拍照/高扫 4960x3507 等）时必须按比例缩放，否则裁剪区域错位

        Returns:
            裁剪结果列表，每项包含 q, image_url, page_index
        """
        if not regions:
            return []

        # 模板坐标 → 扫描图坐标：按页缩放
        regions = self._scale_regions(regions, images, reference_images)

        # 按题号分组
        groups: Dict[int, List[Dict[str, Any]]] = {}
        for region in regions:
            q = int(region.get("q", 0))
            groups.setdefault(q, []).append(region)

        results = []
        for q, group in groups.items():
            group = sorted(group, key=lambda r: int(r.get("page_index", 0)))
            chains = self._collect_stitch_chains(images, group)
            for chain_idx, (stitched, stitch_regions, page_index) in enumerate(chains):
                if stitched is None:
                    continue
                crop_img, crop_page_index = self._crop_stitched(stitched, stitch_regions)
                if crop_img is None:
                    continue
                # 多链时增加序号后缀，保证同一题多段裁剪都可返回
                suffix = f"_c{chain_idx}" if len(chains) > 1 else ""
                filename = f"q_{q}{suffix}_{uuid.uuid4().hex[:8]}.jpg"
                file_path = self._save_image(crop_img, namespace, filename)
                results.append({
                    "q": q,
                    "image_url": self._to_url(file_path),
                    "page_index": crop_page_index,
                })
        return results

    @staticmethod
    def _scale_regions(
        regions: List[Dict[str, Any]],
        images: List[np.ndarray],
        reference_images: Optional[Dict[int, np.ndarray]],
    ) -> List[Dict[str, Any]]:
        """把模板坐标缩放到扫描图坐标（模板与扫描图尺寸不一致时）。

        模板坐标以黄金模板图（如 2339x1654）为基准；手机拍照/高扫常见 4960x3507，
        如果不缩放，主观题会裁到左上角一小块，表现就是“Q22 裁出条码区、Q21 裁出选择题区”。
        """
        if not reference_images:
            return regions

        scaled: List[Dict[str, Any]] = []
        for region in regions:
            page = int(region.get("page_index", 0))
            ref = reference_images.get(page)
            if ref is None or page >= len(images):
                scaled.append(dict(region))
                continue
            target = images[page]
            ref_h, ref_w = ref.shape[:2]
            tgt_h, tgt_w = target.shape[:2]
            if ref_w <= 0 or ref_h <= 0 or (ref_w == tgt_w and ref_h == tgt_h):
                scaled.append(dict(region))
                continue
            scale_x = tgt_w / ref_w
            scale_y = tgt_h / ref_h
            item = dict(region)
            for key in ("x1", "x2"):
                if key in item:
                    item[key] = int(round(float(item[key]) * scale_x))
            for key in ("y1", "y2"):
                if key in item:
                    item[key] = int(round(float(item[key]) * scale_y))
            scaled.append(item)
        return scaled

    def crop_by_regions(
        self,
        image: np.ndarray,
        regions: List[Dict[str, Any]],
        namespace: str,
    ) -> List[Dict[str, Any]]:
        """从单张图片按区域列表裁剪（不含跨页拼接）"""
        results = []
        for region in regions:
            q = int(region.get("q", 0))
            crop = self._crop_one(image, region)
            if crop is None or crop.size == 0:
                continue
            filename = f"q_{q}_{uuid.uuid4().hex[:8]}.jpg"
            file_path = self._save_image(crop, namespace, filename)
            results.append({
                "q": q,
                "image_url": self._to_url(file_path),
                "page_index": int(region.get("page_index", 0)),
            })
        return results

    def _collect_stitch_chains(
        self,
        images: List[np.ndarray],
        group: List[Dict[str, Any]],
    ) -> List[Tuple[Optional[np.ndarray], List[Tuple[Dict[str, Any], int]], int]]:
        """按 stitch_with_next 把区域拆分为多个连续链，每链单独拼接裁剪。

        Returns:
            [(拼接后的图片, [(region, y_offset), ...], 起始页索引), ...]
        """
        chains: List[List[Dict[str, Any]]] = []
        current: List[Dict[str, Any]] = []
        for i, region in enumerate(group):
            current.append(region)
            if not bool(region.get("stitch_with_next", False)) or i == len(group) - 1:
                chains.append(current)
                current = []
        if current:
            chains.append(current)

        results = []
        for chain in chains:
            stitched, region_offsets = self._build_stitched(images, chain)
            start_page_index = int(chain[0].get("page_index", 0)) if chain else 0
            results.append((stitched, region_offsets, start_page_index))
        return results

    def _build_stitched(
        self,
        images: List[np.ndarray],
        chain: List[Dict[str, Any]],
    ) -> Tuple[Optional[np.ndarray], List[Tuple[Dict[str, Any], int]]]:
        """拼接单条连续链对应的页面并计算 y 偏移。"""
        pages_to_stitch: List[int] = []
        for region in chain:
            page_index = int(region.get("page_index", 0))
            pages_to_stitch.append(page_index)

        # 去重并保持顺序
        seen = set()
        ordered_pages = []
        for p in pages_to_stitch:
            if p not in seen:
                seen.add(p)
                ordered_pages.append(p)

        valid_pages = [p for p in ordered_pages if 0 <= p < len(images)]
        if not valid_pages:
            return None, []

        page_images = [images[p] for p in valid_pages]
        # 统一宽度后垂直拼接
        target_width = min(img.shape[1] for img in page_images) or 1
        resized = []
        for img in page_images:
            if img.shape[1] != target_width:
                scale = target_width / img.shape[1]
                target_height = int(img.shape[0] * scale)
                resized.append(cv2.resize(img, (target_width, target_height)))
            else:
                resized.append(img)
        stitched = cv2.vconcat(resized)

        # 计算每个原始页面对应的 y 偏移
        y_offset = 0
        page_y_offsets = {}
        for p, img in zip(valid_pages, resized):
            page_y_offsets[p] = y_offset
            y_offset += img.shape[0]

        region_offsets = []
        for region in chain:
            page_index = int(region.get("page_index", 0))
            if page_index in page_y_offsets:
                region_offsets.append((region, page_y_offsets[page_index]))

        return stitched, region_offsets

    @staticmethod
    def _crop_stitched(
        stitched: np.ndarray,
        region_offsets: List[Tuple[Dict[str, Any], int]],
    ) -> Tuple[Optional[np.ndarray], int]:
        """从拼接图中裁剪出覆盖所有区域的并集"""
        if not region_offsets:
            return None, 0

        h, w = stitched.shape[:2]
        x1_list, y1_list, x2_list, y2_list = [], [], [], []
        for region, y_offset in region_offsets:
            x1 = int(region.get("x1", 0))
            raw_y1 = int(region.get("y1", 0))
            x2 = int(region.get("x2", x1 + 1))
            raw_y2 = int(region.get("y2", raw_y1 + 1))
            y1 = raw_y1 + y_offset
            y2 = raw_y2 + y_offset
            x1_list.append(max(0, x1))
            y1_list.append(max(0, y1))
            x2_list.append(min(w, x2))
            y2_list.append(min(h, y2))

        x1, y1 = min(x1_list), min(y1_list)
        x2, y2 = max(x2_list), max(y2_list)
        if x2 <= x1 or y2 <= y1:
            return None, 0
        return stitched[y1:y2, x1:x2], int(region_offsets[0][0].get("page_index", 0))

    @staticmethod
    def _crop_one(image: np.ndarray, region: Dict[str, Any]) -> Optional[np.ndarray]:
        h, w = image.shape[:2]
        x1 = max(0, int(region.get("x1", 0)))
        y1 = max(0, int(region.get("y1", 0)))
        x2 = min(w, int(region.get("x2", x1 + 1)))
        y2 = min(h, int(region.get("y2", y1 + 1)))
        if x2 <= x1 or y2 <= y1:
            return None
        return image[y1:y2, x1:x2]

    def _save_image(self, image: np.ndarray, namespace: str, filename: str) -> str:
        """保存图片到输出目录，返回本地绝对路径。

        对 namespace 做路径穿越校验，确保最终文件位于 self.output_dir 下。
        """
        save_dir = self.output_dir / namespace
        resolved_dir = save_dir.resolve()
        resolved_output = self.output_dir.resolve()
        # 路径穿越防护：解析后的目录必须位于输出目录之下
        if not str(resolved_dir).startswith(str(resolved_output) + os.sep):
            raise ValueError(f"非法的 namespace 路径: {namespace}")
        save_dir.mkdir(parents=True, exist_ok=True)
        file_path = save_dir / filename
        ok = cv2.imwrite(str(file_path), image, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            raise RuntimeError(f"图片保存失败: {file_path}")
        return str(file_path.absolute())

    def _to_url(self, file_path: str) -> str:
        """本地路径转 URL，配置了 base_url 时返回可访问 URL"""
        if self.base_url:
            relative = Path(file_path).relative_to(self.output_dir)
            return f"{self.base_url}/{relative.as_posix()}"
        return f"file://{file_path}"
