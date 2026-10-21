"""本地主观题裁图保留期清理。"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CropCleanupResult:
    """一次清理的结果统计。"""

    deleted_files: int = 0
    freed_bytes: int = 0


def _is_link(path: Path) -> bool:
    """识别符号链接及 Windows junction 等重解析点。"""
    if path.is_symlink():
        return True
    if os.name != "nt":
        return False
    try:
        return bool(os.lstat(path).st_file_attributes & 0x400)
    except OSError:
        return False


def clean_expired_crops(output_dir: str | Path, retention_days: int, now: float | None = None) -> CropCleanupResult:
    """删除超期本地文件与空目录，不跟随任何符号链接。"""
    if retention_days <= 0:
        logger.info("本地裁图清理已停用，保留期配置为 %s 天", retention_days)
        return CropCleanupResult()

    base = Path(output_dir)
    if _is_link(base):
        logger.warning("本地裁图输出目录是符号链接，已跳过清理: %s", base)
        return CropCleanupResult()
    if not base.exists():
        logger.info("本地裁图输出目录不存在，无需清理: %s", base)
        return CropCleanupResult()

    cutoff = (time.time() if now is None else now) - retention_days * 86400
    deleted_files = 0
    freed_bytes = 0
    directories = []
    for root, dirs, files in os.walk(base, topdown=True, followlinks=False):
        current_dir = Path(root)
        dirs[:] = [name for name in dirs if not _is_link(current_dir / name)]
        directories.append(current_dir)
        for filename in files:
            path = current_dir / filename
            if _is_link(path) or not path.is_file():
                continue
            try:
                stat = path.stat()
                if stat.st_mtime >= cutoff:
                    continue
                path.unlink()
            except FileNotFoundError:
                continue
            except OSError as exc:
                logger.warning("过期裁图删除失败: %s，原因: %s", path, exc)
                continue
            deleted_files += 1
            freed_bytes += stat.st_size

    for current_dir in reversed(directories):
        if current_dir == base or _is_link(current_dir):
            continue
        try:
            if not any(current_dir.iterdir()):
                current_dir.rmdir()
        except FileNotFoundError:
            continue
        except OSError as exc:
            logger.warning("裁图空目录删除失败: %s，原因: %s", current_dir, exc)

    result = CropCleanupResult(deleted_files=deleted_files, freed_bytes=freed_bytes)
    logger.info(
        "本地裁图清理完成：目录=%s，删除文件=%d，释放空间=%d 字节",
        base,
        result.deleted_files,
        result.freed_bytes,
    )
    return result
