"""图片加载器 EXIF 方向测试。"""
from __future__ import annotations

from pathlib import Path

from PIL import Image

from omr_service.loader.image_loader import ImageLoader


def _make_rotated_jpeg(path: Path) -> None:
    """生成 200x400 竖向像素、EXIF Orientation=6（顺时针 90°），模拟手机横向拍摄。"""
    image = Image.new("RGB", (200, 400), (255, 255, 255))
    exif = image.getexif()
    exif[274] = 6
    image.save(path, format="JPEG", exif=exif)


def test_load_local_applies_exif_orientation(tmp_path: Path) -> None:
    path = tmp_path / "wechat.jpg"
    _make_rotated_jpeg(path)

    loader = ImageLoader()
    try:
        loaded = loader.load(str(path))
    finally:
        loader.close()

    assert loaded is not None
    # 存储像素 200x400 竖向；EXIF 6 顺时针旋转 90° 后应恢复成 400x200 横向。
    height, width = loaded.shape[:2]
    assert width == 400
    assert height == 200


def test_load_local_without_exif_unchanged(tmp_path: Path) -> None:
    path = tmp_path / "plain.jpg"
    Image.new("RGB", (320, 240), (255, 255, 255)).save(path, format="JPEG")

    loader = ImageLoader()
    try:
        loaded = loader.load(str(path))
    finally:
        loader.close()

    assert loaded is not None
    assert loaded.shape[:2] == (240, 320)
