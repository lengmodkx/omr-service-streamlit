"""POST /v1/ocr/handwriting、/v1/ocr/formula 路由测试（mock OCR 引擎）。

覆盖：URL/base64 两种图片来源、data URI 前缀、非法入参（都缺/都传/非法 base64/
非图片）、URL 下载失败、识别为空、引擎不可用；全部场景不得返回 500。
"""
import base64
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from omr_service.api.deps import register_dependencies
from omr_service.api.errors import register_error_handlers
from omr_service.api.routers.ocr import router
from omr_service.engine.ocr import PersonalInfoOcr

# PaddleOCR 返回结构：[[[box], (text, confidence)], ...]
FAKE_OCR_RESULT = [[
    [None, ("手写答案一", 0.98)],
    [None, ("第二行文本", 0.90)],
]]


def _make_image(width: int = 64, height: int = 64) -> np.ndarray:
    """生成一张可编码的测试图片（白底 + 黑块）。"""
    img = np.full((height, width, 3), 255, dtype=np.uint8)
    img[16:48, 16:48] = 0
    return img


def _image_base64(img: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return base64.b64encode(buf.tobytes()).decode("ascii")


class _FakePaddleEngine:
    """mock PaddleOCR 引擎：results 为 None 时模拟识别为空。"""

    def __init__(self, results=None):
        self.results = results
        self.calls = 0

    def ocr(self, image, cls=True):
        self.calls += 1
        if self.results is None:
            return None
        return self.results


@pytest.fixture
def app():
    app = FastAPI()
    app.include_router(router)
    register_error_handlers(app)
    settings = MagicMock()
    service = MagicMock()
    service.image_loader = MagicMock()
    register_dependencies(app, settings=settings, service=service, task_registry=MagicMock())
    return app, service


@pytest.fixture
def fake_engine(monkeypatch):
    """把 mock PaddleOCR 引擎挂到 PersonalInfoOcr 单例（不触发真实初始化）。"""
    engine = _FakePaddleEngine(FAKE_OCR_RESULT)
    monkeypatch.setattr(PersonalInfoOcr, "_ocr_engine", engine)
    return engine


# ---------- URL 来源 ----------

def test_handwriting_url_200(app, fake_engine):
    app_obj, service = app
    service.image_loader.load.return_value = _make_image()

    r = TestClient(app_obj).post("/v1/ocr/handwriting", json={"image_url": "http://x/a.png"})
    assert r.status_code == 200
    data = r.json()
    assert data["text"] == "手写答案一\n第二行文本"
    assert data["lines"] == [
        {"text": "手写答案一", "confidence": 0.98},
        {"text": "第二行文本", "confidence": 0.9},
    ]
    assert data["engine"] == "paddle"
    assert data["latex"] is None
    # URL 走现有 loader 加载
    service.image_loader.load.assert_called_once_with("http://x/a.png")


def test_formula_url_200_latex_null(app, fake_engine):
    app_obj, service = app
    service.image_loader.load.return_value = _make_image()

    r = TestClient(app_obj).post("/v1/ocr/formula", json={"image_url": "http://x/f.png"})
    assert r.status_code == 200
    data = r.json()
    assert data["text"] == "手写答案一\n第二行文本"
    assert data["engine"] == "paddle"
    # 公式一期：latex 恒为 null
    assert data["latex"] is None


# ---------- base64 来源 ----------

def test_handwriting_base64_200(app, fake_engine):
    app_obj, service = app
    b64 = _image_base64(_make_image())

    r = TestClient(app_obj).post("/v1/ocr/handwriting", json={"image_base64": b64})
    assert r.status_code == 200
    assert r.json()["text"] == "手写答案一\n第二行文本"
    # base64 分支不应走 loader
    service.image_loader.load.assert_not_called()


def test_base64_with_data_uri_prefix_200(app, fake_engine):
    app_obj, _ = app
    b64 = _image_base64(_make_image())
    payload = f"data:image/png;base64,{b64}"

    r = TestClient(app_obj).post("/v1/ocr/formula", json={"image_base64": payload})
    assert r.status_code == 200
    assert r.json()["text"] == "手写答案一\n第二行文本"


# ---------- 非法入参 ----------

def test_missing_both_sources_400(app):
    app_obj, _ = app
    r = TestClient(app_obj).post("/v1/ocr/handwriting", json={})
    assert r.status_code == 400
    assert r.json()["code"] == 6


def test_both_sources_provided_400(app):
    app_obj, _ = app
    r = TestClient(app_obj).post(
        "/v1/ocr/handwriting",
        json={"image_url": "http://x/a.png", "image_base64": "aGVsbG8="},
    )
    assert r.status_code == 400
    assert r.json()["code"] == 6


def test_invalid_base64_400(app):
    app_obj, _ = app
    r = TestClient(app_obj).post("/v1/ocr/handwriting", json={"image_base64": "!!!not-base64!!!"})
    assert r.status_code == 400
    assert r.json()["code"] == 6


def test_base64_not_an_image_400(app):
    app_obj, _ = app
    b64 = base64.b64encode(b"just text, not an image").decode("ascii")
    r = TestClient(app_obj).post("/v1/ocr/handwriting", json={"image_base64": b64})
    assert r.status_code == 400
    assert r.json()["code"] == 6


# ---------- 图片加载失败 ----------

def test_url_load_failure_502(app, fake_engine):
    app_obj, service = app
    service.image_loader.load.return_value = None  # 下载超时/失败

    r = TestClient(app_obj).post("/v1/ocr/handwriting", json={"image_url": "http://x/a.png"})
    assert r.status_code == 502
    assert r.json()["code"] == 5


# ---------- 识别为空 / 引擎不可用 ----------

def test_empty_ocr_result_200(app, monkeypatch):
    """识别为空（引擎返回 None）→ 200 空文本，不抛 500。"""
    app_obj, service = app
    service.image_loader.load.return_value = _make_image()
    monkeypatch.setattr(PersonalInfoOcr, "_ocr_engine", _FakePaddleEngine(None))

    r = TestClient(app_obj).post("/v1/ocr/handwriting", json={"image_url": "http://x/a.png"})
    assert r.status_code == 200
    assert r.json() == {"text": "", "lines": [], "engine": "paddle", "latex": None}


def test_engine_unavailable_200(app, monkeypatch):
    """引擎不可用（初始化失败）→ 200 空文本，不抛 500。"""
    app_obj, service = app
    service.image_loader.load.return_value = _make_image()
    monkeypatch.setattr(PersonalInfoOcr, "_ocr_engine", None)
    monkeypatch.setattr(PersonalInfoOcr, "_init_failed", True)

    r = TestClient(app_obj).post("/v1/ocr/handwriting", json={"image_url": "http://x/a.png"})
    assert r.status_code == 200
    assert r.json() == {"text": "", "lines": [], "engine": "paddle", "latex": None}


def test_engine_raises_200(app, monkeypatch):
    """引擎推理抛异常 → 200 空文本，不抛 500。"""
    app_obj, service = app
    service.image_loader.load.return_value = _make_image()

    class _BoomEngine:
        def ocr(self, image, cls=True):
            raise RuntimeError("paddle crash")

    monkeypatch.setattr(PersonalInfoOcr, "_ocr_engine", _BoomEngine())

    r = TestClient(app_obj).post("/v1/ocr/handwriting", json={"image_url": "http://x/a.png"})
    assert r.status_code == 200
    assert r.json() == {"text": "", "lines": [], "engine": "paddle", "latex": None}


def test_formula_empty_result_latex_null(app, monkeypatch):
    app_obj, service = app
    service.image_loader.load.return_value = _make_image()
    monkeypatch.setattr(PersonalInfoOcr, "_ocr_engine", _FakePaddleEngine(None))

    r = TestClient(app_obj).post("/v1/ocr/formula", json={"image_url": "http://x/f.png"})
    assert r.status_code == 200
    assert r.json() == {"text": "", "lines": [], "engine": "paddle", "latex": None}
