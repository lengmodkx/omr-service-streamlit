"""Tests for omr_service.main entry point."""
from unittest.mock import MagicMock, patch, ANY
import pytest


def test_main_invokes_uvicorn(monkeypatch):
    """main() should call uvicorn.run with the FastAPI app."""
    monkeypatch.setattr("uvicorn.run", MagicMock())
    monkeypatch.setattr("omr_service.main._setup_dependencies", lambda settings: (MagicMock(), MagicMock(), MagicMock()))
    monkeypatch.setattr("omr_service.main._start_nacos", lambda settings: None)
    monkeypatch.setattr("omr_service.main._start_consumer", lambda settings, service: None)
    monkeypatch.setattr("omr_service.main._deregister_nacos", lambda: None)

    from omr_service import main
    main.main()

    import uvicorn
    assert uvicorn.run.called


def test_main_with_omr_settings(monkeypatch):
    """main() should use OmrSettings (new Pydantic)."""
    monkeypatch.setattr("uvicorn.run", MagicMock())
    monkeypatch.setattr("omr_service.main._setup_dependencies", lambda settings: (MagicMock(), MagicMock(), MagicMock()))
    monkeypatch.setattr("omr_service.main._start_nacos", lambda settings: None)
    monkeypatch.setattr("omr_service.main._start_consumer", lambda settings, service: None)
    monkeypatch.setattr("omr_service.main._deregister_nacos", lambda: None)

    from omr_service import main
    # Should not raise
    main.main()


def test_start_crop_cleaner_returns_stoppable_thread():
    """启动清理线程时返回 (stop_event, thread)，shutdown 才能通知并 join 线程收尾。"""
    from omr_service.main import _start_crop_cleaner

    settings = MagicMock()
    settings.crop_retention_days = 30
    settings.crop_output_dir = "./omr_crops"

    result = _start_crop_cleaner(settings)

    assert result is not None
    stop_event, thread = result
    stop_event.set()
    thread.join(timeout=5)
    assert not thread.is_alive()


def test_start_crop_cleaner_disabled_returns_none():
    """保留期 <= 0 时不启动清理线程。"""
    from omr_service.main import _start_crop_cleaner

    settings = MagicMock()
    settings.crop_retention_days = 0

    assert _start_crop_cleaner(settings) is None
