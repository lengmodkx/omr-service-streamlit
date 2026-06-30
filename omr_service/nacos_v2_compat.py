"""Compatibility shim for nacos-sdk-python v2/v3.

The nacos-sdk-python 3.x series declares a dependency on the ``a2a`` package,
but the current ``a2a-sdk`` releases expose ``AgentCard`` as a protobuf
Message class. The SDK's AI model code tries to subclass it with a Pydantic
model, which raises ``TypeError`` at import time.

This module provides a tiny, deterministic Pydantic-based stub for the two
symbols the SDK actually needs (``a2a._base.A2ABaseModel`` and
``a2a.types.AgentCard``) before importing ``v2.nacos``. The AI client is not
used by this service, so the stub has no runtime impact on naming/config
functionality.
"""
from __future__ import annotations

import asyncio
import logging
import sys
import threading
from types import ModuleType
from typing import Any, Coroutine, TypeVar

from pydantic import BaseModel

logger = logging.getLogger(__name__)
_PATCH_LOCK = threading.Lock()
_PATCH_APPLIED = False

T = TypeVar("T")


def _is_pydantic_model(cls: Any) -> bool:
    """Return True if ``cls`` is a Pydantic BaseModel subclass."""
    try:
        return isinstance(cls, type) and issubclass(cls, BaseModel)
    except TypeError:
        return False


def _patch_a2a_agent_card() -> None:
    """Install a minimal Pydantic-compatible stub for ``a2a.types.AgentCard``."""
    global _PATCH_APPLIED
    if _PATCH_APPLIED:
        return
    with _PATCH_LOCK:
        if _PATCH_APPLIED:
            return

        # If a real a2a-sdk is installed and already exposes AgentCard as a
        # Pydantic model, leave it alone so future SDK versions keep working.
        real_types = sys.modules.get("a2a.types")
        if real_types is not None and _is_pydantic_model(getattr(real_types, "AgentCard", None)):
            _PATCH_APPLIED = True
            logger.debug("a2a.types.AgentCard is already a Pydantic model; no patch needed")
            return

        # Build a fake a2a._base module with a Pydantic A2ABaseModel.
        fake_base = ModuleType("a2a._base")
        fake_base.A2ABaseModel = type("A2ABaseModel", (BaseModel,), {})
        sys.modules["a2a._base"] = fake_base

        # Build a fake a2a.types module with a Pydantic AgentCard.
        fake_types = ModuleType("a2a.types")
        fake_types.AgentCard = type(
            "AgentCard", (fake_base.A2ABaseModel,), {}
        )
        sys.modules["a2a.types"] = fake_types

        _PATCH_APPLIED = True
        logger.debug("Applied nacos-sdk-python a2a compatibility patch")


def import_v2_nacos() -> Any:
    """Import and return the ``v2.nacos`` module after applying the patch."""
    _patch_a2a_agent_card()
    import v2.nacos as _nacos  # noqa: PLC0415

    return _nacos


def import_naming_service() -> Any:
    """Import ``NacosNamingService`` after applying the patch."""
    _patch_a2a_agent_card()
    from v2.nacos.naming.nacos_naming_service import NacosNamingService  # noqa: PLC0415

    return NacosNamingService


def import_config_service() -> Any:
    """Import ``NacosConfigService`` after applying the patch."""
    _patch_a2a_agent_card()
    from v2.nacos.config.nacos_config_service import NacosConfigService  # noqa: PLC0415

    return NacosConfigService


def import_client_config() -> Any:
    """Import ``ClientConfig`` after applying the patch."""
    _patch_a2a_agent_card()
    from v2.nacos.common.client_config import ClientConfig  # noqa: PLC0415

    return ClientConfig


def import_register_instance_param() -> Any:
    """Import ``RegisterInstanceParam`` after applying the patch."""
    _patch_a2a_agent_card()
    from v2.nacos.naming.model.naming_param import RegisterInstanceParam  # noqa: PLC0415
    return RegisterInstanceParam


def import_deregister_instance_param() -> Any:
    """Import ``DeregisterInstanceParam`` after applying the patch."""
    _patch_a2a_agent_card()
    from v2.nacos.naming.model.naming_param import DeregisterInstanceParam  # noqa: PLC0415
    return DeregisterInstanceParam


def import_config_param() -> Any:
    """Import ``ConfigParam`` after applying the patch."""
    _patch_a2a_agent_card()
    from v2.nacos.config.model.config_param import ConfigParam  # noqa: PLC0415
    return ConfigParam


class _SharedAsyncLoop:
    """Single background event loop shared by all Nacos v2 clients.

    nacos-sdk-python v2 is async and expects a running event loop for gRPC
    heartbeats and server-push handling. This class keeps one loop alive in a
    daemon thread so the rest of the service can stay synchronous.
    """

    _instance: _SharedAsyncLoop | None = None
    _lock = threading.Lock()

    def __new__(cls) -> _SharedAsyncLoop:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def _ensure_started(self) -> None:
        if self._initialized:
            return
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()
        self._initialized = True
        logger.debug("Started shared Nacos async event loop thread")

    def run(self, coro: Coroutine[Any, Any, T]) -> T:
        """Schedule ``coro`` on the shared loop and block for the result."""
        self._ensure_started()
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result()

    def stop(self) -> None:
        if not self._initialized:
            return
        try:
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(timeout=3)
        except Exception as exc:
            logger.warning("Error stopping shared Nacos event loop: %s", exc)
        finally:
            self._initialized = False


def shared_loop() -> _SharedAsyncLoop:
    """Return the singleton shared async loop helper."""
    return _SharedAsyncLoop()


