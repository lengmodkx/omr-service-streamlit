"""FastAPI 应用工厂."""
from __future__ import annotations

import logging
import re
import uuid
from typing import TYPE_CHECKING

from fastapi import FastAPI, Request

from omr_service.api.deps import register_dependencies
from omr_service.api.errors import register_error_handlers
from omr_service.api.routers import (
    crops,
    health,
    ocr,
    recognize,
    tasks,
    templates,
)

if TYPE_CHECKING:
    from omr_service.config import OmrSettings
    from omr_service.core.service import OmrService
    from omr_service.core.task_registry import TaskRegistry

logger = logging.getLogger(__name__)


def create_app(
    *,
    settings: "OmrSettings",
    service: "OmrService",
    task_registry: "TaskRegistry",
    lifespan=None,
) -> FastAPI:
    """创建 FastAPI 应用实例.

    OpenAPI / Swagger 路径: /v1/openapi.json, /v1/docs
    """
    app = FastAPI(
        title="OMR Service",
        version="2.0.0",
        openapi_url="/v1/openapi.json",
        docs_url="/v1/docs",
        lifespan=lifespan,
    )

    _REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9_\-]{1,64}$")

    # request_id middleware
    @app.middleware("http")
    async def request_id_middleware(request: Request, call_next):
        header_val = request.headers.get("X-Request-ID", "")
        if _REQUEST_ID_RE.match(header_val):
            rid = header_val
        else:
            rid = str(uuid.uuid4())
        request.state.request_id = rid
        response = await call_next(request)
        response.headers["X-Request-ID"] = rid
        return response

    # 依赖 + 错误处理
    register_dependencies(app, settings=settings, service=service, task_registry=task_registry)
    register_error_handlers(app)

    # 路由
    app.include_router(health.router)
    app.include_router(recognize.router)
    app.include_router(templates.router)
    app.include_router(tasks.router)
    app.include_router(crops.router)
    app.include_router(ocr.router)

    return app
