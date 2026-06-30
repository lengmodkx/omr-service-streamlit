"""gRPC Server 装配"""
import logging
from concurrent import futures

import grpc

from omr_service.config import OmrConfig
from omr_service.loader.image_loader import ImageLoader
from omr_service.loader.template_store import TemplateStore
from omr_service.rpc import omr_pb2_grpc
from omr_service.rpc.omr_service import OmrServiceServicer
from omr_service.worker.pool import WorkerPool

logger = logging.getLogger(__name__)


def create_server(
    cfg: OmrConfig,
    template_store: TemplateStore,
    image_loader: ImageLoader,
    worker_pool: WorkerPool,
) -> grpc.Server:
    """创建并配置 gRPC server"""
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=cfg.worker_count),
        options=[("grpc.max_send_message_length", 50 * 1024 * 1024),
                 ("grpc.max_receive_message_length", 50 * 1024 * 1024)],
    )
    servicer = OmrServiceServicer(cfg, template_store, image_loader, worker_pool)
    omr_pb2_grpc.add_OmrServiceServicer_to_server(servicer, server)
    server.add_insecure_port(f"[::]:{cfg.dubbo_port}")
    return server
