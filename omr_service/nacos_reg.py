"""Nacos 自注册（基于 nacos-sdk-python v2/v3 gRPC 协议）。"""
from __future__ import annotations

import json
import logging
from typing import Optional

from omr_service.config import OmrConfig
from omr_service.nacos_v2_compat import (
    import_client_config,
    import_deregister_instance_param,
    import_naming_service,
    import_register_instance_param,
    shared_loop,
)

logger = logging.getLogger(__name__)

INTERFACE_SERVICE_NAME = "providers:omr.OmrService::"


class NacosRegistrator:
    """Nacos 实例注册器（gRPC 协议）。

    同时注册：
    1. 应用级服务发现：serviceName = omr-service
    2. 接口级服务发现：serviceName = providers:omr.OmrService::

    Java 消费端推荐配置（interface 级）：
      dubbo.application.service-discovery.migration=FORCE_INTERFACE
      dubbo.consumer.protocol=tri
    """

    def __init__(self, cfg: OmrConfig):
        self.cfg = cfg
        self._naming_service: Optional = None
        self._loop = shared_loop()

    def _build_metadata(self, interface: bool = False) -> dict:
        """Dubbo Triple 需要的公共元数据。"""
        meta = {
            "side": "provider",
            "release": "3.0.0_py",
            "protocol": "tri",
            "application": self.cfg.nacos_service_name,
            "dubbo.endpoints": json.dumps(
                [{"port": self.cfg.dubbo_port, "protocol": "tri"}]
            ),
            "dubbo.metadata.storage-type": "local",
            "meta-v": "2.0.0",
        }
        if interface:
            meta.update(
                {
                    "interface": "omr.OmrService",
                    "version": self.cfg.service_version,
                    "group": self.cfg.nacos_group_name,
                    "methods": "ParseGoldenTemplate,RecognizeByTemplate,VerifyRecognitionRate,ReverifyPaper",
                    "generic": "false",
                    "revision": self.cfg.service_version,
                    "tri.service": "omr.OmrService",
                }
            )
        return meta

    def _make_client_config(self):
        ClientConfig = import_client_config()
        return ClientConfig(
            server_addresses=self.cfg.nacos_server,
            namespace_id=self.cfg.nacos_namespace or "public",
            username=self.cfg.nacos_username or "",
            password=self.cfg.nacos_password or "",
            log_level=logging.INFO,
        )

    def _register_one(self, service_name: str) -> bool:
        Param = import_register_instance_param()
        metadata = self._build_metadata(interface=service_name.startswith("providers:"))
        coro = self._naming_service.register_instance(
            Param(
                service_name=service_name,
                group_name=self.cfg.nacos_group_name,
                ip=self.cfg.local_ip,
                port=self.cfg.dubbo_port,
                metadata=metadata,
                healthy=True,
                enabled=True,
                weight=1.0,
                ephemeral=True,
            )
        )
        try:
            ok = self._loop.run(coro)
            if ok:
                logger.info(
                    "Nacos 注册成功: %s@%s:%s",
                    service_name,
                    self.cfg.local_ip,
                    self.cfg.dubbo_port,
                )
            else:
                logger.warning("Nacos 注册 %s 返回 False", service_name)
            return ok
        except Exception as exc:
            logger.error("Nacos 注册 %s 失败: %s", service_name, exc)
            return False

    def register(self) -> bool:
        """注册实例到 Nacos。"""
        NacosNamingService = import_naming_service()
        client_config = self._make_client_config()

        async def _create():
            return await NacosNamingService.create_naming_service(client_config)

        try:
            self._naming_service = self._loop.run(_create())
        except Exception as exc:
            logger.error("Nacos NamingService 初始化失败: %s", exc)
            return False

        results = [
            self._register_one(self.cfg.nacos_service_name),
            self._register_one(INTERFACE_SERVICE_NAME),
        ]
        return any(results)

    def deregister(self) -> bool:
        """从 Nacos 注销实例。"""
        if self._naming_service is None:
            return True

        DeregisterInstanceParam = import_deregister_instance_param()
        ok_all = True
        for service_name in (self.cfg.nacos_service_name, INTERFACE_SERVICE_NAME):
            coro = self._naming_service.deregister_instance(
                DeregisterInstanceParam(
                    service_name=service_name,
                    group_name=self.cfg.nacos_group_name,
                    ip=self.cfg.local_ip,
                    port=self.cfg.dubbo_port,
                    ephemeral=True,
                )
            )
            try:
                ok = self._loop.run(coro)
                if ok:
                    logger.info("Nacos 注销成功: %s", service_name)
                else:
                    logger.warning("Nacos 注销 %s 返回 False", service_name)
                    ok_all = False
            except Exception as exc:
                logger.error("Nacos 注销 %s 失败: %s", service_name, exc)
                ok_all = False
        return ok_all

    def close(self) -> None:
        """注销实例并释放资源。"""
        try:
            self.deregister()
        finally:
            if self._naming_service is not None:
                try:
                    self._loop.run(self._naming_service.shutdown())
                except Exception as exc:
                    logger.warning("Nacos NamingService 关闭异常: %s", exc)
                self._naming_service = None
