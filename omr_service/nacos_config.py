"""Nacos 配置中心客户端（基于 nacos-sdk-python v2/v3 gRPC 协议）。"""
from __future__ import annotations

import json
import logging
from typing import Any, Callable, Dict, Optional

import yaml

from omr_service.config import OmrConfig
from omr_service.nacos_v2_compat import (
    import_client_config,
    import_config_param,
    import_config_service,
    shared_loop,
)

logger = logging.getLogger(__name__)


def _parse_content(content: str, data_type: str) -> Dict[str, Any]:
    """根据配置类型解析内容。"""
    if not content:
        return {}
    data_type = (data_type or "").lower()
    if data_type.endswith(".yaml") or data_type.endswith(".yml"):
        return yaml.safe_load(content) or {}
    if data_type.endswith(".json"):
        return json.loads(content)
    if data_type.endswith(".properties"):
        return _parse_properties(content)
    # 默认按 yaml 尝试
    try:
        return yaml.safe_load(content) or {}
    except Exception:
        return {}


def _parse_properties(content: str) -> Dict[str, Any]:
    """简单 properties 解析。"""
    result = {}
    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("!"):
            continue
        if "=" in line:
            k, v = line.split("=", 1)
            result[k.strip()] = v.strip()
    return result


def flatten_dict(d: Dict[str, Any], parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
    """把嵌套 dict 打平，例如 redis.host -> value。"""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


class NacosConfigClient:
    """Nacos 配置中心客户端（gRPC 协议）。

    支持：
    - 启动时拉取配置
    - 服务端推送的配置变更监听
    """

    def __init__(self, cfg: OmrConfig):
        self.cfg = cfg
        self._data_id = cfg.nacos_config_data_id
        self._group = cfg.nacos_config_group
        self._config_service: Optional = None
        self._loop = shared_loop()
        self._on_change_callbacks: list = []

    def _make_client_config(self):
        ClientConfig = import_client_config()
        return ClientConfig(
            server_addresses=self.cfg.nacos_server,
            namespace_id=self.cfg.nacos_namespace or "public",
            username=self.cfg.nacos_username or "",
            password=self.cfg.nacos_password or "",
            log_level=logging.INFO,
        )

    def load(self) -> Dict[str, Any]:
        """从 Nacos 拉取配置。"""
        ConfigService = import_config_service()
        ConfigParam = import_config_param()
        client_config = self._make_client_config()

        async def _load():
            service = await ConfigService.create_config_service(client_config)
            try:
                return await service.get_config(
                    ConfigParam(data_id=self._data_id, group=self._group)
                )
            finally:
                await service.shutdown()

        try:
            content = self._loop.run(_load())
            parsed = _parse_content(content, self._data_id)
            logger.info(
                "Nacos 配置拉取成功: dataId=%s group=%s keys=%d",
                self._data_id,
                self._group,
                len(parsed),
            )
            return parsed
        except Exception as exc:
            logger.error("Nacos 配置拉取失败: %s", exc)
            return {}

    def start_listener(self) -> bool:
        """启动配置变更监听（服务端推送）。"""
        ConfigService = import_config_service()
        client_config = self._make_client_config()

        async def _create_and_listen():
            service = await ConfigService.create_config_service(client_config)
            await service.add_listener(self._data_id, self._group, self._on_config_change)
            return service

        try:
            self._config_service = self._loop.run(_create_and_listen())
            logger.info("Nacos 配置监听已启动")
            return True
        except Exception as exc:
            logger.warning("Nacos 配置监听启动失败: %s", exc)
            return False

    def _on_config_change(self, tenant: str, group: str, data_id: str, content: str) -> None:
        """SDK 推送回调。

        Signature expected by nacos-sdk-python v2:
        ``listener(tenant, group, data_id, content)``.
        """
        logger.info("Nacos 配置发生变更，重新拉取")
        parsed = _parse_content(content, data_id)
        for cb in self._on_change_callbacks:
            try:
                cb(parsed)
            except Exception as exc:
                logger.error("配置变更回调执行失败: %s", exc)

    def add_change_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        self._on_change_callbacks.append(callback)

    def close(self) -> None:
        if self._config_service is not None:
            try:
                self._loop.run(self._config_service.shutdown())
            except Exception as exc:
                logger.warning("Nacos ConfigService 关闭异常: %s", exc)
            self._config_service = None
