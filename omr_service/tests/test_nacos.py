"""Nacos 注册器与配置中心单元测试"""
import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from omr_service.config import OmrConfig
from omr_service.nacos_config import NacosConfigClient, _parse_content
from omr_service.nacos_reg import INTERFACE_SERVICE_NAME, NacosRegistrator


class TestMetadata(unittest.TestCase):
    def test_build_metadata(self):
        cfg = OmrConfig.from_env()
        reg = NacosRegistrator(cfg)
        meta = reg._build_metadata()
        self.assertEqual(meta["side"], "provider")
        self.assertEqual(meta["protocol"], "tri")
        self.assertIn("dubbo.endpoints", meta)

    def test_build_metadata_interface(self):
        cfg = OmrConfig.from_env()
        reg = NacosRegistrator(cfg)
        meta = reg._build_metadata(interface=True)
        self.assertEqual(meta["interface"], "omr.OmrService")
        self.assertEqual(meta["version"], cfg.service_version)
        self.assertIn("tri.service", meta)


class _SimpleParam:
    """Minimal stand-in for SDK param dataclasses."""

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class TestNacosRegistrator(unittest.TestCase):
    @patch("omr_service.nacos_reg.import_naming_service")
    @patch("omr_service.nacos_reg.import_client_config")
    @patch("omr_service.nacos_reg.import_register_instance_param")
    @patch("omr_service.nacos_reg.import_deregister_instance_param")
    @patch("omr_service.nacos_reg.shared_loop")
    def test_register_and_deregister(
        self,
        mock_shared_loop,
        mock_dereg_param,
        mock_reg_param,
        mock_cfg_cls,
        mock_naming_cls,
    ):
        cfg = OmrConfig.from_env()
        cfg.dubbo_port = 52527

        mock_reg_param.return_value = _SimpleParam
        mock_dereg_param.return_value = _SimpleParam

        mock_service = MagicMock()
        mock_service.register_instance = AsyncMock(return_value=True)
        mock_service.deregister_instance = AsyncMock(return_value=True)
        mock_service.shutdown = AsyncMock()
        mock_naming_cls.return_value.create_naming_service = AsyncMock(
            return_value=mock_service
        )

        mock_loop = MagicMock()
        mock_loop.run = MagicMock(side_effect=lambda coro: asyncio.run(coro))
        mock_shared_loop.return_value = mock_loop

        reg = NacosRegistrator(cfg)
        self.assertTrue(reg.register())
        self.assertIs(reg._naming_service, mock_service)

        # register_instance should be called for both service names
        calls = mock_service.register_instance.await_args_list
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0].args[0].service_name, cfg.nacos_service_name)
        self.assertEqual(calls[1].args[0].service_name, INTERFACE_SERVICE_NAME)

        self.assertTrue(reg.deregister())
        reg.close()
        mock_service.shutdown.assert_awaited()


class TestConfigParse(unittest.TestCase):
    def test_yaml(self):
        content = "redis:\n  host: 127.0.0.1\n  port: 6379\n"
        parsed = _parse_content(content, "omr-service.yaml")
        self.assertEqual(parsed["redis"]["host"], "127.0.0.1")

    def test_json(self):
        content = '{"key": "value"}'
        parsed = _parse_content(content, "config.json")
        self.assertEqual(parsed["key"], "value")


class TestNacosConfigClient(unittest.TestCase):
    @patch("omr_service.nacos_config.import_config_service")
    @patch("omr_service.nacos_config.import_client_config")
    @patch("omr_service.nacos_config.import_config_param")
    @patch("omr_service.nacos_config.shared_loop")
    def test_load(
        self,
        mock_shared_loop,
        mock_config_param,
        mock_cfg_cls,
        mock_config_service_cls,
    ):
        cfg = OmrConfig.from_env()
        mock_config_param.return_value = _SimpleParam

        mock_service = MagicMock()
        mock_service.get_config = AsyncMock(return_value="key: value")
        mock_service.shutdown = AsyncMock()
        mock_config_service_cls.return_value.create_config_service = AsyncMock(
            return_value=mock_service
        )

        mock_loop = MagicMock()
        mock_loop.run = MagicMock(side_effect=lambda coro: asyncio.run(coro))
        mock_shared_loop.return_value = mock_loop

        client = NacosConfigClient(cfg)
        result = client.load()
        self.assertEqual(result["key"], "value")


if __name__ == "__main__":
    unittest.main()
