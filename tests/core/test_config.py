from omr_service.config import OmrSettings


def test_default_http_port(monkeypatch):
    # 防止本机环境变量（如开发机设置了 OMR_HTTP_PORT=9501）干扰默认值断言
    monkeypatch.delenv("OMR_HTTP_PORT", raising=False)
    s = OmrSettings(_env_file=None)
    assert s.http_port == 8080


def test_env_override(monkeypatch):
    monkeypatch.setenv("OMR_HTTP_PORT", "9999")
    s = OmrSettings(_env_file=None)
    assert s.http_port == 9999


def test_legacy_dubbo_port_alias(monkeypatch):
    monkeypatch.setenv("OMR_LEGACY_DUBBO_PORT", "20884")
    s = OmrSettings(_env_file=None)
    assert s.legacy_dubbo_port == 20884


def test_redis_result_hash_prefix():
    s = OmrSettings(_env_file=None)
    assert s.redis_result_hash_prefix == "omr:batch:result:hash"


def test_crop_retention_days_env(monkeypatch):
    monkeypatch.setenv("OMR_CROP_RETENTION_DAYS", "14")
    s = OmrSettings(_env_file=None)
    assert s.crop_retention_days == 14


def test_crop_retention_days_nacos_override():
    from omr_service.nacos_config import apply_nacos_config

    s = OmrSettings(_env_file=None, crop_retention_days=14)
    apply_nacos_config(s, {"crop": {"retention-days": 7}})
    assert s.crop_retention_days == 7
