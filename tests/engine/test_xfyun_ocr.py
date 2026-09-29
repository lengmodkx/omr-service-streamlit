# -*- coding: utf-8 -*-
"""讯飞 OCR 引擎封装单元测试（mock HTTP，不发真实请求）"""
import base64
import hashlib
import json
import unittest
from unittest.mock import patch

import numpy as np

from omr_service.engine.xfyun_ocr import (
    XfyunDocOcr,
    XfyunHandwritingOcr,
    build_xfyun_clients,
    _extract_text_lines,
)


class _FakeResponse:
    def __init__(self, data, status_code=200):
        self._data = data
        self.status_code = status_code
        self.text = json.dumps(data, ensure_ascii=False)

    def json(self):
        return self._data


class TestXfyunDocOcr(unittest.TestCase):
    def setUp(self):
        self.client = XfyunDocOcr("appid", "key", "secret")

    def test_available_requires_credentials(self):
        self.assertTrue(self.client.available)
        self.assertFalse(XfyunDocOcr("", "key", "secret").available)

    def test_recognize_lines_plain_text(self):
        text_b64 = base64.b64encode("东乌一中 敖其泰 班级:四班\n802200231 考场:341班".encode()).decode()
        resp = _FakeResponse({
            "header": {"code": 0, "message": "ok", "sid": "s1"},
            "payload": {"result": {"text": text_b64}},
        })
        with patch.object(self.client._session, "post", return_value=resp) as mock_post:
            lines = self.client.recognize_lines(np.full((50, 100, 3), 255, dtype=np.uint8))
        self.assertIn("东乌一中 敖其泰 班级:四班", lines)
        # 鉴权 URL 应包含 host/date/authorization
        url = mock_post.call_args[0][0]
        self.assertIn("host=cbm01.cn-huabei-1.xf-yun.com", url)
        self.assertIn("authorization=", url)

    def test_recognize_lines_json_text(self):
        inner = json.dumps({"pages": [{"lines": [{"content": "东乌一中"}, {"content": "敖其泰"}]}]},
                           ensure_ascii=False)
        text_b64 = base64.b64encode(inner.encode()).decode()
        resp = _FakeResponse({
            "header": {"code": 0},
            "payload": {"result": {"text": text_b64}},
        })
        with patch.object(self.client._session, "post", return_value=resp):
            lines = self.client.recognize_lines(np.full((50, 100, 3), 255, dtype=np.uint8))
        self.assertIn("敖其泰", lines)

    def test_business_error_raises(self):
        resp = _FakeResponse({"header": {"code": 11201, "message": "licc failed", "sid": "s"}},
                             status_code=500)
        with patch.object(self.client._session, "post", return_value=resp):
            with self.assertRaises(RuntimeError):
                self.client.recognize_lines(np.full((50, 100, 3), 255, dtype=np.uint8))


class TestXfyunHandwritingOcr(unittest.TestCase):
    def setUp(self):
        self.client = XfyunHandwritingOcr("appid", "apikey123")

    def test_checksum_header(self):
        resp = _FakeResponse({"code": "0", "data": {"block": [
            {"line": [{"word": [{"content": "敖"}, {"content": "其"}, {"content": "泰"}]}]}
        ]}})
        with patch.object(self.client._session, "post", return_value=resp) as mock_post:
            text = self.client.recognize_text(np.full((50, 100, 3), 255, dtype=np.uint8))
        headers = mock_post.call_args[1]["headers"]
        expected = hashlib.md5((self.client.api_key + headers["X-CurTime"] + headers["X-Param"]).encode()).hexdigest()
        self.assertEqual(headers["X-CheckSum"], expected)
        self.assertEqual(text, "敖 其 泰")

    def test_business_error_raises(self):
        resp = _FakeResponse({"code": "40203", "desc": "illegal X-CheckSum", "sid": "s"})
        with patch.object(self.client._session, "post", return_value=resp):
            with self.assertRaises(RuntimeError):
                self.client.recognize_text(np.full((50, 100, 3), 255, dtype=np.uint8))


class TestBuildXfyunClients(unittest.TestCase):
    def test_disabled_by_default(self):
        class Cfg:
            xfyun_enabled = False
        self.assertEqual(build_xfyun_clients(Cfg()), (None, None))

    def test_enabled_without_credentials(self):
        class Cfg:
            xfyun_enabled = True
            xfyun_app_id = ""
            xfyun_doc_api_key = ""
            xfyun_doc_api_secret = ""
            xfyun_hw_api_key = ""
        self.assertEqual(build_xfyun_clients(Cfg()), (None, None))

    def test_enabled_with_credentials(self):
        class Cfg:
            xfyun_enabled = True
            xfyun_app_id = "appid"
            xfyun_doc_api_key = "k"
            xfyun_doc_api_secret = "s"
            xfyun_hw_api_key = "hk"
            xfyun_doc_url = ""
            xfyun_hw_url = ""
            xfyun_timeout_seconds = 10.0
            xfyun_verify_ssl = True
        doc, hw = build_xfyun_clients(Cfg())
        self.assertIsNotNone(doc)
        self.assertIsNotNone(hw)


class TestExtractTextLines(unittest.TestCase):
    def test_plain_text(self):
        self.assertEqual(_extract_text_lines("甲\n乙"), ["甲", "乙"])

    def test_empty(self):
        self.assertEqual(_extract_text_lines(""), [])


if __name__ == "__main__":
    unittest.main()
