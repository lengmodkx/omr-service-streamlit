"""HTTP 健康检查服务"""
import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

logger = logging.getLogger(__name__)


class HealthHandler(BaseHTTPRequestHandler):
    """简单的健康检查 handler"""

    def log_message(self, format, *args):
        logger.debug(format, *args)

    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "UP"}).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()


class HealthServer:
    """HTTP 健康检查服务（用于 K8s / Docker HEALTHCHECK）"""

    def __init__(self, port: int = 8080):
        self.port = port
        self._server: HTTPServer = None
        self._thread: threading.Thread = None

    def start(self):
        self._server = HTTPServer(("0.0.0.0", self.port), HealthHandler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        logger.info("健康检查服务已启动: :%s/health", self.port)

    def stop(self):
        if self._server:
            self._server.shutdown()
            self._server.server_close()
            logger.info("健康检查服务已停止")
