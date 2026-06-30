"""MQ 批量任务处理"""
import logging
import time
from typing import Any, Dict, List, Optional

from omr_service.config import OmrConfig
from omr_service.engine.recognizer import RecognizeContext
from omr_service.engine.recognizers import StandardTemplateRecognizer
from omr_service.loader.image_loader import ImageLoader
from omr_service.loader.template_store import TemplateStore
from omr_service.mq.producer import MqProducer
from omr_service.worker.pool import WorkerPool

logger = logging.getLogger(__name__)


class BatchJobHandler:
    """处理批量识别任务"""

    def __init__(
        self,
        cfg: OmrConfig,
        template_store: TemplateStore,
        image_loader: ImageLoader,
        worker_pool: WorkerPool,
        producer: MqProducer,
    ):
        self.cfg = cfg
        self.template_store = template_store
        self.image_loader = image_loader
        self.worker_pool = worker_pool
        self.producer = producer

    def _get_or_load_template(self, template_id: int) -> Optional[Any]:
        tpl = self.template_store.get(template_id)
        if tpl is not None:
            return tpl
        # 当前实现要求模板必须先通过 ParseGoldenTemplate 解析并缓存
        logger.warning("模板未找到: template_id=%s", template_id)
        return None

    def _recognize_one(self, template_id: int, image_url: str) -> Dict[str, Any]:
        """识别单张图片，返回可序列化的 dict"""
        start = time.perf_counter()
        tpl = self._get_or_load_template(template_id)
        if tpl is None:
            return {
                "scan_image_url": image_url,
                "code": 4,
                "message": "模板未找到",
                "answers": [],
            }

        img = self.image_loader.load(image_url)
        if img is None:
            return {
                "scan_image_url": image_url,
                "code": 5,
                "message": "图片加载失败",
                "answers": [],
            }

        recognizer = StandardTemplateRecognizer(standard_template=tpl)
        ctx = RecognizeContext()
        result = recognizer.recognize(img, ctx)

        return {
            "scan_image_url": image_url,
            "code": 0,
            "message": "ok",
            "template_id": template_id,
            "answers": [
                {
                    "q": q,
                    "answer": info.get("answer") or "",
                    "status": info.get("status", "empty"),
                    "correct": info.get("correct") or False,
                }
                for q, info in result.answers.items()
            ],
            "total": result.total,
            "empty_count": result.empty_count,
            "multi_count": result.multi_count,
            "card_flag": result.card_flag or "",
            "duration_ms": int(result.duration_ms),
        }

    def handle(self, job: Dict[str, Any]) -> Dict[str, Any]:
        """处理一个批量任务"""
        job_id = job.get("job_id", "")
        template_id = int(job.get("template_id", 0))
        image_urls: List[str] = job.get("image_urls", []) or []

        logger.info("[mq] 开始批量任务 job_id=%s template_id=%s images=%d", job_id, template_id, len(image_urls))

        completed = 0
        failed = 0
        results: List[Dict[str, Any]] = []

        def _task(url: str) -> Dict[str, Any]:
            try:
                return self._recognize_one(template_id, url)
            except Exception as e:
                logger.exception("[mq] 单张识别失败: %s", url)
                return {"scan_image_url": url, "code": 99, "message": str(e), "answers": []}

        # 并发识别
        futures = [self.worker_pool.submit(_task, url) for url in image_urls]
        for f in futures:
            res = f.result()
            if res.get("code", 0) == 0:
                completed += 1
            else:
                failed += 1
            results.append(res)

        payload = {
            "job_id": job_id,
            "template_id": template_id,
            "completed": completed,
            "failed": failed,
            "results": results,
        }

        # 发送结果
        result_stream = job.get("result_stream") or self.cfg.redis_result_stream
        try:
            self.producer.send_result(stream=result_stream, payload=payload)
        except Exception:
            logger.exception("[mq] 任务结果发送失败 job_id=%s", job_id)

        logger.info("[mq] 批量任务完成 job_id=%s completed=%d failed=%d", job_id, completed, failed)
        return payload
