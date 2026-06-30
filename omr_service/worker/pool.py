"""识别任务线程池"""
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, TypeVar

T = TypeVar("T")


class WorkerPool:
    """基于 ThreadPoolExecutor 的轻量线程池"""

    def __init__(self, max_workers: int = 4):
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="omr_worker")
        self.max_workers = max_workers

    def submit(self, fn: Callable[..., T], *args, **kwargs) -> T:
        return self._executor.submit(fn, *args, **kwargs)

    def map(self, fn: Callable[..., T], iterable):
        return self._executor.map(fn, iterable)

    def shutdown(self, wait: bool = True):
        self._executor.shutdown(wait=wait)
