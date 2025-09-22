"""Dynamic worker discovery."""

from __future__ import annotations

import importlib
from typing import Iterable, List, Sequence

from ..services.logging import get_logger


class WorkerLoader:
    """Load worker classes defined in configuration."""

    def __init__(self, worker_paths: Sequence[str], symbols: Sequence[str]) -> None:
        self._worker_paths = worker_paths
        self._symbols = list(symbols)
        self._logger = get_logger(__name__)

    def load(self) -> List[object]:
        workers: List[object] = []
        for dotted_path in self._worker_paths:
            module_name, class_name = dotted_path.rsplit(".", 1)
            module = importlib.import_module(module_name)
            worker_cls = getattr(module, class_name)
            worker = worker_cls(symbols=self._symbols)
            workers.append(worker)
            self._logger.info("Worker %s loaded", dotted_path)
        return workers
