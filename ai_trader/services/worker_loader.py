"""Dynamic worker discovery."""

from __future__ import annotations

import importlib
import inspect
from typing import Any, Dict, List, Sequence, Tuple

from ..services.logging import get_logger


class WorkerLoader:
    """Load worker classes defined in configuration."""

    def __init__(self, worker_config: Dict[str, Any], symbols: Sequence[str]) -> None:
        self._worker_config = worker_config
        self._symbols = list(symbols)
        self._logger = get_logger(__name__)

    def load(self, shared_services: Dict[str, Any]) -> Tuple[List[object], List[object]]:
        workers: List[object] = []
        researchers: List[object] = []
        definitions = self._worker_config.get("definitions", {})
        for worker_name, definition in definitions.items():
            if not definition.get("enabled", True):
                self._logger.info("Worker %s disabled via configuration", worker_name)
                continue
            dotted_path = definition.get("module")
            if not dotted_path:
                self._logger.warning("Worker %s missing module path", worker_name)
                continue
            module_name, class_name = dotted_path.rsplit(".", 1)
            module = importlib.import_module(module_name)
            worker_cls = getattr(module, class_name)
            kwargs: Dict[str, Any] = {}
            symbols = definition.get("symbols", self._symbols)
            params = definition.get("parameters", {})
            risk_cfg = definition.get("risk", {})
            signature = inspect.signature(worker_cls)
            for param_name in signature.parameters:
                if param_name == "symbols":
                    kwargs[param_name] = symbols
                elif param_name == "config":
                    kwargs[param_name] = params
                elif param_name == "risk_config":
                    kwargs[param_name] = risk_cfg
                elif param_name in shared_services:
                    kwargs[param_name] = shared_services[param_name]
            worker = worker_cls(**kwargs)
            target = researchers if getattr(worker, "is_researcher", False) else workers
            target.append(worker)
            self._logger.info("Worker %s loaded (%s)", worker.name, dotted_path)
        return workers, researchers
