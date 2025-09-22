"""Dynamic worker discovery."""

from __future__ import annotations

import importlib
import inspect
from typing import Any, Dict, Iterable, Iterator, List, Sequence, Tuple

from ai_trader.services.logging import get_logger


class WorkerLoader:
    """Load worker classes defined in configuration."""

    def __init__(self, worker_config: Dict[str, Any], symbols: Sequence[str]) -> None:
        self._worker_config = worker_config
        self._symbols = list(symbols)
        self._logger = get_logger(__name__)

    def load(self, shared_services: Dict[str, Any]) -> Tuple[List[object], List[object]]:
        workers: List[object] = []
        researchers: List[object] = []
        for worker_name, definition, force_researcher in self._iter_definitions():
            if not definition.get("enabled", True):
                self._logger.info("Worker %s disabled via configuration", worker_name)
                continue
            dotted_path = definition.get("module")
            if not dotted_path:
                self._logger.warning("Worker %s missing module path", worker_name)
                continue
            try:
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
            except Exception as exc:  # pragma: no cover - defensive logging
                self._logger.exception("Failed to load worker %s (%s): %s", worker_name, dotted_path, exc)
                continue
            target = researchers if force_researcher or getattr(worker, "is_researcher", False) else workers
            target.append(worker)
            self._logger.info("Worker %s loaded (%s)", worker.name, dotted_path)
        return workers, researchers

    def _iter_definitions(self) -> Iterator[Tuple[str, Dict[str, Any], bool]]:
        """Yield worker definitions along with research hints.

        Returns tuples of ``(worker_key, definition, force_researcher)`` so the
        loader can route researcher-style workers correctly even if the class
        lacks the ``is_researcher`` attribute. The existing ``definitions``
        mapping is processed first to maintain backwards compatibility, and the
        legacy top-level ``researcher`` block is normalised into the same
        structure.
        """

        definitions = self._worker_config.get("definitions", {})
        if isinstance(definitions, dict):
            for worker_name, definition in definitions.items():
                yield worker_name, definition, False

        researcher_cfg = self._worker_config.get("researcher")
        if not researcher_cfg:
            return

        for worker_name, definition in self._coerce_researchers(researcher_cfg):
            yield worker_name, definition, True

    @staticmethod
    def _coerce_researchers(
        researcher_cfg: Any,
    ) -> Iterable[Tuple[str, Dict[str, Any]]]:
        """Normalise researcher configuration into iterable key/config pairs."""

        if isinstance(researcher_cfg, dict):
            module = researcher_cfg.get("module")
            # Single researcher configuration at the top level.
            if module or researcher_cfg.get("parameters"):
                yield researcher_cfg.get("name", "researcher"), researcher_cfg
                return
            for worker_name, definition in researcher_cfg.items():
                if isinstance(definition, dict):
                    yield worker_name, definition
            return

        if isinstance(researcher_cfg, list):
            for index, definition in enumerate(researcher_cfg):
                if isinstance(definition, dict):
                    yield definition.get("name", f"researcher_{index}"), definition

