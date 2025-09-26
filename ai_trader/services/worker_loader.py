"""Dynamic worker discovery."""

from __future__ import annotations

import importlib
import inspect
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

from ai_trader.services.logging import get_logger


class WorkerLoader:
    """Load worker classes defined in configuration."""

    def __init__(
        self,
        worker_config: Optional[Dict[str, Any]],
        symbols: Sequence[str],
        researcher_config: Any | None = None,
    ) -> None:
        self._worker_config: Dict[str, Any] = worker_config or {}
        self._symbols = self._normalize_symbols(symbols)
        self._logger = get_logger(__name__)
        self._forced_researchers: Set[str] = self._normalize_researchers(researcher_config)
        self._auto_researcher_key: str | None = None
        self._ensure_researcher_placeholder()
        if not self._forced_researchers:
            self._logger.warning(
                "No market researcher configured; ML feature engineering will remain idle."
            )

    def load(self, shared_services: Dict[str, Any]) -> Tuple[List[object], List[object]]:
        workers: List[object] = []
        researchers: List[object] = []
        ml_enabled_workers: List[object] = []
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
                symbols = self._normalize_symbols(definition.get("symbols", self._symbols))
                if not symbols:
                    symbols = list(self._symbols)
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
                self._logger.exception(
                    "Failed to load worker %s (%s): %s", worker_name, dotted_path, exc
                )
                continue
            is_researcher = force_researcher or getattr(worker, "is_researcher", False)
            display_name = str(definition.get("display_name", "")).strip()
            if display_name:
                setattr(worker, "name", display_name)
            emoji = str(definition.get("emoji", "")).strip()
            if emoji:
                setattr(worker, "emoji", emoji)
            target = researchers if is_researcher else workers
            target.append(worker)
            self._logger.info("Worker %s loaded (%s)", worker.name, dotted_path)
            if not is_researcher and getattr(worker, "_ml_service", None) is not None:
                ml_enabled_workers.append(worker)
        if not researchers:
            self._logger.warning(
                "No MarketResearchWorker instances were loaded; ML feature engineering will remain offline."
            )
        if ml_enabled_workers and not researchers:
            injected = self._inject_default_researcher(shared_services)
            if injected is not None:
                researchers.append(injected)
                self._logger.warning(
                    "Injected fallback MarketResearchWorker because %d ML worker(s) require research data.",
                    len(ml_enabled_workers),
                )
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
        if not isinstance(definitions, dict):
            return

        for worker_name, definition in definitions.items():
            force_researcher = worker_name in self._forced_researchers
            yield worker_name, definition, force_researcher

    def _ensure_researcher_placeholder(self) -> None:
        """Inject a default researcher definition when ML workers are configured."""

        definitions = self._worker_config.setdefault("definitions", {})
        if not isinstance(definitions, dict):
            return

        has_researcher = any(self._looks_like_researcher(defn) for defn in definitions.values())
        ml_workers_present = any(
            self._definition_requires_research(defn) for defn in definitions.values()
        )
        if ml_workers_present and not has_researcher:
            self._auto_researcher_key = "auto_researcher"
            definitions.setdefault(
                self._auto_researcher_key,
                {
                    "module": "ai_trader.workers.researcher.MarketResearchWorker",
                    "enabled": True,
                    "parameters": {"warmup_candles": 2, "log_every_n_snapshots": 1},
                },
            )
            self._forced_researchers.add(self._auto_researcher_key)

    def _definition_requires_research(self, definition: Dict[str, Any]) -> bool:
        module = str(definition.get("module", ""))
        if not module:
            return False
        if bool(definition.get("ml_required")):
            return True
        lowered = module.lower()
        if "ml_" in lowered or lowered.endswith("mlworker") or "mlshort" in lowered:
            return True
        return False

    def _normalize_researchers(self, external_config: Any | None) -> Set[str]:
        """Normalise legacy researcher config blocks into worker definitions."""

        forced: Set[str] = set()
        definitions = self._worker_config.setdefault("definitions", {})
        if not isinstance(definitions, dict):
            self._logger.warning(
                "Worker definitions must be a mapping, received %s", type(definitions).__name__
            )
            definitions = {}
            self._worker_config["definitions"] = definitions

        candidate_cfgs: List[Any] = []
        if external_config is not None:
            candidate_cfgs.append(external_config)
        worker_level_cfg = self._worker_config.get("researcher")
        if worker_level_cfg is not None:
            candidate_cfgs.append(worker_level_cfg)

        for cfg in candidate_cfgs:
            for worker_name, definition in self._coerce_researchers(cfg):
                if not isinstance(definition, dict):
                    continue
                normalized = dict(definition)
                normalized.setdefault("module", "ai_trader.workers.researcher.MarketResearchWorker")
                normalized.setdefault("enabled", True)
                existing = definitions.get(worker_name, {})
                merged = {**normalized, **existing}
                definitions[worker_name] = merged
                forced.add(worker_name)

        for worker_name, definition in definitions.items():
            if self._looks_like_researcher(definition):
                forced.add(worker_name)

        return forced

    def _normalize_symbols(self, symbols: Iterable[str] | str | None) -> List[str]:
        """Return a de-duplicated, upper-cased list of market symbols."""

        if symbols is None:
            return []
        if isinstance(symbols, str):
            candidates: Iterable[str] = [symbols]
        else:
            candidates = symbols
        normalised: List[str] = []
        seen: Set[str] = set()
        for candidate in candidates:
            if candidate is None:
                continue
            text = str(candidate).strip().upper()
            if not text or "/" not in text:
                self._logger.warning("Ignoring malformed worker symbol '%s'", candidate)
                continue
            base, quote = text.split("/", 1)
            if base == "XBT":
                base = "BTC"
            symbol = f"{base}/{quote}"
            if symbol not in seen:
                seen.add(symbol)
                normalised.append(symbol)
        return normalised

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

    @staticmethod
    def _looks_like_researcher(definition: Dict[str, Any]) -> bool:
        """Return True when the worker definition maps to a researcher class."""

        module = definition.get("module")
        if isinstance(module, str) and module.endswith("MarketResearchWorker"):
            return True
        return bool(definition.get("is_researcher"))

    def _inject_default_researcher(self, shared_services: Dict[str, Any]) -> object | None:
        """Instantiate an auto-injected MarketResearchWorker when needed."""

        if self._auto_researcher_key is None:
            return None
        definitions = self._worker_config.get("definitions", {})
        definition = definitions.get(self._auto_researcher_key)
        if not isinstance(definition, dict):
            return None

        try:
            module_name, class_name = str(definition["module"]).rsplit(".", 1)
            module = importlib.import_module(module_name)
            worker_cls = getattr(module, class_name)
            params = definition.get("parameters", {})
            signature = inspect.signature(worker_cls)
            kwargs: Dict[str, Any] = {}
            for param_name in signature.parameters:
                if param_name == "symbols":
                    kwargs[param_name] = self._symbols
                elif param_name == "config":
                    kwargs[param_name] = params
                elif param_name in shared_services:
                    kwargs[param_name] = shared_services[param_name]
            worker = worker_cls(**kwargs)
        except Exception as exc:  # pragma: no cover - defensive logging
            self._logger.exception("Failed to inject default MarketResearchWorker: %s", exc)
            return None
        self._logger.info(
            "Worker %s loaded (%s) [auto-injected]",
            worker.name,
            definition.get("module"),
        )
        return worker
