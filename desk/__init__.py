"""High-level package export for the trading desk runtime."""

__all__ = ["TradingRuntime"]


def __getattr__(name: str):  # pragma: no cover - import hook
    if name == "TradingRuntime":
        from .runtime import TradingRuntime

        return TradingRuntime
    raise AttributeError(name)
