"""Service layer exports for the trading desk runtime."""

__all__ = [
    "BrokerCCXT",
    "ExecutionEngine",
    "PositionStore",
    "OpenTrade",
    "FeedHandler",
    "Learner",
    "EventLogger",
    "PortfolioManager",
    "RiskEngine",
    "TelemetryClient",
    "Intent",
    "VetoResult",
    "Worker",
]


def __getattr__(name: str):  # pragma: no cover - import hook
    if name == "BrokerCCXT":
        from .broker import BrokerCCXT

        return BrokerCCXT
    if name == "ExecutionEngine":
        from .execution import ExecutionEngine

        return ExecutionEngine
    if name == "PositionStore":
        from .execution import PositionStore

        return PositionStore
    if name == "OpenTrade":
        from .execution import OpenTrade

        return OpenTrade
    if name == "FeedHandler":
        from .feed import FeedHandler

        return FeedHandler
    if name == "Learner":
        from .learner import Learner

        return Learner
    if name == "EventLogger":
        from .logger import EventLogger

        return EventLogger
    if name == "PortfolioManager":
        from .portfolio import PortfolioManager

        return PortfolioManager
    if name == "RiskEngine":
        from .risk import RiskEngine

        return RiskEngine
    if name == "TelemetryClient":
        from .telemetry import TelemetryClient

        return TelemetryClient
    if name == "Intent":
        from .worker import Intent

        return Intent
    if name == "VetoResult":
        from .worker import VetoResult

        return VetoResult
    if name == "Worker":
        from .worker import Worker

        return Worker
    if name == "broker":
        from importlib import import_module

        return import_module(".broker", __name__)
    raise AttributeError(name)
