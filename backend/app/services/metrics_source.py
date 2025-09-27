"""Interfaces for sourcing equity and balance metrics from trading engines."""
from __future__ import annotations

from decimal import Decimal
from typing import Protocol

from loguru import logger


class MetricsSource(Protocol):
    """Protocol describing the metrics source contract."""

    async def fetch_equity_snapshot(self) -> Decimal:
        """Return the latest total equity for snapshot storage."""

    async def fetch_current_balance(self) -> Decimal:
        """Return the current account balance for heartbeat notifications."""


class DefaultMetricsSource:
    """Default metrics source returning zero values with warnings."""

    async def fetch_equity_snapshot(self) -> Decimal:
        logger.warning(
            "metrics_source_default_equity",
            message="Default metrics source in use; returning neutral equity value.",
        )
        return Decimal("0")

    async def fetch_current_balance(self) -> Decimal:
        logger.warning(
            "metrics_source_default_balance",
            message="Default metrics source in use; returning neutral balance value.",
        )
        return Decimal("0")


_metrics_source: MetricsSource = DefaultMetricsSource()


def configure_metrics_source(source: MetricsSource) -> None:
    """Override the default metrics source implementation."""

    global _metrics_source
    _metrics_source = source


def get_metrics_source() -> MetricsSource:
    """Return the active metrics source implementation."""

    return _metrics_source

