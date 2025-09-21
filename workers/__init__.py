"""Strategy worker exports for modular runtime wiring."""

from desk.services.worker import Intent, VetoResult, Worker

__all__ = ["Worker", "Intent", "VetoResult"]
