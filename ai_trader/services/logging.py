"""Centralised logging configuration for the trading bot."""

from __future__ import annotations

import logging
from typing import Optional

from colorama import Fore, Style, init as colorama_init

colorama_init(autoreset=True)


class ColorFormatter(logging.Formatter):
    """Format log records with subtle colour accents."""

    LEVEL_MAP = {
        logging.DEBUG: Fore.BLUE,
        logging.INFO: Fore.GREEN,
        logging.WARNING: Fore.YELLOW,
        logging.ERROR: Fore.RED,
        logging.CRITICAL: Fore.MAGENTA,
    }

    def format(self, record: logging.LogRecord) -> str:  # noqa: D401
        colour = self.LEVEL_MAP.get(record.levelno, "")
        prefix = f"{colour}{record.levelname:<8}{Style.RESET_ALL}"
        message = super().format(record)
        return f"{prefix} {message}"


def configure_logging(level: int = logging.INFO) -> None:
    """Configure the root logger for the application."""

    logger = logging.getLogger()
    if logger.handlers:
        return

    handler = logging.StreamHandler()
    formatter = ColorFormatter("%(asctime)s | %(name)s | %(message)s", "%H:%M:%S")
    handler.setFormatter(formatter)
    logger.setLevel(level)
    logger.addHandler(handler)


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a module-scoped logger."""

    configure_logging()
    return logging.getLogger(name if name else "ai_trader")
