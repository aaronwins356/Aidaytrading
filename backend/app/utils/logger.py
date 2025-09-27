"""Logging helpers."""
from __future__ import annotations

import logging
from logging.config import dictConfig

_LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s %(levelname)s [%(name)s] %(message)s",
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "default",
        }
    },
    "root": {
        "level": "INFO",
        "handlers": ["console"],
    },
}

dictConfig(_LOGGING_CONFIG)


def get_logger(name: str) -> logging.Logger:
    """Return configured logger."""

    return logging.getLogger(name)
