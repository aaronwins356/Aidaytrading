"""Time helpers for timezone aware scheduling."""
from __future__ import annotations

from datetime import datetime

import pytz

from ..config import get_settings


def now_tz() -> datetime:
    """Return timezone-aware now using configured timezone."""

    tz = pytz.timezone(get_settings().timezone)
    return datetime.now(tz)
