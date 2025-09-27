#!/usr/bin/env python3

"""Utility for creating consistent SQLite backups using the native backup API."""

from __future__ import annotations

import argparse
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional


LOGGER = logging.getLogger(__name__)
DEFAULT_DATABASE = Path("ai_trader/data/trades.db")


def backup_sqlite(source: Path, destination: Path) -> Path:
    """Copy ``source`` to ``destination`` using the SQLite backup API."""

    if not source.exists():
        raise FileNotFoundError(f"SQLite database not found: {source}")

    destination.parent.mkdir(parents=True, exist_ok=True)

    try:
        with sqlite3.connect(source) as source_conn:
            with sqlite3.connect(destination) as destination_conn:
                source_conn.backup(destination_conn)
                destination_conn.commit()
    except sqlite3.Error as exc:
        raise RuntimeError(f"Failed to backup SQLite database: {exc}") from exc

    LOGGER.info("SQLite backup written to %s", destination)
    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a hot backup of the SQLite trade log.")
    parser.add_argument(
        "db_path",
        type=Path,
        nargs="?",
        default=DEFAULT_DATABASE,
        help="Path to the source SQLite database (default: ai_trader/data/trades.db)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional destination path. Defaults to ai_trader/data/backups/<db>-<timestamp>.bak",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()

    source = args.db_path.resolve()
    destination: Optional[Path]
    if args.output:
        destination = args.output.resolve()
    else:
        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        destination = source.parent / "backups" / f"{source.stem}-{timestamp}.bak"

    try:
        backup_sqlite(source, destination)
    except Exception as exc:  # noqa: BLE001 - surface the failure for operator awareness
        LOGGER.exception("SQLite backup failed: %s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
