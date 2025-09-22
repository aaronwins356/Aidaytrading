"""CLI entrypoint that boots the trading runtime."""

from __future__ import annotations

import argparse

from desk import TradingRuntime
from desk.services.pretty_logger import pretty_logger


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the AI trading desk runtime")
    parser.add_argument(
        "--config",
        dest="config_path",
        default=None,
        help="Optional path to a runtime configuration file.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose debug logging (raw WebSocket/REST output).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    pretty_logger.configure(verbose=args.verbose)
    runtime = TradingRuntime(config_path=args.config_path)
    runtime.start_services()
    runtime.run()


if __name__ == "__main__":
    main()
