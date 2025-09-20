"""CLI entrypoint that boots the trading runtime."""

from desk import TradingRuntime


def main() -> None:
    runtime = TradingRuntime()
    runtime.start_services()
    runtime.run()


if __name__ == "__main__":
    main()
