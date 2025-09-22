# Kraken Live-Readiness Checklist

The trading runtime now executes a deterministic readiness audit each time it
boots. The checks run before the main loop starts to ensure that the desk is
safe to operate in Kraken live mode. A condensed summary is printed to the
console and a full JSON payload is persisted to the dashboard SQLite database
so it can be displayed inside the Streamlit command center.

## Checks Performed

1. **ccxt Exchange Handle** – Confirms that the Kraken `ccxt` client is loaded
   and markets have been initialised.
2. **Symbol Resolution** – Verifies that every configured symbol resolves to a
   canonical Kraken market with cached metadata.
3. **Precision & Minimums** – Ensures price/amount precision and minimum order
   information is present so that the sizing engine can respect Kraken limits.
4. **Credentials** – Validates that mandatory environment variables (e.g.
   `KRAKEN_KEY`, `KRAKEN_SECRET`) are present.
5. **Dashboard Database** – Checks the Streamlit dashboard SQLite database is
   writable, creating it on demand when required.
6. **Clock Drift** – Compares local time to the Kraken server time (when
   available) and raises an error if the drift exceeds two seconds.

Each item is tagged as `ok`, `warning`, or `error`. Any error blocks live
readiness and should be rectified before trading.

## Troubleshooting

- **Missing Metadata** – Call `KrakenBroker.normalise_symbols()` and ensure
  `ccxt` is installed so that markets are cached correctly.
- **Precision Warnings** – Reload markets via `ccxt` or override the minimums
  within the strategy configuration.
- **Clock Drift Errors** – Synchronise the host clock (e.g. via `ntp`) and
  restart the runtime.
- **Database Failures** – Verify file permissions for `desk/db/live_trading.sqlite`.

The dashboard exposes the latest report under `⚙️ Settings & Readiness`. Use it
as a pre-flight checklist before enabling execution.
