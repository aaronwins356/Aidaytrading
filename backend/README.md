# Aidaytrading Backend

This directory houses the production-ready FastAPI backend for the Aidaytrading project. The legacy trading bot and research code remains in the top-level `ai_trader/` package; only the backend services have been reorganised under `backend/` in this prompt.

## What Prompt 1 Delivers

- Fully typed FastAPI application exposing `/api/v1/signup`, `/api/v1/login`, `/api/v1/refresh`, `/api/v1/logout`, and `/api/v1/me` endpoints with consistent error envelopes.
- Secure authentication/authorisation flow with bcrypt password hashing, access and refresh JWTs, and a blacklist table for revocation.
- Deterministic Alembic migrations and asynchronous SQLAlchemy models for `users` and `token_blacklist` tables.
- Pydantic Settings-driven configuration that validates required environment variables and supports `.env` files.
- Structured JSON logging middleware safe for PII.
- Developer tooling via Poetry, Makefile, and pre-commit along with Black, isort, Flake8, MyPy, and pytest + coverage.
- Comprehensive async API tests (`pytest-asyncio` + `httpx`) with >80% coverage of new modules.

## Prompt 2 Enhancements

- Production Brevo SMTP integration with templated HTML notifications and exponential retry logic.
- Admin-only API surface for approving, disabling, and re-assigning users with enforced role/status checks.
- Token versioning strategy that revokes existing access/refresh tokens when roles or status change.
- Append-only audit logging of privileged actions with pagination for compliance review.

## Prompt 3 Enhancements

- Expanded data model with equity snapshots, per-day P/L rollups, trade history, device tokens, and singleton system status.
- Reporting APIs for equity curves, profitability summaries, calendar heatmaps, trades, and device registration.
- Secure WebSocket feeds for status, equity, trades, and balances with token authentication and in-memory pub/sub.
- APScheduler-driven jobs for equity snapshots, midnight (America/Chicago) daily P&L rollups, and twice-daily heartbeats.
- Firebase push notification integration with device token fan-out and structured logging.
- IP-based login rate limiting (5 attempts per 15 minutes) to throttle brute-force attacks.

## Project Structure

```
backend/
  app/
    api/
      v1/
    core/
    models/
    schemas/
    services/
    migrations/
    tests/
  pyproject.toml
  Makefile
  README.md
  .env.example
  .flake8
  .editorconfig
  .pre-commit-config.yaml
```

`ai_trader/` and other existing directories are untouched and continue to operate independently of the backend service.

## Requirements

- Python 3.11+
- [Poetry](https://python-poetry.org/) 1.7+

## Quick Start

1. **Create and populate an environment file**

   Copy `.env.example` to `.env` and update the values as needed:

   ```bash
   cp .env.example .env
   ```

   - `DB_URL`: Async SQLAlchemy URL. Defaults to a local SQLite file (`sqlite+aiosqlite:///./backend.db`).
   - `JWT_SECRET`: Replace with a strong random string (e.g. `openssl rand -hex 32`).
   - `JWT_ALGORITHM`: Defaults to `HS256`.
   - `ACCESS_TOKEN_EXPIRES_MIN`: Minutes before access tokens expire (default `15`).
   - `REFRESH_TOKEN_EXPIRES_DAYS`: Days before refresh tokens expire (default `7`).
   - `ENV`: `local`, `dev`, or `prod`. Non-local environments must not use `CHANGE_ME` secrets.
   - `BREVO_API_KEY`: Brevo transactional email API key (used as the SMTP password).
   - `BREVO_SMTP_SERVER`: Brevo SMTP hostname (e.g. `smtp-relay.brevo.com`).
   - `BREVO_PORT`: Brevo SMTP port (typically `587`).
   - `BREVO_SENDER_EMAIL`: Verified Gmail/Brevo sender address for outbound mail.
   - `BREVO_SENDER_NAME`: Friendly display name shown in email headers.
   - `ADMIN_NOTIFICATION_EMAIL`: Destination mailbox for new-signup alerts.

2. **Install dependencies**

   ```bash
   make install
   ```

3. **Run database migrations**

   ```bash
   make migrate
   ```

4. **Start the development server**

   ```bash
   make run
   ```

   The API is available at http://localhost:8000 and ships with interactive docs at `/docs`.

## Tooling Commands

| Command | Description |
| ------- | ----------- |
| `make install` | Install project dependencies via Poetry. |
| `make fmt` | Format code with isort and Black. |
| `make lint` | Run Flake8 and MyPy. |
| `make test` | Execute pytest with coverage (async tests included). |
| `make run` | Start the FastAPI development server with Uvicorn. |
| `make migrate` | Apply Alembic migrations (`alembic upgrade head`). |
| `make revision` | Create a new auto-generated migration (`alembic revision --autogenerate`). |

## Database Migrations

Alembic is configured for async SQLAlchemy. To create a new migration after modifying models:

```bash
make revision MESSAGE="add_new_table"
```

To run migrations in production environments, ensure `DB_URL` points to the correct database and run `make migrate`.

## Testing

All tests are asynchronous and use a temporary SQLite database with migrations applied automatically.

```bash
make test
```

Coverage reports are emitted to the terminal; CI should enforce ≥80% coverage for the modules introduced here.

## API Overview

All responses use a consistent error format: `{ "error": { "code": str, "message": str, "details": optional } }`.

### POST `/api/v1/signup`

Request:

```json
{
  "username": "alice",
  "email": "alice@example.com",
  "password": "Str0ngPass!"
}
```

Response (`201 Created`):

```json
{
  "message": "Signup received. Await approval.",
  "status": "pending"
}
```

### POST `/api/v1/login`

Request:

```json
{
  "username": "alice",
  "password": "Str0ngPass!"
}
```

Response (`200 OK`):

```json
{
  "access_token": "<JWT>",
  "refresh_token": "<JWT>",
  "token_type": "bearer"
}
```

Inactive or disabled accounts return `401` with code `inactive_account` and the generic message “Invalid credentials or inactive account.”

### POST `/api/v1/refresh`

Request body or `Authorization: Bearer <refresh token>` header containing the refresh token.

Response (`200 OK`):

```json
{
  "access_token": "<JWT>",
  "token_type": "bearer"
}
```

### POST `/api/v1/logout`

Requires the access token in the `Authorization` header. Optionally include the refresh token in the body.

Response (`200 OK`):

```json
{
  "message": "Logged out"
}
```

The provided token JTIs are persisted to the blacklist table; subsequent requests with the same token fail with `401 token_revoked`.

### GET `/api/v1/me`

Requires a valid (non-blacklisted) access token. Returns the authenticated user's profile:

```json
{
  "id": 1,
  "username": "alice",
  "email": "alice@example.com",
  "role": "viewer",
  "status": "active",
  "created_at": "2024-04-01T12:34:56+00:00",
  "updated_at": "2024-04-01T12:34:56+00:00"
}
```

### Admin API (all routes require an active admin access token)

| Method & Path | Description |
| --- | --- |
| `GET /api/v1/admin/pending-users` | List users awaiting approval. |
| `POST /api/v1/admin/approve/{id}` | Promote a pending user to `active`. |
| `POST /api/v1/admin/disable/{id}` | Disable an account and immediately revoke existing tokens. |
| `POST /api/v1/admin/role/{id}` | Change a user's role between `viewer` and `admin`. |
| `GET /api/v1/admin/audit-logs?limit=&offset=` | Paginated view of admin actions (append-only). |

All admin mutations run inside a single transaction, write an `admin_actions` audit entry, and bump the target user's
`token_version` so any existing tokens become invalid. Safeguards prevent self-disabling and ensure at least one active admin
remains.

### Reporting & Portfolio APIs

- **GET `/api/v1/equity-curve`** – Returns ordered `[timestamp, equity]` pairs. Supports `start`, `end`, and `limit` query parameters; defaults to the most recent 30 UTC days when omitted.
- **GET `/api/v1/profit`** – Provides the latest balance, total P/L amount and percentage (relative to `BASELINE_EQUITY` or the earliest snapshot), and the trade win-rate.
- **GET `/api/v1/calendar`** – Produces a date-keyed dictionary of daily P/L and colour classification (`green|red|neutral`). Aggregation is always executed in `America/Chicago` to match trading hours.
- **GET `/api/v1/status`** – Exposes `{ running, uptime_seconds }` reflecting the singleton system status row.
- **GET `/api/v1/trades`** – Paginated trade ledger (`page`, `page_size`, `symbol`, `side`, `start`, `end`) sorted by timestamp descending.
- **POST `/api/v1/register-device`** – Upserts a push-notification device token for the authenticated user (`platform` defaults to `ios`).

> **Login rate limiting:** `POST /api/v1/login` is throttled to 5 attempts per IP per 15 minutes. Excess attempts receive `429 Too Many Requests` with a `retry_after` hint (seconds) for client backoff logic.

## Email Notifications

- New user signups persist with `pending` status and trigger an HTML email to `ADMIN_NOTIFICATION_EMAIL`.
- Templates are rendered with Jinja2 and include a branded header, body copy, and footer.
- Delivery uses Brevo's SMTP relay (`username=apikey`) with TLS. Transient failures retry with exponential backoff and jitter;
  persistent failures emit structured logs but do not fail the originating request.

## Audit Logging

- `admin_actions` captures `{admin_id, action, target_user_id, metadata, created_at}` with append-only semantics.
- `metadata` stores contextual JSON (previous/new role or status) to simplify investigations.
- Indexes on `admin_id`, `target_user_id`, and `created_at` enable efficient filtering for compliance reviews.

## WebSocket Feeds

- `/ws/status` – Broadcasts bot lifecycle updates. Clients must supply a valid access token via `?token=` or `Authorization: Bearer` header; unauthenticated connections are closed with `1008`.
- `/ws/equity` – Streams the latest equity snapshot every scheduler tick plus on-connect priming. Payloads follow `{ "event": "equity", "data": { "timestamp": ISO8601, "equity": "<Decimal>" } }`.
- `/ws/trades` – Designed for near-real-time trade ingest. Call `app.services.trade_feed.broadcast_trade` after inserting a trade to fan-out updates.
- `/ws/balance` – Emits balance heartbeats in sync with the scheduled push notifications.

All feeds share an in-memory pub/sub implementation that can be swapped for Redis in production. Connections are read-only and respect viewer-role access tokens.

## Scheduler & Time Zones

`APScheduler` starts when the FastAPI lifespan begins (disabled automatically in tests with `SCHEDULER_ENABLED=false`). Jobs:

- Equity snapshots every `EQUITY_SNAPSHOT_INTERVAL_MINUTES` (default 10) via `services.metrics_source`.
- Daily P/L rollup at midnight `America/Chicago` using trade timestamps and persisting into `daily_pnl`.
- Heartbeat push notifications at `HEARTBEAT_CRON` (default `0 2,8,14,20 * * *`, i.e., 2am/8am/2pm/8pm Central).

All scheduler timestamps are stored in UTC, while calendar aggregation explicitly converts to Central Time for day boundaries.

## Push Notifications

- Set `FIREBASE_CREDENTIALS` to either the path of the Google service-account JSON file or the JSON string itself. When omitted, push delivery is disabled gracefully.
- Register devices through `POST /api/v1/register-device`; tokens are deduplicated via a unique index.
- `services.push.send_push_to_user` fans out to every active token for a user and removes invalid tokens (e.g., unregistered APNs keys).
- Heartbeat job payloads are formatted as `Balance: $X, P/L: Y%` and also broadcast over `/ws/balance`.

## CORS Configuration

Use `CORS_ALLOWED_ORIGINS` to provide a comma-separated list of permitted origins (e.g., iOS WebViews). Defaults allow `http://localhost` and `http://localhost:3000` for development. Tighten this list for production deployments.

## Security Decisions

- Emails are normalised to lowercase (`email_canonical`) with a unique index for case-insensitive deduplication.
- JWTs embed `token_version`; any role/status change increments the value so stale access/refresh tokens are rejected. Logout
  still blacklists explicit JTIs for immediate revocation of the active session.
- Usernames are trimmed and validated against `^[A-Za-z0-9_]+$` to eliminate unsafe characters.
- `require_active_user` / `require_admin_active_user` dependencies centralise JWT decoding, blacklist checks, role validation,
  and status enforcement.

## Logging

Requests are logged as single-line JSON with fields for timestamp, log level, message, request ID, method, path, status code, and latency. Sensitive information (passwords, full email addresses, tokens) is never written to logs. Use `tail -f` on your server logs to observe structured output in real time.

## Pre-commit Hooks

Install the hooks once dependencies are installed:

```bash
poetry run pre-commit install
```

The hooks enforce formatting (Black, isort), linting (Flake8), and guard against large files or accidentally committed secrets.

## Next Steps

Future work can focus on richer admin tooling (e.g. frontend UI), granular token/session management, and automated escalation
rules for high-risk signups.
