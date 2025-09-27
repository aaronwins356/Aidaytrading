# Aidaytrading Backend

This package provides the production-grade FastAPI backend for the live trading bot control centre that powers the iOS client. It exposes authenticated REST and WebSocket APIs for investors and administrators, integrates with Brevo for email notifications, uses Firebase Cloud Messaging for push notifications, and ships with observability, risk controls, and automated QA.

## Features

- **Authentication & lifecycle** – bcrypt password hashing, short-lived access tokens, refresh token storage with revocation, signup approval workflow, and role-based access (viewer vs. admin).
- **Admin tooling** – approve, disable, and re-role users; manage global risk controls; start/stop the trading bot; switch between paper/live modes; trigger password resets via Brevo.
- **Investor APIs** – portfolio status, PnL summary, equity curve, trade ledger, calendar heatmap, and balance history with strict role enforcement.
- **Realtime streaming** – token-authenticated WebSocket feeds for status, equity, trades, and balances with broadcast deduplication.
- **Notifications** – Brevo email hooks and Firebase push fan-out with a 6-hour heartbeat scheduler (08:00, 14:00, 20:00, 02:00 America/Chicago).
- **Monitoring** – `/health` JSON endpoint and `/metrics` Prometheus exposition for trades, websockets, notifications, and scheduler runs.
- **Tooling** – Black, Flake8, and pytest automation via GitHub Actions, Docker build validation, and extensive unit/regression tests.

## Project Structure

```
backend/
  app/
    api/
      __init__.py
      auth.py
      bot.py
      devices.py
      equity.py
      monitoring.py
      risk.py
      trades.py
      users.py
      websocket.py
    config.py
    database.py
    main.py
    models/
      __init__.py
      risk.py
      trade.py
      user.py
    schemas/
      __init__.py
      auth.py
      risk.py
      trade.py
      user.py
    security/
      __init__.py
      auth.py
      jwt.py
    services/
      __init__.py
      bot.py
      brevo.py
      notifications.py
      push.py
      risk.py
    utils/
      __init__.py
      logger.py
      time.py
  tests/
    conftest.py
    test_auth.py
    test_bot.py
    test_risk.py
    test_users.py
    test_ws.py
    regression/
      test_regression.py
  .env.example
  pyproject.toml
  README.md
```

## Prerequisites

- Python 3.11+
- [Poetry 1.7+](https://python-poetry.org/) or `pip` if you prefer requirements files
- Optional: Docker for containerised deployments

## Installation

```bash
cd backend
poetry install --no-root
```

To work with `pip`, install from `pyproject.toml`:

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

## Configuration

Create a local environment file using the provided template:

```bash
cp .env.example .env
```

Set the following values:

| Variable | Description |
| --- | --- |
| `DATABASE_URL` | SQLAlchemy database URL (SQLite default). |
| `JWT_SECRET` | 32+ character secret used to sign JWTs. |
| `JWT_ALGORITHM` | Signing algorithm (defaults to `HS256`). |
| `JWT_ACCESS_EXPIRES` | Access token lifetime in seconds. |
| `JWT_REFRESH_EXPIRES` | Refresh token lifetime in seconds. |
| `BREVO_API_KEY` | Brevo transactional email API key. |
| `OWNER_EMAIL` | Destination address for signup approvals. |
| `FIREBASE_CREDENTIALS_PATH` | Path to Firebase service account JSON file. |
| `TIMEZONE` | Scheduler timezone (defaults to `America/Chicago`). |

## Running the API

```bash
poetry run uvicorn app.main:app --reload
```

The OpenAPI docs are available at [http://localhost:8000/docs](http://localhost:8000/docs).

## Quality Gates

```bash
poetry run black app tests
poetry run flake8 app tests
poetry run pytest -q --maxfail=1 --disable-warnings --timeout=20
```

The repository includes a GitHub Actions workflow (`.github/workflows/qa.yml`) that enforces formatting, linting, testing, and Docker image builds on every push.

## Integrations

- **Brevo** – Email alerts for new signups and password resets (`services/brevo.py`).
- **Firebase Admin SDK** – Push notifications with graceful degradation when credentials are absent (`services/push.py`).
- **APScheduler** – Heartbeat scheduling and logging (`services/notifications.py`).
- **Prometheus** – Text-based metrics endpoint for ops tooling (`api/monitoring.py`).

## Testing Notes

The pytest suite provisions an in-memory SQLite database, disables external integrations, and covers:

- Full auth lifecycle (signup → approval → login → refresh → logout)
- Admin CRUD and password reset hooks
- Risk validation boundary enforcement
- Bot start/stop/mode transitions
- WebSocket authentication and streaming
- Regression coverage for investor endpoints with deterministic fixtures

Execute the suite with:

```bash
poetry run pytest -q --maxfail=1 --disable-warnings --timeout=20
```

## Deployment

1. Ensure `.env` contains production secrets and paths.
2. Build the container image (the CI pipeline already validates this):
   ```bash
   docker build -t aidaytrading-backend .
   ```
3. Run the image with environment variables mounted or managed via your orchestrator (Docker Compose, Kubernetes, etc.).
4. Configure HTTPS termination, database migrations, and monitoring according to your infrastructure standards.

## Further Improvements

- Replace the in-memory WebSocket pub/sub with Redis for horizontal scaling.
- Persist scheduled notification history to a dedicated analytics store.
- Integrate fine-grained audit logging for all admin operations.
- Add rate limiting to authentication endpoints using Redis or an API gateway.
