# Aidaytrading Backend

FastAPI-powered authentication and admin service for the Aidaytrading platform. The legacy trading runtime (`ai_trader/`) ships alongside this backend; both can be deployed independently.

## Quick Start (≈5 minutes)

1. **Clone & enter the backend**
   ```bash
   git clone https://github.com/your-org/aidaytrading.git
   cd aidaytrading/backend
   ```
2. **Copy the environment template**
   ```bash
   cp .env.example .env
   ```
   Update secrets (`JWT_SECRET`, Brevo SMTP credentials) before running in anything other than local mode.
3. **Install dependencies with Poetry**
   ```bash
   make install
   ```
   This installs runtime and developer dependencies (lint, tests). For production images use `poetry install --without dev` instead.
4. **Apply migrations**
   ```bash
   make migrate
   ```
5. **Run the API**
   ```bash
   make run
   ```
   Visit http://localhost:8000/health to confirm the service is healthy, then explore http://localhost:8000/docs.

## Configuration Matrix

| Variable | Description | Default | Required |
| --- | --- | --- | --- |
| `DB_URL` | Async SQLAlchemy URL for the primary database. | `sqlite+aiosqlite:///./backend.db` | ✅ |
| `JWT_SECRET` | Symmetric signing secret for JWTs (>=32 random bytes). | `CHANGE_ME` | ✅ |
| `JWT_ALGORITHM` | JWT signing algorithm. | `HS256` | ✅ |
| `ACCESS_TOKEN_EXPIRES_MIN` | Minutes before access tokens expire. | `15` | ✅ |
| `REFRESH_TOKEN_EXPIRES_DAYS` | Days before refresh tokens expire. | `7` | ✅ |
| `ENV` | Deployment environment (`local`, `dev`, `prod`). | `local` | ✅ |
| `BREVO_API_KEY` | Brevo SMTP API key (used as password). | – | ✅ |
| `BREVO_SMTP_SERVER` | Brevo SMTP host. | `smtp-relay.brevo.com` | ✅ |
| `BREVO_PORT` | Brevo SMTP port. | `587` | ✅ |
| `BREVO_SENDER_EMAIL` | From email address for notifications. | `alerts@example.com` | ✅ |
| `BREVO_SENDER_NAME` | Friendly sender display name. | `Aidaytrading Alerts` | ✅ |
| `ADMIN_NOTIFICATION_EMAIL` | Inbox for new-signup alerts. | `ops@example.com` | ✅ |
| `LOGIN_RATE_LIMIT_ATTEMPTS` | Allowed login attempts per window. | `5` | ✅ |
| `LOGIN_RATE_LIMIT_WINDOW_SECONDS` | Sliding window length in seconds. | `60` | ✅ |
| `LOGIN_RATE_LIMIT_BLOCK_SECONDS` | Cooldown applied after limit exceeded. | `300` | ✅ |
| `GIT_SHA` | Git revision injected at build time. | – | Optional |
| `CORS_ORIGINS` | JSON array of allowed origins (via `.env`). | `http://localhost`, `http://localhost:3000` | Optional |

Secrets are read from the environment; never commit `.env` files or keys. Production deployments should inject configuration via your secrets manager (AWS SSM, Vault, etc.).

## Developer Tooling

| Command | Purpose |
| --- | --- |
| `make install` | Install application and dev dependencies with Poetry. |
| `make fmt` | Run Black/Isort in check mode (used by CI). |
| `make fmt-apply` | Auto-format the codebase. |
| `make lint` | Run Flake8 and MyPy (strict mode). |
| `make test` | Execute pytest with coverage (HTML + XML reports under `htmlcov/`). |
| `make migrate` | Apply database migrations. |
| `make revision MESSAGE="..."` | Generate a new Alembic migration. |
| `poetry run docs/export_openapi.py` | Export `docs/openapi.json` for API consumers. |

## API Catalog

All API examples assume the backend is running on `http://localhost:8000`.

### Auth

#### `POST /api/v1/signup`
```bash
curl -X POST http://localhost:8000/api/v1/signup \
  -H 'Content-Type: application/json' \
  -d '{"username":"alice","email":"alice@example.com","password":"Str0ngPass1"}'
```
Response `201 Created`:
```json
{"message": "Signup received. Await approval.", "status": "pending"}
```

#### `POST /api/v1/login`
```bash
curl -X POST http://localhost:8000/api/v1/login \
  -H 'Content-Type: application/json' \
  -d '{"username":"alice","password":"Str0ngPass1"}'
```
Response `200 OK` includes `access_token` and `refresh_token`. Rate limiting (default 5 attempts/minute) returns `429 rate_limited` with a `Retry-After` header.

#### `POST /api/v1/refresh`
```bash
curl -X POST http://localhost:8000/api/v1/refresh \
  -H 'Content-Type: application/json' \
  -d '{"refresh_token":"<refresh-jwt>"}'
```
Response: `{"access_token": "<new-access-token>"}`.

#### `POST /api/v1/logout`
```bash
curl -X POST http://localhost:8000/api/v1/logout \
  -H "Authorization: Bearer <access-token>" \
  -H 'Content-Type: application/json' \
  -d '{"refresh_token":"<refresh-jwt>"}'
```
Tokens are added to the blacklist immediately.

### Users

#### `GET /api/v1/me`
```bash
curl http://localhost:8000/api/v1/me \
  -H "Authorization: Bearer <access-token>"
```
Returns the authenticated user profile.

### Admin
All routes require an active admin bearer token.

- `GET /api/v1/admin/pending-users`
- `POST /api/v1/admin/approve/{user_id}`
- `POST /api/v1/admin/disable/{user_id}`
- `POST /api/v1/admin/role/{user_id}` with body `{ "role": "admin" | "viewer" }`
- `GET /api/v1/admin/audit-logs?limit=20&offset=0`

Each mutation bumps the target user’s `token_version`, writes an audit record, emits structured logs (`event=admin.action`), and increments observability counters.

### Status

`GET /status` reports business-facing counts:
```bash
curl http://localhost:8000/status
```
Response:
```json
{
  "timestamp": "2024-05-01T12:00:00Z",
  "environment": "local",
  "version": "unknown",
  "users": {"pending": 3, "active": 12, "disabled": 1},
  "admins_active": 2
}
```

### Health

`GET /health` is tuned for load balancers/Kubernetes probes.

```bash
curl http://localhost:8000/health
```
Key fields:
- `uptime_seconds`
- `db_status.state` (`ok`, `degraded`, `down` + `reason`)
- `scheduler_status` (`equity_heartbeat`, `daily_rollup`) with ISO timestamps + lag seconds
- `version`

### Metrics

`GET /metrics` emits Prometheus text format. Exposed metrics include:
- `http_requests_total{path,method,status}`
- `http_request_duration_seconds_bucket|sum|count`
- `auth_logins_total{outcome}`
- `push_events_total{type,outcome}` (email dispatches, Firebase when added)
- `ws_clients_gauge{channel}`

Restrict this endpoint via the provided Nginx sample (`deploy/nginx.conf`).

### Reporting & Trades

Trading analytics live in the legacy runtime (`ai_trader/api_service.py`). When that service runs (e.g. `uvicorn ai_trader.api_service:app --port 9000`), it exposes:

```bash
curl http://localhost:9000/status
curl http://localhost:9000/trades
curl http://localhost:9000/profit?days=30
```

These endpoints provide equity curves, trade history, ML metrics, and risk-state snapshots used by dashboards.

### WebSockets

The backend does not yet publish public WebSocket channels, but middleware tracks connections for future channels via `ws_clients_gauge`. When enabling WebSockets, mount routes under `/ws/*`; the supplied Nginx config already forwards upgrade headers.

### Device Registration

Device/push registration is delegated to upcoming mobile services. Email notifications are currently handled through Brevo (`app/services/brevo_email.py`). Push delivery instrumentation is ready via `push_events_total` and structured logs (`event=push.notification`).

## Operations & Observability

- **Structured logs** – Loguru outputs single-line JSON with `req_id`, `ip`, `user_id`, path, method, status, latency, and sanitized error fields. In containers/systemd, forward stdout/stderr to your log collector (Loki, CloudWatch). For bare-metal installs, use journald or configure `logrotate` on `/var/log/nginx/aidaytrading*.log`.
- **Tail logs** – `journalctl -u aidaytrading-backend -f` or `docker logs -f <container>`.
- **Metrics** – Scrape `/metrics` with Prometheus (per-instance) or ship to Grafana Cloud via `remote_write`. Alert on `auth_logins_total{outcome="failure"}` spikes, `http_requests_total{status="500"}` growth, or stale scheduler ticks.
- **Health checks** – Kubernetes/Load balancers should hit `/health`. For business dashboards, consume `/status`.
- **Scheduler telemetry** – Background jobs call `record_scheduler_tick` (see `app/core/health.py`). Lag exceeding your SLO should trigger alarms.

## Deployment Cookbook

1. **Build & install** – Copy the repository to `/opt/aidaytrading/backend`, set ownership to `www-data`, and create a Python virtualenv via Poetry (`poetry install --without dev`).
2. **Systemd service** – Drop `deploy/aidaytrading-backend.service` into `/etc/systemd/system/`, adjust `WorkingDirectory` and `ExecStart` paths, then run:
   ```bash
   sudo systemctl daemon-reload
   sudo systemctl enable --now aidaytrading-backend
   ```
3. **TLS termination** – Install Nginx using `deploy/nginx.conf` (update `server_name` + upstream). Generate certificates with `deploy/generate_tls.sh api.example.com ops@example.com` and schedule `certbot renew`.
4. **WebSockets** – Nginx stanza handles `Upgrade`/`Connection` headers automatically; expose WS endpoints under `/ws/` in FastAPI.
5. **Rolling restarts** – `sudo systemctl restart aidaytrading-backend` (systemd) or `docker service update --force` (containers). Because the app is stateless, restarts are safe once the DB migration step has succeeded.
6. **Log shipping** – Journald captures JSON logs; forward via `systemd-journal-remote` or agents (Vector/Fluent Bit). Avoid local file rotation for app logs by piping stdout directly to collectors.

## Security Notes

- **Password policy** – `validate_email_format` + `hash_password` enforce strong secrets (>=8 chars, complexity). Rejections are logged via `validation_error` without exposing PII.
- **JWT safety** – Access & refresh tokens embed `token_version`; admin mutations bump it so stale tokens fail. Logout adds JTIs to `token_blacklist`.
- **Rate limiting** – Login attempts use a sliding window limiter (`login_rate_limiter`). Rate-limited responses emit structured logs (`event=auth.login`, `outcome=rate_limited`).
- **Secrets** – Only read from env vars; `.env` is gitignored. Inject secrets through your platform-specific store.
- **CORS & CSP** – Configure `CORS_ORIGINS` for approved frontends. The provided Nginx config applies strict CSP/Referrer policies suitable for API + WebSocket usage.
- **TLS** – Terminate TLS at the reverse proxy (Nginx/Caddy). Enforce HSTS and modern cipher suites as shown.

## Troubleshooting

| Symptom | Likely Cause | Resolution |
| --- | --- | --- |
| `OperationalError: no such table` | Migrations not applied. | Run `make migrate` with the correct `DB_URL`. |
| `401 invalid_token` on every request | `JWT_SECRET` mismatch between issuers and API. | Ensure all API nodes share the same secret; rotate tokens if changed. |
| `429 rate_limited` despite low traffic | Shared IP hitting limit (e.g., load tests). | Increase `LOGIN_RATE_LIMIT_*` env vars or provide unique `X-Forwarded-For`. |
| Browser blocked by CORS | Missing origin in `CORS_ORIGINS`. | Update `.env` to include frontend origin(s). |
| WebSocket upgrade fails | Proxy not forwarding `Upgrade`/`Connection`. | Use provided Nginx/Caddy config; verify upstream port (default 8000). |
| `/health` shows `db_status.down` | Database unreachable. | Check DB credentials/network, run migrations after restore. |

## Further Reading

- `docs/export_openapi.py` – generate OpenAPI documentation for client teams.
- `docs/openapi.json` – published schema (generate as part of release process).
- `docs/adr/0001-architecture.md` – architecture decisions and rationale.
- `SECURITY.md` – vulnerability disclosure process.
- `CONTRIBUTING.md` – coding standards and workflow.

## Legacy Trading Runtime

The historical AI trading engine remains under `ai_trader/`. It exposes its own FastAPI service (`ai_trader/api_service.py`) plus a Streamlit dashboard. Refer to the root-level `README.md` for trader-specific documentation.
