# ADR 0001: Backend Architecture Overview

## Status
Accepted – April 2024

## Context
We are building a production-capable REST API for the Aidaytrading trading platform. Requirements include:

- Clear separation between core domain (users, auth, admin) and infrastructure concerns.
- Asynchronous I/O to coexist with trading workloads and scheduled jobs.
- Token-based authentication with revocation controls.
- Operational observability (structured logs, metrics, health checks) without introducing heavy dependencies.

## Decision

1. **Framework** – Use FastAPI running on Uvicorn. FastAPI offers async-first request handling, automatic OpenAPI generation, and tight integration with Pydantic for validation.
2. **Data Layer** – Adopt SQLAlchemy 2.0 async ORM with Alembic migrations. SQLite is used in local/dev environments; production deployments target PostgreSQL. `async_sessionmaker` provides per-request sessions while background tasks reuse the same factory.
3. **Authentication** – JSON Web Tokens (JWT) issued via `python-jose` with:
   - Access + refresh pairs.
   - Token blacklist table storing JTIs for explicit logout.
   - `token_version` column on users to revoke all sessions on role/status change.
4. **Email/Push Delivery** – Pluggable service objects (e.g. Brevo SMTP) encapsulate retries, logging, and metrics so that delivery failures never crash the request path.
5. **Scheduling** – Lightweight cooperative scheduler hooks (equity heartbeat, daily rollup) publish heartbeat timestamps into shared state for `/health` and metrics. Cron/systemd timers or external orchestrators can drive the jobs without coupling to the web workers.
6. **Observability** – Loguru configured for JSON logs, Prometheus client library for metrics, `/health` for infra probes. Middleware enriches every log entry with `req_id`, user context, and latencies; metrics track HTTP, auth, push, and websocket gauges.
7. **Configuration** – Pydantic Settings reads environment variables with `.env` support. Secrets are never committed; `.env.example` documents all required fields.

## Consequences

- Async model requires awareness when integrating blocking libraries; new dependencies should expose async APIs or run in threadpools.
- Stateless JWT issuance keeps API nodes horizontally scalable, but revocation depends on database availability (for blacklist/token version checks).
- Prometheus metrics registry is in-process; for multi-worker deployments, scrape each worker or configure a metrics gateway.
- The documented reverse proxy + systemd units define the “production contract” for ops teams; containerised deployments should replicate TLS headers and health/metrics exposure.
