# Contributing Guidelines

Thank you for your interest in contributing to Aidaytrading! This document describes the expected workflow and coding standards.

## Branching & Commits

- Fork the repository or create a feature branch from `main`.
- Use descriptive branch names (`feature/structured-logging`, `fix/rate-limit-bug`).
- Write meaningful commits using the Conventional Commits style when possible (`feat: add status endpoint`, `fix: tighten rate limiter`).
- Keep commits small and focused; rebase onto `main` before opening a PR.

## Development Workflow

1. Install prerequisites (`Python 3.11`, Poetry ≥1.7).
2. Create/update your `.env` using `backend/.env.example`.
3. Install dependencies: `cd backend && make install`.
4. Apply migrations for local testing: `make migrate`.
5. Run the full QA suite before pushing:
   ```bash
   make fmt-apply      # or run make fmt for checks only
   make lint
   make test
   ```
6. Generate the OpenAPI schema if API contracts changed: `poetry run docs/export_openapi.py`.
7. Document noteworthy changes in `backend/README.md`, `CHANGELOG` (if present), and update tests.

## Code Style & Quality

- **Python** – Black (line length 100), Isort (profile `black`), Flake8, and MyPy (strict) are enforced.
- **Type hints** – Required for all new Python code. Prefer precise types over `Any`.
- **Logging** – Use Loguru with structured fields. Do not log secrets or raw PII.
- **Error handling** – Surface errors via FastAPI `HTTPException` with consistent envelopes. Avoid silent failures.
- **Testing** – Write deterministic tests (use `freezegun`, random seeds, isolated DB). Coverage must remain ≥80% for `app/`.
- **Docs** – Update README/ADR when architectural decisions shift. Keep code comments concise but explain non-obvious trading logic.

## Pull Requests

- Open PRs against `main` with a summary, testing evidence, and screenshots when UI changes are involved.
- All GitHub Actions checks must pass (format, lint, migrations, tests under 10 minutes).
- Request review from at least one maintainer. Address feedback promptly and keep the PR scope focused.
- Squash merge or rebase merge after approval.

## Community Expectations

- Follow the [Security Policy](SECURITY.md) for vulnerability disclosures.
- Be respectful and collaborative. Code reviews aim to improve quality, not assign blame.
- When unsure, start a discussion via issues before large refactors.

Happy shipping! 🚀
