PYTHON=python3
POETRY=poetry
APP_MODULE=app.main:app
UVICORN=uvicorn

install:
	$(POETRY) install --no-root

fmt:
	$(POETRY) run isort app tests
	$(POETRY) run black app tests

lint:
	$(POETRY) run flake8 app tests
	$(POETRY) run mypy app

test:
	$(POETRY) run pytest -q --maxfail=1 --disable-warnings --timeout=20

run:
	$(POETRY) run $(UVICORN) $(APP_MODULE) --reload --host 0.0.0.0 --port 8000

migrate:
	$(POETRY) run alembic upgrade head

revision:
	$(POETRY) run alembic revision --autogenerate -m "Auto"

.PHONY: install fmt lint test run migrate revision
