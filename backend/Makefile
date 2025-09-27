PYTHON=python3
POETRY=poetry
APP_MODULE=app.main:app
UVICORN=uvicorn

install:
	$(POETRY) install

fmt:
	$(POETRY) run isort --check-only app
	$(POETRY) run black --check app

fmt-apply:
	$(POETRY) run isort app
	$(POETRY) run black app

lint:
	$(POETRY) run flake8 app
	$(POETRY) run mypy app

test:
	$(POETRY) run pytest

run:
	$(POETRY) run $(UVICORN) $(APP_MODULE) --reload --host 0.0.0.0 --port 8000

migrate:
	$(POETRY) run alembic upgrade head

revision:
	$(POETRY) run alembic revision --autogenerate -m "Auto"

.PHONY: install fmt fmt-apply lint test run migrate revision
