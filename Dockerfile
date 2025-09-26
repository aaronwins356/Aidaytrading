# syntax=docker/dockerfile:1.7

FROM python:3.11-slim AS builder
ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*
COPY requirements-core.txt requirements-ml.txt requirements.txt ./
RUN python -m pip install --upgrade pip
RUN pip wheel --no-deps --wheel-dir /opt/wheels -r requirements-core.txt
RUN pip wheel --no-deps --wheel-dir /opt/wheels -r requirements-ml.txt

FROM python:3.11-slim AS runtime
ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    APP_HOME=/app
WORKDIR $APP_HOME
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && \
    rm -rf /var/lib/apt/lists/*
COPY --from=builder /opt/wheels /opt/wheels
COPY requirements-core.txt requirements-ml.txt requirements.txt ./
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir --no-index --find-links=/opt/wheels -r requirements-core.txt -r requirements-ml.txt
COPY . .
EXPOSE 8000 8501
VOLUME ["/app/ai_trader/data"]
CMD ["python", "-m", "ai_trader.main", "--mode", "live", "--config", "configs/config.yaml"]
