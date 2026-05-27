# Predictor + workers (executor, drift_watchdog, kill_switch) — mesma imagem, comandos diferentes.
# Build context = raiz do repo (../..).

FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# libgomp1 = LightGBM, curl = healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# deps primeiro pra cachear layer
COPY requirements.txt /tmp/requirements.txt
COPY proposals/live_ops_stack/requirements-extra.txt /tmp/requirements-extra.txt
RUN pip install -r /tmp/requirements.txt -r /tmp/requirements-extra.txt

# código (pipeline + novos módulos predictor/executor/scripts)
COPY pipeline /app/pipeline
COPY proposals/live_ops_stack/predictor /app/predictor
COPY proposals/live_ops_stack/executor /app/executor
COPY proposals/live_ops_stack/scripts /app/scripts

# usuário não-root (boa prática mas opcional pra MVP)
RUN useradd -u 1001 -m runner && chown -R runner /app
USER runner

EXPOSE 8080 9101

# default: predictor FastAPI. Outros serviços sobrescrevem em `command:`.
CMD ["uvicorn", "predictor.app:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
