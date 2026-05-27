"""FastAPI predictor service — paper-live stack.

Endpoints:
  GET  /healthz   liveness (process up)
  GET  /readyz    readiness (modelo carregado + redis up)
  GET  /metrics   Prometheus exposition
  POST /predict   força um predict no bar atual (idempotente por bar_open_time)

APScheduler interno dispara /predict no minuto :05 das velas 4h
(00, 04, 08, 12, 16, 20 UTC). Publica sinal em Redis pubsub "signal:new"
pra o executor consumir.

Diferenças vs pipeline/predict_now.py (CLI):
- Não commita parquet diretamente — escreve via Redis + SQLite
- Checa redis.get("killed") antes de publicar (kill-switch)
- Expõe métricas Prometheus (latency, signal counts, errors)
- Não envia Telegram aqui — quem alerta é o executor após fill

TODOs deixados claros:
- [ ] persistir signals em SQLite WAL (pipeline/storage_sqlite.py)
- [ ] integrar dynamic_sizing_risk.compute_position_size (célula Risk)
- [ ] integrar microstructure_features (célula Quant Microestrutura)
- [ ] auth header simples nos endpoints (basic auth ou token estático)
"""
from __future__ import annotations

import json
import logging
import os
import sys
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import redis.asyncio as redis_async
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

sys.path.insert(0, "/app")  # container path do repo

from pipeline import model as mdl  # noqa: E402

logger = logging.getLogger("predictor")
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format='{"ts":"%(asctime)s","lvl":"%(levelname)s","msg":"%(message)s"}',
)

REDIS_URL = os.environ.get("REDIS_URL", "redis://redis:6379/0")
KILLED_KEY = "killed"
LAST_SIGNAL_KEY = "signal:last"
PUBSUB_CHANNEL = "signal:new"
PREDICT_DEDUP_TTL = 60 * 60 * 4 * 2  # 8h, > 1 bar 4h

predict_latency = Histogram("predict_latency_seconds", "predict() wall time")
predict_total = Counter("predict_total", "predict calls", ["outcome"])
signal_emitted = Counter("signal_emitted_total", "signals published to redis", ["signal"])
proba_last = Gauge("predict_proba_long_last", "last proba_long")
last_predict_ts = Gauge("predict_last_unix", "last predict unix ts")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.redis = redis_async.from_url(REDIS_URL, decode_responses=True)
    try:
        await app.state.redis.ping()
        logger.info("redis ok")
    except Exception as e:
        logger.error(f"redis fail: {e}")

    sched = AsyncIOScheduler(timezone="UTC")
    # Dispara aos 5 min das velas 4h (alinhado c/ predict_4h.yml atual)
    sched.add_job(
        _scheduled_predict,
        CronTrigger(hour="0,4,8,12,16,20", minute=5),
        args=[app],
        id="predict_4h",
    )
    sched.start()
    app.state.scheduler = sched
    logger.info("scheduler started")

    yield

    sched.shutdown(wait=False)
    await app.state.redis.aclose()


app = FastAPI(lifespan=lifespan, title="btc-forecast predictor")


async def _scheduled_predict(app: FastAPI) -> None:
    try:
        await _do_predict(app.state.redis, source="cron")
    except Exception as e:
        logger.error(f"scheduled predict failed: {e}")
        predict_total.labels(outcome="error").inc()


async def _do_predict(r, source: str = "manual") -> dict:
    killed = await r.get(KILLED_KEY)
    if killed and killed not in ("0", "false", ""):
        logger.warning(f"killed flag set ({killed}) — skipping predict")
        predict_total.labels(outcome="killed").inc()
        return {"status": "killed", "reason": killed}

    with predict_latency.time():
        pred = mdl.predict_dual_horizon()

    dedup_key = f"predict:done:{pred['open_time']}"
    if await r.set(dedup_key, "1", ex=PREDICT_DEDUP_TTL, nx=True) is None:
        logger.info(f"bar {pred['open_time']} já processado — dedup hit")
        predict_total.labels(outcome="dedup").inc()
        return {"status": "dedup", "open_time": pred["open_time"]}

    payload = {
        "open_time": int(pred["open_time"]),
        "close": float(pred["close"]),
        "proba_long": float(pred["proba_long"]),
        "signal": int(pred["signal"]),
        "confidence_pct": float(pred["confidence_pct"]),
        "source": source,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    await r.set(LAST_SIGNAL_KEY, json.dumps(payload))
    await r.publish(PUBSUB_CHANNEL, json.dumps(payload))

    proba_last.set(payload["proba_long"])
    last_predict_ts.set(datetime.now(timezone.utc).timestamp())
    signal_emitted.labels(signal=str(payload["signal"])).inc()
    predict_total.labels(outcome="ok").inc()
    logger.info(f"signal published: {payload}")
    return {"status": "ok", **payload}


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


@app.get("/readyz")
async def readyz():
    try:
        await app.state.redis.ping()
    except Exception:
        raise HTTPException(503, "redis down")
    return {"status": "ready"}


@app.get("/metrics")
async def metrics():
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict")
async def predict_endpoint():
    result = await _do_predict(app.state.redis, source="manual")
    return JSONResponse(result)


@app.get("/")
async def root():
    return {
        "service": "predictor",
        "endpoints": ["/healthz", "/readyz", "/metrics", "POST /predict"],
        "killed_key": KILLED_KEY,
        "pubsub_channel": PUBSUB_CHANNEL,
    }
