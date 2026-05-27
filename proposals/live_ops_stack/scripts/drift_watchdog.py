"""Drift watchdog — PSI por feature, exporta gauges Prometheus, alerta Telegram, seta kill flag.

Stub funcional. Roda em loop (--loop) ou one-shot.

Uso:
    python -m scripts.drift_watchdog --loop --interval-min 60

Fluxo:
    1. Carrega features atuais: últimos 30d a partir de /data/features.duckdb
       (ou fallback: rebuild via pipeline.features.build_v2_from_parquets).
    2. Carrega reference: /data/reference_features.parquet (snapshot do treino confiável).
    3. Pra cada feature numérica em FEATURES_TO_WATCH, calcula PSI (10 buckets equal-frequency
       na reference, evita zero-prob com smoothing eps).
    4. Exporta:
         drift_psi{feature="funding_z90"} 0.07
         drift_psi_max 0.18
         drift_ks_pvalue{feature="..."} (opcional, KS two-sample)
         drift_check_timestamp
         drift_check_errors_total
    5. Decisão:
         max(psi) < PSI_WARN              → log INFO, nada
         PSI_WARN <= max < PSI_TRIP       → log WARN, Telegram normal
         max >= PSI_TRIP                  → log ERROR, Telegram P0, redis.set("killed", "1", ex=86400)

NÃO depende do FastAPI predictor; é cron worker autônomo. Métricas via HTTPServer próprio.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict

import numpy as np
import polars as pl
from prometheus_client import Gauge, Counter, start_http_server

# Smoothing pra evitar log(0) no PSI
PSI_EPS = 1e-6
N_BUCKETS = 10

# Features-chave (cross-domain). Ajustar conforme feature_importance estável.
FEATURES_TO_WATCH = [
    "funding_z90",
    "funding_ema8d",
    "rv_1d",
    "rv_1w",
    "atr_14",
    "rsi_14",
    "bb_pos",
    "dist_ma_30d",
    "dist_ma_90d",
    "dxy_z30",
    "vix_z30",
    "fg",
    "fg_z30",
    "drawdown_30d",
    "vol_z30d",
]

DATA_DIR = Path(os.environ.get("DATA_DIR", "/data"))
REFERENCE_PATH = Path(os.environ.get("REFERENCE_PATH", str(DATA_DIR / "reference_features.parquet")))
PSI_WARN = float(os.environ.get("PSI_WARN", "0.10"))
PSI_TRIP = float(os.environ.get("PSI_TRIP", "0.25"))
METRICS_PORT = int(os.environ.get("METRICS_PORT", "9101"))

# --- Prometheus metrics ---
g_psi = Gauge("drift_psi", "PSI por feature vs reference", ["feature"])
g_psi_max = Gauge("drift_psi_max", "PSI máximo entre features monitoradas")
g_ts = Gauge("drift_check_timestamp", "Unix ts do último check completo")
c_errors = Counter("drift_check_errors_total", "Erros no check (exceções, dados ausentes)")
c_trips = Counter("drift_psi_trip_total", "Trips de PSI >= PSI_TRIP")

log = logging.getLogger("drift_watchdog")


def compute_psi(reference: np.ndarray, current: np.ndarray, n_buckets: int = N_BUCKETS) -> float:
    """PSI = Σ (cur_i - ref_i) * ln(cur_i / ref_i). Buckets via quantis da reference."""
    ref = reference[~np.isnan(reference)]
    cur = current[~np.isnan(current)]
    if len(ref) < 100 or len(cur) < 20:
        raise ValueError(f"amostras insuficientes ref={len(ref)} cur={len(cur)}")

    # Quantis da reference (equal-frequency)
    qs = np.linspace(0, 1, n_buckets + 1)
    edges = np.unique(np.quantile(ref, qs))
    if len(edges) < 3:
        # variável quase constante — PSI por contagem direta
        edges = np.array([ref.min() - 1e-9, ref.mean(), ref.max() + 1e-9])

    ref_hist, _ = np.histogram(ref, bins=edges)
    cur_hist, _ = np.histogram(cur, bins=edges)

    ref_pct = ref_hist / max(ref_hist.sum(), 1)
    cur_pct = cur_hist / max(cur_hist.sum(), 1)

    # Smoothing
    ref_pct = np.clip(ref_pct, PSI_EPS, None)
    cur_pct = np.clip(cur_pct, PSI_EPS, None)

    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def _telegram_alert(text: str, silent: bool = False) -> None:
    """Envio direto (não importa pipeline.telegram pra manter watchdog independente)."""
    import requests
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    chat = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
    if not token or not chat:
        log.warning("telegram não configurado — skip alert")
        return
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat, "text": text, "parse_mode": "Markdown",
                  "disable_notification": silent},
            timeout=10,
        )
        if r.status_code != 200:
            log.error("telegram falhou %s: %s", r.status_code, r.text[:200])
    except Exception as exc:  # noqa: BLE001
        log.exception("telegram exception: %s", exc)


def _set_kill_flag(reason: str) -> None:
    """Seta redis killed=1 com TTL 24h. Predictor checa antes de gerar sinal."""
    try:
        import redis
        r = redis.from_url(os.environ.get("REDIS_URL", "redis://redis:6379/0"))
        r.set("killed", "1", ex=86400)
        r.set("killed_reason", reason, ex=86400)
        r.set("killed_at", str(int(time.time())), ex=86400)
        log.error("kill flag SET: %s", reason)
    except Exception as exc:  # noqa: BLE001
        log.exception("redis set falhou: %s", exc)


def _load_current_features() -> pl.DataFrame:
    """Carrega features atuais. Preferência: features.duckdb;
    fallback: rebuild via pipeline.features.build_v2_from_parquets (mesma lógica do treino)."""
    duck_path = DATA_DIR / "features.duckdb"
    if duck_path.exists():
        import duckdb
        con = duckdb.connect(str(duck_path), read_only=True)
        df = con.execute(
            "SELECT * FROM features WHERE open_time > (extract(epoch from now()) - 30*86400)*1000"
        ).pl()
        con.close()
        return df
    # Fallback: usar pipeline existente
    from pipeline import features as feat
    full = feat.build_v2_from_parquets(timeframe_min=240, lag=1)
    # últimos 30d em bars 4h = 180 bars
    return full.tail(180)


def _load_reference() -> pl.DataFrame:
    if not REFERENCE_PATH.exists():
        raise FileNotFoundError(
            f"reference em {REFERENCE_PATH} ausente. "
            "Gerar com: python -m scripts.snapshot_reference (treino confiável)."
        )
    return pl.read_parquet(REFERENCE_PATH)


def run_once() -> Dict[str, float]:
    g_ts.set(time.time())
    try:
        current = _load_current_features()
        reference = _load_reference()
    except Exception as exc:  # noqa: BLE001
        c_errors.inc()
        log.exception("load falhou: %s", exc)
        return {}

    psis: Dict[str, float] = {}
    for col in FEATURES_TO_WATCH:
        if col not in current.columns or col not in reference.columns:
            log.warning("feature %s ausente em current ou reference — skip", col)
            continue
        try:
            psi = compute_psi(
                reference[col].drop_nulls().to_numpy(),
                current[col].drop_nulls().to_numpy(),
            )
        except Exception as exc:  # noqa: BLE001
            c_errors.inc()
            log.warning("psi(%s) falhou: %s", col, exc)
            continue
        psis[col] = psi
        g_psi.labels(feature=col).set(psi)

    if not psis:
        log.error("nenhum PSI computado")
        return {}

    max_psi = max(psis.values())
    max_feat = max(psis, key=psis.get)
    g_psi_max.set(max_psi)

    summary = ", ".join(f"{k}={v:.3f}" for k, v in sorted(psis.items(), key=lambda x: -x[1])[:5])
    log.info("PSI max=%.3f (%s). top5: %s", max_psi, max_feat, summary)

    if max_psi >= PSI_TRIP:
        c_trips.inc()
        _set_kill_flag(f"PSI_TRIP feature={max_feat} psi={max_psi:.3f}")
        _telegram_alert(
            f"🚨 *DRIFT TRIP* — kill flag ON\n"
            f"feature `{max_feat}` PSI=*{max_psi:.3f}* (limiar {PSI_TRIP})\n"
            f"Top: {summary}\n"
            f"Predictor congelado por 24h. Investigar e rodar `unkill` manual."
        )
    elif max_psi >= PSI_WARN:
        _telegram_alert(
            f"⚠️ Drift WARN — feature `{max_feat}` PSI={max_psi:.3f} "
            f"(warn {PSI_WARN}, trip {PSI_TRIP})\nTop: {summary}",
            silent=True,
        )

    return psis


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--loop", action="store_true", help="rodar continuamente")
    parser.add_argument("--interval-min", type=int, default=60)
    parser.add_argument("--metrics-port", type=int, default=METRICS_PORT)
    args = parser.parse_args()

    # JSON logs simples (formato compatível com promtail.yml)
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format='{"timestamp":"%(asctime)s","level":"%(levelname)s","service":"drift_watchdog","event":"%(message)s"}',
    )

    start_http_server(args.metrics_port)
    log.info("drift_watchdog up, metrics on :%d", args.metrics_port)

    if not args.loop:
        run_once()
        return 0

    while True:
        try:
            run_once()
        except Exception as exc:  # noqa: BLE001
            c_errors.inc()
            log.exception("run_once exception: %s", exc)
        time.sleep(args.interval_min * 60)


if __name__ == "__main__":
    sys.exit(main())
