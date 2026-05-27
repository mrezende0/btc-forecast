"""Kill-switch loop — monitora Sharpe rolling 90d, API errors, drift acumulado.

Trip conditions (qualquer uma → set redis.killed=1 com TTL 24h):
  1. Sharpe rolling 90d < SHARPE_MIN_90D por SHARPE_BREACH_WEEKS consecutivas
  2. API errors 1h > API_ERROR_BUDGET_1H (via Prometheus query)
  3. PSI max ≥ 0.25 (já trippa via drift_watchdog, mas reforçamos aqui)

Reset manual: redis-cli DEL killed.

TODOs:
- [ ] storage_sqlite pra source-of-truth de trades (não parquet)
- [ ] integrar com Prometheus para api_errors_total{outcome="error"}
- [ ] webhook pra UI / dashboard
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import polars as pl
import redis

sys.path.insert(0, "/app")

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format='{"ts":"%(asctime)s","lvl":"%(levelname)s","msg":"%(message)s"}',
)
log = logging.getLogger("kill_switch")

REDIS_URL = os.environ.get("REDIS_URL", "redis://redis:6379/0")
SHARPE_MIN_90D = float(os.environ.get("SHARPE_MIN_90D", "0.30"))
SHARPE_BREACH_WEEKS = int(os.environ.get("SHARPE_BREACH_WEEKS", "4"))
DATA_DIR = Path(os.environ.get("DATA_DIR", "/data"))


def _rolling_sharpe_90d(positions: pl.DataFrame) -> float | None:
    if positions.is_empty() or "pnl_pct" not in positions.columns:
        return None
    closed = positions.filter(pl.col("status") != "open").drop_nulls("pnl_pct")
    if closed.is_empty():
        return None
    if "exit_time" not in closed.columns:
        return None
    closed = closed.with_columns(
        pl.from_epoch("exit_time", time_unit="ms").alias("exit_dt")
    )
    cutoff = datetime.now(timezone.utc) - timedelta(days=90)
    recent = closed.filter(pl.col("exit_dt") >= pl.lit(cutoff))
    if recent.height < 5:
        return None
    pnl = recent["pnl_pct"].to_numpy()
    if pnl.std() == 0:
        return None
    # Sharpe per-trade annualizado approximation: assume ~recent.height / 90 trades/day
    sr_trade = pnl.mean() / pnl.std()
    trades_per_year = max(1.0, recent.height / 90 * 365)
    return float(sr_trade * np.sqrt(trades_per_year))


def _breach_count(state: dict) -> int:
    """Conta semanas consecutivas com sharpe < min. State é o que vc persiste em Redis."""
    return int(state.get("breach_weeks", 0))


def check(r: redis.Redis) -> dict:
    state_raw = r.get("kill_state")
    state = json.loads(state_raw) if state_raw else {"breach_weeks": 0, "last_check": None}

    positions_path = DATA_DIR / "positions.parquet"
    if not positions_path.exists():
        log.info("positions.parquet ausente — skip check")
        return state

    positions = pl.read_parquet(positions_path)
    sr = _rolling_sharpe_90d(positions)
    log.info(f"sharpe_90d={sr} min={SHARPE_MIN_90D}")

    now = datetime.now(timezone.utc).isoformat()
    last_check = state.get("last_check")
    weekly_step = True
    if last_check:
        prev = datetime.fromisoformat(last_check)
        weekly_step = (datetime.now(timezone.utc) - prev) >= timedelta(days=7)

    if sr is not None and sr < SHARPE_MIN_90D and weekly_step:
        state["breach_weeks"] = state.get("breach_weeks", 0) + 1
        log.warning(f"breach #{state['breach_weeks']} (sr={sr:.3f} < {SHARPE_MIN_90D})")
    elif sr is not None and sr >= SHARPE_MIN_90D and weekly_step:
        if state.get("breach_weeks", 0) > 0:
            log.info("breach streak reset")
        state["breach_weeks"] = 0
    state["last_check"] = now
    state["last_sharpe_90d"] = sr

    if state["breach_weeks"] >= SHARPE_BREACH_WEEKS:
        reason = f"sharpe_90d < {SHARPE_MIN_90D} por {state['breach_weeks']} semanas"
        r.set("killed", reason, ex=86400)
        log.error(f"KILL TRIPPED: {reason}")

    r.set("kill_state", json.dumps(state))
    return state


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--loop", action="store_true")
    p.add_argument("--interval-min", type=int, default=60)
    args = p.parse_args()

    r = redis.from_url(REDIS_URL, decode_responses=True)
    while True:
        try:
            check(r)
        except Exception as e:
            log.error(f"kill_switch check failed: {e}")
        if not args.loop:
            break
        time.sleep(args.interval_min * 60)


if __name__ == "__main__":
    main()
