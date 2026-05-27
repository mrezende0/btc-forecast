"""Gestão de posições — abre/fecha trades baseados em sinal + triple-barrier.

Schema positions.parquet:
  entry_time     : ms timestamp (chave única)
  entry_price    : preço de entrada
  atr            : ATR no momento da entrada (pra recriar barreiras)
  atr_mult       : multiplicador (3.0)
  target_price   : entry + atr_mult × atr
  stop_price     : entry − atr_mult × atr
  timeout_at     : ms timestamp do timeout
  horizon_hours  : 48 default
  proba_long     : confiança do modelo na entrada
  status         : open | closed_target | closed_stop | closed_timeout
  exit_time      : ms ou null
  exit_price     : preço de saída ou null
  pnl_pct        : retorno realizado líquido de custo, ou null se aberta
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import polars as pl

from pipeline import storage

POSITIONS = Path("data") / "positions.parquet"
COST_ROUND = 0.0008  # 0.08% round-trip


def read() -> pl.DataFrame:
    return storage.read(POSITIONS)


def get_open() -> pl.DataFrame:
    df = read()
    if df.is_empty():
        return df
    return df.filter(pl.col("status") == "open")


def has_open() -> bool:
    return not get_open().is_empty()


def open_position(
    entry_time_ms: int,
    entry_price: float,
    atr: float,
    proba_long: float,
    atr_mult: float = 3.0,
    horizon_hours: int = 48,
) -> dict:
    """Cria nova posição e persiste. Retorna o dict da posição."""
    target = entry_price + atr_mult * atr
    stop = entry_price - atr_mult * atr
    timeout_at = entry_time_ms + horizon_hours * 3600 * 1000

    row = {
        "entry_time": entry_time_ms,
        "entry_price": float(entry_price),
        "atr": float(atr),
        "atr_mult": float(atr_mult),
        "target_price": float(target),
        "stop_price": float(stop),
        "timeout_at": int(timeout_at),
        "horizon_hours": int(horizon_hours),
        "proba_long": float(proba_long),
        "status": "open",
        "exit_time": None,
        "exit_price": None,
        "pnl_pct": None,
    }
    df = pl.DataFrame([row])
    storage.upsert(POSITIONS, df, "entry_time")
    return row


def close_position(
    entry_time_ms: int,
    exit_time_ms: int,
    exit_price: float,
    status: str,
) -> dict:
    """Fecha uma posição existente. Atualiza status, exit_*, pnl_pct."""
    df = read()
    if df.is_empty():
        raise ValueError("Nenhuma posição existe")
    mask = df["entry_time"] == entry_time_ms
    if not mask.any():
        raise ValueError(f"Posição {entry_time_ms} não encontrada")

    pos = df.filter(mask).to_dicts()[0]
    pnl_raw = exit_price / pos["entry_price"] - 1
    pnl_net = pnl_raw - COST_ROUND  # round-trip cost

    # Polars não tem update in-place; recompõe
    others = df.filter(~mask)
    updated = {
        **pos,
        "status": status,
        "exit_time": int(exit_time_ms),
        "exit_price": float(exit_price),
        "pnl_pct": float(pnl_net),
    }
    new_df = pl.concat(
        [others, pl.DataFrame([updated])], how="vertical_relaxed"
    ).sort("entry_time")
    POSITIONS.parent.mkdir(parents=True, exist_ok=True)
    new_df.write_parquet(POSITIONS)
    return updated


def evaluate_position(pos: dict, ohlcv: pl.DataFrame, now_ms: int) -> dict | None:
    """Verifica se a posição deve ser fechada usando OHLCV >= entry_time.

    Retorna dict com status novo + exit_time + exit_price, ou None se segue aberta.

    Regras (ordem de prioridade):
      1. timeout: se now >= timeout_at → fecha pelo close mais recente
      2. stop: se algum low <= stop_price desde entrada → fecha em stop
      3. target: se algum high >= target_price desde entrada → fecha em target

    Em conflito (target E stop na mesma vela), aplica regra conservadora:
    stop antes (cenário pior pro position).
    """
    # janela: open_time >= entry_time
    window = ohlcv.filter(pl.col("open_time") > pos["entry_time"]).sort("open_time")
    if window.is_empty():
        if now_ms >= pos["timeout_at"]:
            # sem OHLCV mas timeout — fecha no entry_price (sem PnL real)
            return {
                "status": "closed_timeout",
                "exit_time": pos["timeout_at"],
                "exit_price": pos["entry_price"],
            }
        return None

    # 1. STOP: primeiro low que bate stop
    stop_hits = window.filter(pl.col("low") <= pos["stop_price"])
    # 2. TARGET: primeiro high que bate target
    target_hits = window.filter(pl.col("high") >= pos["target_price"])

    stop_time = int(stop_hits["open_time"][0]) if stop_hits.height > 0 else None
    target_time = int(target_hits["open_time"][0]) if target_hits.height > 0 else None

    # Cenários
    if stop_time is not None and target_time is not None:
        # Mesma vela: conservador = stop
        if stop_time <= target_time:
            return {"status": "closed_stop", "exit_time": stop_time, "exit_price": pos["stop_price"]}
        else:
            return {"status": "closed_target", "exit_time": target_time, "exit_price": pos["target_price"]}
    if stop_time is not None:
        return {"status": "closed_stop", "exit_time": stop_time, "exit_price": pos["stop_price"]}
    if target_time is not None:
        return {"status": "closed_target", "exit_time": target_time, "exit_price": pos["target_price"]}

    # 3. TIMEOUT
    if now_ms >= pos["timeout_at"]:
        last = window.tail(1)
        return {
            "status": "closed_timeout",
            "exit_time": int(last["close_time"][0]),
            "exit_price": float(last["close"][0]),
        }

    return None


def summary() -> dict:
    """Métricas das posições fechadas."""
    df = read()
    if df.is_empty():
        return {"total": 0}
    closed = df.filter(pl.col("status") != "open")
    if closed.is_empty():
        return {"total": df.height, "open": df.height, "closed": 0}
    n_target = closed.filter(pl.col("status") == "closed_target").height
    n_stop = closed.filter(pl.col("status") == "closed_stop").height
    n_timeout = closed.filter(pl.col("status") == "closed_timeout").height
    pnl = closed["pnl_pct"].to_numpy()
    return {
        "total": df.height,
        "open": (df["status"] == "open").sum(),
        "closed": closed.height,
        "n_target": n_target,
        "n_stop": n_stop,
        "n_timeout": n_timeout,
        "win_rate": float((pnl > 0).mean()),
        "total_pnl_pct": float(pnl.sum()),
        "avg_pnl_pct": float(pnl.mean()),
        "best": float(pnl.max()),
        "worst": float(pnl.min()),
    }
