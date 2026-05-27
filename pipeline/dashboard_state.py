"""Gera data/dashboard_state.json — snapshot consumido pelo Cowork dashboard.

Usa os Parquets existentes pra computar tudo que cabe num JSON pequeno:
preço atual, retornos, vol, funding, macro, F&G, sentiment, model status.

Rodar localmente ou via GH Actions após ingest_daily.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl

DATA = Path("data")
OUT = DATA / "dashboard_state.json"


def _safe_last(df: pl.DataFrame, col: str):
    if df.is_empty() or col not in df.columns:
        return None
    val = df[col].drop_nulls().tail(1)
    return val.item() if not val.is_empty() else None


def _price_block(ohlcv: pl.DataFrame) -> dict:
    o = ohlcv.sort("open_time")
    last_close = _safe_last(o, "close")
    last_ts = _safe_last(o, "open_time")
    last_dt = datetime.fromtimestamp(last_ts / 1000, tz=timezone.utc) if last_ts else None

    if last_close is None:
        return {"available": False}

    # Retornos em diferentes janelas (em bars de 15m: 96=1d, 96*7=1sem, 96*30=1mês)
    def pct_change(bars_back: int) -> float | None:
        if o.height <= bars_back:
            return None
        prev = o["close"][o.height - 1 - bars_back]
        return (last_close / prev - 1) if prev else None

    return {
        "available": True,
        "last_close": float(last_close),
        "last_dt": last_dt.isoformat() if last_dt else None,
        "ret_24h": pct_change(96),
        "ret_7d": pct_change(96 * 7),
        "ret_30d": pct_change(96 * 30),
        "ret_90d": pct_change(96 * 90),
        "high_30d": float(o["high"].tail(96 * 30).max() or 0),
        "low_30d": float(o["low"].tail(96 * 30).min() or 0),
        "n_bars": o.height,
    }


def _vol_block(ohlcv: pl.DataFrame) -> dict:
    o = ohlcv.sort("open_time").with_columns(
        (pl.col("close").log() - pl.col("close").shift(1).log()).alias("lr")
    )
    lr = o["lr"].drop_nulls()
    if lr.is_empty():
        return {"available": False}

    rv_1d = lr.tail(96).std() * np.sqrt(96 * 365) if len(lr) >= 96 else None
    rv_1w = lr.tail(96 * 7).std() * np.sqrt(96 * 365) if len(lr) >= 96 * 7 else None
    rv_30d = lr.tail(96 * 30).std() * np.sqrt(96 * 365) if len(lr) >= 96 * 30 else None

    return {
        "available": True,
        "rv_1d_ann": float(rv_1d) if rv_1d else None,
        "rv_1w_ann": float(rv_1w) if rv_1w else None,
        "rv_30d_ann": float(rv_30d) if rv_30d else None,
    }


def _funding_block(funding: pl.DataFrame) -> dict:
    if funding.is_empty():
        return {"available": False}
    f = funding.sort("funding_time")
    last = f["funding_rate"][-1]
    last_ts = f["funding_time"][-1]
    last_dt = datetime.fromtimestamp(last_ts / 1000, tz=timezone.utc)
    # 30d rolling stats (90 funding points ~= 30d)
    recent = f["funding_rate"].tail(90).to_numpy()
    mu = float(recent.mean())
    sd = float(recent.std())
    z = (float(last) - mu) / sd if sd > 0 else None
    return {
        "available": True,
        "last": float(last),
        "last_dt": last_dt.isoformat(),
        "mean_30d": mu,
        "z_30d": z,
        "q05_30d": float(np.percentile(recent, 5)),
        "q95_30d": float(np.percentile(recent, 95)),
    }


def _macro_block(macro: pl.DataFrame) -> dict:
    if macro.is_empty():
        return {"available": False}
    m = macro.sort("date").tail(60)
    out = {"available": True, "last_date": str(m["date"][-1])}
    for col in ["dxy", "vix", "spx"]:
        if col in m.columns and not m[col].drop_nulls().is_empty():
            cur = float(m[col][-1] or 0)
            vals = m[col].drop_nulls().to_numpy()
            chg_5d = float(vals[-1] / vals[-6] - 1) if len(vals) >= 6 else None
            mu = float(vals.mean())
            sd = float(vals.std())
            z = (cur - mu) / sd if sd > 0 else None
            out[col] = {"last": cur, "z_30d": z, "chg_5d": chg_5d}
    return out


def _fg_block(fg: pl.DataFrame) -> dict:
    if fg.is_empty():
        return {"available": False}
    fg = fg.sort("date")
    last = int(fg["fg_value"][-1])
    last_class = str(fg["fg_class"][-1])
    last_date = str(fg["date"][-1])
    # 7d / 30d ago
    prev_7d = int(fg["fg_value"][-8]) if fg.height >= 8 else None
    prev_30d = int(fg["fg_value"][-31]) if fg.height >= 31 else None
    return {
        "available": True,
        "last": last,
        "last_class": last_class,
        "last_date": last_date,
        "chg_7d": (last - prev_7d) if prev_7d is not None else None,
        "chg_30d": (last - prev_30d) if prev_30d is not None else None,
    }


def _sentiment_block(sd: pl.DataFrame) -> dict:
    if sd.is_empty():
        return {"available": False, "reason": "GDELT backfill pendente ou sentiment_agg não rodou"}
    sd = sd.sort("date")
    last_date = str(sd["date"][-1])
    last_net = float(sd["net_sentiment"][-1]) if "net_sentiment" in sd.columns else None
    last_count = int(sd["news_count"][-1]) if "news_count" in sd.columns else None
    # 7d média
    tail7 = sd.tail(7)
    avg_net_7d = float(tail7["net_sentiment"].mean()) if "net_sentiment" in sd.columns else None
    avg_count_7d = float(tail7["news_count"].mean()) if "news_count" in sd.columns else None
    return {
        "available": True,
        "last_date": last_date,
        "net_sentiment_today": last_net,
        "net_sentiment_7d_avg": avg_net_7d,
        "news_count_today": last_count,
        "news_count_7d_avg": avg_count_7d,
        "days_covered": sd.height,
        "date_range": [str(sd["date"][0]), str(sd["date"][-1])],
    }


def build_state() -> dict:
    ohlcv = pl.read_parquet(DATA / "ohlcv_15m.parquet") if (DATA / "ohlcv_15m.parquet").exists() else pl.DataFrame()
    funding = pl.read_parquet(DATA / "funding.parquet") if (DATA / "funding.parquet").exists() else pl.DataFrame()
    macro = pl.read_parquet(DATA / "macro_daily.parquet") if (DATA / "macro_daily.parquet").exists() else pl.DataFrame()
    fg = pl.read_parquet(DATA / "fg_daily.parquet") if (DATA / "fg_daily.parquet").exists() else pl.DataFrame()
    sd_path = DATA / "sentiment_daily.parquet"
    sd = pl.read_parquet(sd_path) if sd_path.exists() else pl.DataFrame()

    return {
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "price": _price_block(ohlcv),
        "vol": _vol_block(ohlcv),
        "funding": _funding_block(funding),
        "macro": _macro_block(macro),
        "fg": _fg_block(fg),
        "sentiment_news": _sentiment_block(sd),
        "data_health": {
            "ohlcv_15m_rows": ohlcv.height,
            "funding_rows": funding.height,
            "macro_days": macro.height,
            "fg_days": fg.height,
            "news_days": sd.height,
        },
    }


def run() -> None:
    state = build_state()
    OUT.write_text(json.dumps(state, indent=2, default=str))
    print(f"[dashboard]  state.json gerado em {OUT}  ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    run()
