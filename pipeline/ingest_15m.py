"""Job de 15min: OHLCV + funding + perp via Binance. Multi-asset via --asset.

Uso:
  python -m pipeline.ingest_15m                              # default BTC
  python -m pipeline.ingest_15m --asset ETH
  python -m pipeline.ingest_15m --asset BTC --backfill       # usa DEFAULT_START
  python -m pipeline.ingest_15m --asset BTC --start 2017-08-17
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone

from pipeline import assets, binance, binance_deriv, storage

DEFAULT_START_MS = int(datetime(2021, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)


def _parse_start(s: str | None) -> int | None:
    if not s:
        return None
    dt = datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


DERIV_BACKFILL_MS = 29 * 24 * 60 * 60 * 1000  # 29 dias (30 retorna 400)


def _deriv_start(path, full_backfill: bool, start_override: int | None) -> int:
    """Start time pra deriv: incremental se houver dado, senão últimos 30 dias (limite Binance)."""
    last = storage.last_ts(path, "open_time")
    if last is not None and not full_backfill:
        return last + binance_deriv.PERIOD_MS
    if start_override is not None:
        from time import time as _t
        max_back = int(_t() * 1000) - DERIV_BACKFILL_MS
        return max(start_override, max_back)
    from time import time as _t
    return int(_t() * 1000) - DERIV_BACKFILL_MS


def run(asset: str = "BTC", backfill: bool = False, start: str | None = None,
        skip_deriv: bool = False) -> None:
    cfg = assets.get(asset)
    symbol = cfg["symbol"]
    ohlcv_path = cfg["ohlcv"]
    fund_path = cfg["funding"]
    perp_path = cfg["perp"]
    oi_path = cfg["oi"]
    ls_path = cfg["long_short"]
    taker_path = cfg["taker_ratio"]

    start_override = _parse_start(start)
    full_backfill = backfill or start_override is not None

    print(f"[ingest] asset={asset} symbol={symbol} start={start or 'incremental'}", flush=True)

    last_ot = storage.last_ts(ohlcv_path, "open_time")
    if full_backfill:
        start_o = start_override if start_override is not None else DEFAULT_START_MS
    else:
        start_o = (last_ot + binance.INTERVAL_MS) if last_ot is not None else DEFAULT_START_MS
    k = binance.fetch_klines(start_o, symbol=symbol)
    print(f"[ohlcv]   +{storage.upsert(ohlcv_path, k, 'open_time')} velas → {ohlcv_path.name}")

    last_ft = storage.last_ts(fund_path, "funding_time")
    if full_backfill:
        start_f = start_override if start_override is not None else DEFAULT_START_MS
    else:
        start_f = (last_ft + 1) if last_ft is not None else DEFAULT_START_MS
    f = binance.fetch_funding(start_f, symbol=symbol)
    print(f"[funding] +{storage.upsert(fund_path, f, 'funding_time')} pontos → {fund_path.name}")

    last_pt = storage.last_ts(perp_path, "open_time")
    if full_backfill:
        start_p = start_override if start_override is not None else DEFAULT_START_MS
    else:
        start_p = (last_pt + binance.INTERVAL_MS) if last_pt is not None else DEFAULT_START_MS
    try:
        p = binance.fetch_perp_klines(start_p, symbol=symbol)
        print(f"[perp]    +{storage.upsert(perp_path, p, 'open_time')} velas → {perp_path.name}")
    except Exception as e:
        print(f"[perp]    ⚠️ falha: {e}")

    if skip_deriv:
        return

    # Derivativos — Binance fapi retém só ~30 dias. Cron acumula histórico.
    try:
        s = _deriv_start(oi_path, full_backfill, start_override)
        oi = binance_deriv.fetch_open_interest_hist(s, symbol=symbol)
        print(f"[oi]      +{storage.upsert(oi_path, oi, 'open_time')} pontos → {oi_path.name}")
    except Exception as e:
        print(f"[oi]      ⚠️ falha: {e}")

    try:
        s = _deriv_start(ls_path, full_backfill, start_override)
        ls_top_acc = binance_deriv.fetch_top_long_short_account(s, symbol=symbol)
        ls_top_pos = binance_deriv.fetch_top_long_short_position(s, symbol=symbol)
        ls_global = binance_deriv.fetch_global_long_short_account(s, symbol=symbol)
        merged = ls_top_acc
        for other in (ls_top_pos, ls_global):
            if not other.is_empty():
                merged = merged.join(other, on="open_time", how="full", coalesce=True) if not merged.is_empty() else other
        print(f"[ls]      +{storage.upsert(ls_path, merged, 'open_time')} pontos → {ls_path.name}")
    except Exception as e:
        print(f"[ls]      ⚠️ falha: {e}")

    try:
        s = _deriv_start(taker_path, full_backfill, start_override)
        tk = binance_deriv.fetch_taker_ratio(s, symbol=symbol)
        print(f"[taker]   +{storage.upsert(taker_path, tk, 'open_time')} pontos → {taker_path.name}")
    except Exception as e:
        print(f"[taker]   ⚠️ falha: {e}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--asset", default="BTC", help="Símbolo: BTC ou ETH (default BTC)")
    p.add_argument("--backfill", action="store_true", help="Força fetch desde DEFAULT_START (2021-01-01)")
    p.add_argument("--start", default=None, help="Override start date (YYYY-MM-DD). Implica backfill.")
    p.add_argument("--skip-deriv", action="store_true", help="Pula coleta de OI/long-short/taker.")
    args = p.parse_args()
    run(asset=args.asset, backfill=args.backfill, start=args.start, skip_deriv=args.skip_deriv)
