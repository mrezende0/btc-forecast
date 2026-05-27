"""Job de 15min: OHLCV + funding + perp via Binance. Multi-asset via --asset.

Uso:
  python -m pipeline.ingest_15m                  # default BTC
  python -m pipeline.ingest_15m --asset ETH
  python -m pipeline.ingest_15m --asset BTC --backfill
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from pipeline import assets, binance, storage

DEFAULT_START_MS = int(datetime(2021, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)


def run(asset: str = "BTC", backfill: bool = False) -> None:
    cfg = assets.get(asset)
    symbol = cfg["symbol"]
    ohlcv_path = cfg["ohlcv"]
    fund_path = cfg["funding"]
    perp_path = cfg["perp"]

    print(f"[ingest] asset={asset} symbol={symbol}", flush=True)

    last_ot = storage.last_ts(ohlcv_path, "open_time")
    start = (
        DEFAULT_START_MS
        if (backfill or last_ot is None)
        else last_ot + binance.INTERVAL_MS
    )
    k = binance.fetch_klines(start, symbol=symbol)
    print(f"[ohlcv]   +{storage.upsert(ohlcv_path, k, 'open_time')} velas → {ohlcv_path.name}")

    last_ft = storage.last_ts(fund_path, "funding_time")
    start_f = DEFAULT_START_MS if (backfill or last_ft is None) else last_ft + 1
    f = binance.fetch_funding(start_f, symbol=symbol)
    print(f"[funding] +{storage.upsert(fund_path, f, 'funding_time')} pontos → {fund_path.name}")

    last_pt = storage.last_ts(perp_path, "open_time")
    start_p = (
        DEFAULT_START_MS
        if (backfill or last_pt is None)
        else last_pt + binance.INTERVAL_MS
    )
    try:
        p = binance.fetch_perp_klines(start_p, symbol=symbol)
        print(f"[perp]    +{storage.upsert(perp_path, p, 'open_time')} velas → {perp_path.name}")
    except Exception as e:
        print(f"[perp]    ⚠️ falha: {e}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--asset", default="BTC", help="Símbolo: BTC ou ETH (default BTC)")
    p.add_argument("--backfill", action="store_true")
    run(**vars(p.parse_args()))
