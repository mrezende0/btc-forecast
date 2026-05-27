"""Job de 15min: OHLCV + funding via Binance, upsert Parquet."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from pipeline import binance, storage

DATA = Path("data")
OHLCV = DATA / "ohlcv_15m.parquet"
FUND = DATA / "funding.parquet"

DEFAULT_START_MS = int(datetime(2021, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)


def run(backfill: bool = False) -> None:
    last_ot = storage.last_ts(OHLCV, "open_time")
    start = (
        DEFAULT_START_MS
        if (backfill or last_ot is None)
        else last_ot + binance.INTERVAL_MS
    )
    k = binance.fetch_klines(start)
    print(f"[ohlcv]   +{storage.upsert(OHLCV, k, 'open_time')} velas")

    last_ft = storage.last_ts(FUND, "funding_time")
    start_f = DEFAULT_START_MS if (backfill or last_ft is None) else last_ft + 1
    f = binance.fetch_funding(start_f)
    print(f"[funding] +{storage.upsert(FUND, f, 'funding_time')} pontos")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--backfill", action="store_true")
    run(**vars(p.parse_args()))
