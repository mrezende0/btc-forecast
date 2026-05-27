"""Job diário: macro (yfinance), F&G (alternative.me), CoinDesk incremental.

Sentiment scoring é separado (sentiment_agg.py) pra não rodar FinBERT em todo cron
caso queira separar workflows.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from pipeline import coindesk, macro, sentiment_fg, storage

DATA = Path("data")
MACRO = DATA / "macro_daily.parquet"
FG = DATA / "fg_daily.parquet"
NEWS = DATA / "news_raw.parquet"


def _run_macro() -> None:
    last_existing = storage.read(MACRO)
    start = (
        "2021-01-01"
        if last_existing.is_empty()
        else last_existing["date"].max().isoformat()
    )
    df = macro.fetch_macro(start=start)
    print(f"[macro]   +{storage.upsert(MACRO, df, 'date')} dias")


def _run_fg() -> None:
    df = sentiment_fg.fetch_fg(limit=0)
    print(f"[f&g]     +{storage.upsert(FG, df, 'date')} dias")


def _run_news() -> None:
    df = coindesk.fetch_latest(pages=1)
    if df.is_empty():
        print("[news]    0 novos artigos")
        return
    # garante schema mínimo (mesmo do GDELT pra mesclar depois)
    keep = [
        "id", "url", "title", "body", "published_ts",
        "domain", "language", "sentiment_src", "categories", "source",
    ]
    df = df.select([c for c in keep if c in df.columns])
    n = storage.upsert(NEWS, df, "id")
    print(f"[news]    +{n} artigos CoinDesk")


def run(skip_news: bool = False) -> None:
    _run_macro()
    _run_fg()
    if not skip_news:
        try:
            _run_news()
        except Exception as e:
            print(f"[news]    ⚠️  pulei CoinDesk: {e}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--skip-news", action="store_true")
    run(**vars(p.parse_args()))
