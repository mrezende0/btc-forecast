"""Aplica FinBERT em news_raw.parquet e agrega por dia em sentiment_daily.parquet.

Idempotente: por padrão, pontua só artigos sem score. `--recompute-all` força tudo.

Output (sentiment_daily.parquet):
    date | news_count | sent_pos | sent_neg | sent_neu | net_sentiment

`net_sentiment` = média dos scores no dia (em [-1,+1]).
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

from pipeline import sentiment_score, storage

DATA = Path("data")
NEWS = DATA / "news_raw.parquet"
DAILY = DATA / "sentiment_daily.parquet"
SCORED = DATA / "news_scored.parquet"  # cache de scores por artigo


def _ensure_scores(recompute_all: bool = False) -> pl.DataFrame:
    news = storage.read(NEWS)
    if news.is_empty():
        return pl.DataFrame()

    scored_existing = storage.read(SCORED) if not recompute_all else pl.DataFrame()

    to_score = news
    if not scored_existing.is_empty():
        already = set(scored_existing["id"].to_list())
        to_score = news.filter(~pl.col("id").is_in(list(already)))

    print(f"[score]  {to_score.height} artigos a pontuar (já: {scored_existing.height})")
    if to_score.is_empty():
        return scored_existing

    # Texto: title + body (se houver). Title puro é o robusto.
    titles = to_score["title"].to_list()
    bodies = to_score["body"].to_list() if "body" in to_score.columns else [None] * len(titles)
    texts = [
        (t or "") + (". " + b if b else "")
        for t, b in zip(titles, bodies)
    ]

    scores = sentiment_score.score(texts, batch_size=32)
    new_scored = to_score.select(["id", "published_ts"]).with_columns(
        pl.Series("sentiment", scores)
    )

    merged = (
        pl.concat([scored_existing, new_scored], how="vertical_relaxed")
        if not scored_existing.is_empty()
        else new_scored
    ).unique(subset=["id"], keep="last")
    SCORED.parent.mkdir(parents=True, exist_ok=True)
    merged.write_parquet(SCORED)
    print(f"[score]  total {merged.height} artigos com score salvo")
    return merged


def _aggregate(scored: pl.DataFrame) -> pl.DataFrame:
    if scored.is_empty():
        return pl.DataFrame()

    df = scored.with_columns(
        pl.from_epoch(pl.col("published_ts"), time_unit="s")
        .cast(pl.Date)
        .alias("date")
    )
    agg = (
        df.group_by("date")
        .agg(
            pl.len().alias("news_count"),
            (pl.col("sentiment") > 0.15).sum().alias("sent_pos"),
            (pl.col("sentiment") < -0.15).sum().alias("sent_neg"),
            ((pl.col("sentiment") >= -0.15) & (pl.col("sentiment") <= 0.15))
            .sum()
            .alias("sent_neu"),
            pl.col("sentiment").mean().alias("net_sentiment"),
        )
        .sort("date")
    )
    return agg


def run(recompute_all: bool = False) -> None:
    scored = _ensure_scores(recompute_all=recompute_all)
    if scored.is_empty():
        print("[agg]    sem dados — rodar news_backfill / ingest_daily primeiro")
        return
    agg = _aggregate(scored)
    n = storage.upsert(DAILY, agg, "date")
    print(f"[agg]    +{n} dias em sentiment_daily.parquet  (total {storage.read(DAILY).height})")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--recompute-all", action="store_true")
    run(**vars(p.parse_args()))
