"""Fetcher macro daily via yfinance: DXY, VIX, SPX."""
from __future__ import annotations

from datetime import date

import polars as pl
import yfinance as yf

TICKERS = {
    "dxy": "DX-Y.NYB",
    "vix": "^VIX",
    "spx": "^GSPC",
}


def fetch_macro(start: str = "2021-01-01", end: str | None = None) -> pl.DataFrame:
    end = end or date.today().isoformat()
    frames: list[pl.DataFrame] = []
    for name, ticker in TICKERS.items():
        hist = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
        if hist is None or hist.empty:
            continue
        # Achata MultiIndex de colunas (yfinance >= 0.2.40)
        if hasattr(hist.columns, "get_level_values"):
            hist.columns = hist.columns.get_level_values(0)
        # Index pode ser "Date" ou "Datetime"; reset e renomeia 1ª coluna
        hist = hist.reset_index()
        date_col = hist.columns[0]
        hist = hist[[date_col, "Close"]].rename(columns={date_col: "date", "Close": name})
        frames.append(pl.from_pandas(hist).with_columns(pl.col("date").cast(pl.Date)))

    if not frames:
        return pl.DataFrame()

    out = frames[0]
    for f in frames[1:]:
        out = out.join(f, on="date", how="full", coalesce=True)
    return out.sort("date")
