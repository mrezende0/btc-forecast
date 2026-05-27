"""Fetcher Fear & Greed Index via alternative.me (gratuito, sem auth)."""
from __future__ import annotations

from datetime import datetime, timezone

import polars as pl
import requests

URL = "https://api.alternative.me/fng/"


def fetch_fg(limit: int = 0) -> pl.DataFrame:
    """limit=0 puxa o histórico completo."""
    r = requests.get(URL, params={"limit": limit, "format": "json"}, timeout=20)
    r.raise_for_status()
    data = r.json().get("data", [])
    if not data:
        return pl.DataFrame()

    rows = [
        {
            "date": datetime.fromtimestamp(int(d["timestamp"]), tz=timezone.utc).date(),
            "fg_value": int(d["value"]),
            "fg_class": d["value_classification"],
        }
        for d in data
    ]
    return pl.DataFrame(rows).sort("date")
