"""Configuração multi-asset. Centraliza símbolos e caminhos por ativo.

Mantém retrocompat: BTC usa os caminhos legados (data/ohlcv_15m.parquet).
Novos ativos usam prefixo (data/{symbol_lower}_ohlcv_15m.parquet).
"""
from __future__ import annotations

from pathlib import Path

DATA = Path("data")


def _paths_for(symbol: str, legacy: bool = False) -> dict:
    """Retorna paths dos parquets desse ativo."""
    if legacy:
        return {
            "ohlcv": DATA / "ohlcv_15m.parquet",
            "funding": DATA / "funding.parquet",
            "perp": DATA / "perp_15m.parquet",
            "oi": DATA / "oi_15m.parquet",
            "long_short": DATA / "long_short_15m.parquet",
            "taker_ratio": DATA / "taker_ratio_15m.parquet",
        }
    base = symbol.replace("USDT", "").lower()
    return {
        "ohlcv": DATA / f"{base}_ohlcv_15m.parquet",
        "funding": DATA / f"{base}_funding.parquet",
        "perp": DATA / f"{base}_perp_15m.parquet",
        "oi": DATA / f"{base}_oi_15m.parquet",
        "long_short": DATA / f"{base}_long_short_15m.parquet",
        "taker_ratio": DATA / f"{base}_taker_ratio_15m.parquet",
    }


ASSETS = {
    "BTC": {
        "symbol": "BTCUSDT",
        "legacy": True,  # mantém os parquets atuais sem renomear
        **_paths_for("BTCUSDT", legacy=True),
    },
    "ETH": {
        "symbol": "ETHUSDT",
        "legacy": False,
        **_paths_for("ETHUSDT", legacy=False),
    },
}


def get(asset: str) -> dict:
    asset = asset.upper()
    if asset not in ASSETS:
        raise ValueError(f"Asset {asset} não configurado. Disponíveis: {list(ASSETS)}")
    return ASSETS[asset]
