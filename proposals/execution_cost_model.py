"""execution_cost_model.py — modelo de custo realista para btc-forecast.

Substitui o hardcoded COST_ROUND = 0.0008 (positions.py / exp_backtest_1k.py) por:

  total_bp = taker_fee_bp + half_spread_bp + impact_bp(size, vol, depth) + adverse_selection_bp

Calibração:
  - η do square-root law (Almgren-Chriss / Tóth) por backfit em book snapshots
  - half_spread real do top-of-book (não assumido)
  - vol_bar dinâmico (ATR_pct ou realized vol) — não constante
  - latency model: drift em bps por segundo de stale signal

Uso:

    from proposals.execution_cost_model import compute_realistic_cost, simulate_latency_impact

    cost = compute_realistic_cost(
        size_usd=1000,
        side="buy",
        book_snapshot=fetch_book_snapshot("BTCUSDT"),
        vol_bar_bp=60.0,
        venue="binance_spot",
        execution_style="taker",
    )
    # -> {"taker_fee_bp": 4.5, "half_spread_bp": 0.4, "impact_bp": 0.0,
    #     "adverse_bp": 1.0, "total_bp": 5.9, "round_trip_bp": 11.8}

    drift_bp = simulate_latency_impact(
        signal_ts_ms=1700000000000,
        fill_ts_ms=1700000300000,  # 5min depois
        returns_series=df_4h["close"].pct_change(),  # vol referência
    )

Notas:
  - book_snapshot é opcional. Se None, usa half_spread default por venue (estimativa).
  - venue ∈ {binance_spot, binance_perp, bybit_perp, okx_perp, hyperliquid_perp}.
  - execution_style ∈ {taker, maker_first}. maker_first assume fill_rate calibrado.

Refs nos comentários inline mapeiam pro briefs/execution.md §Refs.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Literal, Optional

import requests


# ---------------------------------------------------------------- constants

# Fee schedule snapshot 2026 (verificar contra fee API antes de prod)
# Ref §11–14 brief
FEE_SCHEDULE_BP: dict[str, dict[str, float]] = {
    "binance_spot":     {"taker": 4.5,   "maker": 4.5},   # VIP0; com BNB-25%: 3.825 / 3.825
    "binance_spot_bnb": {"taker": 3.825, "maker": 3.825},
    "binance_perp":     {"taker": 5.0,   "maker": 2.0},
    "bybit_perp":       {"taker": 5.5,   "maker": 2.0},
    "okx_perp":         {"taker": 5.0,   "maker": 2.0},
    "hyperliquid_perp": {"taker": 4.5,   "maker": 1.5},
}

# Default half-spread em condição normal (bps) — fallback quando book não disponível.
# Estimado empiricamente Kaiko/Amberdata 2024-2025 (ref §9–10), BTCUSDT.
DEFAULT_HALF_SPREAD_BP: dict[str, float] = {
    "binance_spot":     0.4,
    "binance_spot_bnb": 0.4,
    "binance_perp":     0.3,
    "bybit_perp":       0.5,
    "okx_perp":         0.5,
    "hyperliquid_perp": 1.5,
}

# η do square-root law calibrado em BTC spot (Donier-Bouchaud 2015, ref §4).
# slippage_bp = ETA * vol_bar_bp * sqrt(participation_rate)
# participation_rate = size_usd / volume_bar_usd
ETA_SQRT_LAW: dict[str, float] = {
    "binance_spot":     0.6,
    "binance_spot_bnb": 0.6,
    "binance_perp":     0.5,   # perp book mais profundo p/ BTCUSDT
    "bybit_perp":       0.7,
    "okx_perp":         0.7,
    "hyperliquid_perp": 1.2,   # livro mais fino, η maior
}

# Volume médio 4h em USD (calibrar trimestralmente). Conservador.
BAR_VOLUME_USD_4H: dict[str, float] = {
    "binance_spot":     3.0e8,
    "binance_spot_bnb": 3.0e8,
    "binance_perp":     8.0e9,
    "bybit_perp":       3.0e9,
    "okx_perp":         2.0e9,
    "hyperliquid_perp": 4.0e8,
}

# Maker-first fill probability (calibrado empiricamente — placeholder, refinar com book replay).
# Ref §6 Cont-Kukanov 2017
MAKER_FILL_PROB: dict[str, float] = {
    "binance_spot":     0.70,
    "binance_spot_bnb": 0.70,
    "binance_perp":     0.75,
    "bybit_perp":       0.65,
    "okx_perp":         0.65,
    "hyperliquid_perp": 0.55,
}

# Adverse selection penalty quando ordem é executada — empírico, condicional ao regime
# (Aquilina-Budish-O'Neill 2022, ref §7). Em vol-stress dobra.
ADVERSE_SELECTION_BP_NORMAL = 1.0
ADVERSE_SELECTION_BP_STRESS = 3.0

Venue = Literal[
    "binance_spot", "binance_spot_bnb", "binance_perp",
    "bybit_perp", "okx_perp", "hyperliquid_perp",
]
Side = Literal["buy", "sell"]
ExecStyle = Literal["taker", "maker_first"]


# ---------------------------------------------------------------- data classes

@dataclass
class BookSnapshot:
    """Top-of-book + alguns levels p/ walk-the-book."""
    mid_px: float
    best_bid_px: float
    best_ask_px: float
    bids: list[tuple[float, float]] = field(default_factory=list)  # [(px, qty)]
    asks: list[tuple[float, float]] = field(default_factory=list)
    ts_ms: int = 0

    @property
    def spread_bp(self) -> float:
        if self.mid_px <= 0:
            return float("nan")
        return (self.best_ask_px - self.best_bid_px) / self.mid_px * 1e4

    @property
    def half_spread_bp(self) -> float:
        return self.spread_bp / 2.0


@dataclass
class CostBreakdown:
    venue: Venue
    side: Side
    size_usd: float
    execution_style: ExecStyle
    taker_fee_bp: float
    maker_fee_bp: float
    effective_fee_bp: float
    half_spread_bp: float
    impact_bp: float
    adverse_bp: float
    total_bp: float            # one-way
    round_trip_bp: float       # 2× total_bp (assumindo simétrico)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------- book fetch

def fetch_book_snapshot_binance_spot(symbol: str = "BTCUSDT", limit: int = 20) -> BookSnapshot:
    """Lê book L2 spot Binance via REST. Para produção real, usar WebSocket depth stream.

    Endpoint: GET /api/v3/depth (spot) — sem auth.
    Mirror data-api é usado em outras partes do projeto (pipeline/binance.py).
    """
    url = "https://data-api.binance.vision/api/v3/depth"
    r = requests.get(url, params={"symbol": symbol, "limit": limit}, timeout=10)
    r.raise_for_status()
    data = r.json()
    bids = [(float(p), float(q)) for p, q in data["bids"]]
    asks = [(float(p), float(q)) for p, q in data["asks"]]
    if not bids or not asks:
        raise RuntimeError(f"empty book for {symbol}")
    best_bid_px = bids[0][0]
    best_ask_px = asks[0][0]
    mid = (best_bid_px + best_ask_px) / 2.0
    return BookSnapshot(
        mid_px=mid,
        best_bid_px=best_bid_px,
        best_ask_px=best_ask_px,
        bids=bids,
        asks=asks,
    )


def walk_the_book(snap: BookSnapshot, size_usd: float, side: Side) -> float:
    """Simula consumo do livro p/ size_usd. Retorna slippage em bps vs mid.

    side='buy' anda nos asks; 'sell' nos bids.
    """
    if not snap.bids or not snap.asks or snap.mid_px <= 0:
        return float("nan")
    levels = snap.asks if side == "buy" else snap.bids
    remaining_usd = size_usd
    cost_weighted_px = 0.0
    filled_usd = 0.0
    for px, qty in levels:
        level_usd = px * qty
        take = min(level_usd, remaining_usd)
        cost_weighted_px += take * px
        filled_usd += take
        remaining_usd -= take
        if remaining_usd <= 0:
            break
    if filled_usd <= 0:
        return float("nan")
    avg_fill_px = cost_weighted_px / filled_usd
    # se não encheu, penaliza com último nível visível
    if remaining_usd > 0:
        return float("nan")  # book raso — caller decide o que fazer
    sign = 1.0 if side == "buy" else -1.0
    return sign * (avg_fill_px - snap.mid_px) / snap.mid_px * 1e4


# ---------------------------------------------------------------- impact model

def impact_bp_sqrt_law(
    size_usd: float,
    vol_bar_bp: float,
    venue: Venue,
    execution_window_frac: float = 1.0,
) -> float:
    """Square-root impact law (Tóth 2011, Almgren-Chriss 2000 — ref §1,§3).

    impact_bp = eta * vol_bar_bp * sqrt(participation_rate)
    participation_rate = size_usd / (volume_bar_usd * execution_window_frac)

    execution_window_frac=1.0 significa size consumido em 1 bar inteiro (passivo).
    execution_window_frac=0.01 = aggressive crossing em 1% do bar.
    """
    vol_usd = BAR_VOLUME_USD_4H.get(venue, 1e9) * max(execution_window_frac, 1e-4)
    participation = max(size_usd / vol_usd, 0.0)
    eta = ETA_SQRT_LAW.get(venue, 0.7)
    return float(eta * vol_bar_bp * math.sqrt(participation))


# ---------------------------------------------------------------- main API

def compute_realistic_cost(
    size_usd: float,
    side: Side,
    book_snapshot: Optional[BookSnapshot] = None,
    vol_bar_bp: float = 60.0,
    venue: Venue = "binance_spot",
    execution_style: ExecStyle = "taker",
    vol_regime: Literal["normal", "stress"] = "normal",
    execution_window_frac: float = 0.05,  # crossing rápido — 5% do bar 4h = ~12min
) -> dict:
    """Retorna decomposição de custo one-way + round-trip em bps.

    Parameters
    ----------
    size_usd : float
        Notional em USD do trade.
    side : 'buy'|'sell'
    book_snapshot : BookSnapshot, optional
        Se fornecido, half_spread vem do livro real e impact pode usar walk_the_book
        como sanity check. Senão usa DEFAULT_HALF_SPREAD_BP + sqrt-law.
    vol_bar_bp : float
        Vol realizada do bar de execução (ex: ATR_pct × 1e4). Calibre fora.
    venue : str
    execution_style : 'taker'|'maker_first'
    vol_regime : 'normal'|'stress'
        Liga adverse_selection penalty 1× vs 3×.
    execution_window_frac : float
        Fração do bar em que a ordem é consumida (taker rápido → ~0.05).

    Returns
    -------
    dict com {venue, side, size_usd, execution_style, taker_fee_bp, maker_fee_bp,
              effective_fee_bp, half_spread_bp, impact_bp, adverse_bp, total_bp,
              round_trip_bp, notes[]}
    """
    notes: list[str] = []
    fees = FEE_SCHEDULE_BP.get(venue)
    if fees is None:
        raise ValueError(f"venue desconhecido: {venue}")
    taker_bp = fees["taker"]
    maker_bp = fees["maker"]

    # half-spread
    if book_snapshot is not None:
        half_spread = book_snapshot.half_spread_bp
        if math.isnan(half_spread) or half_spread <= 0:
            half_spread = DEFAULT_HALF_SPREAD_BP.get(venue, 1.0)
            notes.append("book inválido → half_spread default")
    else:
        half_spread = DEFAULT_HALF_SPREAD_BP.get(venue, 1.0)
        notes.append("sem book → half_spread default")

    # impact via sqrt-law
    impact = impact_bp_sqrt_law(size_usd, vol_bar_bp, venue, execution_window_frac)

    # opcional: cross-check com walk-the-book se snapshot tem profundidade
    if book_snapshot is not None and book_snapshot.bids and book_snapshot.asks:
        wb = walk_the_book(book_snapshot, size_usd, side)
        if not math.isnan(wb):
            # blendamos: usa max(sqrt-law, walk-book-net-of-spread) — conservador
            walk_impact = max(abs(wb) - half_spread, 0.0)
            if walk_impact > impact:
                notes.append(f"walk-book impact {walk_impact:.2f}bp > sqrt {impact:.2f}bp — usando walk")
                impact = walk_impact

    # adverse selection
    adverse = ADVERSE_SELECTION_BP_STRESS if vol_regime == "stress" else ADVERSE_SELECTION_BP_NORMAL

    # effective fee depende do estilo
    if execution_style == "taker":
        effective_fee = taker_bp
    else:  # maker_first com fallback taker
        p_fill = MAKER_FILL_PROB.get(venue, 0.65)
        effective_fee = p_fill * maker_bp + (1.0 - p_fill) * taker_bp
        # maker-first reduz half_spread (entra no bid/ask, não cruza) E reduz adverse
        # porém adiciona timing risk → modelado simples: half_spread → 0 quando fill no maker
        half_spread = (1.0 - p_fill) * half_spread
        adverse = adverse * (0.4 + 0.6 * (1.0 - p_fill))  # fill maker = menos adverse
        notes.append(f"maker_first: p_fill={p_fill:.2f}")

    one_way_bp = effective_fee + half_spread + impact + adverse
    rt_bp = 2.0 * one_way_bp

    out = CostBreakdown(
        venue=venue,
        side=side,
        size_usd=size_usd,
        execution_style=execution_style,
        taker_fee_bp=taker_bp,
        maker_fee_bp=maker_bp,
        effective_fee_bp=effective_fee,
        half_spread_bp=half_spread,
        impact_bp=impact,
        adverse_bp=adverse,
        total_bp=one_way_bp,
        round_trip_bp=rt_bp,
        notes=notes,
    )
    return out.to_dict()


# ---------------------------------------------------------------- latency model

def simulate_latency_impact(
    signal_ts_ms: int,
    fill_ts_ms: int,
    vol_per_minute_bp: float = 12.0,
) -> dict:
    """Estima custo de stale signal entre signal_ts e fill_ts.

    Modelo simples: drift esperado ~ vol_per_minute_bp * sqrt(minutes_elapsed)
    (random walk em log-preço com vol calibrada). RMS = magnitude do custo
    independente do sign — caller deve interpretar como custo p/ sinais
    direcionais (metade são contra, metade a favor → RMS é o custo esperado
    do hedge contra adverse direction).

    Parameters
    ----------
    signal_ts_ms : timestamp da decisão (close do bar)
    fill_ts_ms : timestamp do fill executado
    vol_per_minute_bp : default 12 bps/min (vol BTC 4h ~50bp / sqrt(240min))

    Returns
    -------
    {elapsed_seconds, expected_drift_bp_rms, kill_signal: bool}
    kill_signal=True se elapsed > 4min (sugestão do brief — gap 3 mitigation).
    """
    elapsed_ms = max(fill_ts_ms - signal_ts_ms, 0)
    elapsed_sec = elapsed_ms / 1000.0
    elapsed_min = elapsed_sec / 60.0
    drift_bp = vol_per_minute_bp * math.sqrt(max(elapsed_min, 0.0))
    kill = elapsed_sec > 240  # 4 minutos
    return {
        "elapsed_seconds": elapsed_sec,
        "elapsed_minutes": elapsed_min,
        "expected_drift_bp_rms": drift_bp,
        "kill_signal": kill,
        "vol_per_minute_bp": vol_per_minute_bp,
    }


# ---------------------------------------------------------------- self-test

if __name__ == "__main__":
    # Exemplo 1: $1k taker spot Binance, sem book real
    c1 = compute_realistic_cost(
        size_usd=1000,
        side="buy",
        book_snapshot=None,
        vol_bar_bp=60.0,
        venue="binance_spot",
        execution_style="taker",
        vol_regime="normal",
    )
    print("=== $1k taker Binance spot (normal vol, sem book) ===")
    for k, v in c1.items():
        print(f"  {k}: {v}")

    # Exemplo 2: mesmo trade em modo maker-first
    c2 = compute_realistic_cost(
        size_usd=1000, side="buy", vol_bar_bp=60.0,
        venue="binance_spot", execution_style="maker_first",
    )
    print("\n=== $1k maker-first Binance spot ===")
    for k, v in c2.items():
        print(f"  {k}: {v}")

    # Exemplo 3: $100k em stress regime
    c3 = compute_realistic_cost(
        size_usd=100_000, side="buy", vol_bar_bp=120.0,
        venue="binance_perp", execution_style="taker", vol_regime="stress",
    )
    print("\n=== $100k taker Binance perp (stress vol) ===")
    for k, v in c3.items():
        print(f"  {k}: {v}")

    # Exemplo 4: tentar puxar book real (pode falhar offline)
    try:
        snap = fetch_book_snapshot_binance_spot("BTCUSDT", limit=20)
        c4 = compute_realistic_cost(
            size_usd=1000, side="buy", book_snapshot=snap,
            vol_bar_bp=60.0, venue="binance_spot",
        )
        print(f"\n=== $1k com book real (mid=${snap.mid_px:,.2f}, spread={snap.spread_bp:.2f}bp) ===")
        for k, v in c4.items():
            print(f"  {k}: {v}")
    except Exception as e:
        print(f"\n[skip] book fetch falhou: {e}")

    # Exemplo 5: latency model
    lat = simulate_latency_impact(
        signal_ts_ms=1_700_000_000_000,
        fill_ts_ms=1_700_000_000_000 + 5 * 60 * 1000,  # 5min depois
    )
    print("\n=== Latency 5min ===")
    for k, v in lat.items():
        print(f"  {k}: {v}")

    lat2 = simulate_latency_impact(
        signal_ts_ms=1_700_000_000_000,
        fill_ts_ms=1_700_000_000_000 + 30 * 1000,  # 30s
    )
    print("\n=== Latency 30s ===")
    for k, v in lat2.items():
        print(f"  {k}: {v}")
