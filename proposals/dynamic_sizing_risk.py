"""Dynamic position sizing — proposal stub.

Combina 5 multiplicadores ortogonais em um `size_pct` ∈ [0, 1]:

    size_pct = clamp(f_vol * f_kelly * f_regime * f_dd * f_conf, 0, SIZE_MAX)

Referências (ver briefs/risk_manager.md):
  - Carver (2015) Systematic Trading — vol targeting, f_vol
  - MacLean-Thorp-Ziemba (2010) — half-Kelly como compromisso growth/vol
  - Grossman-Zhou (1993) — drawdown-conditional surplus → f_dd
  - Sinclair (2020) Positional Option Trading cap 9 — Kelly com p incerto
  - López de Prado (2018) AFML cap 10 — bet sizing pela confiança

Integração:
  pipeline/positions.open_position(...) chama compute_position_size(...) ANTES
  de gravar; se size_pct == 0, NÃO abre. PnL em USD passa a usar size_usd.

Teste self-contained no final usa `data/positions.parquet` + `data/ohlcv_15m.parquet`
para retro-aplicar sizing aos trades fechados (replay) e comparar Sharpe / MaxDD
contra full-notional.

Não importa nada de pipeline.* — fica isolado em proposals/ até validação.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Literal

# ----------------------------------------------------------------- constantes
SIZE_MAX = 1.0                # política: sem alavancagem
TARGET_VOL_ANN = 0.25         # 25%/ano — abaixo do BTC long-run (~60%)
VOL_FLOOR_ANN = 0.10          # piso pra divisão segura
VOL_CAP_ANN = 1.50            # acima disso = black-swan → size 0

# Kelly
KELLY_FRAC = 0.5              # half-Kelly
KELLY_BASELINE = 0.05         # kelly "típico" do projeto (p=0.55, b=1)

# Regime
REGIME_MULT = {"bull": 1.20, "chop": 0.80, "bear": 0.00}
RET_30D_BULL = 0.05
RET_30D_BEAR = -0.05

# Drawdown
DD_FLOOR = -0.20              # -20% → size 0
DD_REACTIVATE = -0.05         # reabre quando equity > 95% peak

# Signal threshold (acoplado a pipeline.model.SIGNAL_THRESHOLD)
SIGNAL_THRESHOLD = 0.35

Regime = Literal["bull", "chop", "bear"]


@dataclass
class SizeBreakdown:
    """Saída auditável: cada componente do sizing."""
    size_pct: float
    size_usd: float
    f_vol: float
    f_kelly: float
    f_regime: float
    f_dd: float
    f_conf: float
    regime: Regime
    realized_vol_ann: float
    equity_dd: float
    notes: str = ""

    def asdict(self) -> dict:
        return asdict(self)


# ----------------------------------------------------------------- componentes
def f_vol_target(realized_vol_ann: float,
                 target: float = TARGET_VOL_ANN,
                 floor: float = VOL_FLOOR_ANN,
                 cap_mult: float = 1.5) -> float:
    """Carver vol-targeting. f_vol = target_vol / realized_vol, clamped."""
    if realized_vol_ann <= 0 or math.isnan(realized_vol_ann):
        return 0.0
    if realized_vol_ann > VOL_CAP_ANN:
        return 0.0  # black-swan filter
    raw = target / max(realized_vol_ann, floor)
    return max(0.1, min(raw, cap_mult))


def f_kelly_norm(p: float, b: float = 1.0,
                 frac: float = KELLY_FRAC,
                 baseline: float = KELLY_BASELINE) -> float:
    """Half-Kelly normalizado contra baseline do projeto.

    Kelly bruto: f* = (p*(1+b) - 1) / b.
    Half-Kelly: 0.5 * f*. Como kelly típico do projeto ≈ 0.05 e isso seria
    irrealisticamente conservador como sizing absoluto, normalizamos contra
    baseline → multiplicador relativo (1.0 = sinal típico).
    """
    if not (0.0 < p < 1.0) or b <= 0:
        return 0.5
    kelly = (p * (1 + b) - 1) / b
    if kelly <= 0:
        return 0.0
    half = frac * kelly
    rel = half / baseline
    return max(0.5, min(rel, 1.5))


def f_regime_classify(ret_30d: float) -> tuple[Regime, float]:
    """Classifica regime via retorno 30d. Retorna (regime, multiplier)."""
    if math.isnan(ret_30d):
        return "chop", REGIME_MULT["chop"]
    if ret_30d > RET_30D_BULL:
        return "bull", REGIME_MULT["bull"]
    if ret_30d < RET_30D_BEAR:
        return "bear", REGIME_MULT["bear"]
    return "chop", REGIME_MULT["chop"]


def f_drawdown(equity: float, peak: float,
               floor: float = DD_FLOOR) -> tuple[float, float]:
    """Grossman-Zhou linear. Retorna (f_dd, dd_atual)."""
    if peak <= 0:
        return 1.0, 0.0
    dd = (equity - peak) / peak  # ≤ 0
    if dd >= 0:
        return 1.0, 0.0
    if dd <= floor:
        return 0.0, dd
    # linear entre 0 (em dd=floor) e 1 (em dd=0)
    f = 1.0 + dd / abs(floor)
    return max(0.0, min(1.0, f)), dd


def f_confidence(signal_prob: float, threshold: float = SIGNAL_THRESHOLD) -> float:
    """López de Prado bet sizing: confiança escalável acima do threshold."""
    if signal_prob <= threshold or signal_prob >= 1.0:
        return 0.5 if signal_prob <= threshold else 1.5
    m = (signal_prob - threshold) / (1.0 - threshold)
    return max(0.5, min(0.5 + m, 1.5))


# ----------------------------------------------------------------- API principal
def compute_position_size(
    signal_prob: float,
    current_regime_ret_30d: float,
    recent_vol_daily: float,
    equity: float,
    peak_equity: float,
    avg_win_loss_ratio: float = 1.0,
) -> SizeBreakdown:
    """Stub principal — chamado em positions.open_position().

    Args:
        signal_prob:           proba do sinal (min(proba_mid, proba_long) sugerido)
        current_regime_ret_30d: BTC return 30d (já existe em pipeline.model)
        recent_vol_daily:      stdev de logret_1 últimas 1d (== rv_1d na feature matrix)
        equity:                capital atual (dólares)
        peak_equity:           maior equity já atingido
        avg_win_loss_ratio:    avg_win / |avg_loss| histórico (≈1.0 no projeto)

    Returns:
        SizeBreakdown com size_pct, size_usd e auditoria dos 5 fatores.
    """
    realized_vol_ann = recent_vol_daily * math.sqrt(365.0)

    fv = f_vol_target(realized_vol_ann)
    fk = f_kelly_norm(signal_prob, b=avg_win_loss_ratio)
    regime, fr = f_regime_classify(current_regime_ret_30d)
    fd, dd = f_drawdown(equity, peak_equity)
    fc = f_confidence(signal_prob)

    raw = fv * fk * fr * fd * fc
    size_pct = max(0.0, min(raw, SIZE_MAX))

    notes = []
    if realized_vol_ann > VOL_CAP_ANN:
        notes.append("black-swan-vol")
    if dd <= DD_FLOOR:
        notes.append("dd-kill")
    if regime == "bear":
        notes.append("bear-zero")
    if fk == 0.0:
        notes.append("kelly-neg")

    return SizeBreakdown(
        size_pct=size_pct,
        size_usd=size_pct * equity,
        f_vol=fv,
        f_kelly=fk,
        f_regime=fr,
        f_dd=fd,
        f_conf=fc,
        regime=regime,
        realized_vol_ann=realized_vol_ann,
        equity_dd=dd,
        notes=",".join(notes),
    )


# ----------------------------------------------------------------- shim integração
def shim_for_positions_open(
    capital: float,
    peak_capital: float,
    entry_price: float,
    proba_mid: float,
    proba_long: float,
    rv_1d: float,
    ret_30d: float,
) -> dict:
    """Drop-in shim para positions.open_position().

    Use o min das duas probas como p efetiva (mais conservador, e é o que
    'confidence_pct' já usa em model.predict_dual_horizon).

    Retorna dict com 'size_pct', 'size_usd', 'size_btc' + auditoria — para gravar
    nos campos novos do schema. Se size_pct == 0, caller deve abortar abertura.
    """
    p = min(proba_mid, proba_long)
    bd = compute_position_size(
        signal_prob=p,
        current_regime_ret_30d=ret_30d,
        recent_vol_daily=rv_1d,
        equity=capital,
        peak_equity=max(capital, peak_capital),
    )
    out = bd.asdict()
    out["size_btc"] = bd.size_usd / entry_price if entry_price > 0 else 0.0
    return out


# ----------------------------------------------------------------- replay/teste
def _replay_backtest():
    """Replay sobre trades fechados do exp_backtest_1k (ou positions.parquet).

    Reaplica sizing dinâmico aos retornos já realizados (`ret_net`) e compara
    Sharpe / MaxDD / Calmar contra full-notional. Não retreina nada.
    """
    import sys
    try:
        import polars as pl
        import numpy as np
        import pandas as pd
    except ImportError:
        print("[replay] polars/pandas indisponíveis — pulando")
        return

    root = Path(__file__).resolve().parent.parent
    eq_path = root / "data" / "backtest_equity.parquet"
    pos_path = root / "data" / "positions.parquet"
    ohlcv_path = root / "data" / "ohlcv_15m.parquet"

    # Caminho 1: usar positions.parquet (trades reais de produção/paper)
    if not pos_path.exists():
        print(f"[replay] {pos_path} não existe — rode pipeline antes")
        return

    pos = pl.read_parquet(pos_path)
    closed = pos.filter(pl.col("status") != "open").to_pandas()
    if closed.empty:
        print("[replay] nenhuma posição fechada — nada a fazer")
        return

    closed = closed.sort_values("entry_time").reset_index(drop=True)
    closed["dt"] = pd.to_datetime(closed["entry_time"], unit="ms", utc=True)

    # Precisa de rv_1d e ret_30d no momento de cada entry — recompõe do OHLCV
    if not ohlcv_path.exists():
        print(f"[replay] {ohlcv_path} ausente — só roda full-notional baseline")
        rv_map = {}
        r30_map = {}
    else:
        oh = pl.read_parquet(ohlcv_path).sort("open_time").to_pandas()
        oh["logret"] = np.log(oh["close"] / oh["close"].shift(1))
        # rv_1d em 15m bars = 96 bars
        oh["rv_1d"] = oh["logret"].rolling(96).std()
        # ret_30d em 15m bars = 2880 bars
        oh["ret_30d"] = oh["close"] / oh["close"].shift(2880) - 1
        rv_map = dict(zip(oh["open_time"].astype("int64"), oh["rv_1d"]))
        r30_map = dict(zip(oh["open_time"].astype("int64"), oh["ret_30d"]))

    INITIAL = 1000.0
    cap_full = INITIAL
    cap_dyn = INITIAL
    peak_dyn = INITIAL
    equity_full = [INITIAL]
    equity_dyn = [INITIAL]

    for _, t in closed.iterrows():
        ret = float(t["pnl_pct"]) if t["pnl_pct"] is not None else 0.0
        # full
        cap_full *= (1 + ret)
        equity_full.append(cap_full)
        # dinâmico
        rv = rv_map.get(int(t["entry_time"]), 0.02)
        r30 = r30_map.get(int(t["entry_time"]), 0.0)
        p = float(t.get("proba_long", 0.55))
        bd = compute_position_size(
            signal_prob=p,
            current_regime_ret_30d=r30 if r30 is not None and not pd.isna(r30) else 0.0,
            recent_vol_daily=rv if rv is not None and not pd.isna(rv) else 0.02,
            equity=cap_dyn,
            peak_equity=peak_dyn,
        )
        cap_dyn *= (1 + bd.size_pct * ret)
        peak_dyn = max(peak_dyn, cap_dyn)
        equity_dyn.append(cap_dyn)

    def metrics(curve: list[float]) -> dict:
        arr = np.asarray(curve, dtype=float)
        rets = np.diff(arr) / arr[:-1]
        if len(rets) < 2 or rets.std(ddof=1) == 0:
            return {"final": arr[-1], "sharpe": 0.0, "maxdd": 0.0}
        # Sharpe trade-by-trade (aprox; idealmente daily)
        sh = float(rets.mean() / rets.std(ddof=1) * math.sqrt(len(rets) / max(1, len(rets) // 52)))
        peak = np.maximum.accumulate(arr)
        dd = (arr - peak) / peak
        return {"final": float(arr[-1]), "sharpe": sh, "maxdd": float(dd.min())}

    mf = metrics(equity_full)
    md = metrics(equity_dyn)
    print("\n" + "=" * 70)
    print(f"REPLAY  ({len(closed)} closed trades)")
    print("=" * 70)
    print(f"{'scheme':>10s}  {'final':>10s}  {'ret %':>8s}  {'sharpe':>7s}  {'maxDD %':>9s}")
    print(f"{'FULL':>10s}  ${mf['final']:>9,.2f}  {100*(mf['final']/INITIAL-1):>+7.2f}  "
          f"{mf['sharpe']:>+6.2f}  {100*mf['maxdd']:>+8.2f}")
    print(f"{'DYNAMIC':>10s}  ${md['final']:>9,.2f}  {100*(md['final']/INITIAL-1):>+7.2f}  "
          f"{md['sharpe']:>+6.2f}  {100*md['maxdd']:>+8.2f}")
    lift_sh = md["sharpe"] - mf["sharpe"]
    print(f"\n  Δ Sharpe: {lift_sh:+.2f}   Δ MaxDD: {100*(md['maxdd']-mf['maxdd']):+.2f}pp")


if __name__ == "__main__":
    # Smoke test: imprime sizing para alguns cenários canônicos
    print("Smoke test — compute_position_size:\n")
    scenarios = [
        ("bull-low-vol-strong",   0.65, 0.10, 0.015, 1000.0, 1000.0),
        ("bull-high-vol",         0.55, 0.10, 0.045, 1000.0, 1000.0),
        ("chop-mid",              0.50, 0.00, 0.022, 1000.0, 1000.0),
        ("bear-block",            0.70, -0.08, 0.020, 1000.0, 1000.0),
        ("post-DD-10pct",         0.55, 0.03, 0.022,  900.0, 1000.0),
        ("near-DD-floor",         0.55, 0.03, 0.022,  820.0, 1000.0),
        ("black-swan-vol",        0.60, 0.10, 0.090, 1000.0, 1000.0),
        ("threshold-borderline",  0.36, 0.02, 0.020, 1000.0, 1000.0),
    ]
    hdr = f"{'cenário':>22s}  {'size %':>7s}  {'fvol':>5s}  {'fkel':>5s}  {'freg':>5s}  {'fdd':>5s}  {'fconf':>6s}  {'reg':>5s}  {'notes':>15s}"
    print(hdr)
    print("-" * len(hdr))
    for name, p, r30, rv, eq, pk in scenarios:
        b = compute_position_size(p, r30, rv, eq, pk)
        print(f"{name:>22s}  {100*b.size_pct:>6.1f}%  "
              f"{b.f_vol:>5.2f}  {b.f_kelly:>5.2f}  {b.f_regime:>5.2f}  "
              f"{b.f_dd:>5.2f}  {b.f_conf:>5.2f}  {b.regime:>5s}  {b.notes:>15s}")
    print()
    _replay_backtest()
