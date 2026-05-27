"""Triple-Barrier labeling — implementação pura sem mlfinpy.

López de Prado, AFML cap.3. Para cada barra t:
  - Barreira superior:  close[t] * (1 + upper_mult * ATR[t]/close[t])
  - Barreira inferior:  close[t] * (1 - lower_mult * ATR[t]/close[t])
  - Barreira temporal:  t + horizon_bars

Olhamos o caminho [t+1, t+horizon_bars]. Label:
  +1 = long_win  (atinge barreira superior antes)
   0 = timeout   (nenhuma barreira atingida)
  -1 = stop      (atinge barreira inferior antes)

Retorna também:
  - hit_bar: bar index relativo do hit (NaN se timeout)
  - barrier_ret: retorno realizado até o hit ou no fim do horizonte
  - upper_px / lower_px: as barreiras absolutas (debug)
"""
from __future__ import annotations

import numpy as np
import polars as pl


def triple_barrier(
    df: pl.DataFrame,
    upper_mult: float = 1.5,
    lower_mult: float = 1.5,
    horizon_bars: int = 32,
    price_col: str = "close",
    atr_col: str = "atr_14",
    high_col: str = "high",
    low_col: str = "low",
) -> pl.DataFrame:
    """Recebe um DF ordenado por tempo com colunas close/high/low/atr.

    Implementação vetorizada: para cada t, percorre janela [t+1, t+horizon] de
    máximas e mínimas em numpy puro. ~1-2s para 189k velas.
    """
    df = df.sort("open_time")
    n = df.height
    close = df[price_col].to_numpy()
    high = df[high_col].to_numpy()
    low = df[low_col].to_numpy()
    atr = df[atr_col].to_numpy()

    upper_px = close + upper_mult * atr
    lower_px = close - lower_mult * atr

    label = np.zeros(n, dtype=np.int8)
    hit_bar = np.full(n, np.nan)
    barrier_ret = np.full(n, np.nan)

    for t in range(n):
        if np.isnan(atr[t]) or close[t] <= 0:
            label[t] = 0
            continue
        end = min(t + horizon_bars + 1, n)
        # caminho futuro entre t+1 e end
        window_high = high[t + 1 : end]
        window_low = low[t + 1 : end]
        if len(window_high) == 0:
            label[t] = 0
            continue

        up_hits = np.where(window_high >= upper_px[t])[0]
        dn_hits = np.where(window_low <= lower_px[t])[0]
        up_first = up_hits[0] if len(up_hits) else np.inf
        dn_first = dn_hits[0] if len(dn_hits) else np.inf

        if up_first == np.inf and dn_first == np.inf:
            # timeout — usa preço no fim do horizonte
            label[t] = 0
            hit_bar[t] = len(window_high)
            barrier_ret[t] = close[t + len(window_high)] / close[t] - 1
        elif up_first < dn_first:
            label[t] = 1
            hit_bar[t] = up_first + 1
            barrier_ret[t] = upper_px[t] / close[t] - 1
        elif dn_first < up_first:
            label[t] = -1
            hit_bar[t] = dn_first + 1
            barrier_ret[t] = lower_px[t] / close[t] - 1
        else:
            # empate na mesma vela — conservador: tratar como stop (pior caso)
            label[t] = -1
            hit_bar[t] = up_first + 1
            barrier_ret[t] = lower_px[t] / close[t] - 1

    return df.with_columns(
        pl.Series("label", label),
        pl.Series("hit_bar", hit_bar),
        pl.Series("barrier_ret", barrier_ret),
        pl.Series("upper_px", upper_px),
        pl.Series("lower_px", lower_px),
    )


def avg_uniqueness(
    hit_bar: np.ndarray,
    horizon_bars: int,
) -> np.ndarray:
    """López de Prado, AFML eq.4.2 — peso por sobreposição de labels.

    Triple-barrier labels com horizonte longo se sobrepõem: label `i` ainda
    está "vivo" enquanto labels `i+1, i+2, ...` começam. Treinar com peso
    uniforme conta a mesma informação várias vezes → otimismo.

    Para cada label i:
        t_0 = i  (entrada na barra i)
        t_1 = i + hit_bar[i]  se NaN → horizonte completo (i + horizon_bars)
        concorrência(t) = #labels ativos em t
        uniqueness_i = média_t∈[t_0, t_1] de (1/concorrência(t))

    Retorna vetor de pesos ∈ (0, 1]. 1 = label não sobrepõe nenhum outro.
    """
    n = len(hit_bar)
    spans = np.where(np.isnan(hit_bar), horizon_bars, hit_bar).astype(np.int64)
    starts = np.arange(n, dtype=np.int64)
    ends = np.minimum(starts + spans, n - 1)

    diff = np.zeros(n + 1, dtype=np.int64)
    np.add.at(diff, starts, 1)
    end_plus1 = ends + 1
    np.add.at(diff, end_plus1[end_plus1 < n + 1], -1)
    concurrency = np.maximum(np.cumsum(diff)[:n], 1)

    inv_conc = 1.0 / concurrency
    csum = np.concatenate([[0.0], np.cumsum(inv_conc)])
    lengths = (ends - starts + 1).astype(np.float64)
    return (csum[ends + 1] - csum[starts]) / lengths


def attach_uniqueness(df: pl.DataFrame, horizon_bars: int) -> pl.DataFrame:
    """Adiciona coluna `uniqueness_weight` ao DF com saída de triple_barrier."""
    if "hit_bar" not in df.columns:
        raise ValueError("DF precisa ter coluna hit_bar (rodar triple_barrier antes)")
    w = avg_uniqueness(df["hit_bar"].to_numpy(), horizon_bars=horizon_bars)
    return df.with_columns(pl.Series("uniqueness_weight", w))
