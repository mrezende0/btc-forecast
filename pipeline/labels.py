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
