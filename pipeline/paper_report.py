"""Paper-trade reporting — agrega positions.parquet em estatísticas observáveis.

Compara realidade contra expectativas do backtest:
  Backtest HOLDOUT 2025+:  Sharpe 1.55  Win 54.2%  Avg 0.43%/trade  $1k → $1290

Uso:
  python -m pipeline.paper_report                       # print sumário
  python -m pipeline.paper_report --send-telegram       # envia resumo
  python -m pipeline.paper_report --since 2026-05-27    # janela específica
"""
from __future__ import annotations

import argparse
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl

from pipeline import positions

POSITIONS = Path("data") / "positions.parquet"
SIGNALS = Path("data") / "signals.parquet"

# Expectativas do backtest HOLDOUT 2025+ (Caminho A1-A, validado por Red Team M5)
BACKTEST_SHARPE_HO = 1.55
BACKTEST_WIN_RATE = 0.542
BACKTEST_AVG_PNL = 0.0043
BACKTEST_PSR_0 = 0.952
BACKTEST_MAXDD = -0.12


def _read(path: Path) -> pl.DataFrame:
    return pl.read_parquet(path) if path.exists() else pl.DataFrame()


def _t_stat_test(pnl_arr: np.ndarray) -> tuple[float, float]:
    """t-stat de Sharpe > 0. Retorna (sharpe, p_value_one_sided)."""
    if len(pnl_arr) < 5:
        return 0.0, 1.0
    mu = pnl_arr.mean()
    sd = pnl_arr.std(ddof=1)
    if sd <= 0:
        return 0.0, 1.0
    # Sharpe anualizado assumindo trades independentes — proxy razoável pra teste
    # 4h horizon × 365 dias / 12 bars = ~182 trades/year se sempre tem signal
    # Como cada trade dura ~48h, ~182 trades/year possíveis. Usaremos amostra real.
    sharpe = mu / sd
    t = sharpe * math.sqrt(len(pnl_arr))
    # one-sided p-value via aproximação normal
    p = 0.5 * (1 - math.erf(t / math.sqrt(2)))
    return float(sharpe), float(p)


def build_report(since: datetime | None = None) -> dict:
    pos = _read(POSITIONS)
    sig = _read(SIGNALS)
    if pos.is_empty():
        return {"empty": True, "msg": "Nenhuma posição ainda — aguardando 1º sinal."}

    if since:
        since_ms = int(since.timestamp() * 1000)
        pos = pos.filter(pl.col("entry_time") >= since_ms)

    closed = pos.filter(pl.col("status") != "open")
    open_ = pos.filter(pl.col("status") == "open")

    if closed.is_empty():
        return {
            "empty": False,
            "n_closed": 0,
            "n_open": open_.height,
            "msg": "Nenhuma posição fechada ainda.",
            "open_positions": open_,
        }

    pnl = closed["pnl_pct"].to_numpy()
    n = len(pnl)
    win = (pnl > 0).mean()
    avg = pnl.mean()
    total = pnl.sum()  # somatório simples (não composto)
    # Compounded: prod(1 + pnl) - 1
    compounded = float(np.prod(1 + pnl) - 1)
    best = pnl.max()
    worst = pnl.min()

    # Breakdown por status
    n_target = closed.filter(pl.col("status") == "closed_target").height
    n_stop = closed.filter(pl.col("status") == "closed_stop").height
    n_timeout = closed.filter(pl.col("status") == "closed_timeout").height

    # t-test do edge
    sharpe_per_trade, p_value = _t_stat_test(pnl)
    # Sharpe anualizado proxy: trades de 48h → 182 trades/ano se 100% exposto
    bars_per_year = 365 * 24 / 48  # 182.5
    sharpe_ann = sharpe_per_trade * math.sqrt(min(bars_per_year, n / max(1, (datetime.now(tz=timezone.utc) - datetime.fromtimestamp(closed["entry_time"][0] / 1000, tz=timezone.utc)).days) * 365))

    # Diff vs backtest
    win_diff = win - BACKTEST_WIN_RATE
    avg_diff = avg - BACKTEST_AVG_PNL

    first_entry = datetime.fromtimestamp(int(closed["entry_time"].min()) / 1000, tz=timezone.utc)
    last_exit_field = "exit_time"
    last_exit_ms = int(closed[last_exit_field].max())
    last_exit = datetime.fromtimestamp(last_exit_ms / 1000, tz=timezone.utc)
    days_active = (last_exit - first_entry).days or 1

    return {
        "empty": False,
        "n_closed": n,
        "n_open": open_.height,
        "n_target": n_target,
        "n_stop": n_stop,
        "n_timeout": n_timeout,
        "win_rate": float(win),
        "avg_pnl": float(avg),
        "total_pnl_sum": float(total),
        "compounded": compounded,
        "best": float(best),
        "worst": float(worst),
        "sharpe_per_trade": sharpe_per_trade,
        "sharpe_ann_proxy": float(sharpe_ann) if sharpe_ann else 0.0,
        "p_value_edge_positive": p_value,
        "win_diff_vs_bt": float(win_diff),
        "avg_diff_vs_bt": float(avg_diff),
        "first_entry": first_entry,
        "last_exit": last_exit,
        "days_active": days_active,
        "open_positions": open_,
    }


def format_report(r: dict) -> str:
    if r.get("empty"):
        return f"📊 *Paper Trade Report*\n\n{r.get('msg', 'Sem dados.')}"

    if r["n_closed"] == 0:
        msg = (
            "📊 *Paper Trade Report*\n\n"
            f"⏳ {r['n_open']} posição(ões) aberta(s), nenhuma fechada ainda.\n"
            f"Aguarde target/stop/timeout."
        )
        return msg

    lines = [
        "📊 *Paper Trade Report*",
        "",
        f"📅 Período: {r['first_entry']:%Y-%m-%d} → {r['last_exit']:%Y-%m-%d}  ({r['days_active']}d)",
        f"📈 Trades fechados: *{r['n_closed']}*  |  abertos: {r['n_open']}",
        f"   {r['n_target']} 🟢 target  ·  {r['n_stop']} 🔴 stop  ·  {r['n_timeout']} ⏱️ timeout",
        "",
        "🎯 *Performance acumulada*",
        f"   Win rate:        *{r['win_rate']*100:.1f}%*  (backtest esperava {BACKTEST_WIN_RATE*100:.1f}%)",
        f"   Avg PnL/trade:   *{r['avg_pnl']*100:+.2f}%*  (backtest esperava {BACKTEST_AVG_PNL*100:+.2f}%)",
        f"   Total PnL:       *{r['compounded']*100:+.1f}%* (composto)",
        f"   Melhor / Pior:   {r['best']*100:+.2f}% / {r['worst']*100:+.2f}%",
        "",
        "📐 *Estatística do edge*",
        f"   Sharpe/trade:    {r['sharpe_per_trade']:+.2f}",
        f"   p-value (Sharpe>0): *{r['p_value_edge_positive']:.3f}*",
    ]

    # Veredito estatístico
    p = r["p_value_edge_positive"]
    if r["n_closed"] < 20:
        lines.append(f"   _Ainda poucos trades pra conclusão (precisa ≥20 pra confiança razoável)_")
    elif p < 0.05:
        lines.append(f"   ✅ Edge estatisticamente significativo (p<0.05)")
    elif p < 0.20:
        lines.append(f"   ⚠️ Edge sugestivo mas não confirmado (p<0.20)")
    else:
        lines.append(f"   ❌ Edge não detectado em paper trading ainda")

    # Comparação vs backtest
    win_diff = r["win_diff_vs_bt"] * 100
    avg_diff = r["avg_diff_vs_bt"] * 100
    lines.extend([
        "",
        "🔬 *vs Backtest HOLDOUT (Sharpe 1.55 esperado)*",
        f"   Δ win rate:      {win_diff:+.1f}pp",
        f"   Δ avg PnL:       {avg_diff:+.2f}pp",
    ])
    if abs(win_diff) < 5 and abs(avg_diff) < 0.2:
        lines.append(f"   ✓ Paper alinhado com backtest")
    elif win_diff < -10 or avg_diff < -0.5:
        lines.append(f"   ⚠️ Paper underperforma backtest — possível regime shift ou overfit")

    # Open positions
    if r["n_open"] > 0:
        lines.append("")
        lines.append(f"🔓 *Posição(ões) aberta(s):*")
        for p_row in r["open_positions"].to_dicts():
            entry_dt = datetime.fromtimestamp(p_row["entry_time"] / 1000, tz=timezone.utc)
            age_h = (datetime.now(tz=timezone.utc) - entry_dt).total_seconds() / 3600
            lines.append(
                f"   • entry ${p_row['entry_price']:,.0f}  ·  target ${p_row['target_price']:,.0f}  ·  "
                f"stop ${p_row['stop_price']:,.0f}  ·  {age_h:.0f}h aberta"
            )

    lines.extend([
        "",
        "_Paper trade — sinais reais, sem execução. Use pra validar edge antes de operar com dinheiro._",
    ])
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--since", default=None, help="ISO date pra filtrar trades desde X")
    p.add_argument("--send-telegram", action="store_true", help="Envia o relatório no Telegram")
    args = p.parse_args()

    since = datetime.fromisoformat(args.since).replace(tzinfo=timezone.utc) if args.since else None
    rep = build_report(since=since)
    text = format_report(rep)
    print(text.replace("*", ""))  # plain no terminal

    if args.send_telegram:
        from pipeline import telegram
        telegram.send(text, silent=False)
        print("\n[paper] ✓ enviado no Telegram")


if __name__ == "__main__":
    main()
