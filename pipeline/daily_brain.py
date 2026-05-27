"""Daily Brain — relatório diário automático do sistema.

Lê snapshots já gerados (dashboard_state.json, signals.parquet, backtest_equity.parquet)
e produz markdown commitado em reports/daily_YYYY-MM-DD.md.

Não faz chamada de LLM por padrão — gera tudo determinístico em Python.
Se `OPENAI_API_KEY` ou `ANTHROPIC_API_KEY` estiverem no env, anexa síntese LLM
opcional no final (ver TODO).

Roda local (`python -m pipeline.daily_brain`) ou via GH Actions
(workflow .github/workflows/daily_brain.yml).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from textwrap import dedent

import numpy as np
import polars as pl

DATA = Path("data")
REPORTS = Path("reports")
DASHBOARD_STATE = DATA / "dashboard_state.json"
SIGNALS = DATA / "signals.parquet"
EQUITY = DATA / "backtest_equity.parquet"


@dataclass
class Section:
    title: str
    body: str
    flag: str = ""  # "ok" | "warn" | "alert"


def _load_state() -> dict | None:
    if not DASHBOARD_STATE.exists():
        return None
    return json.loads(DASHBOARD_STATE.read_text())


def _section_market(state: dict | None) -> Section:
    if not state or not state.get("price", {}).get("available"):
        return Section("Mercado", "_dashboard_state.json indisponível_", "warn")

    p = state["price"]
    v = state.get("vol", {})
    f = state.get("funding", {})
    fg = state.get("fg", {})

    def pct(x):
        return f"{x*100:+.2f}%" if x is not None else "n/d"

    flag = ""
    flags = []
    if f.get("z_30d") is not None and abs(f["z_30d"]) > 2:
        flags.append(f"funding z={f['z_30d']:+.1f}σ (extremo)")
        flag = "alert"
    if fg.get("last") is not None:
        if fg["last"] < 25:
            flags.append(f"F&G {fg['last']} ({fg.get('last_class')})")
            flag = flag or "warn"
        elif fg["last"] > 80:
            flags.append(f"F&G {fg['last']} ({fg.get('last_class')})")
            flag = flag or "warn"
    if v.get("rv_1d_ann") and v.get("rv_30d_ann"):
        ratio = v["rv_1d_ann"] / v["rv_30d_ann"]
        if ratio > 1.5:
            flags.append(f"vol 1d/30d = {ratio:.1f}× (expandindo)")
            flag = flag or "warn"

    flags_md = "\n".join(f"- ⚠️ {x}" for x in flags) if flags else ""
    body = dedent(f"""
        - Preço: **${p['last_close']:,.0f}** | 24h {pct(p.get('ret_24h'))} | 7d {pct(p.get('ret_7d'))} | 30d {pct(p.get('ret_30d'))}
        - Range 30d: ${p.get('low_30d', 0):,.0f} — ${p.get('high_30d', 0):,.0f}
        - Vol ann (1d / 1w / 30d): {(v.get('rv_1d_ann') or 0)*100:.0f}% / {(v.get('rv_1w_ann') or 0)*100:.0f}% / {(v.get('rv_30d_ann') or 0)*100:.0f}%
        - Funding: {f.get('last', 0)*1e4:+.1f}bp | z30d {f.get('z_30d', 0):+.2f}σ
        - F&G: **{fg.get('last', 'n/d')}** ({fg.get('last_class', 'n/d')}) | Δ7d {fg.get('chg_7d', 0):+d}
        {flags_md}
    """).strip()
    return Section("Mercado", body, flag)


def _section_signals() -> Section:
    if not SIGNALS.exists():
        return Section("Sinais", "_signals.parquet não existe ainda_", "warn")

    s = pl.read_parquet(SIGNALS)
    if s.is_empty():
        return Section("Sinais", "_signals.parquet vazio_", "warn")

    if "timestamp" in s.columns:
        s = s.with_columns(pl.col("timestamp").cast(pl.Datetime, strict=False))
    elif "ts" in s.columns:
        s = s.with_columns(pl.col("ts").alias("timestamp").cast(pl.Datetime, strict=False))

    cutoff = datetime.now(timezone.utc) - timedelta(days=7)
    if "timestamp" in s.columns:
        recent = s.filter(pl.col("timestamp") >= pl.lit(cutoff))
    else:
        recent = s.tail(50)

    body_lines = [
        f"- Sinais 7d: **{recent.height}** | total registrado: {s.height}",
    ]

    if "prob" in recent.columns:
        body_lines.append(f"- Prob média 7d: {recent['prob'].mean():.3f}")
    if "signal" in recent.columns:
        sig_counts = recent.group_by("signal").len().sort("len", descending=True)
        for row in sig_counts.iter_rows(named=True):
            body_lines.append(f"- {row['signal']}: {row['len']}")

    last_row = s.tail(1).to_dicts()[0] if s.height else {}
    if last_row:
        body_lines.append(f"- Último sinal: `{last_row}`")

    flag = "warn" if recent.height == 0 else ""
    return Section("Sinais", "\n".join(body_lines), flag)


def _section_equity() -> Section:
    if not EQUITY.exists():
        return Section("Equity / PnL", "_backtest_equity.parquet não existe (backtest não rodado)_", "warn")

    e = pl.read_parquet(EQUITY)
    if e.is_empty():
        return Section("Equity / PnL", "_equity vazio_", "warn")

    col_equity = None
    for c in ["equity", "equity_curve", "pnl_cumulative", "value", "capital"]:
        if c in e.columns:
            col_equity = c
            break
    if not col_equity:
        return Section("Equity / PnL", f"_colunas: {e.columns}_", "warn")

    eq = e[col_equity].to_numpy()
    if len(eq) < 2:
        return Section("Equity / PnL", "_série muito curta_", "warn")

    rets = np.diff(np.log(np.clip(eq, 1e-9, None)))
    sharpe = float(np.mean(rets) / np.std(rets) * np.sqrt(96 * 365)) if np.std(rets) > 0 else 0
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak
    max_dd = float(dd.min())
    cur_dd = float(dd[-1])
    cagr = float((eq[-1] / eq[0]) ** (365 / max(1, len(eq) / 96)) - 1) if eq[0] > 0 else 0

    rolling_n = min(96 * 30, len(rets))
    if rolling_n > 10:
        recent_rets = rets[-rolling_n:]
        sharpe_30d = float(np.mean(recent_rets) / np.std(recent_rets) * np.sqrt(96 * 365)) if np.std(recent_rets) > 0 else 0
    else:
        sharpe_30d = None

    flag = ""
    if sharpe_30d is not None and sharpe_30d < 0.3:
        flag = "alert"
    elif cur_dd < -0.2:
        flag = "warn"

    body = dedent(f"""
        - Sharpe full: **{sharpe:.2f}** | Sharpe 30d: {sharpe_30d:.2f} (gate morte: < 0.3 por 4sem)
        - MaxDD: {max_dd*100:.1f}% | DD atual: {cur_dd*100:.1f}%
        - CAGR estimado: {cagr*100:+.1f}%
        - Equity: {eq[0]:.2f} → {eq[-1]:.2f} ({(eq[-1]/eq[0]-1)*100:+.1f}%)
    """).strip()
    return Section("Equity / PnL", body, flag)


def _section_data_health(state: dict | None) -> Section:
    if not state:
        return Section("Saúde dos dados", "_dashboard_state ausente_", "warn")
    h = state.get("data_health", {})
    age_hours = None
    if state.get("generated_at"):
        try:
            ts = datetime.fromisoformat(state["generated_at"].replace("Z", "+00:00"))
            age_hours = (datetime.now(timezone.utc) - ts).total_seconds() / 3600
        except Exception:
            pass

    lines = []
    if age_hours is not None:
        flag_age = "alert" if age_hours > 30 else ("warn" if age_hours > 12 else "")
        lines.append(f"- Snapshot idade: **{age_hours:.1f}h** {'⚠️ stale' if flag_age else ''}")
    lines.append(f"- OHLCV 15m: {h.get('ohlcv_15m_rows', 0):,} bars")
    lines.append(f"- Funding: {h.get('funding_rows', 0):,} pts")
    lines.append(f"- Macro: {h.get('macro_days', 0)} dias")
    lines.append(f"- F&G: {h.get('fg_days', 0)} dias")
    lines.append(f"- News: {h.get('news_days', 0)} dias")

    flag = "alert" if age_hours and age_hours > 30 else ("warn" if age_hours and age_hours > 12 else "")
    return Section("Saúde dos dados", "\n".join(lines), flag)


def _section_anomalies(state: dict | None) -> Section:
    """Heurísticas baratas que sinalizam algo a investigar."""
    if not state:
        return Section("Anomalias", "_sem dados_", "")

    alerts = []
    f = state.get("funding", {})
    fg = state.get("fg", {})
    v = state.get("vol", {})
    macro = state.get("macro", {})

    if f.get("z_30d") is not None and abs(f["z_30d"]) > 1.5:
        alerts.append(f"Funding z={f['z_30d']:+.2f}σ — posicionamento {'esticado' if abs(f['z_30d']) < 2 else 'extremo'}")
    if fg.get("chg_7d") and abs(fg["chg_7d"]) > 25:
        alerts.append(f"F&G mexeu {fg['chg_7d']:+d} em 7d — shift de regime")
    if v.get("rv_1d_ann") and v.get("rv_30d_ann"):
        if v["rv_1d_ann"] / v["rv_30d_ann"] > 1.5:
            alerts.append(f"Vol 1d/30d = {v['rv_1d_ann']/v['rv_30d_ann']:.2f}× — expansão")
        elif v["rv_1d_ann"] / v["rv_30d_ann"] < 0.6:
            alerts.append(f"Vol 1d/30d = {v['rv_1d_ann']/v['rv_30d_ann']:.2f}× — compressão (breakout pendente?)")
    if isinstance(macro.get("vix"), dict) and macro["vix"].get("z_30d", 0) > 1.5:
        alerts.append(f"VIX z={macro['vix']['z_30d']:+.2f}σ — risk-off macro")

    if not alerts:
        return Section("Anomalias", "_nenhuma flag heurística disparou_", "")

    body = "\n".join(f"- 🚨 {a}" for a in alerts)
    return Section("Anomalias", body, "alert")


def _section_actions(sections: list[Section]) -> Section:
    """Síntese determinística do que olhar hoje, baseada nos flags das outras seções."""
    alerts = [s for s in sections if s.flag == "alert"]
    warns = [s for s in sections if s.flag == "warn"]
    actions = []
    if any("Anomalias" in s.title for s in alerts):
        actions.append("Revisar anomalias antes de qualquer trade — não automatizar nessa janela.")
    if any("Equity" in s.title for s in alerts):
        actions.append("Equity em zona crítica — verificar critério de morte (Sharpe 30d < 0.3).")
    if any("Saúde" in s.title for s in alerts):
        actions.append("Pipeline de dados stale — checar GH Actions ingest_daily / ingest_15m.")
    if any("Sinais" in s.title for s in warns):
        actions.append("Volume baixo de sinais — investigar threshold ou regime filter.")
    if not actions:
        actions.append("Nenhuma flag — sistema operando dentro do esperado.")
    body = "\n".join(f"- {a}" for a in actions)
    return Section("O que olhar hoje", body, "")


def build_report() -> str:
    state = _load_state()
    sections = [
        _section_market(state),
        _section_signals(),
        _section_equity(),
        _section_data_health(state),
        _section_anomalies(state),
    ]
    sections.append(_section_actions(sections))

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    header_flag = "🚨" if any(s.flag == "alert" for s in sections) else ("⚠️" if any(s.flag == "warn" for s in sections) else "✅")

    parts = [
        f"# Daily Brain — {today} {header_flag}",
        f"_gerado em {datetime.now(timezone.utc).isoformat()}_",
        "",
    ]
    for sec in sections:
        emoji = {"alert": "🚨", "warn": "⚠️", "": ""}[sec.flag]
        parts.append(f"## {emoji} {sec.title}".rstrip())
        parts.append(sec.body)
        parts.append("")
    parts.append("---")
    parts.append("_Daily Brain rodando via .github/workflows/daily_brain.yml — relatório commitado em reports/_")

    return "\n".join(parts)


def run() -> None:
    REPORTS.mkdir(exist_ok=True)
    report = build_report()
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    out = REPORTS / f"daily_{today}.md"
    out.write_text(report)
    latest = REPORTS / "latest.md"
    latest.write_text(report)
    print(f"[daily_brain]  relatório gerado em {out}  ({out.stat().st_size} bytes)")


if __name__ == "__main__":
    run()
