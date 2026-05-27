"""Sender Telegram — formata e envia alertas do modelo v2.

Requer envs:
  TELEGRAM_BOT_TOKEN — do BotFather
  TELEGRAM_CHAT_ID   — seu chat ID (numérico)
"""
from __future__ import annotations

import os
from datetime import datetime, timezone

import requests

API = "https://api.telegram.org/bot{token}/sendMessage"


def _creds() -> tuple[str, str]:
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
    if not token or not chat_id:
        raise RuntimeError(
            "Faltam TELEGRAM_BOT_TOKEN ou TELEGRAM_CHAT_ID. "
            "Veja dashboard/SETUP_TELEGRAM.md"
        )
    return token, chat_id


def send(text: str, silent: bool = False) -> None:
    token, chat_id = _creds()
    r = requests.post(
        API.format(token=token),
        json={
            "chat_id": chat_id,
            "text": text,
            "parse_mode": "Markdown",
            "disable_notification": silent,
        },
        timeout=15,
    )
    if r.status_code != 200:
        raise RuntimeError(f"Telegram falhou {r.status_code}: {r.text[:200]}")


def format_signal(pred: dict, state: dict | None = None, is_test: bool = False, position: dict | None = None) -> str:
    """Formata alerta legível em Markdown.

    Se `position` (dict do positions.open_position) vier, mostra target/stop reais.
    """
    ts = datetime.fromtimestamp(pred["open_time"] / 1000, tz=timezone.utc)
    proba_pct = pred["proba_long"] * 100
    conf_pct = pred["confidence_pct"]
    price = pred["close"]
    edge_sign = "+" if conf_pct >= 0 else ""  # negative já vem com "-"

    if is_test:
        header = "🧪 *TESTE — sem sinal real*"
        sub = f"_proba {proba_pct:.1f}% está abaixo do threshold 35%, em produção isto NÃO teria sido enviado._"
    else:
        header = "🟢 *SINAL DE COMPRA — BTC*"
        sub = None

    lines = [
        header,
        "",
        f"📊 Vela: `{ts:%Y-%m-%d %H:%M} UTC` (4h)",
        f"💵 Preço: *${price:,.0f}*",
    ]
    # Mostra proba do modelo MID (regra de produção) + long_h informativo se houver
    if "proba_mid" in pred:
        rule = pred.get("ensemble_rule", "MID")
        proba_mid_pct = pred["proba_mid"] * 100
        if pred.get("proba_long_horizon") is not None:
            lines.append(
                f"🎯 Modelos: mid 48h *{proba_mid_pct:.1f}%*  ·  long 72h *{pred['proba_long_horizon']*100:.1f}%*  "
                f"(regra: {rule})"
            )
        else:
            lines.append(f"🎯 Modelo mid 48h: *{proba_mid_pct:.1f}%*  (threshold 35%, regra: {rule})")
        # Filtro de regime
        if pred.get("in_bear"):
            ret_30d_pct = pred.get("ret_30d", 0) * 100
            lines.append(f"🐻 Filtro BEAR ATIVO: BTC {ret_30d_pct:+.1f}% nos últimos 30d (sinal suprimido)")
        elif "ret_30d" in pred:
            lines.append(f"📈 BTC {pred['ret_30d']*100:+.1f}% nos últimos 30d (fora de bear)")
    else:
        lines.append(f"🎯 Confiança modelo: *{proba_pct:.1f}%*  (threshold 35%, edge {edge_sign}{conf_pct:.0f}%)")
    if sub:
        lines.append(sub)
    lines.append("")
    if position:
        tgt_pct = (position["target_price"] / position["entry_price"] - 1) * 100
        stop_pct = (position["stop_price"] / position["entry_price"] - 1) * 100
        lines.extend([
            "🎯 Estratégia (triple-barrier, ATR-based):",
            f"  • Target: *${position['target_price']:,.0f}*  ({tgt_pct:+.2f}%)",
            f"  • Stop: *${position['stop_price']:,.0f}*  ({stop_pct:+.2f}%)",
            f"  • Timeout: {position['horizon_hours']}h",
        ])
        # Sugestão de sizing
        if "size_suggestion" in position:
            sz = position["size_suggestion"]
            mode = sz.get("mode", "full")
            cap_note = " (cap 50%)" if sz.get("capped") else ""
            label = "FULL (margem 100%)" if mode == "full" else "1% risk on stop"
            leverage = sz.get("leverage", 1.0)
            lev_warn = ""
            if leverage >= 3.0:
                lev_warn = "  ⚠️"
            risk_pct = sz.get("risk_pct_of_capital", sz["risk_dollars"]/sz["capital"]) * 100
            # Detalha origem da leverage (dynamic/env/default)
            lev_info = sz.get("leverage_info", {})
            lev_source = lev_info.get("source", "default")
            if lev_source == "dynamic":
                f_conf = lev_info.get("f_conf", 0)
                f_vol = lev_info.get("f_vol", 1)
                lev_label = f"*{leverage:.1f}x* (dinâmica: conf {f_conf*100:.0f}% × vol-brake {f_vol:.2f}){lev_warn}"
            elif lev_source == "env":
                lev_label = f"*{leverage:.1f}x* (manual via TRADING_LEVERAGE){lev_warn}"
            else:
                lev_label = f"*{leverage:.1f}x*{lev_warn}"
            lines.extend([
                "",
                f"💰 Sizing sugerido ({label}):",
                f"  • Capital base: *${sz['capital']:,.0f}*",
                f"  • Alavancagem: {lev_label}",
                f"  • Notional BTC: *{sz['size_btc']:.5f}* ≈ *${sz['notional_usd']:,.0f}*",
                f"  • Margem usada: *${sz.get('margin_usd', sz['size_usd']):,.0f}* ({sz['pct_of_capital']*100:.1f}% capital){cap_note}",
                f"  • Risco se stop: *${sz['risk_dollars']:,.2f}* (*{risk_pct:.1f}% do capital*)",
            ])
            if sz.get("leverage_capped"):
                lines.append(f"  ⚠️ leverage capada em {leverage:.1f}x (máx {leverage:.0f}x por segurança)")
            if leverage >= 3.0:
                lines.append(f"  _⚠️ alta alavancagem: 1 stop = {risk_pct:.0f}% do capital. Errar 4-5 vezes seguidas quebra a conta._")
        lines.append("")
        lines.append("_Saída automática: sistema te avisa quando bater target, stop ou timeout._")
    else:
        lines.extend([
            "🎯 Estratégia (triple-barrier):",
            "  • Target ~+1.1% (barreira superior, varia com ATR)",
            "  • Stop ~−1.1% (barreira inferior, varia com ATR)",
            "  • Timeout: 48h",
        ])

    if state:
        lines.append("")
        lines.append("📈 *Contexto*:")
        if state.get("vol", {}).get("available"):
            rv1d = state["vol"]["rv_1d_ann"] * 100
            lines.append(f"  • Vol 1d: {rv1d:.0f}% ann")
        if state.get("funding", {}).get("available"):
            z = state["funding"]["z_30d"]
            lines.append(f"  • Funding z-30d: {z:+.2f}")
        if state.get("fg", {}).get("available"):
            fg = state["fg"]["last"]
            cls = state["fg"]["last_class"]
            lines.append(f"  • F&G: {fg} ({cls})")
        if state.get("macro", {}).get("vix", {}).get("z_30d") is not None:
            vix_z = state["macro"]["vix"]["z_30d"]
            lines.append(f"  • VIX z: {vix_z:+.2f}")

    lines.append("")
    lines.append("_Uso pessoal. Não é recomendação de investimento._")
    return "\n".join(lines)


def format_no_signal(pred: dict) -> str:
    """Resumo curto quando NÃO houve sinal (notificação silenciosa, opcional)."""
    ts = datetime.fromtimestamp(pred["open_time"] / 1000, tz=timezone.utc)
    return (
        f"⚪ Sem sinal — {ts:%Y-%m-%d %H:%M} UTC\n"
        f"BTC ${pred['close']:,.0f}  ·  proba_long {pred['proba_long']*100:.1f}% (<35%)"
    )
