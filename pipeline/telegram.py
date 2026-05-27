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
    # Se houver pred dual-horizon, mostrar ambos
    if "proba_mid" in pred and "proba_long_horizon" in pred:
        lines.append(
            f"🎯 Modelos: mid 48h *{pred['proba_mid']*100:.1f}%*  +  long 72h *{pred['proba_long_horizon']*100:.1f}%*  "
            f"(both > 35% pra confirmar)"
        )
        # Indica filtro de regime
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
            label = "FULL (100% do capital)" if mode == "full" else "1% risk on stop"
            lines.extend([
                "",
                f"💰 Sizing sugerido ({label}):",
                f"  • Capital base: *${sz['capital']:,.0f}*",
                f"  • Posição: *{sz['size_btc']:.5f} BTC* ≈ *${sz['size_usd']:,.0f}* ({sz['pct_of_capital']*100:.1f}% do capital){cap_note}",
                f"  • Risco se stop: *${sz['risk_dollars']:,.2f}* ({sz['risk_dollars']/sz['capital']*100:.1f}% do capital)",
            ])
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
