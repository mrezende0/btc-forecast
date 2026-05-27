# Setup Telegram Bot — alertas do modelo v2

5 minutos. Gratuito. Você recebe push notification a cada 4h se houver sinal.

## 1. Criar o bot via BotFather

1. No Telegram, busca `@BotFather` e abre conversa
2. Manda `/newbot`
3. Escolhe um nome (ex: "BTC Forecast Alert")
4. Escolhe username terminando em `bot` (ex: `btc_forecast_mr_bot`)
5. BotFather responde com **HTTP API Token**: algo tipo
   `7234567890:AAEhBP-_xxxxxxxxxxxxxxxxxxxxxxxxxxxx`
6. Guarda esse token

## 2. Pegar seu chat_id

1. Inicia conversa com o bot que você acabou de criar (busca o username e
   manda `/start`)
2. Abre no navegador:
   `https://api.telegram.org/bot<SEU_TOKEN>/getUpdates`
   substituindo `<SEU_TOKEN>` pelo token do passo 1
3. Procura no JSON o campo `"chat":{"id": 1234567890, ...}` — esse número é o
   seu chat_id

## 3. Adicionar como GitHub Secrets

No repo:
1. Settings → Secrets and variables → Actions → New repository secret
2. Cria 2 secrets:
   - Nome `TELEGRAM_BOT_TOKEN`  →  valor: token do passo 1
   - Nome `TELEGRAM_CHAT_ID`    →  valor: chat_id do passo 2

## 4. Testar

No GitHub:
1. Actions → "Predict 4h (Telegram alert)" → Run workflow
2. Marca **force_send: true** pra forçar envio (mesmo sem sinal)
3. Click Run workflow
4. Em ~2-3 min você recebe a mensagem no Telegram

Se chegou, está tudo OK. Daí em diante o cron dispara automaticamente a cada 4h
(00, 04, 08, 12, 16, 20 UTC) e só te avisa quando houver sinal real.

## Como ler o alerta

```
🟢 SINAL DE COMPRA — BTC

📊 Vela: 2026-05-27 16:00 UTC (4h)
💵 Preço: $75,400
🎯 Confiança modelo: 42.3%  (threshold 35%, edge +11%)

🎯 Estratégia (triple-barrier):
  • Target +1.1% (barreira superior)
  • Stop −1.1% (barreira inferior)
  • Timeout: 48h

📈 Contexto:
  • Vol 1d: 43% ann
  • Funding z-30d: +1.72
  • F&G: 25 (Extreme Fear)
  • VIX z: -1.03
```

**Como agir:** você decide. Modelo sinaliza "probabilidade > 35% de bater +1.1%
antes de -1.1% em até 48h". Edge histórico Sharpe 1.93 em 2025+, mas é UM
sinal — não vai bater toda vez.

## Quando NÃO houver sinal

Recebe notificação silenciosa (sem som):
```
⚪ Sem sinal — 2026-05-27 16:00 UTC
BTC $75,400  ·  proba_long 28.4% (<35%)
```

Pode desabilitar isso passando `--quiet` no workflow (já é default).

## Parar de receber alertas

Settings → Actions → Disable workflow "Predict 4h" ou deleta os secrets.
