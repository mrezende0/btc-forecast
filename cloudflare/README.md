# Cloudflare Worker proxy

Proxy gratuito pra contornar geo-block (Binance fapi 451, Bybit 403) em runners US do GitHub Actions.

Free tier CF Workers: 100k requests/dia. Uso atual ~2k/dia (192 × 5 endpoints × 2 assets).

## Setup (uma vez)

1. **Conta Cloudflare**: criar grátis em https://dash.cloudflare.com/sign-up
2. **Instalar wrangler** (CLI CF Workers):
   ```bash
   npm install -g wrangler
   ```
3. **Login**:
   ```bash
   wrangler login
   ```
4. **Token de proteção** — gere uma string aleatória (ex `openssl rand -hex 32`) e guarde. Esse `PROXY_TOKEN` evita uso público abusivo.
5. **Deploy + secret**:
   ```bash
   cd cloudflare/
   wrangler secret put PROXY_TOKEN   # cola o token gerado
   wrangler deploy
   ```
   Saída final: `https://btc-forecast-proxy.<seu-subdomain>.workers.dev`
6. **GitHub Secrets** — Settings → Secrets and variables → Actions:
   - `PROXY_BASE` = `https://btc-forecast-proxy.<seu-subdomain>.workers.dev`
   - `PROXY_TOKEN` = (mesmo token do passo 4)

Pronto. Próximo cron `ingest_15m` vai roteamento via worker.

## Testar local

```bash
TOKEN=<seu-token>
curl -H "X-Proxy-Token: $TOKEN" \
  "https://btc-forecast-proxy.<seu-subdomain>.workers.dev/binance-fapi/futures/data/openInterestHist?symbol=BTCUSDT&period=15m&limit=5"
```

Deve retornar JSON. Se 401, token errado. Se 403/451, CF egress ainda bloqueado (raro mas possível).

## Como funciona

Pipeline lê env vars `BINANCE_FAPI_BASE`, `BINANCE_SPOT_BASE`, `BYBIT_BASE`. Sem o proxy configurado, defaults apontam pra upstream direto (funciona local fora-US). Workflow GH define elas pra rotas do worker.
