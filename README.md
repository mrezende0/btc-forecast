# btc-forecast

Sistema de sinais intraday (15m) para BTC. Pipeline de ingestão multi-fonte via GitHub
Actions, dados versionados em Parquet, modelo LightGBM com walk-forward honesto, alertas
no Telegram.

Status: **Fase 1 — Ingestão**.

## Visão geral

```
GH Actions (cron)
   │
   ├── ingest_15m   ──► Binance (OHLCV, funding) ──► data/*.parquet
   └── ingest_daily ──► yfinance (DXY/VIX/SPX)
                       alternative.me (F&G)
                       CoinDesk (notícias)
                       FinBERT scorer ──────────► data/sentiment_daily.parquet

Local (notebook)
   │
   ├── EDA + baseline burro
   ├── features (técnico + derivativos + macro + sentiment)
   ├── labels (triple-barrier)
   ├── walk-forward LightGBM + SHAP
   └── backtest honesto
```

Ver **ROADMAP.md** pro contrato completo do projeto.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # mínimo (Actions)
pip install -r requirements-dev.txt      # local (modelagem)
```

## Backfill inicial (rodar uma vez na máquina local)

```bash
# OHLCV + funding (Binance, ~3-5min)
python -m pipeline.ingest_15m --backfill

# Macro + F&G (yfinance + alternative.me)
python -m pipeline.ingest_daily

# Notícias GDELT (histórico 2021+, demora bastante)
python -m pipeline.news_backfill --start 2021-01-01

# Scorer FinBERT em todo histórico
python -m pipeline.sentiment_agg --recompute-all
```

Daí em diante, GitHub Actions cuida do incremental sozinho.

## Secrets necessários (GitHub repo Settings → Secrets)

- `COINDESK_API_KEY` — developers.coindesk.com (free)

FMP só entra se ativar o dashboard Cowork (não usado no ML).

## Estrutura

```
btc-forecast/
├── pipeline/         # fetchers + jobs
├── data/             # Parquets versionados
├── notebooks/        # EDA + modelagem
├── .github/workflows/
├── ROADMAP.md
└── README.md
```
