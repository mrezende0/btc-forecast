# live_ops_stack — proposta (não-deployada)

Stack mínima pra migrar o `btc-forecast` de "GH Actions + Telegram" pra paper trading testnet
com observability e kill-switch.

Ver brief completo em `briefs/infra_liveops.md`.

## Estrutura

```
proposals/live_ops_stack/
├── docker-compose.yml          # predictor + executor + redis + drift + kill + prom + loki + grafana
├── Dockerfile                  # imagem comum (build context = raiz do repo)
├── requirements-extra.txt      # deltas sobre requirements.txt do repo
├── prometheus.yml              # scrape config
├── promtail.yml                # log shipping pra Loki
├── grafana/provisioning/       # datasources + dashboards auto-load
└── scripts/
    └── drift_watchdog.py       # PSI por feature + kill flag
```

## TODO antes do `docker compose up` real

- [ ] Criar `predictor/app.py` (FastAPI wrapper de `pipeline.predict_now`)
- [ ] Criar `executor/paper.py` (consome sinal do Redis, envia testnet via ccxt sandbox)
- [ ] Criar `scripts/kill_switch.py` (sharpe rolling + api errors + PSI cruzado)
- [ ] Criar `scripts/snapshot_reference.py` (gera `data/reference_features.parquet` a partir do
      período de treino com Sharpe walk-forward validado)
- [ ] Criar dashboards Grafana em `grafana/provisioning/dashboards/overview.json`
- [ ] Migrar `data/positions.parquet` + `data/signals.parquet` pra SQLite WAL
      (`pipeline/storage_sqlite.py`)

## .env esperado

```
TELEGRAM_BOT_TOKEN=...
TELEGRAM_CHAT_ID=...
BINANCE_TESTNET_KEY=...
BINANCE_TESTNET_SECRET=...
GRAFANA_ADMIN_PASSWORD=...
```

## Boot suposto (após TODOs)

```bash
cd proposals/live_ops_stack
cp .env.example .env  # preencher
docker compose build
docker compose up -d
docker compose ps
docker compose logs -f predictor
```

Grafana: `http://127.0.0.1:3000` (NUNCA expor sem Caddy + basic auth).
Prometheus: `http://127.0.0.1:9090`.
Predictor health: `curl http://127.0.0.1:8080/healthz`.
