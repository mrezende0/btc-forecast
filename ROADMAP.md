# Roadmap — btc-forecast

Contrato do projeto. Quando você for tentado a pular EDA ou inflar critério de sucesso
pra atender resultado, esse documento é o adversário.

---

## 1. Filosofia

- **Não prever preço. Identificar regimes e assimetrias de risco/retorno.**
- Edge nasce de **descoberta estatística**, não de opinião de LLM sobre gráficos.
- LLMs **orquestram** (Camada 2: ingestão, explicação, alerta). LLMs **não descobrem edge**.
- Backtest honesto > modelo sofisticado. Bug em validação destrói qualquer ganho.
- Uso pessoal. Não é produto. Não é fundo. Critério de morte explícito.

## 2. Critério de sucesso (sugerido — revisar quando tiver mais maturidade)

Walk-forward 2023–2025, líquido de custos (taker 0.05% + slippage 0.03% = 0.08% round-trip):

| Métrica | Alvo | Mínimo | Benchmark B&H 2021-2026 |
|---|---|---|---|
| Sharpe anualizado | ≥ 1.0 | ≥ 0.7 | **0.60** |
| Max drawdown | ≤ 30% | ≤ 40% | -77.2% |
| Profit factor | ≥ 1.4 | ≥ 1.2 | 1.01 |
| CAGR | ≥ 20% | ≥ 10% | +19.5% |
| Sinais/semana | 2–4 | 1–6 | n/a |
| Bate buy-hold líquido? | sim | sim | — |
| Win rate (long) | ≥ 50% | ≥ 45% | 50.1% |

**Critério de morte:** Sharpe rolling 90d < 0.3 por 4 semanas consecutivas em paper trading.

Esses números saem do `02_baseline.py`. Quando tiver visão de regime (bull vs bear),
revisar — Sharpe 1.0 em chop é difícil; em bull é fácil. Métricas por regime > médias.

## 3. Stack

- **Ingestão:** Python + requests/ccxt-like, GitHub Actions cron, Parquet versionado.
- **Dados:** Binance (OHLCV, funding), yfinance (DXY/VIX/SPX), alternative.me (F&G),
  GDELT (notícias histórico), CoinDesk (notícias forward), FinBERT (sentiment scorer).
- **Modelagem:** Polars, pandas-ta, mlfinpy (triple-barrier + purged CV),
  LightGBM + SHAP, vectorbt.
- **Operação:** GH Actions + Telegram bot.

## 4. Fases

### Fase 1 — Ingestão
- [ ] Pipeline OHLCV 15m + funding (Binance)
- [ ] Pipeline macro daily (yfinance)
- [ ] Pipeline F&G daily
- [ ] Pipeline notícias incremental (CoinDesk)
- [ ] Backfill notícias histórico (GDELT)
- [ ] Sentiment scorer FinBERT
- [ ] Workflows GH Actions rodando estável 48h

### Fase 2 — EDA + Baseline burro
- [ ] Notebook `01_eda.ipynb`: distribuições, correlações, sazonalidade, gaps
- [ ] Definir alvo: triple-barrier (±X×ATR, Yh) — registrar parâmetros
- [ ] Baseline 1: buy-and-hold
- [ ] Baseline 2: regra simples (RSI<30 compra / RSI>70 venda)
- [ ] Baseline 3: sinal aleatório
- [ ] Métricas dos baselines salvas como referência

### Fase 3 — Labels
- [ ] `pipeline/labels.py` com triple-barrier via mlfinpy
- [ ] Sanity check: distribuição de classes (long_win / stop / timeout)
- [ ] Análise de tempo médio até barreira

### Fase 4 — Features
- [ ] Técnico (~15): retornos, vol realizada, ATR, RSI multi-TF, distância VWAP/MAs, Z-vol
- [ ] Derivativos (~10): funding nível/Z/EMA, OI Δ vs preço Δ, basis
- [ ] Macro (~5): DXY Z, real yield, VIX, correlação SPX rolling
- [ ] Sentiment (~5): F&G, news_count_z30d, net_sentiment_z30d
- [ ] **Toda feature com `.shift(1)` rigoroso**
- [ ] Feature registry YAML (nome, fonte, fórmula, lag, versão)

### Fase 5 — Modelo
- [ ] LightGBM + SHAP
- [ ] Hyperopt dentro de cada fold (nunca olhando teste)
- [ ] Análise de feature importance estável across folds

### Fase 6 — Walk-forward honesto
- [ ] Purged k-fold + embargo (mlfinpy)
- [ ] Expanding window mensal
- [ ] Métricas por regime (bull/bear/chop)
- [ ] Custo + slippage realistas

### Fase 7 — Backtest + Position sizing
- [ ] vectorbt com custos
- [ ] Kelly fracional ou volatility targeting
- [ ] Comparar contra baselines da Fase 2

### Fase 8 — Paper trading
- [ ] 3–6 meses gerando sinais reais sem operar
- [ ] Drift detection (KS/PSI nas features)
- [ ] Logging completo: timestamp, features snapshot, prob, threshold, versão modelo

### Fase 9 — Operação
- [ ] Telegram bot
- [ ] Threshold calibrado (2–3 sinais/sem)
- [ ] Kill switch automático (critério da seção 2)

## 5. Anti-armadilhas (checklist antes de cada fase)

- [ ] Toda feature defasada (`.shift(1)`)? Sem look-ahead?
- [ ] Sentiment/macro com `available_at` correto (não `event_time`)?
- [ ] As-of join backward, nunca merge ingênuo?
- [ ] Random split foi banido? Só purged/walk-forward?
- [ ] Hyperopt olhou o teste? Se sim, fold contaminado.
- [ ] Feature importance estável across folds? Senão é overfit.
- [ ] Modelo bate buy-and-hold líquido? Senão não tem edge.
- [ ] Modelo funciona em pelo menos 2 regimes (bull + bear)?
- [ ] Vela em formação foi filtrada (close_time ≤ now - 1min)?
- [ ] Cross-source bias em sentiment (GDELT vs CoinDesk) mitigado?

## 6. Fora de escopo (v1)

- Múltiplos ativos
- Execução automática de ordens
- Otimização HFT
- Reinforcement learning
- Modelos generativos (transformers/LSTM) — só LightGBM
- Dashboard web complexo (Cowork artifact resolve)

## 7. Material de estudo

- López de Prado — *Advances in Financial Machine Learning*, caps. 3, 4, 7
- Repo `tatsath/fin-ml` cap. 6 (case BTC)
- Doc mlfinpy

## 8. Quando matar o projeto

Se após a Fase 6 o modelo não baterconsistentemente buy-and-hold líquido em pelo menos
2 regimes distintos, projeto não tem edge. Arquivar e seguir vida.
