# Paper Trade Phase

Você está aqui. Modelo validado no backtest (Sharpe HOLDOUT 1.55, PSR 0.952),
mas **nunca testado em dados que NUNCA existiram durante o desenvolvimento**.

Esta fase é o último filtro antes de operar com dinheiro real.

## Como funciona

O bot já está rodando — só não executa ordens. A cada 4h:

```
predict_4h (GH Actions)
  ├─ gera proba via dual-horizon MID
  ├─ se > 35% E não-bear E sem posição aberta
  │    ↓
  │  open_position()  ──►  data/positions.parquet  (paper trade!)
  │  envia 🟢 SINAL no Telegram com sizing sugerido
  │
  └─ você anota mentalmente OU executa manual na exchange (opcional)

monitor_positions (a cada 15min)
  ├─ checa posições abertas vs OHLCV recente
  ├─ se bate target/stop/timeout → fecha
  ├─ atualiza positions.parquet com pnl realizado
  └─ envia 🟢/🔴/⏱️ saída no Telegram com PnL final
```

Tudo armazenado em `data/positions.parquet`. Cada trade tem:
- entry_time, entry_price, target, stop
- proba_long do modelo no momento
- exit_time, exit_price, pnl_pct realizado
- status: open / closed_target / closed_stop / closed_timeout

## Como ler

### Status agora mesmo

```bash
python -m pipeline.paper_report
```

Mostra:
- Trades fechados / abertos
- Win rate, avg PnL, total composto
- Sharpe/trade + p-value (edge estatisticamente significativo?)
- Comparação vs backtest esperado
- Posições abertas com idade

### Relatório semanal automático

Workflow `paper_report.yml` envia resumo no Telegram todo **domingo 12:00 UTC**.
Pode disparar manual em Actions → "Paper Trade Report" → Run workflow.

## O que esperar (calibrado APÓS achado E2 — verdade dura)

⚠️ **Métricas anteriores estavam infladas por sorte de seed RNG.**
Análise E2 (ensemble 5 seeds): performance real tem **variância enorme**:

```
Métrica          single-seed enganoso    REAL (E2 mean ± std)
HO Sharpe        1.36 / 1.55             0.59 ± 0.48 (range -0.00 a +1.13)
Final $1k        $1,290                  $862 ± $116 (range $683 a $977)
Win rate         54.2%                   ~50-54% (similar entre seeds)
```

**Produção agora usa ensemble N=5** — converge pra MÉDIA esperada, não pro melhor seed.

Em 30 dias de paper trade você deve ver ~5-15 trades. Em 90 dias, ~25-45.

**Range realista de outcomes em 90 dias (com leverage 1x):**
- Pior cenário (perc 10%): -10% capital ($900)
- Mediana esperada: -5 a +5% capital
- Melhor cenário (perc 90%): +15% capital

**Bull market puro:** B&H continua melhor opção em retorno absoluto.

## Métricas de validação

| Após N trades | Veredito estatístico |
|---|---|
| < 20 | Inconclusivo — amostra muito pequena |
| 20–50 | Tendência detectável se win > 60% ou < 40% |
| 50–100 | p-value razoavelmente confiável |
| 100+ | Edge ou ausência dele bem caracterizados |

**Sinal verde pra trading real:**
- ≥ 30 trades fechados
- Win rate ≥ 50%
- Avg PnL/trade ≥ 0%
- p-value (Sharpe > 0) < 0.20
- Sem desvio severo do backtest (Δ win < -10pp)

**Sinais vermelhos** (suspender e re-analisar):
- Win rate < 40% após 20+ trades
- p-value > 0.5 (provavelmente sem edge)
- Sequência de 7+ losses
- Comportamento muito diferente do backtest

## Frequência atual do bot

```
Estado: bullish (proba > 35% + fora de bear)
   ↓
COMPRA com FULL capital
   ↓
Posição dura ~12h-48h
   ↓
Fecha em target/stop/timeout
```

Hoje proba = 8% (modelo bearish). Bot está **silencioso** — não envia sinal nem
abre posição. Volta a operar quando a config de mercado mudar.

## Setup adicional opcional

### Capital diferente de $1000

GitHub repo → Settings → Variables → New repository variable:
- Nome: `TELEGRAM_USER_CAPITAL`
- Valor: ex `5000`

Mensagens do bot vão mostrar sizing proporcional ao seu capital real.

### Pausar bot

```
GitHub Actions → Predict 4h → Disable workflow
```

Pra retomar: Enable de volta. Posições abertas continuam sendo monitoradas
pelo `monitor_positions` independente.

## Quando promover pra real trading

**Após 60-90 dias de paper trade** com métricas dentro do esperado:
1. Abrir conta na Binance/Bybit
2. Gerar API keys com **apenas trading**, não withdraw
3. Implementar execução em `pipeline/execute_trade.py` (não escrito ainda)
4. Começar com 10-20% do capital pretendido (size testnet)
5. Escalar gradualmente se 30 dias live forem consistentes com paper

NÃO promova ainda. Espera os dados primeiro.

## Riscos honestos

- Modelo testado em **bull market puro** (2023-2026). Em bear, comportamento desconhecido
- Custos podem subir em volatility extrema
- Slippage em sinais simultâneos com outros traders algorítmicos
- Edge pode degradar com tempo (alpha decay clássico)
- Você NÃO bate Buy-and-hold em bull (sacrifica retorno por menor drawdown)
