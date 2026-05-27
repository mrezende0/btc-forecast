# Red-Team Brief — btc-forecast
**Alvo:** Sharpe 1.29 reportado no commit `bff285c` (dual-horizon AND ensemble).
**Postura:** céticismo máximo. Auditoria forense linha-a-linha de `pipeline/` + `notebooks/exp_*` + `notebooks/0[5-9]_*`.
**Data:** 2026-05-27.

---

## Veredito top-line

**O Sharpe 1.29 NÃO sobrevive a uma validação rigorosa. Confiança que o número *real* OOS é ≤0.7: ~80%.**

Sharpe "verdadeiro" estimado depois de corrigir os bugs catalogados: **0.4 – 0.8** (banda larga por causa de selection bias em cascata). Razões em ordem de impacto:

1. **Custo subestimado** (0.08% vs realista 0.15–0.20% round-trip) — derruba Sharpe em ~30–50% sozinho.
2. **Annualização de Sharpe por trade × √(trades/year)** assume IID; sobreposição massiva de barreiras (horizon=12 bars, overlap 11 bars) viola IID e infla Sharpe ~1.5–2×.
3. **Threshold (0.35), filtro bear (-5%) e dual-horizon AND** foram escolhidos *olhando o mesmo pool walk-forward 2023+*. Não há holdout separado para essas decisões → selection bias clássico (López de Prado, "Selection bias under multiple testing").
4. **Filtro "no-bear"** não foi *stress-tested* fora de 2023+ (regime majoritariamente bull). O `bff285c` ("+47%") provavelmente captura ajuste a um path único.
5. **Sample uniqueness não ponderado** (López de Prado AFML cap.4): treino vê labels altamente correlacionados como independentes.
6. **Hyperopt em `07_hyperopt.py`** separa VAL/HOLDOUT, mas os parâmetros realmente em produção (`pipeline/model.py:23-35`) NÃO vêm desse holdout — vêm de `06_model_v2.py` que rodou no pool inteiro. Os params "validados" são na verdade in-sample.

---

## Bugs encontrados — ordenados por severidade

### HIGH-1 — Custo round-trip subestimado em ~2×
- **Arquivo:** `pipeline/model.py` (não tem custo, é só prod), `notebooks/exp_ensemble.py:38`, `notebooks/exp_multi_horizon.py:45`, `notebooks/exp_threshold_grid.py:47`, `notebooks/exp_backtest_1k.py:40`, `notebooks/06_model_v2.py:38`, `notebooks/08_model_v3_sentiment.py:77`.
- **Linha exata (exp_ensemble):** `COST = 0.0008`.
- **Descrição:** Binance taker fee spot = 0.10% por lado = **0.20% round-trip** (preço cheio, sem BNB/VIP). Mesmo com slippage zero. Slippage realista em BTCUSDT spot para market order pequeno: 0.02–0.05%. Total realístico round-trip: **0.12–0.20%**. `COST = 0.0008` (= 0.08%) assume preço fechado de BNB-discount + zero slippage.
- **Fix sugerido:** subir COST para 0.0015 (taker padrão sem desconto) ou 0.002. Roteiro: rodar `exp_backtest_1k.py` com COST=0.0015 e COST=0.002 e reportar Sharpe.
- **Impacto estimado em Sharpe:** −0.3 a −0.5. Com avg_pnl da ordem de 0.5–1% por trade e 0.07% a mais de custo, isso é ~10–15% do PnL líquido por trade.

### HIGH-2 — Sharpe anualizado infla por sobreposição de trades (não-IID)
- **Arquivo:** `notebooks/exp_ensemble.py:198-201`, `notebooks/exp_multi_horizon.py:203-206`, `notebooks/exp_threshold_grid.py:199-202`.
- **Snippet (exp_threshold_grid):**
  ```python
  def sharpe_trades(pnls: np.ndarray, trades_per_year: float) -> float:
      if len(pnls) < 2 or pnls.std(ddof=1) == 0:
          return 0.0
      return float(pnls.mean() / pnls.std(ddof=1) * np.sqrt(trades_per_year))
  ```
- **Descrição:** Sharpe = média(pnl)/std(pnl) × √(n_trades/years). Assume trades IID. Mas:
  - Triple-barrier com `horizon_bars=12` produz labels com até 11 bars de overlap.
  - Trade aberto em bar t cobre [t, t+12]; trade em bar t+1 cobre [t+1, t+13] — 92% overlap em PnL exposure.
  - Se trades se aglomeram em regiões de baixa vol → std(pnl) cai → Sharpe sobe espuriamente.
  - Anualização por √(trades/year) trata 200 trades em 3 anos como 67 trades anuais; com forte autocorrelação serial entre PnLs, fator inflacionário típico é 1.3–2.0×.
- **Fix sugerido:** (a) computar Sharpe em base de **barra** (eq_df) como em `exp_backtest_1k.py:285-290`, que é o padrão certo; (b) ou usar bootstrap em blocos / Sharpe deflated (López de Prado, "The Deflated Sharpe Ratio").
- **Impacto estimado:** Sharpe per-trade × √(tpy) consistente em ~1.5× o Sharpe bar-based real. Sharpe 1.29 → real ~0.7–0.9.

### HIGH-3 — Threshold 0.35 escolhido no mesmo test pool (data snooping)
- **Arquivo:** `notebooks/06_model_v2.py:185-196` (introduz threshold sweep), `notebooks/exp_threshold_grid.py:251-385` (varre grid 5×5 no test pool).
- **Linha exata (06):** `for thr in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:` — agregado em `proba_all` (linha 179) que é o pool inteiro 2023+.
- **Descrição:** O threshold 0.35 (mais o filtro AND mid/long) virou hyperparâmetro fixo em `pipeline/model.py:19` (`SIGNAL_THRESHOLD = 0.35`). A escolha desse 0.35 saiu de inspecionar o pool de teste 2023+ → 0.35 dá melhor Sharpe. **Esse é o teste contaminando hyperparam.**
- **Fix sugerido:** rodar threshold sweep só em janela VAL (ex.: 2023), congelar 0.35, e reportar Sharpe em HOLDOUT (2024–2026) sem retocar. `07_hyperopt.py` faz isso para params LGB mas NÃO para o threshold.
- **Impacto estimado:** −0.2 a −0.4 em Sharpe (incrustado dentro do número 1.29).

### HIGH-4 — Filtro `NO_BEAR_THRESHOLD = -0.05` calibrado no test set
- **Arquivo:** `pipeline/model.py:21`, validado em `notebooks/exp_regime_analysis.py` (untracked).
- **Snippet:** `NO_BEAR_THRESHOLD = -0.05  # se BTC caiu >5% no último mês → suprime sinal (validado em exp_regime_analysis)`
- **Descrição:** O parâmetro -5% / 30d veio de inspecionar o mesmo histórico em que se mede Sharpe. Em 2023+ (majoritariamente bull) esse filtro raramente dispara e quando dispara, evita exatamente os drawdowns conhecidos. É overfit por construção: o "Δ Sharpe +0.4" reportado depende do path único.
- **Fix sugerido:** validar -5% em ≥ 2 regimes históricos (ex.: 2022 bear) com purge entre VAL/test. Ou substituir por filtro paramétrico (ex.: drawdown rolling como z-score) calibrado em sub-period.
- **Impacto estimado:** −0.1 a −0.3 quando avaliado em regime fora-de-amostra.

### HIGH-5 — Selection bias na escolha do ensemble vencedor
- **Arquivo:** `notebooks/exp_ensemble.py:230-232`.
- **Snippet:**
  ```python
  best_name = max(results, key=lambda r: r['sharpe'] if not np.isnan(r['sharpe']) else -np.inf)['name']
  best_proba = combos[best_name]
  ```
- **Descrição:** Testa 5 combos (LGB only, XGB only, mean, weighted, max) no mesmo pool e escolhe o melhor por Sharpe. Pelo "multiple testing problem" (LdP, 2014), com 5 combos e 7 thresholds = **35 hipóteses testadas**, Sharpe-spurious-best esperado ~0.3–0.4 mesmo sob H0.
- **Fix sugerido:** Bonferroni-correct ou Deflated Sharpe Ratio. Para reporte honesto, fixar UMA combinação a priori.
- **Impacto estimado:** −0.2 a −0.3.

### HIGH-6 — Sample uniqueness não ponderado (overlap massivo em treino)
- **Arquivo:** `pipeline/labels.py:57-72`, e qualquer chamada de `lgb.train(...)` sem `weight=`.
- **Descrição:** López de Prado AFML cap.4: com horizon=12 bars e 1 sample por bar, cada label sobrepõe 11 outras → uniqueness média ≈ 1/12. Modelo treina como se 7.000 amostras fossem independentes; effective sample size ≈ 580. Resultado: confiança do modelo overshoot + Sharpe inflado.
- **Fix sugerido:** computar `avg_uniqueness` por amostra (LdP eq.4.2) e passar como `weight=` no `lgb.Dataset`. Ou usar `seq_bootstrap` para o bagging.
- **Impacto estimado:** Difícil quantificar; literatura sugere correção tipicamente reduz Sharpe 0.1–0.3 em modelos com overlap pesado.

### HIGH-7 — Meta-labeling tem leak entre as duas metades (09_meta_labeling.py)
- **Arquivo:** `notebooks/09_meta_labeling.py:89-99`.
- **Snippet:**
  ```python
  half = len(train_idx) // 2
  fold_a = train_idx[:half]
  fold_b = train_idx[half:]
  ...
  m_a = lgb.train(PARAMS_PRIM, lgb.Dataset(Xa, ya), num_boost_round=500)
  m_b = lgb.train(PARAMS_PRIM, lgb.Dataset(Xb, yb), num_boost_round=500)
  pa = m_b.predict(Xa)  # fold_a predito por modelo SEM fold_a
  pb = m_a.predict(Xb)
  ```
- **Descrição:** As duas metades são adjacentes sem purge. Labels do final de `fold_a` (horizon=12 bars à frente) tocam preços do início de `fold_b`. `m_a` treinado em fold_a "viu" features que predizem retornos cujo período-realizado vaza para fold_b. Quando `m_a` prediz `fold_b`, há leak temporal.
- **Fix sugerido:** Inserir purge de `HORIZON_BARS` entre as metades; ou usar PurgedKFold do mlfinpy.
- **Impacto estimado:** Não afeta diretamente o número 1.29 (que vem do ensemble dual-horizon, não do meta). Mas se meta-labeling for futuramente integrado, Sharpe vai inflar artificialmente.

### HIGH-8 — Hyperopt VAL/HOLDOUT separa params mas threshold/horizons NÃO foram validados nesse split
- **Arquivo:** `notebooks/07_hyperopt.py:52-55, 91-108`.
- **Descrição:** Hyperopt rouba split honesto: VAL=2024, HOLDOUT=2025+. Otimiza Sharpe em VAL com TPE. Mas em `pipeline/model.py` os LGB_PARAMS em produção (linhas 23-35) **não são os do hyperopt** — são os hardcoded do `06_model_v2.py:91-104`. Os params "validados" são in-sample (06 rodou em 2023+ inteiro).
- **Fix sugerido:** copiar os best_params do study para `pipeline/model.py`; ou aceitar os defaults com nota explícita que NÃO foram validados em holdout.
- **Impacto estimado:** −0.1 a −0.2.

### HIGH-9 — Dual-horizon "AND" ganho de Sharpe não validado em holdout próprio
- **Arquivo:** `notebooks/exp_multi_horizon.py:295-313` (regras "consensus/majority/any"), `pipeline/model.py:78-136` (produção usa AND mid+long).
- **Descrição:** O salto Sharpe 0.88 → 1.29 (+47%) reportado no commit `bff285c` vem de selecionar AND(mid, long) no mesmo pool 2023+. Não há holdout post-selection. A escolha "AND" sobre 3 regras (any/majority/consensus) e 3 horizontes possíveis = ao menos 6 testes implícitos.
- **Fix sugerido:** validar combinação dual-horizon em 2024 only, reportar 2025+ sem retoque.
- **Impacto estimado:** −0.15 a −0.30. Plausível que o ganho "+47%" seja parcialmente espúrio.

### MED-1 — Backtest live ≠ produção: retreino a cada 90d vs a cada execução
- **Arquivo:** `notebooks/exp_backtest_1k.py:42, 142` vs `pipeline/predict_now.py:51-53`.
- **Snippet (backtest):** `RETRAIN_EVERY_BARS = 90 * BARS_PER_DAY` (linha 42); `pipeline/model.py:predict_dual_horizon` retreina **toda execução** (linha 89-90).
- **Descrição:** Backtest assume stale model 90 dias; produção retreina cada bar. Discrepância vira drift: produção pode ter accuracy diferente da backtest. Não é leak, mas é unrealistic alignment.
- **Fix sugerido:** alinhar retraining cadence backtest = produção, ou aceitar e medir diferença.
- **Impacto:** Direção desconhecida.

### MED-2 — GDELT `seendate` ≠ `published_date` real (timestamp leak potencial)
- **Arquivo:** `pipeline/gdelt.py:88-104`, `pipeline/sentiment_agg.py:99-103`.
- **Snippet:** `pl.col("seendate").str.strptime(...).alias("published_at")` (gdelt.py:91-95) — `seendate` é "when GDELT first ingested", não exatamente publicação do artigo. Caso real: artigo publicado às 23:55, indexado pelo GDELT às 00:05 do dia seguinte → date salta.
- **Descrição:** Não vaza futuro (seendate ≥ true publish), mas pode atrasar sinal — efeito anti-leak (conservador). Risco oposto: se algum site backdates URL (ex.: artigo de "2023-01-15" foi escrito em 2023-01-20), `seendate` reflete real, mas títulos podem citar eventos futuros visualmente identificáveis pelo FinBERT.
- **Fix sugerido:** auditar 100 artigos amostrados; comparar `seendate` vs metadata real. Aceitável manter como está; documentar.
- **Impacto:** Baixo (efeito conservador).

### MED-3 — Threshold grid `exp_threshold_grid.py` legaliza data snooping no log
- **Arquivo:** `notebooks/exp_threshold_grid.py:313-377`.
- **Descrição:** Imprime "MELHOR (por Sharpe)" no mesmo pool 2023+. Recomenda trocar threshold se Δ > 0.15. **Isso é deliberada otimização in-sample.** O notebook decide pela manutenção (0.35) mas a infraestrutura para data snooping está pronta.
- **Fix:** dividir grid em VAL/HOLDOUT como `07_hyperopt.py`.
- **Impacto:** Direto se for usado para mudar threshold em prod.

### MED-4 — `exp_backtest_1k.py:212-217`: predição usa `fc_mid_arr[i]` (features defasadas em t-1), entrada em `closes[i]`
- **Arquivo:** `notebooks/exp_backtest_1k.py:129, 212-222`.
- **Snippet:**
  ```python
  fc_mid_arr = mat_mid[fc_mid].to_numpy()  # já com lag=1 do features.py
  ...
  x_mid = fc_mid_arr[i : i + 1]
  proba_mid = float(model_mid.predict(x_mid)[0])
  ...
  entry_px = closes[i]  # close da barra i
  ```
- **Descrição:** Features defasadas em 1 bar (apply_lag em features.py:258). Entrada no `closes[i]` da mesma bar. Isso significa: na bar t você usa features computadas até t-1 e abre no close de t. Na execução real, decisão precisa ser tomada ANTES do close de t. Marginal but tem look-ahead intrabar de 1 bar de 4h (~uma quase-prazerosa janela).
- **Fix:** entrar no `closes[i+1]` (=close do próximo bar) ou em `opens[i+1]` (mais realista, com slippage). Subir COST para refletir.
- **Impacto:** Pequeno se features pré-bar; mas se o modelo aprende algo do `close[t-1]` que muda com volume de t, pode haver 0.05–0.1% de "borrowed alpha" por trade. Pequeno mas existe.

### MED-5 — `model.py:107-108` filtro bear olha `mat_mid["close"][-1]` (close da vela atual em formação?)
- **Arquivo:** `pipeline/model.py:107-108`.
- **Snippet:**
  ```python
  close_now = float(mat_mid["close"][-1])
  close_30d_ago = float(mat_mid["close"][-1 - bars_per_month])
  ```
- **Descrição:** `mat_mid` vem de `build_v2_from_parquets` que lê `ohlcv_15m.parquet`, que em `binance.py:71` filtra `close_time <= now - 1min`. Então `close[-1]` é a última vela já fechada, **não** vela em formação. OK causal.
- **Impacto:** Nenhum. Mas a justificativa precisa ficar clara no código.

### LOW-1 — `pipeline/labels.py:83-87` empate na mesma vela tratado como STOP
- **Snippet:** se `up_first == dn_first`, label=-1 (conservador). OK.

### LOW-2 — `features.py:243-249` LAG_SAFE_EXCLUDE inclui calendário, OK
Calendário não precisa de defasagem (hora atual da vela é conhecida no momento dela).

### LOW-3 — `pipeline/binance.py:71` filtra velas em formação corretamente
`close_time <= now - 1min` — bom guardrail.

---

## Bugs suspeitos não confirmados (precisam de teste)

### SUS-1 — Sentiment cross-source bias entre GDELT (histórico) e CoinDesk (live)
- **Arquivo:** `pipeline/sentiment_agg.py` agrega ambas as fontes na mesma `net_sentiment`.
- **Suspeita:** Distribuição de FinBERT scores em manchetes do GDELT (multi-portal, multi-quality) vs CoinDesk (curado, cripto-focado) pode diferir sistematicamente. Treino em 2021–2023 (~70% GDELT) e teste em 2024+ (mistura ou só CoinDesk) → covariate shift estilo "test set distribution ≠ train".
- **Teste:** plotar histogramas de `sentiment` separados por `source` em janelas overlap. Se médias divergem > 1σ, há viés.

### SUS-2 — Features de hour/dow podem capturar correlação espúria
- **Arquivo:** `features.py:228-239`.
- **Suspeita:** `is_us_session` + `is_weekend` em dataset 2021+ que inclui rally pós-ETF Approval (janeiro 2024) podem aprender padrões sazonais coincidentes. Importância dessas features no modelo > esperado seria red flag.
- **Teste:** rodar feature importance por fold e verificar instabilidade.

### SUS-3 — yfinance `Close` ajustado retroativamente?
- **Arquivo:** `pipeline/macro.py:20` (`auto_adjust=False`) — bom, mas yfinance pode retornar dados de SPX/VIX que diferem entre re-runs. Se `macro_daily.parquet` foi populado em datas diferentes e contém valores parcialmente revisados, o histórico não é point-in-time consistente.
- **Teste:** comparar `macro_daily.parquet` atual com snapshot de meses atrás. Diff > 0 = leak retrospectivo de revisão.

### SUS-4 — Position sizing `RISK_PER_TRADE = 0.01` foi calibrado in-sample
- **Arquivo:** `pipeline/model.py:20`, `notebooks/exp_position_sizing.py` (untracked).
- **Suspeita:** Em backtest com full notional 100% por trade (exp_backtest_1k.py) o Sharpe 1.29 não usa esse 1%. Mas se algum experiment usou 1% Kelly-like calibrado em 2023+, mesmo padrão de data snooping.

### SUS-5 — Bias em `news_count` quando GDELT 2.0 mudou quotas/cobertura
- **Arquivo:** `pipeline/gdelt.py` puxa `bitcoin` keyword. GDELT mudou indexing several times. `news_count` em 2021 vs 2024 pode refletir mudança de quota, não interesse real.
- **Teste:** plotar `news_count` mensal — quebras estruturais visíveis = artefato.

---

## Top-3 testes adversariais a rodar

Os 3 testes estão implementados no stub `proposals/red_team_tests.py`.

1. **Shuffle labels test** — permuta `y` aleatoriamente preservando estrutura temporal. Treina modelo idêntico, mede Sharpe. **Sharpe deve colapsar a 0 ± ruído.** Se Sharpe > 0.3, há leak ou bug onde features "veem" o label de outra forma (ex.: target encoding, normalização do dataset inteiro).

2. **Time-reversed test** — inverte ordem temporal: treina em "futuro" (2024+), testa em "passado" (2023). Se Sharpe permanece alto, modelo está aprendendo correlação não-causal (mean-reversion estática) e não regime preditivo. Se Sharpe quebra para perto de 0 ou negativo, o pipeline é minimamente causal (mas isso não exclui os outros bugs).

3. **Noise feature injection** — adiciona uma feature gaussian iid `noise ~ N(0,1)` ao set. Se `feature_importance(noise) > 0.05` em qualquer fold, há leak no pipeline (ex.: normalização cross-fold, feature engineered sobre dataset inteiro, etc.). Esperado: noise rank no fim da lista.

Bônus se houver tempo:

4. **Custo realista** — rodar `exp_backtest_1k.py` com `COST = 0.002` (taker padrão sem desconto). Sharpe esperado: 0.6–0.8.

5. **Deflated Sharpe Ratio (López de Prado 2014)** — com N=35 hipóteses testadas (5 ensembles × 7 thresholds + 5×5 grid + 3 regras horizon), aplicar correção. Esperado: Sharpe deflated ~0.7–0.9.

---

## Refs (verificadas)

1. López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley. Cap. 3 (triple-barrier + meta-labeling), cap. 4 (sample weights / uniqueness), cap. 7 (cross-validation in finance, purge & embargo).

2. López de Prado, M. (2014). "The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting and Non-Normality." *Journal of Portfolio Management*, 40(5), 94-107. — fórmula para corrigir Sharpe ótimo encontrado entre N hipóteses.

3. López de Prado, M. (2018). "The 10 Reasons Most Machine Learning Funds Fail." *Journal of Portfolio Management*, 44(6). — "Seven sins of quantitative investing" (overfitting, look-ahead, backtest data snooping, leakage, survivorship, NaN/outliers, transaction-cost misspecification).

4. Bailey, D. H., Borwein, J., López de Prado, M., Zhu, Q. J. (2014). "Pseudo-Mathematics and Financial Charlatanism: The Effects of Backtest Overfitting on Out-of-Sample Performance." *Notices of the American Mathematical Society*, 61(5). — quantifica inflação esperada do Sharpe ótimo entre N hipóteses.

5. Bailey, D. H., López de Prado, M. (2014). "The Sharpe Ratio Efficient Frontier." *Journal of Risk*, 15(2). — distribuição amostral de Sharpe sob não-normalidade.

6. Microsoft LightGBM docs — `sample_weight` parameter (https://lightgbm.readthedocs.io/en/latest/Parameters.html#sample_weight). — confirmação de que weights são suportados para corrigir uniqueness.

7. GDELT 2.0 Documentation (https://www.gdeltproject.org/data.html#documentation) — `seendate` é "first seen by GDELT", não publish_date estrita. Esquema confirma o risco descrito em MED-2.

8. Binance Spot Trading Fees (https://www.binance.com/en/fee/schedule) — Regular taker fee: 0.10% por lado. Com BNB discount: 0.075%. Sem desconto: 0.20% round-trip. Confirmação do HIGH-1.

9. Bouchaud, J.-P., Bonart, J., Donier, J., Gould, M. (2018). *Trades, Quotes and Prices: Financial Markets Under the Microscope*. Cambridge University Press. — slippage realista em BTC market orders cap.5.

10. Mlfinpy docs (https://mlfinpy.readthedocs.io) — `PurgedKFold`, `seq_bootstrap`, `avg_uniqueness` — todas as ferramentas que o projeto declara usar mas não usa em walk-forward.

---

## Apêndice: ordem recomendada de ação

1. **Imediato (1 dia):** rodar os 3 testes em `proposals/red_team_tests.py`. Resultado esperado: shuffle Sharpe ≈ 0; time-reversed Sharpe baixo ou negativo; noise feature importance < 0.01.
2. **Antes de operar real:** subir COST para 0.0015 mínimo e refazer dual-horizon backtest. Se Sharpe cair abaixo de 0.7, NÃO operar.
3. **Médio prazo (semana):** implementar `avg_uniqueness` weighting (LdP cap.4) — pouco trabalho, grande impacto em honestidade.
4. **Antes do próximo commit "+47%":** dividir histórico em VAL (≤2024) e HOLDOUT (2025+). Selecionar ensemble + threshold + filtro bear APENAS em VAL. Reportar HOLDOUT sem retoque.
