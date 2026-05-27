# Validação estatística — auditoria do Sharpe 1.29 (dual-horizon AND)

Auditor: quant pesquisa (linha López de Prado / Bailey / Harvey-Liu).
Alvo: commit `bff285c` — sinal intraday 4h BTC, LightGBM walk-forward, dual-horizon `mid h=12 AND long h=18` @ thr 0.35.

---

## Diagnóstico do Sharpe 1.29 sob lente DSR/PBO

Sharpe 1.29 é **point estimate sem intervalo, num único path, escolhido depois de ≥10 experimentos correlacionados**. A literatura (Bailey & López de Prado 2014; Harvey & Liu 2015) é unânime: nessas condições o haircut esperado é grande e o número raramente sobrevive.

Estimativa rápida — pool ≈ 3.4 anos × ~26 trades/ano (visto em `exp_threshold_grid`) ≈ N≈90 trades. Variância em torno do Sharpe estimado V[SR̂]≈(1+0.5·SR²)/(N-1) ≈ 0.018 → desvio ≈ 0.13 anualizado num grid 5×5 thr. Com K=25 trials só no grid de thr e correlação alta entre elas, o **Expected Max SR sob hipótese nula** já bate 0.4–0.6. Somando hyperopt (30 trials, `07_hyperopt.py`), 3 horizontes, 5 regras de ensemble, 6 experimentos exp_* extras → **K_efetivo entre 50 e 200**. Haircut de Harvey-Liu nessa zona corta 40–70% do t-stat. **Probabilidade de o Sharpe deflacionado (DSR) cruzar o threshold de 0.95 é baixa — provavelmente DSR ≤ 0.5 e PBO > 0.5.** Sobrevive? Sob critério rigoroso, não. Sob critério "PSR(0) > 0.95" (true SR > 0), provavelmente sim — mas isso não é o que o ROADMAP pede.

---

## Top-3 Gaps

**1. Multiple testing nunca foi tratado (gap mais grave).**
Evidência no repo: `exp_threshold_grid.py` varre 25 combos sobre o mesmo pool; `07_hyperopt.py` roda 30 trials Optuna otimizando Sharpe direto na VAL; `exp_ensemble.py` testa 5 regras; `exp_multi_horizon.py` testa 3 horizontes × 3 regras de voto; mais `exp_position_sizing`, `exp_regime_analysis`, `exp_drawdown_analysis`, `exp_ema200_veto`, `exp_wick_filter`, `exp_asym_barriers`. **N_trials_efetivo ≥ 50.** A literatura padrão (López de Prado, "10 Reasons Most ML Funds Fail", JPM 2018; Harvey-Liu "Backtesting", JPM 2015) classifica isso como overfitting de seleção. Nenhum dos arquivos aplica Bonferroni, Holm, BHY, Romano-Wolf stepdown ou White's Reality Check.

**2. Ausência de Deflated Sharpe Ratio e Probabilistic Sharpe Ratio.**
Em todo o repo, Sharpe é calculado como `mean/std * sqrt(trades_per_year)` (ver `exp_ensemble.py:201`, `exp_threshold_grid.py:202`, `exp_multi_horizon.py:206`). Não há ajuste por skew/kurtose (PSR — Bailey & López de Prado 2012) nem por nº de trials (DSR — Bailey & López de Prado 2014). Sem isso, retornos com skew negativo e cauda gorda (típico de BTC pós-stop) **inflam Sharpe sistematicamente**. mlfinlab implementa ambos (`backtest_overfitting/seven_reasons`); `mlfinpy` (já no stack) tem CPCV mas não DSR — gap precisa ser preenchido.

**3. CV inadequado: walk-forward expanding quarterly = single path; faltam CPCV, embargo dimensionado, PBO via CSCV.**
`05_model.py:99` e todos os exp_* usam `train_end = test_start - HORIZON_BARS`. **Purge correto, mas embargo nominal de 12–18 bars (48–72h) é exatamente o horizonte do label — sem folga adicional**. López de Prado (AFML cap. 7) recomenda embargo ≥ horizon × (1+α) com α≈0.01 da amostra. Mais grave: walk-forward quarterly gera **1 único path de PnL**. CPCV (López de Prado 2018; mlfinlab `cross_validation/combinatorial.py`) gera φ[N,k] paths combinatórios → permite estimar **distribuição do Sharpe** em vez de point estimate. Sem distribuição, não dá pra computar PBO (Bailey, Borwein, López de Prado & Zhu 2017) nem aplicar Romano-Wolf nos exp_*.

---

## Experimento concreto: protocolo de validação a aplicar no projeto

Passo a passo, ordem importa.

**0. Congelar a research path (pre-registration).**
Antes de mexer em qualquer coisa: criar `research_log.md` listando os ≥10 experimentos já rodados, params variados, métrica usada na decisão. Esse é o `K` que vai entrar no DSR. **Sem esse número honesto, DSR fica subestimado e a auditoria perde credibilidade.**

**1. Reimplementar Sharpe com IC bootstrap estacionário.**
Substituir todo `sharpe = mean/std * sqrt(trades_per_year)` por:
- Bootstrap estacionário (Politis & Romano 1994, lib `arch.bootstrap.StationaryBootstrap`, block_size ≈ √N) sobre os retornos por trade.
- 5000 reamostragens → IC 95% para Sharpe.
- **Critério**: Sharpe é "real" se limite inferior do IC 95% > 0.5.

**2. Computar PSR(SR* = 0) e PSR(SR* = 1.0).**
Formula Bailey-López de Prado 2012:
`PSR(SR*) = Φ( (SR̂ - SR*) · √(n-1) / √(1 - γ₃·SR̂ + (γ₄-1)/4 · SR̂²) )`
onde γ₃, γ₄ são skew/kurtose dos retornos por trade.
**Critério**: PSR(0) > 0.95 (rejeita SR=0). Bônus: PSR(1) > 0.5.

**3. Computar Deflated Sharpe Ratio com K honesto.**
- K = nº de configurações testadas (≥50 estimado).
- V[SR̂] = variância dos Sharpes observados entre as configurações já rodadas.
- `SR_0 = √V[SR̂] · ((1-γ_em)·Φ⁻¹(1-1/K) + γ_em·Φ⁻¹(1-1/(K·e)))` (γ_em = Euler-Mascheroni ≈ 0.5772).
- `DSR = PSR(SR_0)`.
- **Critério**: DSR > 0.95.

**4. Migrar walk-forward para CPCV via `mlfinpy`.**
- N=10 grupos sequenciais, k=2 grupos de teste → C(10,2)=45 splits → φ[10,2] ≈ 9 backtest paths.
- Purge: remover do treino observações com label_endtime sobrepondo teste.
- Embargo: bars adicionais após cada teste (default 1% da amostra, ~30 bars no projeto).
- **Output**: 9 Sharpes (distribuição) em vez de 1.

**5. PBO via CSCV (combinatorially symmetric cross-validation).**
- Bailey-Borwein-López de Prado-Zhu 2017.
- Divide T observações em S=16 sub-grupos; para cada combinação de S/2 IS vs S/2 OOS (C(16,8) = 12870 splits), rankeia configurações em IS e mede performance OOS daquela #1 IS.
- **PBO = P(rank_OOS da melhor IS estar na metade inferior)**.
- **Critério**: PBO < 0.5 (idealmente < 0.3).

**6. Aplicar Romano-Wolf stepdown nos exp_*.**
Tratar cada `exp_*.py` como hipótese H_i : "estratégia i bate baseline B&H". Bootstrap conjunto (mesma reamostragem para todas), stepdown sequencial controlando FWER 5%. **Quais experimentos sobrevivem é a resposta honesta sobre quantas descobertas reais existem.** Implementação: `arch.bootstrap.StepM` ou Hansen SPA test.

**7. Conformal prediction para tamanho de posição.**
- Split conformal sobre as probabilidades calibradas do LightGBM (Angelopoulos & Bates 2023 cap. 2).
- Conformal pi para retornos (calibration set = último trimestre OOS).
- Em vez de threshold fixo 0.35, **abster do trade quando intervalo conformal inclui zero ou negativo** (Kato 2024, Conformal Predictive Portfolio Selection).

**8. Critério de morte estatístico explícito (atualizar ROADMAP §2).**
Atual: "Sharpe rolling 90d < 0.3 por 4 sem". Adicionar:
- DSR < 0.6 em qualquer relabel mensal.
- PBO > 0.5 no re-run de CSCV trimestral.
- Median Sharpe dos paths CPCV < 0.5.

**9. Documentar K antes de cada novo experimento.**
Toda nova `exp_*.py` incrementa o contador `K_trials` em `research_log.md`. Antes de rodar, computar haircut esperado (Harvey-Liu): se mesmo no upside o Sharpe pós-haircut não bate 0.7, **não rode**.

---

## Refs (15 — todas reais e verificáveis)

1. Bailey, D. H., & López de Prado, M. (2014). *The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting and Non-Normality*. Journal of Portfolio Management 40(5), 94–107. SSRN 2460551.
2. Bailey, D. H., & López de Prado, M. (2012). *The Sharpe Ratio Efficient Frontier*. Journal of Risk 15(2). SSRN 1821643. (Probabilistic Sharpe Ratio.)
3. Bailey, D. H., Borwein, J. M., López de Prado, M., & Zhu, Q. J. (2017). *The Probability of Backtest Overfitting*. Journal of Computational Finance 20(4), 39–69. SSRN 2326253.
4. López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley. Caps. 4 (sampling/purging), 7 (cross-validation in finance), 11 (backtest statistics), 14 (combinatorial backtesting).
5. López de Prado, M. (2018). *The 10 Reasons Most Machine Learning Funds Fail*. Journal of Portfolio Management 44(6), 120–133. (Versão estendida do talk QuantCon 2018 "7 Reasons".) SSRN 3104847.
6. López de Prado, M. (2018). *A Practical Solution to the Multiple-Testing Crisis in Financial Research*. Journal of Financial Data Science 1(1).
7. Harvey, C. R., & Liu, Y. (2015). *Backtesting*. Journal of Portfolio Management 42(1), 13–28. SSRN 2345489. (Haircut Sharpe Ratio.)
8. Harvey, C. R., Liu, Y., & Zhu, H. (2016). *…and the Cross-Section of Expected Returns*. Review of Financial Studies 29(1), 5–68. (316 fatores; multiple testing em finance.)
9. White, H. (2000). *A Reality Check for Data Snooping*. Econometrica 68(5), 1097–1126.
10. Romano, J. P., & Wolf, M. (2005). *Stepwise Multiple Testing as Formalized Data Snooping*. Econometrica 73(4), 1237–1282.
11. Hansen, P. R. (2005). *A Test for Superior Predictive Ability*. Journal of Business & Economic Statistics 23(4), 365–380. (SPA test, refina White's Reality Check.)
12. Politis, D. N., & Romano, J. P. (1994). *The Stationary Bootstrap*. Journal of the American Statistical Association 89(428), 1303–1313.
13. Ledoit, O., & Wolf, M. (2008). *Robust Performance Hypothesis Testing with the Sharpe Ratio*. Journal of Empirical Finance 15(5), 850–859. (Bootstrap IC para diferença de Sharpes.)
14. Angelopoulos, A. N., & Bates, S. (2023). *Conformal Prediction: A Gentle Introduction*. Foundations and Trends in Machine Learning 16(4), 494–591. arXiv 2107.07511.
15. Kato, M. (2024). *Conformal Predictive Portfolio Selection*. arXiv 2410.16333.

Implementações de referência:
- mlfinlab (Hudson & Thames): `cross_validation/combinatorial.py`, `backtest_statistics/backtests.py` (DSR, PSR, haircut). https://github.com/hudson-and-thames/mlfinlab
- mlfinpy (baobach) — já no stack: `https://github.com/baobach/mlfinpy` — tem CPCV; **falta DSR/PSR (adicionar).**
- rubenbriones/Probabilistic-Sharpe-Ratio (PSR + DSR Python puro).
- `arch.bootstrap.StationaryBootstrap` e `StepM` (Sheppard).
- aangelopoulos/conformal-prediction (notebooks oficiais do tutorial).
