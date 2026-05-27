# Brief — Risk Manager / Position Sizing

Audit & framework. Sharpe atual 1.29 com sizing fixo "full" → telhado baixo. Existe ferramental de pesquisa pronto, falta engenheirar o uplift até produção.

---

## Diagnóstico (estado atual)

`pipeline/positions.py` é uma state-machine de barreiras tripla (`open`/`closed_target`/`closed_stop`/`closed_timeout`) mas **não persiste tamanho** — `open_position()` grava `entry_price`, `atr`, `target`, `stop`, `proba_long`; **não há `size_usd`, `size_btc`, `risk_dollars`**. PnL em `close_position()` calcula `pnl_pct = exit/entry - 1 - cost_round`: é retorno do ativo, não do book. Equivalente a *full notional, 1× leverage, 0% reserve*.

`pipeline/model.py` tem `position_size(capital, entry, stop, risk_pct=1%, max_pct=50%)` implementada e validada em `exp_position_sizing.py` (RISK-1PCT teve melhor Calmar entre 5 esquemas) — mas a função **não é chamada por `monitor_positions.py` nem por `predict_now.py`**. Código órfão.

Faltam três camadas inteiras: (1) **vol-targeting** — sizing usa `entry-stop` (≈ 3×ATR) como única medida de risco; ignora vol realizada de horizonte curto (`rv_1d`, `rv_1w` já estão no feature set); (2) **regime-conditional**: `model.py:113` tem flag `in_bear` (suprime sinal) mas é binário on/off; não há `size_multiplier(regime)` apesar de `exp_regime_analysis.py` mostrar Sharpe BULL >> CHOP >> BEAR; (3) **drawdown-conditional**: kill-switch do ROADMAP é manual ("Sharpe 90d < 0.3 por 4 semanas") — não há feedback dinâmico de equity peak para `size`. Resultado: vol-targeting + half-Kelly + regime gating dariam aproximação Grossman-Zhou/Carver completa. **Telhado realista 1.6–1.9 Sharpe** sem mexer no sinal.

---

## Top-3 Gaps

| # | Gap | Lift esperado | Custo eng. |
|---|---|---|---|
| **1** | Sizing fixo "full notional" em vez de **vol-targeting** (Carver). Trades em vol alta (≈4-5% diário, mar/2024) recebem mesmo size que vol baixa (≈1.5%, set/2023). | **Sharpe +0.20-0.35** (Moskowitz-Ooi-Pedersen 2012: vol scaling explica boa parte do alfa do TSMOM); **MaxDD -25-40% relativo** ao full. Em literatura crypto recente Sharpe sobe ~30%. | Baixo — usar `rv_1d` * `sqrt(365)` como vol anualizada; alvo 25% vol anual (~1.3× vol BTC long-run de ~60%, deslevera). |
| **2** | Sem **regime-conditional size**. `exp_regime_analysis.py` revela Sharpe BULL e CHOP positivos, BEAR negativo — mas filtro é binário (suprime). Em CHOP, paga reduzir size pra metade em vez de operar full ou zero. | **Sharpe +0.10-0.20**, **Calmar +0.3-0.5**. Sinclair (Positional Option Trading, cap 9): Kelly com `p` mal estimado → meio-Kelly; o mesmo lógica em regime — quando a distribuição de retornos é incerta, reduzir size. | Médio — precisa de detector de regime ao vivo (BTC return 30d, já está em `model.py:108`); mapeamento `regime → multiplier`. |
| **3** | **Drawdown-conditional sizing ausente**. Kill-switch atual é binário e lento (4 semanas). Grossman-Zhou (1993): size ótima sob constraint de drawdown é proporcional ao *surplus* `W_t − α·max_W`. Hoje, dois meses ruins → mesma alavancagem do dia 1. | **MaxDD -10-15 pp absoluto** (de ~30% para ~17%, faixa "psicologicamente operável" do `exp_drawdown_analysis.py`); Sharpe marginal +0.05; **Calmar +0.5-0.8**. | Baixo — ler equity curve do `data/backtest_equity.parquet` ou recomputar de `positions.parquet` (closed trades). |

**Soma teórica**: Sharpe 1.29 → 1.6-1.9. MaxDD reduz ~40-50% relativo. Calmar dobra.

---

## Experimento concreto: sizing dinâmico

### Fórmula composta (multiplicadores ortogonais, multiplicativos, capped)

```
size_pct = clamp(
    f_vol  *  f_kelly  *  f_regime  *  f_dd  *  f_conf,
    0.0,
    SIZE_MAX
)
```

`SIZE_MAX = 1.0` (sem alavancagem, política do projeto). Cada fator é um multiplicador ∈ [0, ≈2].

#### 1) `f_vol` — Volatility targeting (Carver)
```
realized_vol_ann = rv_1d * sqrt(365)        # já em features
target_vol_ann   = 0.25                      # 25%/ano — abaixo do BTC LR ~0.60
f_vol            = target_vol_ann / max(realized_vol_ann, 0.10)
f_vol            = clamp(f_vol, 0.10, 1.50)  # piso/teto sanidade
```
Em vol alta (rv_1d ≈ 4%/dia → 76% anual): f_vol ≈ 0.33. Vol baixa (1.5%/dia → 29%): f_vol ≈ 0.86. Carver, *Systematic Trading* cap 9 — "position size ∝ target / instrument_risk". Cap em 1.5 evita alavancar em "vol-of-vol" baixa enganosa.

#### 2) `f_kelly` — Half-Kelly base de credibilidade
```
p     = signal_prob (proba_mid · proba_long, ou min)
b     = avg_win / avg_loss  do walk-forward histórico (≈ 1.0 no projeto: barreiras simétricas ±3 ATR)
kelly = (p*(1+b) - 1) / b
f_kelly = clamp(0.5 * kelly, 0.0, 1.0)        # half-Kelly
```
Half-Kelly: 75% do growth com ~50% da volatilidade (MacLean-Thorp-Ziemba 2010). Importante: `p` aqui é a `proba_long` do modelo, mas `p` "verdadeiro" é desconhecido — half-Kelly absorve a incerteza (Sinclair cap 9). Em `THRESHOLD=0.35`, p≈0.55 e b≈1: kelly ≈ 0.10 → f_kelly ≈ 0.05 — muito baixo. Usar como **piso de credibilidade** multiplicado, não como sizing absoluto: re-escalar `f_kelly_norm = f_kelly / 0.05` (referência), capped em [0.5, 1.5].

#### 3) `f_regime` — regime multiplier
```
ret_30d = close / close[-180_bars_4h] - 1     # já em model.py
regime  = "bull" if ret_30d >  0.05
         else "bear" if ret_30d < -0.05
         else "chop"
f_regime = {"bull": 1.20, "chop": 0.80, "bear": 0.00}[regime]
```
Calibrar com `exp_regime_analysis.py` (PnL/trade por regime). Hoje BEAR já é zero (filtro), mas via multiplier vira escalável e explicável.

#### 4) `f_dd` — drawdown-conditional (Grossman-Zhou simplified)
```
peak     = max(equity até agora)
dd       = (equity - peak) / peak             # ≤ 0
DD_FLOOR = -0.20                              # zero a -20%
f_dd     = clamp(1 + dd / abs(DD_FLOOR), 0.0, 1.0)
# dd=  0%   → f_dd = 1.0
# dd= -10%  → f_dd = 0.50
# dd= -20%  → f_dd = 0.0   (kill-switch suave)
```
Linear; Grossman-Zhou 1993 mostra que sob CRRA o ótimo é proporcional ao *surplus*. Versão linear é boa aproximação prática (He 2001, Carver *Leveraged Trading* cap 22 — "ratchet").

#### 5) `f_conf` — confiança do sinal (López de Prado cap 10)
```
m       = (p - THRESHOLD) / (1 - THRESHOLD)   # 0 no threshold, 1 em p=1.0
f_conf  = clamp(0.5 + m, 0.5, 1.5)
```
Reforça quando ambos modelos estão >> threshold. Evita binário on/off em `proba=0.351` vs `proba=0.95`.

### Gatilhos / kill-switches discretos (em cima do `size_pct` contínuo)

| Trigger | Ação |
|---|---|
| `f_regime == "bear"` | `size_pct = 0` (já era). |
| `equity_dd ≤ -20%` (DD_FLOOR) | `size_pct = 0` por 14 dias; só reativa após `equity > 0.95·peak`. |
| Sharpe rolling 90d < 0.3 por 4 semanas | Kill-switch ROADMAP (manual). |
| Streak ≥ 5 losses consecutivas | Halve size por 7 dias (cooling). `exp_drawdown_analysis.py` já flagga 7 como dor psicológica. |
| `realized_vol_ann > 1.50` (vol > 150%/ano) | `size_pct = 0` — black-swan filter; ex: maio/2021, nov/2022. |

### Integração em `pipeline/positions.py`

1. **Schema bump** — adicionar a `positions.parquet`:
   - `size_pct: float` (0.0–1.0)
   - `size_usd: float`
   - `size_btc: float`
   - `equity_at_open: float`
   - `regime_at_open: str`
   - `f_vol, f_kelly, f_regime, f_dd, f_conf: float` (auditoria)

2. **`open_position()`** chama `compute_position_size(...)` (em `proposals/dynamic_sizing_risk.py`) **antes** de gravar. Se `size_pct == 0`, **não abre** (`return None`).

3. **`close_position()`** — `pnl_pct` continua sendo retorno do ativo. Adicionar `pnl_dollars = size_usd * pnl_pct`. Equity = soma de `pnl_dollars`.

4. **`predict_now.py`** — ler equity atual de `positions.parquet` (sum de `pnl_dollars` em closed) antes de chamar sizing. Telegram passa a mostrar `size: X% capital` em vez de só `proba`.

5. **`monitor_positions.py`** — sem mudança (saídas já são corretas; só PnL em USD passa a depender de `size_usd`).

### Validação empírica (a rodar antes do merge)

Re-rodar `exp_backtest_1k.py` substituindo o sizing fixo pelo composto:
- Métricas alvo: Sharpe ≥ 1.6, MaxDD ≤ 22%, Calmar ≥ 1.4, % tempo underwater ≤ 50%.
- Se Sharpe cair < 1.4 → erro de implementação (vol-targeting deveria entregar 0.2+).
- A/B: comparar (a) só `f_vol`, (b) `f_vol·f_regime`, (c) composto inteiro — decomposição limpa do lift.

---

## Refs

1. Thorp, E. O. (1997/2006). *The Kelly Criterion in Blackjack, Sports Betting, and the Stock Market.* In *Handbook of Asset and Liability Management, Vol 1.* [PDF](https://gwern.net/doc/statistics/decision/2006-thorp.pdf)
2. MacLean, L. C., Thorp, E. O., & Ziemba, W. T. (2010). *Good and Bad Properties of the Kelly Criterion.* Quantitative Finance, 10(7), 681–687. [PDF](https://www.stat.berkeley.edu/~aldous/157/Papers/Good_Bad_Kelly.pdf)
3. Carver, R. (2015). *Systematic Trading: A unique new method for designing trading and investing systems.* Harriman House. Caps 9 (vol targeting), 11 (FDM/IDM). [Publisher](https://www.harriman-house.com/systematic-trading)
4. Carver, R. (2019). *Leveraged Trading.* Harriman House. Caps 10, 13, 22 (position adjustment / ratchet). [Goodreads](https://www.goodreads.com/book/show/48611879-leveraged-trading)
5. Moskowitz, T., Ooi, Y. H., & Pedersen, L. H. (2012). *Time Series Momentum.* Journal of Financial Economics, 104(2), 228–250. [PDF](http://docs.lhpedersen.com/TimeSeriesMomentum.pdf)
6. Kim, A. Y., et al. (2016). *Time Series Momentum and Volatility Scaling.* Journal of Financial Markets, 30, 103–124. [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S1386418116301379)
7. Grossman, S. J., & Zhou, Z. (1993). *Optimal Investment Strategies for Controlling Drawdowns.* Mathematical Finance, 3(3), 241–276. [Wiley](https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1467-9965.1993.tb00044.x)
8. Cvitanic, J., & Karatzas, I. (1995). *On Portfolio Optimization under Drawdown Constraints.* IMA Lecture Notes 65, 35–46.
9. Martin, P. G., & McCann, B. B. (1989). *The Investor's Guide to Fidelity Funds.* (Origem do Ulcer Index & UPI.) [Reference — Investopedia/StockCharts](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/ulcer-index)
10. Young, T. W. (1991). *Calmar Ratio: A Smoother Tool.* Futures Magazine, Oct 1991.
11. Sinclair, E. (2020). *Positional Option Trading: An Advanced Guide.* Wiley. Cap 9 (Kelly + estimation uncertainty + stop losses). [Wiley](https://www.wiley.com/en-us/Positional+Option+Trading%3A+An+Advanced+Guide-p-9781119583530)
12. López de Prado, M. (2018). *Advances in Financial Machine Learning.* Wiley. Cap 10 — Bet Sizing. [O'Reilly](https://www.oreilly.com/library/view/advances-in-financial/9781119482086/c10.xhtml)
13. He, G. (2001). *Drawdown-Controlled Optimal Portfolio Selection with Linear Constraints on Portfolio Weights.* SSRN 288321. [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=288321)
14. Hocquard, A., Ng, S., & Papageorgiou, N. (2013). *A Constant-Volatility Framework for Managing Tail Risk.* Journal of Portfolio Management, 39(2), 28–40.
15. Yang, Z. G., & Zhong, L. (2012). *Optimal Portfolio Strategy to Control Maximum Drawdown — The Case of Risk Based Dynamic Asset Allocation.* SSRN 2053854. [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2053854)
16. (Crypto vol-targeting) Bianchi, D., & Babiak, M. (2024). *Vol-Managed Cryptocurrency Portfolios.* Journal of Empirical Finance (relacionado: [arXiv survey 2025](https://arxiv.org/html/2510.14435v4)).
