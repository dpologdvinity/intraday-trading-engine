# Reinforcement-learning trader: design and pre-registration

Status: draft for owner approval. Written before any reinforcement-learning result on
real data. Once approved and committed, the search space, splits, costs and pass rule
below are fixed; every configuration trained counts as a trial.

## Goal

Give reinforcement learning a fair, serious try at beating simply holding the same
stocks, after costs, on data it never trained on or was selected on. Two tracks: an
intraday trader that is flat every night (the owner's day-trading goal), and a daily
trader that may hold for days, where published reinforcement-learning results are
strongest.

## What the evidence says (research, 2026-11-11)

- The best-known wins are slow: deep RL on 50 daily futures (Zhang, Zohren and
  Roberts 2020, [arXiv 1911.10107](https://arxiv.org/abs/1911.10107)) and TDQN on 30
  daily stocks (Théate and Ernst 2021, [arXiv 2004.06627](https://arxiv.org/abs/2004.06627),
  average Sharpe 0.40 against 0.37 for buy-and-hold). They survive costs mostly by
  trading rarely.
- Intraday equity evidence is weak: a 2025 review finds overfitting, little
  out-of-sample testing and lagging buy-and-hold in rising markets
  ([MDPI](https://www.mdpi.com/2076-3417/16/20/10060)); a 2026 survey finds 86% of
  papers claim to beat a benchmark without a statistical test and 90% model costs
  unrealistically ([SSRN 7363738](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7363738)).
  FinRL's headline results use hindsight stock lists
  ([FinRL PR 1462](https://github.com/AI4Finance-Foundation/FinRL/pull/1462)).
- What RL can add over prediction models: it handles costs that depend on its own
  position, and it can maximize risk-adjusted return directly. A differential Sharpe
  ratio reward gave more consistent agents than raw profit (Moody and Saffell 1998,
  [NeurIPS](http://papers.neurips.cc/paper/1551-reinforcement-learning-for-trading.pdf)).
- Overfitting is the main failure; the probability of backtest overfitting
  ([Bailey et al.](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2326253)) and the
  deflated Sharpe ratio measure it.
- Retail day traders: in Brazil 97% of those trading 300+ days lost money and 1.1%
  earned more than the minimum wage ([SSRN 3423101](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3423101));
  in Taiwan fewer than 1% are predictably profitable
  ([J. Financial Markets 2014](https://www.sciencedirect.com/science/article/abs/pii/S1386418113000190)).

The design follows the evidence: fewer, larger decisions; a risk-adjusted reward;
costs randomized in training; strict overfitting checks; strong baselines.

## 1. Data (no new download)

The 68 tradable symbols; SIP 1-minute bars and daily bars from 2016
(`data/ml/1Min`, `data/ml/1Day`); SPY and QQQ as market context; the exchange
calendar; the nine pre-open news features (`hft/ml/news_features.py`). All inputs are
causal; a test fails if any value uses a later bar.

## 2. Track A: intraday, flat every night

- **Episode:** one symbol on one session. Decisions every 5 minutes (78 a day), each
  filling at the next minute's open as in `hft/ml/fills.py`.
- **Observation:** the last 60 minutes of the five minute channels (return, relative
  volume, distance from VWAP, time of day, opening gap) for the symbol and for SPY and
  QQQ; optionally the 23 daily features and 9 news features; position, minutes held,
  unrealized return, minutes to the close, whether today's round trip is used.
- **Actions:** flat or long. Rules enforced by action masks: at most one round trip
  per symbol per day (a cash account cannot reuse unsettled proceeds the same day);
  hold at least 15 minutes once in; no entries in the last 15 minutes; flat by the
  last minute.

## 3. Track B: daily, may hold for days

- **Episode:** one symbol over 60 consecutive sessions. A decision at each session's
  open fills at that open (with the open's half-spread); a position may be held
  overnight for up to 20 sessions.
- **Observation:** the last 60 sessions of daily returns, ranges and relative volume
  for the symbol, SPY and QQQ; optionally the news features; position, sessions held,
  unrealized return.
- **Actions:** flat or long; hold at least 2 sessions once in.

## 4. Shared rules

- **Costs:** each side pays the symbol's half-spread (at least half a cent) plus 1 bp
  slippage, as `hft/ml/fills.py`. In training each episode draws a cost multiple
  uniformly from 1x to 3x, so the agent must earn its trades under uncertain costs.
  Evaluation reports 1x and 3x.
- **Reward variants:** (a) net log return per step; (b) the differential Sharpe
  ratio (Moody and Saffell), an online, per-step version of the Sharpe ratio.
- **Parity test:** for random action sequences, the environment's returns equal those
  of `fills.simulate_day` (track A) and of a direct open-to-open calculation (track
  B), and evaluation uses the training code path.

## 5. Agent and training

- PPO with generalized advantage estimation in PyTorch, CleanRL style, no new
  dependency; the environment is tensorized so 4,096 episodes step together on a
  Kaggle T4.
- Network: a 1-D convolution or GRU encoder over the window, concatenated with the
  other inputs, separate policy and value heads.
- Reward and advantage normalization, entropy annealing, gradient clipping,
  learning-rate decay; early stopping on the inner validation year.
- 50 million decisions per configuration for track A, 20 million for track B; three
  seeds per configuration, ensembled by averaging action probabilities; per-seed
  results reported.

## 6. Search space (16 configurations, fixed in advance)

| Choice | Values |
| --- | --- |
| Track | A (intraday), B (daily) |
| Encoder | 1-D convolution, GRU |
| Inputs | prices + market context; prices + market + daily (A) or + news (A and B) |
| Reward | net log return; differential Sharpe ratio |

2 x 2 x 2 x 2 = 16 configurations, counted as trials 109 to 124 after the 108 ML
variants. Nothing is added after results are seen.

## 7. Splits, baselines and pass rule

- **Train** 2016-2021; **inner validation** 2022 (early stopping, checkpoint choice);
  **validation** 2023-2024 (selection); **test** 2025 onward, once, only for a
  configuration that passes.
- **Portfolio for scoring:** equal slots in all 68 symbols; a slot's return is the
  policy's net return in that symbol (0 when flat).
- **Baselines the agent must beat on validation:** holding the same 68 symbols; buy at
  the open and sell at the close every day (track A) or hold every session (track B);
  random entries with the agent's own trade count and holding times; the frozen
  supervised news model (7f7b87ac489e); and always flat, as a sanity check.
- **Pass rule (validation):** edge over holding with a 95% interval (days resampled)
  above zero; beats every baseline; deflated Sharpe ratio over 124 trials at least
  0.95; probability of backtest overfitting below 0.2 (combinatorially symmetric
  cross-validation over the 16 configurations, 16 blocks of validation days);
  positive at 3x costs.
- **Robustness (reported):** the chosen policy scored on SPY, QQQ, IWM and DIA alone,
  which carry no hindsight from the stock list.
- **Test:** a passing configuration is frozen, committed and run once on 2025 onward
  (`.state/` registry); the hindsight note of the ML day trader applies. A test pass
  leads to forward paper trading before any real money.

## 8. Build and run

1. `hft/ml/rl_env.py`: tensor environments for both tracks, masks, parity and
   causality tests (local CPU, small subset).
2. `hft/ml/rl_train.py`: PPO, checkpoints, inner validation, ensemble evaluation with
   `hft/ml/evaluate.py`, baselines, PBO; CPU smoke run on two symbols.
3. `notebooks/kaggle_rl.ipynb`: the 16 configurations on a Kaggle T4, one result file
   per configuration so an interrupted session resumes (about 6-10 GPU hours).
4. Results in `docs/ml-daytrader-results.md` and `docs/project-history.md`.

## Not in scope

Order-book data (paid, and it mostly explains concurrent rather than future moves),
offline RL and decision transformers (no evidence of gains here), earnings-date
inputs (no verified free history; a later pre-registration if obtained), synthetic
data augmentation.
