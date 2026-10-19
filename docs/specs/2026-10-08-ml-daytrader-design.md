# ML day trader: design

Status: approved by the owner on 2026-10-08. Research agent (news reading) deferred.

## Goal and success criterion

Build a bot that does a day trader's job: each morning it picks which stocks to
trade, and during the day it decides minute by minute when to buy and sell them,
flat by the close. Neural networks and other models learn both decisions from
history, and many variants are trained in parallel and combined.

**Success** means the combined system beats simply holding the same symbols, after
costs, on data it never trained on or was selected on. Profit on training data
counts for nothing (see experiment 12 in [project history](https://github.com/dpologdvinity/intraday-trading-engine/blob/master/project-history.md)).
Paper trading after the final test is the only fully clean evidence. Real money
stays locked behind research qualification and 30 paper sessions.

## Constraints

- Cash-account trading: at most one round trip per stock per day, which also avoids
  the pattern day trader rule for accounts under $25,000. Long only, no leverage.
- $0 recurring cost: free Alpaca market data, free Kaggle GPU (about 30 hours a
  week), local CPU limited to one or two cores.
- Every feature uses only information available before its decision.
- Reserved final-test sessions (`hft.training.reserved_final_sessions()`) are
  never read.

## 1. Data and structure

- **Universe** (`hft/ml/universe.py`): 69 symbols, 61 stocks and 8 ETFs: the
  original 46 stocks; DELL, NBIS, LRCX, LITE, SNDK, PLTR, CRWV, ADI, AMAT, ANET,
  CMI, STX, TSM, WDC, SPCX; ETFs SPY, QQQ, IWM, DIA, VLUE, PDBC, SCHD, VOO. Each
  entry records stock or ETF and the first usable date. NBIS starts 2024-10-21
  (earlier data belongs to Yandex). DELL, PLTR, SNDK and CRWV start at listing.
  SPCX (listed 2026-06-12) is trade-only until it has a year of history.
- **Download** (`hft/ml/download.py`, local, read-only, resumable): Alpaca SIP daily
  bars and regular-hours 1-minute bars, 2016 to today, adjustment `all`, stored as
  Parquet per symbol per year under `data/ml/` (not tracked).
- **Features** (`hft/ml/features.py`), computed locally:
  - Daily, per symbol per day, as of the prior close plus the opening gap: returns
    over 1, 5, 20 and 60 days, realized volatility, overnight gap, volume relative
    to its 20-day average, candle body and wicks, the same for SPY and QQQ, day of
    week, and a short-history flag.
  - Minute, per symbol per minute: the last 60 one-minute returns and relative
    volumes, distance from the day's VWAP, minutes since the open, the day's
    opening gap, and the daily model's score.
  - A test for each feature set fails if any value depends on a later bar.
- **Datasets** (`hft/ml/datasets.py`): arrays packaged with a manifest (symbols,
  dates, feature names, split boundaries, SHA-256 fingerprint) for upload to Kaggle
  as one private dataset.

## 2. Models and training

### Track 1: prediction models plus a trading rule (main)

- **Daily model** predicts each symbol's open-to-close return. It picks up to K
  symbols (K from 3 to 5, chosen on validation) whose prediction exceeds the
  round-trip cost; otherwise it stays in cash. Model: an ensemble of gradient-boosted
  trees (LightGBM) and a small multilayer network.
- **Minute model** predicts the return over the next H minutes (H of 5, 15 or 30)
  for the selected symbols. Variants: 1-D convolutional network, GRU, small
  transformer.
- **Trading rule**: buy when the predicted gain exceeds cost plus a margin; sell
  when the prediction turns negative or at 15:55 ET; at most one round trip per
  symbol per day. The margin is tuned only on validation data.

### Track 2: reinforcement learning (comparison)

A PPO agent steps through each selected symbol minute by minute, observing the
minute features plus its position, choosing flat or long, rewarded with profit
after costs, under the same one-round-trip and flat-by-close rules.

### Parallel variants and ensembling

Each track trains about 50 variants (feature sets, horizons, model types, sizes,
margins). Each variant is scored by simulated trading with costs on the validation
years, not by prediction accuracy. The top five are averaged. Every variant tried is
logged so the final result can be discounted for the number of trials.

### Simulated fills (shared by training and evaluation)

Decisions use data up to the end of a minute; orders fill at the next minute's open.
Each side pays an estimated half-spread (1-2 bp for the largest stocks, 3-5 bp for
smaller ones and new listings) plus 1 bp slippage. Results are also reported at 3x
costs.

## 3. Evaluation and safeguards

- Splits: train 2016-2022, validation 2023-2024, final test 2025-01-01 to the latest
  complete session, evaluated once, after the training code and the variant choice
  are frozen and committed.
- Metrics on each split: total and annualized return, return per trade, hit rate,
  maximum drawdown, Sharpe ratio, number of trades, and the same for holding the
  symbols the model was allowed to trade (equal weight, bought at the start of the
  split). The pass test is the model's net return beating holding with a bootstrap
  95% interval (resampling days) above zero for the difference.
- The Sharpe ratio is deflated for the number of variants tried.
- Hindsight-bias note recorded in the results: many symbols were added because they
  rose in 2025-2026, which flatters any long strategy; holding them is the bar.
- Results go to `docs/ml-daytrader-results.md`, and each run to the project history.

## 4. Kaggle workflow and bot integration

- `hft/ml/train.py` runs any variant from a config file on CPU or GPU. The Kaggle
  notebook (`notebooks/kaggle_train.ipynb`) installs the repository from its
  dataset copy, loads the feature dataset, runs the variant search, and writes
  models and a run log. The owner uploads the datasets and runs it from their
  Kaggle account (phone-verified for GPU access).
- Local runs use the same entry point on small configurations to catch bugs before
  spending GPU hours.
- Trained models export to ONNX with a manifest (feature names, data fingerprint,
  variant configs, validation scores).
- New paper strategy `ml-daytrader`: before the open the daily model ranks the
  symbols from the latest daily features; during the day the minute model runs on
  one-minute bars built from the live stream; orders go through the existing
  per-stock budgets, risk gateway and broker. It runs on the paper account only.

## 5. Testing and error handling

- Unit tests: universe dates, feature causality (shuffling or removing future bars
  never changes past features), fill simulation timing and costs, one round trip
  per day, flat by close, split boundaries, manifest fingerprints.
- A tiny end-to-end run (two symbols, two years, a few epochs) in the test suite.
- Download retries with backoff, resumes partial files, and verifies row counts and
  timestamps; a gap is recorded, never filled with made-up prices.
- Training refuses data whose fingerprint does not match the manifest.

## Phases

1. Universe, download, daily features, daily model on CPU, evaluation harness.
2. Minute features, minute models, fill simulator, local small runs.
3. Kaggle packaging and the parallel variant search; frozen final test.
4. `ml-daytrader` paper strategy.
5. Reinforcement-learning track; scaling to about 300-500 symbols.

Deferred: research agent that reads news per industry.
