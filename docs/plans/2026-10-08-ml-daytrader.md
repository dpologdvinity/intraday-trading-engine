# ML Day Trader Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A daily model that picks which symbols to trade and a minute model that times one long round trip per symbol per day, trained as many variants and judged against holding on unseen years.

**Architecture:** New package `hft/ml/` (universe, download, features, datasets, fills, models, train, evaluate). Data is downloaded read-only from Alpaca into `data/ml/`; features and windows are built lazily from compact per-symbol arrays so the minute data fits in memory. Training runs from one entry point on CPU (small configs) or a Kaggle GPU (full search).

**Tech Stack:** Python 3.12, NumPy, PyArrow, PyTorch 2.14 (CPU locally, CUDA on Kaggle), LightGBM (new, pinned), pytest, ruff.

**Spec:** `docs/specs/2026-10-08-ml-daytrader-design.md`

## Global Constraints

- Local runs: `nice -n 19`, one core (`torch.set_num_threads(1)`, LightGBM `num_threads=1`), focused test files; RSS under 2.5 GiB.
- Long only; at most one round trip per symbol per day; flat by 15:55 ET; fills at the next minute's open.
- Costs per side: half-spread 1.5 bp (stocks in the original 46 and the 8 ETFs), 4 bp (all added stocks), plus 1 bp slippage; also report 3x.
- Splits: train 2016-01-01..2022-12-31, validation 2023-01-01..2024-12-31, final test 2025-01-01..latest (run once, after freeze).
- Never read `hft.training.reserved_final_sessions()` sessions; never mention AI tools in tracked files; conventional commits.
- Universe: 69 symbols per the spec, with first usable dates (NBIS 2024-10-21; SPCX trade-only).

## Review Focus

1. Split, half-day and holiday sessions: minute features and fills must handle 13:00 closes (flat 5 minutes before the session's close, not 15:55). Test in Task 7.
2. Symbols that start mid-history (DELL, PLTR, SNDK, CRWV, NBIS): no windows or labels before their first usable date. Test in Task 4.
3. Missing minutes inside a session (no trades in a minute): carry no fake prices; the feature for a missing minute is a zero return with zero volume, and a fill needs a real bar. Test in Task 6.
4. Splits and dividends across the train/validation boundary: labels never span two splits. Test in Task 5.
5. Download interrupted mid-symbol: rerunning resumes without duplicating or dropping bars. Test in Task 2.

---

### Task 1: Universe and dependencies

**Files:** Create `hft/ml/__init__.py`, `hft/ml/universe.py`; modify `requirements-train.txt`, `constraints.txt`; test `tests/test_ml_universe.py`.

**Interfaces:** Produces `Symbol(ticker: str, kind: Literal["stock","etf"], first_date: date, trade_only: bool, half_spread_bps: float)`, `UNIVERSE: tuple[Symbol, ...]`, `tradable(day: date) -> list[Symbol]`.

- [ ] Test: 69 unique tickers, 8 ETFs; NBIS first_date 2024-10-21; SPCX trade_only; DELL/PLTR/SNDK/CRWV first dates 2018-12-28/2020-09-30/2025-02-13/2025-03-28; added stocks half_spread 4.0, others 1.5; `tradable(date(2020,1,2))` excludes PLTR.
- [ ] Implement; pin `lightgbm` and `scipy` (latest wheels for Python 3.12) in requirements-train.txt and constraints.txt; install into .venv.
- [ ] Run tests, ruff; commit `feat(ml): define the 69-symbol trading universe`.

### Task 2: Resumable bar download

**Files:** Create `hft/ml/download.py`; test `tests/test_ml_download.py`.

**Interfaces:** Consumes `hft.history.ReadOnlyClient`, `DATA_BASE`. Produces `download_bars(symbol, timeframe: Literal["1Day","1Min"], start: date, end: date, root: Path, *, client=None) -> list[Path]` writing `root/<timeframe>/<SYMBOL>/<YEAR>.parquet` (columns `t` int64 UTC ns, `o,h,l,c,v,vw` float64, `n` int64), regular hours only for 1Min (09:30-16:00 ET, early closes via the bar timestamps), a year file written atomically only when its pages are complete; `read_bars(root, timeframe, symbol) -> pyarrow.Table` sorted by `t`.

- [ ] Tests with a fake transport: pagination via `next_page_token`; extended-hours minutes dropped; rerun after a simulated failure in year 2 keeps year 1, completes year 2, no duplicate timestamps (Review Focus 5); timestamps strictly increasing.
- [ ] Implement (adjustment `all`, feed `sip`, limit 10000).
- [ ] Run tests, ruff; commit `feat(ml): download daily and minute bars resumably`.

### Task 3: Download the data (operational)

- [ ] `nice -n 19 .venv/bin/python -m hft.ml.download --timeframe 1Day` then `--timeframe 1Min` for the universe from each symbol's first date (run in background; record row counts and duration in the project history).

### Task 4: Daily features and labels

**Files:** Create `hft/ml/features.py`; test `tests/test_ml_features.py`.

**Interfaces:** Produces `DAILY_FEATURES: tuple[str, ...]` and `daily_frame(root: Path, symbols) -> DailyFrame` with arrays `dates (D,)`, `symbols (S,)`, `x (D,S,F) float32` (NaN where unavailable), `label (D,S) float32` = log(close/open) of that day, `valid (D,S) bool`. Features for day d use closes up to d-1 plus day d's open: returns 1/5/20/60d, 20d realized vol, gap log(open_d/close_{d-1}), volume/20d mean volume (d-1), body and wicks of d-1, the same returns for SPY and QQQ, weekday one-hot, short-history flag (< 250 sessions).

- [ ] Tests: changing any bar after day d (including its close) leaves `x[d]` unchanged (causality); `valid` false before each symbol's first_date (Review Focus 2); label equals log(c/o).
- [ ] Implement; run tests; commit `feat(ml): build causal daily features and labels`.

### Task 5: Splits and dataset manifest

**Files:** Create `hft/ml/datasets.py`; test `tests/test_ml_datasets.py`.

**Interfaces:** Produces `SPLITS: dict[str, tuple[date, date]]`, `split_mask(dates, name) -> np.ndarray`, `write_dataset(frame, path) -> dict` (npz + manifest with feature names, symbols, date range, SHA-256), `load_dataset(path) -> DailyFrame` that refuses a fingerprint mismatch.

- [ ] Tests: masks are disjoint and cover each split exactly; a label on the last training day never uses a validation-day price (Review Focus 4); tampered array raises.
- [ ] Implement; commit `feat(ml): split and fingerprint datasets`.

### Task 6: Fill simulator

**Files:** Create `hft/ml/fills.py`; test `tests/test_ml_fills.py`.

**Interfaces:** Produces `simulate_day(minute_open: np.ndarray, minute_valid: np.ndarray, signal: np.ndarray, *, enter: float, half_spread_bps: float, last_minute: int) -> Trade | None` where `signal[t]` is the prediction known at the end of minute t; entry at the first t with `signal[t] > enter`, filled at `minute_open[t+1]` x (1 + cost); exit at the first later t with `signal[t] < 0` or t == last_minute, filled at the next valid open x (1 - cost); one round trip; `Trade(entry_minute, exit_minute, net_return)`. Cost = (half_spread_bps + 1) / 1e4.

- [ ] Tests: never fills at the decision minute's own price; skips missing minutes for fills (Review Focus 3); exits by last_minute; at most one trade; net return includes both costs.
- [ ] Implement; commit `feat(ml): simulate one causal round trip per day`.

### Task 7: Minute arrays and windows

**Files:** Extend `hft/ml/features.py`; test `tests/test_ml_minute.py`.

**Interfaces:** Produces `minute_store(root, symbols) -> MinuteStore` holding per symbol per session arrays `open, close, volume (390,) float32` with a `valid` mask and `last_minute` per session (early closes, Review Focus 1); `windows(store, keys, lookback=60, horizon=H) -> (x (N,60,4), y (N,))` with channels log return, log relative volume, distance from session VWAP, minutes since open / 390; `y` = log(open[t+1+H]/open[t+1]).

- [ ] Tests: window for minute t never reads minute > t; half-day sessions end at the correct minute; missing minutes give zero return and zero volume.
- [ ] Implement; commit `feat(ml): build causal minute windows lazily`.

### Task 8: Models

**Files:** Create `hft/ml/models.py`; test `tests/test_ml_models.py`.

**Interfaces:** Produces `DailyMLP(features: int, width: int)`, `train_lightgbm(x, y, params) -> Booster`, `MinuteCNN(channels=4, width)`, `MinuteGRU`, `MinuteTransformer`; all torch modules map their input batch to a prediction of shape (N,).

- [ ] Tests: output shapes; a model trained on a planted signal (y = x[...,0] at lag 1) beats zero; deterministic with a fixed seed.
- [ ] Implement; commit `feat(ml): add daily and minute models`.

### Task 9: Training and evaluation harness

**Files:** Create `hft/ml/train.py`, `hft/ml/evaluate.py`; test `tests/test_ml_train.py`.

**Interfaces:** `run_variant(config: dict, data_root: Path, out: Path) -> dict` trains on `train`, scores on `validation` by simulated trading (daily top-K selection, minute model timing via Task 6) and writes `out/<variant_id>/` (model, config, metrics); `evaluate.report(trades, holding) -> dict` with total, annualized, per-trade, hit rate, max drawdown, Sharpe, trades, holding comparison and a day-resampled bootstrap 95% interval of the difference; `deflated_sharpe(sharpe, trials, n) -> float`. CLI: `python -m hft.ml.train --config cfg.json --data data/ml --out artifacts/ml`.

- [ ] Tests: tiny end-to-end on synthetic bars (2 symbols, 2 years) runs in under 60 s on one core; refuses the final-test split unless `--final` with a frozen variant list; holding benchmark equals equal-weight buy at split start.
- [ ] Implement; commit `feat(ml): train and evaluate variants against holding`.

### Task 10: Variant search and Kaggle packaging (Phase 3, task level)

- Search config grid (~50 variants per track); `python -m hft.ml.search`; run log with every variant.
- `hft/ml/package.py` builds the Kaggle dataset (features + repository wheel); `notebooks/kaggle_train.ipynb`.
- Local smoke search with 3 variants before any GPU run.

### Later phases (task level)

- Phase 4: `ml-daytrader` paper strategy (daily ranking before the open, minute model on 1-minute bars from the stream, existing budgets, risk and broker).
- Phase 5: PPO track on the minute environment; scaling to 300-500 symbols.
