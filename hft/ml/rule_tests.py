"""Shared pieces of the pre-registered daily rule tests.

Used by `scripts/longer_horizons.py` (experiment 19) and `scripts/earnings_drift.py`
(experiment 22): per-side costs at the daily open, per-trade excess returns against
an equal-weight benchmark, bootstrap intervals and the year splits. Each experiment
pre-registers two rules, so intervals are at the Bonferroni-corrected 98.75% level.
"""

import numpy as np

SLIPPAGE_BPS = 1.0
LEVEL = 98.75  # two rules: Bonferroni-corrected two-sided interval
RESAMPLES, SEED = 10_000, 0
SPLITS = {"development": (2016, 2022), "holdout": (2023, 2024), "test": (2025, 2099)}


def side_costs(half_spreads, multiple=1.0):
    """Cost of one buy or one sell per symbol, as a fraction."""
    return (np.asarray(half_spreads, float) + SLIPPAGE_BPS) * multiple / 1e4


def drift(dates, opens, signal, costs, hold):
    """One record per trade: buy at the open of every row d where `signal[d, s]`, sell
    at the open of row d + hold. Records hold the entry date, symbol index, net return,
    the equal-weight benchmark over the same window (every symbol with both opens,
    without costs) and the excess. Trades without both opens are skipped."""
    trades = []
    for d, s in zip(*np.nonzero(signal), strict=True):
        if d + hold >= len(dates):
            continue
        entry, exit_ = opens[d], opens[d + hold]
        if not (np.isfinite(entry[s]) and np.isfinite(exit_[s])):
            continue
        both = np.isfinite(entry) & np.isfinite(exit_)
        benchmark = float(np.mean(exit_[both] / entry[both]) - 1.0)
        net = float(exit_[s] / entry[s] - 1.0) - 2 * costs[s]
        trades.append(
            {
                "entry": str(dates[d]),
                "symbol": int(s),
                "net": net,
                "benchmark": benchmark,
                "excess": net - benchmark,
            }
        )
    return trades


def block_interval(values, rng, block=3, level=LEVEL):
    """Moving-block bootstrap interval of the mean (blocks of consecutive values)."""
    values = np.asarray(values, float)
    n = len(values)
    if n < block:
        return [float("nan"), float("nan")]
    count = -(-n // block)
    starts = rng.integers(0, n - block + 1, size=(RESAMPLES, count))
    index = (starts[..., None] + np.arange(block)).reshape(RESAMPLES, -1)[:, :n]
    means = values[index].mean(axis=1)
    tail = (100 - level) / 2
    return [float(v) for v in np.percentile(means, [tail, 100 - tail])]


def cluster_interval(groups, values, rng, level=LEVEL):
    """Bootstrap interval of the mean per item, resampling whole groups."""
    values = np.asarray(values, float)
    if not len(values):
        return [float("nan"), float("nan")]
    _, inverse = np.unique(np.asarray(groups), return_inverse=True)
    sums, counts = np.bincount(inverse, weights=values), np.bincount(inverse)
    picks = rng.integers(0, len(sums), size=(RESAMPLES, len(sums)))
    means = sums[picks].sum(axis=1) / counts[picks].sum(axis=1)
    tail = (100 - level) / 2
    return [float(v) for v in np.percentile(means, [tail, 100 - tail])]


def annualized(daily):
    """Compound annual return of daily returns (252 sessions a year)."""
    daily = np.asarray(daily, float)
    return float(np.prod(1 + daily) ** (252 / max(len(daily), 1)) - 1) if len(daily) else 0.0


def summarize(name, values, interval):
    """Count, mean and interval in basis points, and the share of positive values."""
    values = np.asarray(values, float)
    return {
        "n": len(values),
        "mean_bp": float(values.mean() * 1e4) if len(values) else float("nan"),
        f"ci{LEVEL}_bp": [v * 1e4 for v in interval],
        "positive_share": float((values > 0).mean()) if len(values) else float("nan"),
        "statistic": name,
    }


def in_split(date_strings, split):
    """Mask of ISO dates whose year lies in the split."""
    low, high = SPLITS[split]
    years = np.array([int(d[:4]) for d in date_strings])
    return (years >= low) & (years <= high)
