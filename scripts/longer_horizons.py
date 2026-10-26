"""Evaluate the pre-registered longer-horizon rules.

Implements docs/specs/2026-10-25-longer-horizons-preregistration.md: rule 1 holds the
top 10 of 68 symbols by 12-1 momentum for a month; rule 2 holds pre-open analyst
upgrades for 20 sessions. Both trade at daily opens with the universe's half-spread
plus 1 bp slippage per side and are judged against the same symbols. Development
(2016-2022) and holdout (2023-2024) only; the test years run with --final, once,
for a rule that passed.

    .venv/bin/python scripts/longer_horizons.py --output artifacts/longer-horizons/holdout.json
"""

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TOP = 10  # rule 1 portfolio size
HOLD = 20  # rule 2 holding period, sessions
SLOTS = 20  # rule 2 reported portfolio
SLIPPAGE_BPS = 1.0
LEVEL = 98.75  # two rules: Bonferroni-corrected two-sided interval
RESAMPLES, SEED = 10_000, 0
REGISTRY = ROOT / ".state" / "longer-horizons-final.json"
SPLITS = {"development": (2016, 2022), "holdout": (2023, 2024), "test": (2025, 2099)}


def side_costs(half_spreads, multiple=1.0):
    """Cost of one buy or one sell per symbol, as a fraction."""
    return (np.asarray(half_spreads, float) + SLIPPAGE_BPS) * multiple / 1e4


def month_rows(dates):
    """(first session row, last session row) of each calendar month, in order."""
    months = np.asarray(dates).astype("datetime64[M]")
    starts = np.flatnonzero(np.r_[True, months[1:] != months[:-1]])
    ends = np.r_[starts[1:] - 1, len(months) - 1]
    return starts, ends


def momentum(dates, opens, closes, costs, top=TOP):
    """Rule 1. Returns one record per holding month: start date, rule and benchmark
    net returns, their difference, and the rule's traded weight."""
    starts, ends = month_rows(dates)
    records = []
    previous = {"rule": None, "benchmark": None}
    for m in range(12, len(starts) - 2):
        # Decision at the end of month m: 12-1 return from the end of month m-12 to the
        # end of month m-1; hold from the open of month m+1 to the open of month m+2.
        early, late = closes[ends[m - 12]], closes[ends[m - 1]]
        entry, exit_ = opens[starts[m + 1]], opens[starts[m + 2]]
        scored = np.isfinite(early) & np.isfinite(late) & np.isfinite(entry) & np.isfinite(exit_)
        if scored.sum() < top:
            previous = {"rule": None, "benchmark": None}
            continue
        score = np.where(scored, late / early, -np.inf)
        chosen = np.argsort(-score, kind="stable")[:top]
        growth = np.where(scored, exit_ / entry, 1.0)
        out = {"start": str(dates[starts[m + 1]])}
        for name, members in (("rule", chosen), ("benchmark", np.flatnonzero(scored))):
            weights = np.zeros(len(score))
            weights[members] = 1.0 / len(members)
            held = previous[name]
            traded = np.abs(weights - (held if held is not None else 0.0))
            cost = float(traded @ costs)
            gross = float(weights @ growth) - 1.0
            out[name] = gross - cost
            out[f"{name}_turnover"] = float(traded.sum())
            previous[name] = weights * growth / (1.0 + gross)  # weights drift to the exit
        out["excess"] = out["rule"] - out["benchmark"]
        records.append(out)
    return records


def drift(dates, opens, signal, costs, hold=HOLD):
    """Rule 2. One record per trade: entry date, symbol index, net return, the
    equal-weight benchmark over the same window, and the excess."""
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


def slot_portfolio(dates, opens, signal, strength, costs, hold=HOLD, slots=SLOTS):
    """Daily open-to-open returns of rule 2 as a portfolio of equal slots (1/slots of
    equity each, cash otherwise), entering by most analyst-up events. Approximation:
    slots are reset to equal weight daily."""
    held, returns = {}, []  # symbol -> exit row
    for d in range(len(dates) - 1):
        day_cost = 0.0
        for s in [s for s, end in held.items() if end <= d]:
            del held[s]
            day_cost += costs[s] / slots
        candidates = [s for s in np.flatnonzero(signal[d]) if s not in held]
        candidates.sort(key=lambda s: (-strength[d, s], s))
        for s in candidates[: slots - len(held)]:
            if np.isfinite(opens[d, s]) and d + hold < len(dates):
                held[s] = d + hold
                day_cost += costs[s] / slots
        change = 0.0
        for s in held:
            ratio = opens[d + 1, s] / opens[d, s]
            change += (ratio - 1.0) / slots if np.isfinite(ratio) else 0.0
        returns.append(change - day_cost)
    return np.asarray(returns)


def annualized(daily):
    daily = np.asarray(daily, float)
    return float(np.prod(1 + daily) ** (252 / max(len(daily), 1)) - 1) if len(daily) else 0.0


def summarize(name, values, interval):
    values = np.asarray(values, float)
    return {
        "n": len(values),
        "mean_bp": float(values.mean() * 1e4) if len(values) else float("nan"),
        f"ci{LEVEL}_bp": [v * 1e4 for v in interval],
        "positive_share": float((values > 0).mean()) if len(values) else float("nan"),
        "statistic": name,
    }


def in_split(date_strings, split):
    low, high = SPLITS[split]
    years = np.array([int(d[:4]) for d in date_strings])
    return (years >= low) & (years <= high)


def evaluate(root, splits):
    from hft.ml.download import load_calendar
    from hft.ml.features import _aligned, daily_frame
    from hft.ml.news import read_news
    from hft.ml.news_features import NEWS_FEATURES, news_features
    from hft.ml.train import _costs, _holding
    from hft.ml.universe import UNIVERSE

    symbols = [s.ticker for s in UNIVERSE if not s.trade_only]
    frame = daily_frame(root, symbols)
    years = frame.dates.astype("datetime64[Y]").astype(int) + 1970
    keep = years <= max(SPLITS[s][1] for s in splits)  # later rows are never read
    dates = frame.dates[keep]
    bars = [_aligned(root, s, frame.dates)[:, keep] for s in symbols]
    opens = np.stack([b[0] for b in bars], axis=1)
    closes = np.stack([b[3] for b in bars], axis=1)
    analyst = news_features(read_news(root), symbols, dates, load_calendar(root))[
        ..., NEWS_FEATURES.index("news_analyst_net")
    ]
    signal = (analyst > 0) & frame.valid[keep]
    half = _costs(symbols)
    result = {"symbols": len(symbols), "splits": {}}
    for multiple in (1.0, 3.0):
        costs = side_costs(half, multiple)
        months = momentum(dates, opens, closes, costs)
        trades = drift(dates, opens, signal, costs)
        daily = slot_portfolio(dates, opens, signal, analyst, costs)
        for split in splits:
            rng = np.random.default_rng(SEED)
            out = result["splits"].setdefault(split, {})
            key = "costs_1x" if multiple == 1.0 else "costs_3x"
            chosen = [
                m
                for m, ok in zip(months, in_split([m["start"] for m in months], split), strict=True)
                if ok
            ]
            excess = [m["excess"] for m in chosen]
            rule1 = summarize("monthly excess return", excess, block_interval(excess, rng))
            rule1["rule_annual"] = float(
                np.prod([1 + m["rule"] for m in chosen]) ** (12 / max(len(chosen), 1)) - 1
            )
            rule1["benchmark_annual"] = float(
                np.prod([1 + m["benchmark"] for m in chosen]) ** (12 / max(len(chosen), 1)) - 1
            )
            rule1["turnover_per_year"] = float(np.mean([m["rule_turnover"] for m in chosen]) * 12)
            picked = [
                t
                for t, ok in zip(trades, in_split([t["entry"] for t in trades], split), strict=True)
                if ok
            ]
            values = [t["excess"] for t in picked]
            rule2 = summarize(
                "excess return per trade",
                values,
                cluster_interval([t["entry"][:7] for t in picked], values, rng),
            )
            mask = in_split([str(d) for d in dates[:-1]], split)
            rule2["slot_portfolio_annual"] = annualized(daily[mask])
            rule2["trades_per_year"] = len(picked) / max(mask.sum() / 252, 1e-9)
            out[key] = {"momentum": rule1, "analyst_drift": rule2}
    for split in splits:
        days = in_split([str(d) for d in frame.dates], split)
        hold = _holding(root, symbols, frame, days, np.ones(len(symbols), bool))
        result["splits"][split]["buy_and_hold_annual"] = annualized(hold)
    for name in ("momentum", "analyst_drift"):
        if "holdout" in result["splits"] and "development" in result["splits"]:
            holdout = result["splits"]["holdout"]
            result[f"{name}_passes"] = bool(
                holdout["costs_1x"][name]["mean_bp"] > 0
                and holdout["costs_1x"][name][f"ci{LEVEL}_bp"][0] > 0
                and result["splits"]["development"]["costs_1x"][name]["mean_bp"] > 0
                and holdout["costs_3x"][name]["mean_bp"] > 0
            )
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "ml")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--final", action="store_true", help="the one-time test-years run")
    args = parser.parse_args(argv)
    splits = ["development", "holdout"]
    if args.final:
        if REGISTRY.exists():
            parser.error(f"the test years were already used: {REGISTRY.read_text()}")
        REGISTRY.parent.mkdir(parents=True, exist_ok=True)
        REGISTRY.write_text(json.dumps({"started": datetime.now(UTC).isoformat()}) + "\n")
        splits = ["test"]
    result = evaluate(args.data, splits)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
