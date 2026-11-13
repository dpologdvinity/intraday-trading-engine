"""Evaluate the pre-registered post-earnings drift rules.

Implements docs/specs/2026-11-12-earnings-drift-preregistration.md (experiment 22).
Rule 1 buys the open after an EPS headline that beats the consensus; rule 2 buys
the day-1 open after a positive announcement return (day 0's close against the close
before the announcement, net of the equal-weight universe). Both hold five sessions
and pay the universe's half-spread plus 1 bp slippage per side, and are judged
against the equal-weight return of the same 60 stocks. Development (2016-2022) and
holdout (2023-2024) only, reading no bar after 2024; the test years run with
--final --rule NAME, once, for a rule that passed and after the results doc is
committed.

    .venv/bin/python scripts/earnings_drift.py --output artifacts/earnings-drift/holdout.json
"""

import argparse
import json
import subprocess
import sys
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hft.ml.earnings import first_close_after, first_open_after
from hft.ml.rule_tests import (
    LEVEL,
    SEED,
    SPLITS,
    cluster_interval,
    drift,
    in_split,
    side_costs,
    summarize,
)

HOLD = 5  # sessions, open to open
STALE_NS = 4 * 86_400 * 10**9  # rule 1 skips a headline older than this at the entry open
RULES = ("consensus_beat", "announcement_return")
REGISTRY = ROOT / ".state" / "earnings-drift-final.json"
RESULTS_DOC = "docs/earnings-drift-results.md"


def cutoff(splits):
    """Last calendar day whose prices the splits may read."""
    return date(max(SPLITS[s][1] for s in splits), 12, 31)


def load_prices(root, symbols, last_day):
    """Session dates, open and close times (ns) and (D, S) opens and closes, for
    sessions up to `last_day`. Bars after `last_day` are filtered out as the files are
    read. Prices before a stock's first universe date are NaN."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    from hft.ml.download import SCHEMA, load_calendar
    from hft.ml.features import NEW_YORK, _day
    from hft.ml.universe import UNIVERSE

    following = datetime.combine(last_day + timedelta(days=1), datetime.min.time(), NEW_YORK)
    limit = [("t", "<", int(following.timestamp()) * 10**9)]

    sessions = [w for w in load_calendar(root) if w.session_id <= last_day.isoformat()]
    dates = np.array([w.session_id for w in sessions], dtype="datetime64[D]")
    open_ns = np.array([w.open_ns for w in sessions], dtype=np.int64)
    close_ns = np.array([w.close_ns for w in sessions], dtype=np.int64)
    first = {s.ticker: np.datetime64(s.first_date, "D") for s in UNIVERSE}
    opens = np.full((len(dates), len(symbols)), np.nan)
    closes = np.full_like(opens, np.nan)
    row_of = {d: i for i, d in enumerate(dates)}
    for k, symbol in enumerate(symbols):
        files = sorted((Path(root) / "1Day" / symbol).glob("*.parquet"))
        tables = [pq.read_table(f, schema=SCHEMA, filters=limit) for f in files]
        table = pa.concat_tables(tables) if tables else SCHEMA.empty_table()
        for t, o, c in zip(*(table.column(n).to_numpy() for n in "toc"), strict=True):
            day = _day(t)
            if day > dates[-1] or day < first[symbol]:
                continue
            if day not in row_of:
                raise ValueError(f"{symbol} has a bar on {day}, which is not a session")
            opens[row_of[day], k], closes[row_of[day], k] = o, c
    return dates, open_ns, close_ns, opens, closes


def beat_signals(events, symbols, open_ns):
    """Rule 1 and its misses control: (D, S) entry masks at the first open after the
    first matched headline, labelled by the latest headline published before that
    open (a correction out by then supersedes the first report). Headlines older than
    STALE_NS at the open (before the first stored session) are skipped."""
    index = {s: k for k, s in enumerate(symbols)}
    beats = np.zeros((len(open_ns), len(symbols)), bool)
    misses = np.zeros_like(beats)
    for event in events:
        if event.headline_ns is None or event.symbol not in index:
            continue
        row = first_open_after(event.headline_ns, open_ns)
        if row is None or open_ns[row] - event.headline_ns > STALE_NS:
            continue
        beat = event.beat_at(int(open_ns[row]))
        (beats if beat else misses)[row, index[event.symbol]] = True
    return beats, misses


def announcement_entries(events, symbols, close_ns, closes):
    """Rule 2 inputs: one (entry row, symbol index, announcement return) per event
    whose day 0 and preceding close exist. Day 0 is the first session closing after
    the event; the entry is day 1, after day 0's close is known."""
    index = {s: k for k, s in enumerate(symbols)}
    entries = []
    for event in events:
        if event.symbol not in index:
            continue
        day0 = first_close_after(event.event_ns, close_ns)
        if day0 is None or day0 < 1 or day0 + 1 >= len(close_ns):
            continue
        ratio = closes[day0] / closes[day0 - 1]
        s = index[event.symbol]
        if not np.isfinite(ratio[s]):
            continue
        market = float(np.mean(ratio[np.isfinite(ratio)]))
        entries.append((day0 + 1, s, float(ratio[s] - market)))
    return entries


def entry_mask(entries, shape, keep=lambda value: True):
    mask = np.zeros(shape, bool)
    for row, s, value in entries:
        if keep(value):
            mask[row, s] = True
    return mask


def _summary(trades, split):
    chosen = [t for t, ok in zip(trades, in_split([t["entry"] for t in trades], split)) if ok]
    values = [t["excess"] for t in chosen]
    rng = np.random.default_rng(SEED)
    months = [t["entry"][:7] for t in chosen]
    out = summarize("excess return per trade", values, cluster_interval(months, values, rng))
    years = sorted({t["entry"][:4] for t in chosen})
    out["per_year_bp"] = {
        y: float(np.mean([t["excess"] for t in chosen if t["entry"][:4] == y]) * 1e4) for y in years
    }
    out["trades_per_year"] = {y: sum(1 for t in chosen if t["entry"][:4] == y) for y in years}
    return out


def evaluate_arrays(
    dates, open_ns, close_ns, opens, closes, events, symbols, half, splits, rules=RULES
):
    """The chosen rules on prepared arrays, plus the reported-only controls unless a
    single rule is being run on the test years."""
    shape = opens.shape
    beats, misses = beat_signals(events, symbols, open_ns)
    entries = announcement_entries(events, symbols, close_ns, closes)
    signals = {"consensus_beat": beats, "announcement_return": entry_mask(entries, shape, _up)}
    result: dict[str, Any] = {"splits": {}}
    trades_seen = []
    for multiple in (1.0, 3.0):
        costs = side_costs(half, multiple)
        key = "costs_1x" if multiple == 1.0 else "costs_3x"
        trades = {name: drift(dates, opens, signals[name], costs, HOLD) for name in rules}
        for name in rules:
            trades_seen += trades[name]
        for split in splits:
            out = result["splits"].setdefault(split, {})
            out[key] = {name: _summary(t, split) for name, t in trades.items()}
    if tuple(rules) == RULES and "test" not in splits:
        trades_seen += _controls(result, dates, opens, misses, entries, half, splits)
    result["last_exit"] = max((t["exit"] for t in trades_seen), default=None)
    splits_out = result["splits"]
    if "holdout" in splits_out and "development" in splits_out:
        for name in rules:
            holdout = splits_out["holdout"]
            result[f"{name}_passes"] = bool(
                holdout["costs_1x"][name]["mean_bp"] > 0
                and holdout["costs_1x"][name][f"ci{LEVEL}_bp"][0] > 0
                and splits_out["development"]["costs_1x"][name]["mean_bp"] > 0
                and holdout["costs_3x"][name]["mean_bp"] > 0
            )
    return result


def _up(value):
    return value > 0


def _controls(result, dates, opens, misses, entries, half, splits):
    """Reported-only controls at 1x costs: rule 1's misses, every event at the rule-2
    entry, and terciles of the announcement return. Returns every trade made."""
    costs = side_costs(half)
    shape = opens.shape
    controls = {
        "misses": drift(dates, opens, misses, costs, HOLD),
        "all_events": drift(dates, opens, entry_mask(entries, shape), costs, HOLD),
    }
    made = controls["misses"] + controls["all_events"]
    for split in splits:
        out = result["splits"][split]
        out["controls"] = {name: _summary(t, split) for name, t in controls.items()}
        mine = [e for e in entries if in_split([str(dates[e[0]])], split)[0]]
        terciles = []
        if len(mine) >= 3:
            edges = np.quantile([v for _, _, v in mine], [1 / 3, 2 / 3])
            bounds = [(-np.inf, edges[0]), (edges[0], edges[1]), (edges[1], np.inf)]
            for low, high in bounds:
                mask = entry_mask(mine, shape, lambda v, lo=low, hi=high: lo < v <= hi)
                trades = drift(dates, opens, mask, costs, HOLD)
                made += trades
                summary = _summary(trades, split)
                summary["announcement_return_bp"] = [float(low * 1e4), float(high * 1e4)]
                terciles.append(summary)
        out["controls"]["announcement_terciles"] = terciles
    return made


def evaluate(root, earnings_root, splits, rules=RULES):
    from hft.ml.earnings import earnings_events, read_filings
    from hft.ml.news import read_news
    from hft.ml.train import _costs
    from hft.ml.universe import UNIVERSE

    symbols = [s.ticker for s in UNIVERSE if s.kind == "stock" and not s.trade_only]
    last_day = cutoff(splits)
    dates, open_ns, close_ns, opens, closes = load_prices(root, symbols, last_day)
    events = earnings_events(read_filings(earnings_root), read_news(root), symbols)
    events = [e for e in events if e.event_ns <= close_ns[-1]]  # later events: no prices
    result = evaluate_arrays(
        dates, open_ns, close_ns, opens, closes, events, symbols, _costs(symbols), splits, rules
    )
    if result["last_exit"] is not None and result["last_exit"] > last_day.isoformat():
        raise AssertionError(f"read prices after {last_day}: {result['last_exit']}")
    result.update(
        {
            "rules": list(rules),
            "symbols": len(symbols),
            "event_stocks": len({e.symbol for e in events}),
            "events": len(events),
            "events_with_headline": sum(e.headline_ns is not None for e in events),
            "last_session": str(dates[-1]),
        }
    )
    return result


def _committed(path):
    """Whether `path` is in the HEAD commit (staged is not enough)."""
    command = ["git", "cat-file", "-e", f"HEAD:{path}"]
    return subprocess.run(command, cwd=ROOT, capture_output=True, check=False).returncode == 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "ml")
    parser.add_argument("--earnings", type=Path, default=ROOT / "data" / "earnings")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--final", action="store_true", help="the one-time test-years run")
    parser.add_argument("--rule", choices=RULES, help="with --final: the rule that passed")
    args = parser.parse_args(argv)
    splits, rules = ["development", "holdout"], RULES
    if args.final:
        if args.rule is None:
            parser.error("--final runs one rule that passed the holdout: give --rule")
        if not _committed(RESULTS_DOC):
            parser.error(f"commit the holdout report ({RESULTS_DOC}) before the test")
        if REGISTRY.exists():
            parser.error(f"the test years were already used: {REGISTRY.read_text()}")
        REGISTRY.parent.mkdir(parents=True, exist_ok=True)
        REGISTRY.write_text(json.dumps({"started": datetime.now(UTC).isoformat()}) + "\n")
        splits, rules = ["test"], (args.rule,)
    elif args.rule is not None:
        parser.error("--rule is only for --final")
    result = evaluate(args.data, args.earnings, splits, rules)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
