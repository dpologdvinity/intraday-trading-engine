"""Summarize a paper run's completed trades from its hash-chained journals.

Reads `logs/trade/<run>/<SYMBOL>-<start_ns>.jsonl` (every restart of the run writes
a new file), verifies each hash chain, and reports per-stock and per-day results
from the broker fills recorded there. Paper fills are simulated by the broker and
rule-strategy runs never count as graduation evidence.

    .venv/bin/python scripts/paper_report.py --run overnight
    .venv/bin/python scripts/paper_report.py --run news-picks --markdown docs/news-picks-forward.md
"""

import argparse
import sys
import time
from collections import defaultdict
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hft.logs import iter_log

NEW_YORK = ZoneInfo("America/New_York")


def load_trades(log_dir):
    trades, seen = [], set()
    for path in sorted(Path(log_dir).glob("*.jsonl")):
        symbol = path.name.split("-")[0]
        for row in iter_log(path):  # raises on a broken hash chain
            if row.get("event") != "trade":
                continue
            key = (symbol, row["entry_ns"], row["exit_ns"])
            if key in seen:
                continue
            seen.add(key)
            trades.append(
                {
                    "symbol": symbol,
                    "entry_ns": row["entry_ns"],
                    "exit_ns": row["exit_ns"],
                    "pnl": Decimal(str(row["pnl"])),
                    "notional": Decimal(str(row["opening_notional"])),
                }
            )
    return trades


def load_sessions(log_dir):
    """{day: {"picks": [...], "fills": {symbol: [(signed quantity, price)]}}} for every
    New York day on which the run wrote any journal row (it was running)."""
    sessions, seen = defaultdict(lambda: {"picks": None, "fills": defaultdict(list)}), set()
    for path in sorted(Path(log_dir).glob("*.jsonl")):
        symbol = path.name.split("-")[0]
        for row in iter_log(path):
            sessions[exit_day(row["wall_ns"])]  # any row marks the day as running
            if row.get("event") == "picks":
                sessions[row["session"]]["picks"] = row["picks"]
            elif row.get("event") == "fill":
                execution = row["execution"]
                if execution["execution_id"] in seen:
                    continue
                seen.add(execution["execution_id"])
                day = exit_day(execution["timestamp_ns"])
                sessions[day]["fills"][symbol].append(
                    (Decimal(str(execution["signed_quantity"])), Decimal(str(execution["price"])))
                )
    return sessions


def _average(fills, sign):
    rows = [(q, p) for q, p in fills if q * sign > 0]
    quantity = sum((abs(q) for q, _ in rows), Decimal(0))
    return sum((abs(q) * p for q, p in rows), Decimal(0)) / quantity if quantity else None


def markdown(run, trades, sessions, market_days=()):
    """Daily forward record: picks, average fill prices and results per session.

    Market days (session dates, YYYY-MM-DD) from the run's first journal day through
    its last one with no journal row at all are listed as "bot not running".
    """
    by_day = defaultdict(list)
    for t in trades:
        by_day[exit_day(t["exit_ns"])].append(t)
    lines = [
        f"# Paper forward record: `{run}`",
        "",
        "Generated each trading day from the run's hash-chained journals by",
        f"`scripts/paper_report.py --run {run} --markdown`. Paper fills are simulated by the",
        "broker; results over a few days cannot establish an edge.",
        "",
        "| Session | Picks | Stock | Bought | Sold | Net $ | bp |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    total, days = Decimal(0), []
    first, today = min(sessions, default=""), exit_day(time.time_ns())
    shown = {d for d in sessions if d in set(market_days) or sessions[d]["picks"] is not None}
    shown |= {d for d in market_days if first <= d <= today} | set(by_day)
    for day in sorted(shown):
        if day not in sessions and day not in by_day:
            lines.append(f"| {day} | bot not running | - | | | | |")
            continue
        session = sessions.get(day, {"picks": None, "fills": {}})
        picks = ", ".join(session["picks"]) if session["picks"] is not None else "-"
        rows = sorted(by_day.get(day, []), key=lambda t: t["symbol"])
        if not rows:
            lines.append(f"| {day} | {picks or 'none'} | - | | | | |")
            continue
        net = sum(t["pnl"] for t in rows)
        total += net
        days.append((day, net, sum(t["notional"] for t in rows)))
        for t in rows:
            fills = session["fills"].get(t["symbol"], [])
            bought, sold = _average(fills, 1), _average(fills, -1)
            lines.append(
                f"| {day} | {picks} | {t['symbol']} | "
                f"{f'{bought:.2f}' if bought else '-'} | {f'{sold:.2f}' if sold else '-'} | "
                f"{t['pnl']:+.2f} | {t['pnl'] / t['notional'] * 10_000:+.0f} |"
            )
    lines += [
        "",
        "| Session | Net $ | Return on invested | Cumulative $ |",
        "| --- | ---: | ---: | ---: |",
    ]
    running = Decimal(0)
    for day, net, notional in days:
        running += net
        lines.append(f"| {day} | {net:+.2f} | {net / notional * 100:+.2f}% | {running:+.2f} |")
    lines += ["", f"Total {total:+.2f} over {len(trades)} trades on {len(days)} sessions.", ""]
    return "\n".join(lines)


def exit_day(ns):
    return datetime.fromtimestamp(ns / 1e9, UTC).astimezone(NEW_YORK).date().isoformat()


def report(trades):
    if not trades:
        return "no completed trades yet"
    lines = ["stock   trades   net $    mean bp/trade   winners"]
    by_symbol = defaultdict(list)
    for t in trades:
        by_symbol[t["symbol"]].append(t)
    for symbol, rows in sorted(by_symbol.items()):
        pnl = sum(r["pnl"] for r in rows)
        bp = sum(r["pnl"] / r["notional"] for r in rows) / len(rows) * 10_000
        wins = sum(r["pnl"] > 0 for r in rows)
        lines.append(f"{symbol:<7} {len(rows):>6} {pnl:>+8.2f} {bp:>+15.2f} {wins:>9}")
    lines.append("")
    lines.append("exit day     trades   net $   mean bp/trade")
    by_day = defaultdict(list)
    for t in trades:
        by_day[exit_day(t["exit_ns"])].append(t)
    for day, rows in sorted(by_day.items()):
        pnl = sum(r["pnl"] for r in rows)
        bp = sum(r["pnl"] / r["notional"] for r in rows) / len(rows) * 10_000
        lines.append(f"{day}  {len(rows):>6} {pnl:>+7.2f} {bp:>+15.2f}")
    total = sum(t["pnl"] for t in trades)
    days = len(by_day)
    lines.append("")
    lines.append(f"total {total:+.2f} over {len(trades)} trades on {days} exit days")
    lines.append("Broker-simulated paper fills; a few days cannot establish an edge.")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", required=True, help="run name under logs/trade/")
    parser.add_argument("--logs", type=Path, default=ROOT / "logs" / "trade")
    parser.add_argument("--markdown", type=Path, help="also write a daily forward record here")
    args = parser.parse_args()
    trades = load_trades(args.logs / args.run)
    print(report(trades))
    if args.markdown:
        try:
            from hft.runtime import fetch_calendar

            market_days = [r["date"] for r in fetch_calendar()]
        except (RuntimeError, OSError, KeyError, TypeError, ValueError) as error:
            print(f"warning: no market calendar ({error}); missed days are not listed")
            market_days = []
        text = markdown(args.run, trades, load_sessions(args.logs / args.run), market_days)
        args.markdown.write_text(text)


if __name__ == "__main__":
    main()
