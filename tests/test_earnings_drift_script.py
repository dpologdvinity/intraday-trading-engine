"""The experiment-22 evaluation: causal entries, the test-year guard, and a planted
drift that the rules must find."""

import importlib.util
import json
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from hft.ml.download import SCHEMA as BAR_SCHEMA
from hft.ml.earnings import EarningsEvent

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "earnings_drift.py"
HOUR = 3600 * 10**9


def _module():
    spec = importlib.util.spec_from_file_location("earnings_drift", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sessions(start, count):
    """Business days with 14:30-21:00 UTC sessions (winter hours)."""
    days = np.arange(np.datetime64(start), np.datetime64(start) + np.timedelta64(count * 2, "D"))
    days = days[np.is_busday(days)][:count]
    midnight = days.astype("datetime64[ns]").astype(np.int64)
    return days, midnight + 14 * HOUR + HOUR // 2, midnight + 21 * HOUR


def _event(symbol, ns, headline_ns=None, actual=None, estimate=None):
    return EarningsEvent(
        symbol,
        f"{symbol}-{ns}",
        ns,
        min(ns, headline_ns or ns),
        headline_ns,
        "headline" if headline_ns else None,
        actual,
        estimate,
    )


def test_beats_enter_at_the_first_open_after_the_headline_and_misses_are_kept_apart():
    module = _module()
    _, open_ns, _ = _sessions("2023-01-02", 5)
    events = [
        _event("A", open_ns[1] - 3 * HOUR, open_ns[1] - 3 * HOUR, 1.1, 1.0),  # pre-open
        _event("B", open_ns[1] + HOUR, open_ns[1] + HOUR, 1.1, 1.0),  # during the session
        _event("C", open_ns[1] - HOUR, open_ns[2] - 2 * HOUR, 1.1, 1.0),  # headline later
        _event("A", open_ns[3] - HOUR, open_ns[3] - HOUR, 0.9, 1.0),  # a miss
        _event("B", open_ns[3] - HOUR),  # no headline: neither
    ]
    beats, misses = module.beat_signals(events, ["A", "B", "C"], open_ns)
    assert np.argwhere(beats).tolist() == [[1, 0], [2, 1], [2, 2]]
    assert np.argwhere(misses).tolist() == [[3, 0]]


def test_announcement_entries_never_enter_before_day_zero_closes():
    module = _module()
    _, _, close_ns = _sessions("2023-01-02", 6)
    closes = np.ones((6, 3))
    closes[2] = [1.10, 1.00, 0.99]  # session 2: A +10%, B 0%, C -1%; market +3%
    events = [
        _event("A", close_ns[2] - HOUR),  # during session 2: day 0 is session 2
        _event("B", close_ns[1]),  # exactly at session 1's close: day 0 is session 2
        _event("C", close_ns[1] + HOUR),  # after session 1's close: day 0 is session 2
        _event("A", close_ns[5] + HOUR),  # no session left
        _event("C", close_ns[0] - HOUR),  # day 0 is the first session: no prior close
    ]
    entries = module.announcement_entries(events, ["A", "B", "C"], close_ns, closes)
    assert [(row, s) for row, s, _ in entries] == [(3, 0), (3, 1), (3, 2)]
    for row, _, _ in entries:
        assert close_ns[row - 1] > min(e.event_ns for e in events[:3])  # day 0 closed first
    values = {s: v for _, s, v in entries}
    assert values[0] == pytest.approx(0.10 - 0.03)
    assert values[1] == pytest.approx(-0.03)


def test_a_planted_drift_after_good_news_is_found_and_passes():
    """Stock prices are flat except that after every good announcement the stock
    rises 2% over the following five sessions; after bad news it is flat."""
    module = _module()
    rng = np.random.default_rng(1)
    dates, open_ns, close_ns = _sessions("2021-01-04", 4 * 252)  # 2021-2024
    symbols = [f"S{k}" for k in range(6)]
    level = np.ones((len(dates), len(symbols)))
    events = []
    for row in range(10, len(dates) - 10, 7):
        s = int(rng.integers(len(symbols)))
        good = bool(rng.integers(2))
        jump = 1.01 if good else 0.99
        level[row:, s] *= jump  # day-0 announcement move
        if good:
            ramp = np.linspace(1.0, 1.02, 6)  # opens of days 1..6
            level[row + 1 : row + 7, s] *= ramp
            level[row + 7 :, s] *= 1.02
        headline = open_ns[row] - 2 * HOUR  # before day 0's open
        events.append(_event(symbols[s], headline, headline, 1.1 if good else 0.9, 1.0))
    opens = closes = level
    half = np.zeros(len(symbols))
    result = module.evaluate_arrays(
        dates, open_ns, close_ns, opens, closes, events, symbols, half, ["development", "holdout"]
    )
    holdout = result["splits"]["holdout"]["costs_1x"]
    for rule in module.RULES:
        assert holdout[rule]["mean_bp"] > 100
        assert result[f"{rule}_passes"]
    misses = result["splits"]["holdout"]["controls"]["misses"]
    assert abs(misses["mean_bp"]) < 50
    assert result["last_exit"] <= "2024-12-31"


def _write_data(root):
    days = ["2024-10-17", "2024-10-18", "2024-10-21", "2024-12-31", "2025-01-02"]
    sessions = []
    for day in days:
        midnight = int(datetime.fromisoformat(day).replace(tzinfo=UTC).timestamp()) * 10**9
        sessions.append(
            {"session_id": day, "open_ns": midnight + 14 * HOUR, "close_ns": midnight + 21 * HOUR}
        )
    (root / "calendar.json").write_text(json.dumps(sessions))
    for symbol in ("AAPL", "NBIS"):
        folder = root / "1Day" / symbol
        folder.mkdir(parents=True)
        times = [s["open_ns"] - 9 * HOUR for s in sessions]  # 05:00 UTC, the bar's day
        values = [float(k + 1) for k in range(len(days))]
        table = pa.table(
            {
                "t": times,
                "o": values,
                "h": values,
                "l": values,
                "c": values,
                "v": values,
                "vw": values,
                "n": [1] * len(days),
            },
            schema=BAR_SCHEMA,
        )
        pq.write_table(table, folder / "bars.parquet")


def test_prices_stop_at_the_cutoff_and_skip_bars_before_a_listing(tmp_path):
    module = _module()
    _write_data(tmp_path)
    assert module.cutoff(["development", "holdout"]) == date(2024, 12, 31)
    dates, _, _, opens, closes = module.load_prices(tmp_path, ["AAPL", "NBIS"], date(2024, 12, 31))
    assert str(dates[-1]) == "2024-12-31" and len(dates) == 4
    assert opens[:, 0].tolist() == closes[:, 0].tolist() == [1.0, 2.0, 3.0, 4.0]
    assert np.isnan(opens[:2, 1]).all() and opens[2, 1] == 3.0  # NBIS listed 2024-10-21


def test_the_test_years_need_a_committed_report_and_run_once(tmp_path, monkeypatch):
    module = _module()
    monkeypatch.setattr(module, "REGISTRY", tmp_path / "final.json")
    monkeypatch.setattr(module, "RESULTS_DOC", "docs/no-such-report.md")
    monkeypatch.setattr(module, "evaluate", lambda *a: pytest.fail("evaluated test years"))
    output = tmp_path / "out.json"
    with pytest.raises(SystemExit):
        module.main(["--final", "--output", str(output)])
    assert not (tmp_path / "final.json").exists()
    monkeypatch.setattr(module, "RESULTS_DOC", "README.md")  # committed
    (tmp_path / "final.json").write_text("{}")
    with pytest.raises(SystemExit):
        module.main(["--final", "--output", str(output)])
    assert not output.exists()
