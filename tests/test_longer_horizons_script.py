import importlib.util
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "longer_horizons.py"


def _module():
    spec = importlib.util.spec_from_file_location("longer_horizons", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _business_days(start, count):
    days = np.arange(np.datetime64(start), np.datetime64(start) + np.timedelta64(count * 2, "D"))
    return days[np.is_busday(days)][:count]


def test_momentum_holds_last_years_winners_and_beats_the_benchmark_when_they_keep_rising():
    module = _module()
    dates = _business_days("2016-01-01", 15 * 22)
    rows = np.arange(len(dates))
    # Symbol 0 rises steadily, the others are flat: it is always the top pick.
    prices = np.stack([100 * 1.002**rows] + [np.full(len(rows), 50.0)] * 3, axis=1)
    months = module.momentum(dates, prices, prices, np.zeros(4), top=1)
    assert months and all(m["excess"] > 0 for m in months)
    assert months[0]["start"][:7] == "2017-02"  # 12-1 needs a year, then the next month
    costly = module.momentum(dates, prices, prices, np.full(4, 0.01), top=1)
    assert costly[0]["rule"] < months[0]["rule"]  # entry costs
    assert costly[1]["rule_turnover"] < 1e-9  # same pick: no trading


def test_drift_trades_hold_twenty_sessions_and_subtract_the_benchmark():
    module = _module()
    dates = _business_days("2023-01-02", 30)
    opens = np.ones((30, 2))
    opens[20:, 0] = 1.1  # symbol 0 gains 10% by session 20
    signal = np.zeros((30, 2), bool)
    signal[0, 0] = signal[25, 1] = True  # the second has no exit inside the data
    trades = module.drift(dates, opens, signal, np.array([0.001, 0.001]))
    assert len(trades) == 1
    assert abs(trades[0]["net"] - (0.1 - 0.002)) < 1e-12
    assert abs(trades[0]["benchmark"] - 0.05) < 1e-12
    assert abs(trades[0]["excess"] - (0.1 - 0.002 - 0.05)) < 1e-12


def test_intervals_cover_the_mean_and_split_masks_by_year():
    module = _module()
    rng = np.random.default_rng(0)
    values = rng.normal(0.01, 0.001, 24)
    low, high = module.block_interval(values, rng)
    assert low < values.mean() < high and low > 0
    low, high = module.cluster_interval(["a", "a", "b"], [1.0, 1.0, 3.0], rng)
    assert 1.0 <= low <= high <= 3.0
    assert module.in_split(["2022-12-30", "2023-01-03"], "holdout").tolist() == [False, True]
