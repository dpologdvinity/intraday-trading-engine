import math

import numpy as np
import pytest
import torch

from hft.ml.fills import simulate_day
from hft.ml.rl_env import (
    MAX_STEPS_A,
    DailyEnv,
    IntradayEnv,
    _fill_index,
    decision_minutes,
)


def _intraday(n=64, seed=0, gaps=0.05):
    rng = np.random.default_rng(seed)
    opens = 100 * np.exp(np.cumsum(rng.normal(0, 1e-3, (n, 390)), axis=1))
    valid = rng.random((n, 390)) > gaps
    valid[:, 0] = True
    opens = np.where(valid, opens, 0.0).astype(np.float32)
    fill = _fill_index(valid)
    seen = np.maximum.accumulate(np.where(valid, np.arange(390), 0), axis=1)
    last = np.full(n, 385)
    last[: n // 8] = 205  # half days
    tensors = {
        "symbol": torch.zeros(n, dtype=torch.long),
        "date": torch.zeros(n, dtype=torch.long),
        "channels": torch.as_tensor(rng.normal(size=(n, 390, 5)).astype(np.float32)),
        "market": torch.zeros((1, 390, 10)),
        "fill_open": torch.as_tensor(np.take_along_axis(opens, fill, 1)),
        "fill_minute": torch.as_tensor(fill),
        "known_open": torch.as_tensor(np.take_along_axis(opens, seen, 1)),
        "last_minute": torch.as_tensor(last),
        "cost_bps": torch.full((n,), 2.0),
        "extras": torch.zeros((n, 0)),
    }
    return tensors, opens, valid, last


def _plan(env, rng):
    """Random entry and exit decisions that respect the rules (exit -1: forced sale)."""
    plans = []
    minutes = env.minutes.numpy()
    for i in range(len(env.items)):
        steps = int(env.steps[i])
        k_e = int(rng.integers(0, max(steps - 4, 1)))
        k_x = (
            int(rng.integers(k_e + 3, steps)) if rng.random() < 0.7 and k_e + 3 < steps - 1 else -1
        )
        plans.append((k_e, k_x, minutes[i]))
    return plans


def test_intraday_day_returns_match_the_ml_fill_simulator():
    tensors, opens, valid, last = _intraday()
    env = IntradayEnv(tensors)
    env.reset(torch.arange(len(last)), cost_multiple=1.0)
    rng = np.random.default_rng(1)
    plans = _plan(env, rng)
    for k in range(MAX_STEPS_A):
        action = torch.tensor(
            [k_e <= k and (k_x < 0 or k < k_x) for k_e, k_x, _ in plans], dtype=torch.long
        )
        env.step(action)
    checked = 0
    for i, (k_e, k_x, minutes) in enumerate(plans):
        signal = np.zeros(390)
        signal[minutes[k_e]] = 1.0
        if k_x >= 0:
            signal[minutes[k_x]] = -1.0
        trade = simulate_day(
            opens[i], valid[i], signal, enter=0.5, half_spread_bps=2.0, last_minute=int(last[i])
        )
        got = float(env.log_value[i])
        if trade is None:
            assert got == 0.0
            continue
        entry_fill = int(tensors["fill_minute"][i, minutes[k_e]])
        if k_x >= 0 and minutes[k_x] < entry_fill:
            continue  # a 15-minute gap after the order: the simulator reads it differently
        assert got == pytest.approx(math.log1p(trade.net_return), abs=1e-5)
        checked += 1
    assert checked > 40


def test_rules_force_the_sale_and_block_late_entries():
    tensors, _, _, last = _intraday(n=8, gaps=0.0)
    env = IntradayEnv(tensors)
    env.reset(torch.arange(8), cost_multiple=1.0)
    for _ in range(MAX_STEPS_A):
        env.step(torch.ones(8, dtype=torch.long))  # always asks to be long
    assert (env.trades == 1).all()  # one round trip, sold at the end
    assert not env.position.any()
    minutes, steps = decision_minutes(last)
    assert (minutes[np.arange(8), steps - 1] == last - 1).all()


def test_intraday_observation_uses_no_later_minute():
    tensors, *_ = _intraday(n=4, gaps=0.0)
    changed = {k: v.clone() for k, v in tensors.items()}
    changed["channels"][:, 200:] += 5.0
    changed["known_open"][:, 200:] *= 2
    changed["fill_open"][:, 200:] *= 2
    obs = []
    for t in (tensors, changed):
        env = IntradayEnv(t)
        env.reset(torch.arange(4), cost_multiple=1.0)
        seen = []
        for _ in range(39):  # decisions up to minute 194
            window, extra, _ = env.observe()
            seen.append((window.clone(), extra.clone()))
            env.step(torch.ones(4, dtype=torch.long))
        obs.append(seen)
    for (w1, e1), (w2, e2) in zip(*obs, strict=True):
        assert torch.equal(w1, w2) and torch.equal(e1, e2)


def test_daily_episode_returns_match_open_to_open_with_costs():
    days, symbols = 120, 3
    rng = np.random.default_rng(2)
    opens = (50 * np.exp(np.cumsum(rng.normal(0, 0.01, (days, symbols)), axis=0))).astype(
        np.float32
    )
    tensors = {
        "features": torch.zeros((days, symbols, 3)),
        "gap": torch.zeros((days, symbols)),
        "open": torch.as_tensor(opens),
        "valid": torch.ones((days, symbols), dtype=torch.bool),
        "cost_bps": torch.full((symbols,), 1.5),
        "news": torch.zeros((days, symbols, 0)),
        "market": torch.tensor([1, 2]),
    }
    env = DailyEnv(tensors)
    env.reset(torch.tensor([0]), torch.tensor([60]), length=30, cost_multiple=2.0)
    actions = [0, 1, 1, 1, 1, 0] + [0] * 24  # buy at day 61, sell at day 65
    for a in actions:
        env.step(torch.tensor([a]))
    cost = lambda price: 2.0 * (max(1.5, 0.5 / price * 100) + 1.0) / 1e4
    p_in, p_out = float(opens[61, 0]), float(opens[65, 0])
    expected = math.log(p_out / p_in) - math.log1p(cost(p_in)) + math.log1p(-cost(p_out))
    assert float(env.log_value[0]) == pytest.approx(expected, abs=1e-5)
    assert int(env.trades[0]) == 1
