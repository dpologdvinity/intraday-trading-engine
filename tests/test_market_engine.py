from pathlib import Path

import numpy as np
import pytest
from engine_vectors import OPEN, SESSION, engine_vectors, quote, trade

from hft.calendar import SessionWindow
from hft.data import Bar, synthetic_sessions
from hft.features import market_features
from hft.feed import BarAggregator, historical_timeline
from hft.market_engine import BarUpdate, GapEvent, PyMarketEngine, make_market_engine
from hft.strategies import parse_strategy, rule_action

NS = 1_000_000_000


def _session():
    data = synthetic_sessions(1, 200, seed=3)[0]
    return data, SessionWindow(data.session_id, data.open_ns, data.close_ns)


def _run(engine, timeline):
    updates = []
    for event, now in timeline:
        updates += engine.advance_to(now) if event is None else engine.on_event(event, now)
    return updates


def test_engine_bars_match_reference_aggregator():
    data, window = _session()
    engine = PyMarketEngine(data.symbol)
    engine.start_session(window)
    updates = _run(engine, historical_timeline(data))
    reference = BarAggregator(data.symbol, 5, window)
    expected = []
    for event, now in historical_timeline(data):
        expected += reference.advance_to(now) if event is None else reference.add(event, now)
    assert [u.bar for u in updates if isinstance(u, BarUpdate)] == expected


def test_ready_updates_carry_market_features_and_rule_actions():
    data, window = _session()
    engine = PyMarketEngine(data.symbol, "hold-day")
    engine.start_session(window)
    ready = [u for u in _run(engine, historical_timeline(data)) if getattr(u, "ready", False)]
    assert ready, "a 200-bar session must complete the 61-bar warmup"
    first = ready[0]
    assert len(engine.history) == 61
    assert first.action == 1
    assert first.market.dtype == np.float32 and first.market.shape == (10,)
    assert np.array_equal(ready[-1].market, market_features(engine.history))


def test_feed_gap_resets_history_and_warmup():
    engine = PyMarketEngine("X")
    engine.start_session(SESSION)
    _run(engine, [quote(OPEN + NS, i="a"), trade(OPEN + 2 * NS, i="b"), (None, OPEN + 5 * NS)])
    assert len(engine.history) == 1
    updates = engine.on_event(*trade(OPEN + 9 * NS, i="c"))
    assert updates[0] == GapEvent("X", OPEN + 9 * NS, "feed_gap")
    assert len(engine.history) == 0
    assert engine.warmup_after_ns == OPEN + 9 * NS


def test_clock_jump_is_a_runtime_sleep_gap():
    engine = PyMarketEngine("X")
    engine.start_session(SESSION)
    engine.advance_to(OPEN + NS)
    assert engine.advance_to(OPEN + 7 * NS)[0] == GapEvent("X", OPEN + 7 * NS, "runtime_sleep")


def test_carried_bars_before_warmup_are_not_accepted():
    engine = PyMarketEngine("X")
    engine.start_session(SESSION)
    _run(engine, [quote(OPEN + NS, i="a"), trade(OPEN + 2 * NS, i="b")])
    updates = engine.on_event(*trade(OPEN + 13 * NS, i="late"))
    bars = [u for u in updates if isinstance(u, BarUpdate)]
    assert bars and not any(u.accepted for u in bars)


def test_mid_session_start_aligns_to_the_next_boundary():
    engine = PyMarketEngine("X")
    engine.start_session(SESSION, now_ns=OPEN + 3600 * NS - 600 * NS + 2 * NS)
    assert engine.aggregator.start == OPEN + 3000 * NS


def _bars(closes):
    return [
        Bar(
            OPEN + (i + 1) * 5 * NS,
            closes[i],
            closes[i],
            closes[i],
            closes[i],
            1,
            closes[i] - 0.01,
            closes[i] + 0.01,
            100.0,
            100.0,
        )
        for i in range(len(closes))
    ]


def test_rule_actions():
    rising = _bars([float(x) for x in np.linspace(100, 110, 61)])
    falling = _bars([float(x) for x in np.linspace(110, 100, 61)])
    assert rule_action("ema-crossover", rising) == 1
    assert rule_action("ema-crossover", falling) == 0
    assert rule_action("hold-day", falling) == 1
    with pytest.raises(ValueError):
        rule_action("unknown", rising)


def test_parse_strategy():
    assert parse_strategy("ema-crossover").is_rule
    spec = parse_strategy("model:artifacts/x/bundle")
    assert spec.name == "model" and spec.bundle == Path("artifacts/x/bundle")
    scheduled = parse_strategy("momentum")
    assert scheduled.is_scheduled and not scheduled.is_rule
    for bad in ("", "model:", "mean-reversion"):
        with pytest.raises(ValueError):
            parse_strategy(bad)
    with pytest.raises(ValueError):  # scheduled strategies never run on 5 s bars
        PyMarketEngine("X", "momentum")
    with pytest.raises(ValueError):
        make_market_engine("X", implementation="fpga")


@pytest.mark.parametrize("vector", engine_vectors(), ids=lambda v: v[0])
def test_vectors_are_deterministic(vector):
    _, session, symbol, timeline = vector

    def run():
        engine = PyMarketEngine(symbol, "ema-crossover")
        engine.start_session(session)
        return [
            (type(u).__name__, getattr(u, "bar", None), getattr(u, "reason", None))
            for u in _run(engine, timeline)
        ]

    assert run() == run()
