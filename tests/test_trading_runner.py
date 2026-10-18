import asyncio
import json
from decimal import Decimal

import pytest

from hft.account import Costs
from hft.calendar import SessionWindow
from hft.data import NS
from hft.strategies import parse_strategy
from hft.trading.broker import PortfolioBroker
from hft.trading.budget import budget_gateway
from hft.trading.runner import TradeConfig, TradeRunner

OPEN = 1_736_173_800 * NS  # 2025-01-06 14:30 UTC
SESSION = SessionWindow("2025-01-06", OPEN, OPEN + 23_400 * NS)
START = OPEN + 3600 * NS  # the run starts one hour into the session


class Clock:
    def __init__(self, now):
        self.now = now

    def __call__(self):
        return self.now

    async def sleep(self, seconds):
        self.now += int(seconds * NS)
        await asyncio.sleep(0)


class Exchange:
    """Paper account that fills every order immediately at its limit or the last price."""

    mode = "broker-paper"

    def __init__(self, clock):
        self.clock, self.cash, self.positions, self.orders = clock, Decimal(1000), {}, {}
        self.posts = []

    def request(self, method, path, body=None):
        if path.startswith("/v2/account/activities/"):
            return []
        if path == "/v2/account":
            return {"id": "paper-1", "status": "ACTIVE", "cash": str(self.cash)}
        if path == "/v2/clock":
            return {"is_open": True}
        if path.startswith("/v2/assets/"):
            return {"tradable": True, "fractionable": True, "class": "us_equity"}
        if path == "/v2/positions":
            return [{"symbol": s, "qty": str(q)} for s, q in self.positions.items() if q]
        if path.startswith("/v2/orders?"):
            return []
        if method == "POST":
            self.posts.append((body["symbol"], body["side"]))
            price = Decimal(body.get("limit_price", "100"))
            qty, sign = Decimal(body["qty"]), 1 if body["side"] == "buy" else -1
            self.positions[body["symbol"]] = self.positions.get(body["symbol"], 0) + sign * qty
            self.cash -= sign * qty * price
            order = {
                **body,
                "id": f"o{len(self.orders)}",
                "status": "filled",
                "filled_qty": body["qty"],
                "filled_avg_price": str(price),
                "filled_at": "2025-01-06T15:30:00Z",
            }
            self.orders[order["id"]] = order
            return order
        if path.startswith("/v2/orders/"):
            return self.orders[path.rsplit("/", 1)[1]]
        if method == "DELETE":
            return None
        raise AssertionError((method, path, body))


def stream_factory(clock, silent_after_ns):
    async def stream(symbols, *, feed):
        yield "connected", None, clock()
        k = 0
        while True:
            t = START + k * 500_000_000
            while clock() < t:
                await asyncio.sleep(0)
            for symbol in symbols:
                if symbol == "AAPL" and t > START + silent_after_ns:
                    continue
                price = 100 + k * 0.001
                quote = {
                    "T": "q",
                    "S": symbol,
                    "event_ns": t,
                    "bp": price - 0.01,
                    "ap": price + 0.01,
                    "bs": 500.0,
                    "as": 500.0,
                    "sizes_in_shares": True,
                    "i": f"{symbol}q{k}",
                    "arrival_ns": t,
                }
                yield "event", quote, t
                trade = {
                    "T": "t",
                    "S": symbol,
                    "event_ns": t + 1,
                    "p": price,
                    "s": 10.0,
                    "i": f"{symbol}t{k}",
                    "arrival_ns": t + 1,
                }
                yield "event", trade, t + 1
            k += 1

    return stream


def test_runner_trades_one_stock_and_survives_silence_on_another(tmp_path, monkeypatch):
    monkeypatch.setattr("hft.runtime.time.monotonic", lambda: 0.0)
    clock = Clock(START)
    exchange = Exchange(clock)
    budgets = {"NVDA": 300, "AAPL": 200}
    broker = PortfolioBroker(
        exchange,
        budgets,
        Costs(0, 0),
        lambda s: budget_gateway(budgets[s]),
        run_dir=tmp_path / "state",
        identity="run",
        clock=clock,
    )
    config = TradeConfig(
        "test",
        ("NVDA", "AAPL"),
        budgets,
        parse_strategy("hold-day"),
        log_dir=tmp_path / "logs",
        run_dir=tmp_path / "state",
    )
    runner = TradeRunner(
        config,
        broker,
        [SESSION],
        stream_factory=stream_factory(clock, 30 * NS),
        clock=clock,
        sleep=clock.sleep,
    )
    try:
        summary = asyncio.run(runner.run(until_ns=START + 420 * NS))
    finally:
        broker.close()
    assert ("NVDA", "buy") in exchange.posts  # warmed up from a mid-session start and bought
    assert ("NVDA", "sell") in exchange.posts  # shutdown flattened the position
    assert not any(symbol == "AAPL" for symbol, _ in exchange.posts)  # silent stock never traded
    assert Decimal(summary["NVDA"]["position"]) == 0
    aapl = [
        json.loads(line)
        for line in next((tmp_path / "logs").glob("AAPL-*.jsonl")).read_text().splitlines()
    ]
    assert any(r.get("reason") == "feed_silence" for r in aapl)


def test_runner_blocks_entries_while_the_stream_is_disconnected(tmp_path, monkeypatch):
    monkeypatch.setattr("hft.runtime.time.monotonic", lambda: 0.0)
    clock = Clock(START)
    exchange = Exchange(clock)
    events = stream_factory(clock, 10**15)

    async def disconnected(symbols, *, feed):
        async for item in events(symbols, feed=feed):
            yield ("disconnected", None, item[2]) if item[0] == "connected" else item

    budgets = {"NVDA": 300}
    broker = PortfolioBroker(
        exchange,
        budgets,
        Costs(0, 0),
        lambda s: budget_gateway(300),
        run_dir=tmp_path / "state",
        identity="run",
        clock=clock,
    )
    config = TradeConfig(
        "test",
        ("NVDA",),
        budgets,
        parse_strategy("hold-day"),
        log_dir=tmp_path / "logs",
        run_dir=tmp_path / "state",
    )
    runner = TradeRunner(
        config, broker, [SESSION], stream_factory=disconnected, clock=clock, sleep=clock.sleep
    )
    try:
        asyncio.run(runner.run(until_ns=START + 360 * NS))
    finally:
        broker.close()
    assert exchange.posts == []


def test_runner_trades_consecutive_sessions_and_is_flat_between_them(tmp_path, monkeypatch):
    monkeypatch.setattr("hft.runtime.time.monotonic", lambda: 0.0)
    clock = Clock(START)
    exchange = Exchange(clock)
    first = SessionWindow("2025-01-06", START, START + 400 * NS)
    second = SessionWindow("2025-01-07", START + 600 * NS, START + 1000 * NS)
    budgets = {"NVDA": 300}
    broker = PortfolioBroker(
        exchange,
        budgets,
        Costs(0, 0),
        lambda s: budget_gateway(300),
        run_dir=tmp_path / "state",
        identity="run",
        clock=clock,
    )
    config = TradeConfig(
        "test",
        ("NVDA",),
        budgets,
        parse_strategy("hold-day"),
        log_dir=tmp_path / "logs",
        run_dir=tmp_path / "state",
    )
    runner = TradeRunner(
        config,
        broker,
        [first, second],
        clock=clock,
        sleep=clock.sleep,
        stream_factory=stream_factory(clock, 10**15),
    )
    try:
        asyncio.run(runner.run(until_ns=START + 1100 * NS))
    finally:
        broker.close()
    sides = [side for symbol, side in exchange.posts if symbol == "NVDA"]
    assert sides == ["buy", "sell", "buy", "sell"]  # one round trip per session
    assert exchange.positions["NVDA"] == 0
    rows = [
        json.loads(r)
        for r in next((tmp_path / "logs").glob("NVDA-*.jsonl")).read_text().splitlines()
    ]
    assert [r["date"] for r in rows if r["event"] == "session"] == ["2025-01-06", "2025-01-07"]


def alpaca_frames(clock, symbols):
    """Alpaca-shaped frames every 0.7 s; each holds several messages per stock that
    regularly straddle a 5-second bar boundary, so message order matters."""
    import math
    from datetime import UTC, datetime

    def stamp(ns):
        text = datetime.fromtimestamp(ns // NS, UTC).strftime("%Y-%m-%dT%H:%M:%S")
        return f"{text}.{ns % NS:09d}Z"

    async def frames():
        k = 0
        while True:
            t = START + k * 700_000_000
            while clock() < t:
                await asyncio.sleep(0)
            frame = [{"T": "subscription", "quotes": symbols}] if k == 0 else []
            for j, offset in enumerate((650, 300, 50)):
                event_ns = t - offset * 1_000_000
                for n, symbol in enumerate(symbols):
                    price = round(100 + 0.4 * math.sin((3 * k + j) / (40 + 7 * n)), 2)
                    frame.append(
                        {
                            "T": "q",
                            "S": symbol,
                            "t": stamp(event_ns),
                            "bp": price - 0.01,
                            "ap": price + 0.01,
                            "bs": 3,
                            "as": 4,
                        }
                    )
                    frame.append(
                        {
                            "T": "t",
                            "S": symbol,
                            "t": stamp(event_ns + 1),
                            "p": price,
                            "s": 10,
                            "i": k * 10 + j,
                            "c": ["@"],
                        }
                    )
            yield frame, t + 50_000
            k += 1

    return frames


def _run_with_frames(tmp_path, frames_mode):
    clock = Clock(START)
    exchange = Exchange(clock)
    budgets = {"NVDA": 300, "AAPL": 200}
    source = alpaca_frames(clock, list(budgets))

    async def stream(symbols, *, feed, raw=False):
        yield "connected", None, clock()
        async for frame, arrival in source():
            if raw:
                yield "frame", json.dumps(frame).encode(), arrival
            else:
                for message in frame:
                    if message["T"] in ("q", "t"):
                        yield "event", {**message, "arrival_ns": arrival}, arrival

    broker = PortfolioBroker(
        exchange,
        budgets,
        Costs(0, 0),
        lambda s: budget_gateway(budgets[s]),
        run_dir=tmp_path / "state",
        identity="run",
        clock=clock,
    )
    config = TradeConfig(
        "test",
        tuple(budgets),
        budgets,
        parse_strategy("ema-crossover"),
        log_dir=tmp_path / "logs",
        run_dir=tmp_path / "state",
        engine="cpp",
        frames=frames_mode,
    )
    runner = TradeRunner(
        config, broker, [SESSION], stream_factory=stream, clock=clock, sleep=clock.sleep
    )
    try:
        summary = asyncio.run(runner.run(until_ns=START + 900 * NS))
    finally:
        broker.close()
    journals = {}
    for path in sorted((tmp_path / "logs").glob("*.jsonl")):
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        journals[path.name.split("-")[0]] = [
            (r["event"], r.get("event_ns"), r.get("reason"), r.get("action"), r.get("equity"))
            for r in rows
        ]
    return exchange.posts, summary, journals


def test_native_frames_trade_exactly_like_the_python_stream_path(tmp_path, monkeypatch):
    pytest.importorskip("hftcore")
    monkeypatch.setattr("hft.runtime.time.monotonic", lambda: 0.0)
    expected = _run_with_frames(tmp_path / "python", "python")
    actual = _run_with_frames(tmp_path / "native", "native")
    posts, _, journals = expected
    assert ("NVDA", "buy") in posts and ("AAPL", "buy") in posts  # decisions happened
    assert sum(1 for row in journals["NVDA"] if row[0] == "decision") > 50
    assert actual == expected


def test_native_frames_require_the_cpp_engine(tmp_path):
    clock = Clock(START)
    broker = PortfolioBroker(
        Exchange(clock),
        {"NVDA": 300},
        Costs(0, 0),
        lambda s: budget_gateway(300),
        run_dir=tmp_path / "state",
        identity="run",
        clock=clock,
    )
    config = TradeConfig(
        "test",
        ("NVDA",),
        {"NVDA": 300},
        parse_strategy("ema-crossover"),
        log_dir=tmp_path / "logs",
        run_dir=tmp_path / "state",
        engine="python",
        frames="native",
    )
    try:
        with pytest.raises(ValueError, match="C\\+\\+ engine"):
            TradeRunner(config, broker, [SESSION], clock=clock, sleep=clock.sleep)
    finally:
        broker.close()


def test_a_brief_backlog_restarts_warmup_instead_of_halting_the_day(tmp_path, monkeypatch):
    monkeypatch.setattr("hft.runtime.time.monotonic", lambda: 0.0)
    clock = Clock(START)
    exchange = Exchange(clock)
    events = stream_factory(clock, 10**15)

    async def stalled(symbols, *, feed):
        async for kind, message, arrival in events(symbols, feed=feed):
            if kind == "event" and START + 20 * NS <= arrival < START + 22 * NS:
                stale = arrival - 2 * NS  # delivered two seconds late, like a stalled loop
                yield kind, {**message, "arrival_ns": stale}, stale
            else:
                yield kind, message, arrival

    budgets = {"NVDA": 300}
    broker = PortfolioBroker(
        exchange,
        budgets,
        Costs(0, 0),
        lambda s: budget_gateway(300),
        run_dir=tmp_path / "state",
        identity="run",
        clock=clock,
    )
    config = TradeConfig(
        "test",
        ("NVDA",),
        budgets,
        parse_strategy("hold-day"),
        log_dir=tmp_path / "logs",
        run_dir=tmp_path / "state",
    )
    runner = TradeRunner(
        config, broker, [SESSION], stream_factory=stalled, clock=clock, sleep=clock.sleep
    )
    try:
        asyncio.run(runner.run(until_ns=START + 420 * NS))
    finally:
        broker.close()
    rows = [
        json.loads(r)
        for r in next((tmp_path / "logs").glob("NVDA-*.jsonl")).read_text().splitlines()
    ]
    assert [r.get("reason") for r in rows].count("processing_backlog") == 1
    assert ("NVDA", "buy") in exchange.posts  # warmed up again and traded the same day


def test_rest_latency_does_not_age_market_events_into_a_backlog(tmp_path, monkeypatch):
    """The third paper session reset warmup 30 times in three hours: events waited in
    the queue while the loop made ~0.5 s of sequential REST calls."""
    monkeypatch.setattr("hft.runtime.time.monotonic", lambda: 0.0)
    clock = Clock(START)

    class SlowExchange(Exchange):
        def request(self, method, path, body=None):
            self.clock.now += 300_000_000  # 0.3 s per REST call
            return super().request(method, path, body)

    exchange = SlowExchange(clock)
    budgets = {"NVDA": 300}
    broker = PortfolioBroker(
        exchange,
        budgets,
        Costs(0, 0),
        lambda s: budget_gateway(300),
        run_dir=tmp_path / "state",
        identity="run",
        clock=clock,
    )
    config = TradeConfig(
        "test",
        ("NVDA",),
        budgets,
        parse_strategy("hold-day"),
        log_dir=tmp_path / "logs",
        run_dir=tmp_path / "state",
    )
    events = stream_factory(clock, 10**15)

    async def received(symbols, *, feed):
        # Stamp arrival when the receive task gets the event, as the live stream does.
        async for kind, message, _ in events(symbols, feed=feed):
            now = clock()
            yield kind, ({**message, "arrival_ns": now} if message else message), now

    runner = TradeRunner(
        config, broker, [SESSION], stream_factory=received, clock=clock, sleep=clock.sleep
    )
    try:
        asyncio.run(runner.run(until_ns=START + 420 * NS))
    finally:
        broker.close()
    rows = [
        json.loads(r)
        for r in next((tmp_path / "logs").glob("NVDA-*.jsonl")).read_text().splitlines()
    ]
    assert [r.get("reason") for r in rows].count("processing_backlog") == 0
    assert ("NVDA", "buy") in exchange.posts
