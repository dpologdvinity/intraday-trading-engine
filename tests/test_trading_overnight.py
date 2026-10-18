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
from hft.trading.overnight import OvernightBook, OvernightConfig, OvernightRunner

DAY1 = SessionWindow("2025-01-06", 1_736_173_800 * NS, 1_736_197_200 * NS)  # 14:30-21:00 UTC
DAY2 = SessionWindow("2025-01-07", DAY1.open_ns + 86_400 * NS, DAY1.close_ns + 86_400 * NS)
START = DAY1.close_ns - 3600 * NS  # 15:00 ET


class Clock:
    def __init__(self, now):
        self.now = now

    def __call__(self):
        return self.now

    async def sleep(self, seconds):
        self.now += int(seconds * NS)
        await asyncio.sleep(0)


class Exchange:
    """Paper account with auctions: on-close orders fill at the close, on-open orders and
    pre-open day orders at the next open, day orders immediately while the market is open."""

    mode = "broker-paper"

    def __init__(self, clock, prices, cash=Decimal(5000)):
        self.clock, self.prices, self.cash = clock, prices, cash
        self.positions, self.orders, self.posts = {}, {}, []

    def _price(self, symbol, at):
        return self.prices(symbol, at)

    def _settle(self, order):
        if order["status"] != "new":
            return
        now = self.clock()
        due = order["_due"]
        if due is None or now < due:
            return
        price = self._price(order["symbol"], due)
        if (
            order.get("limit_price")
            and order["side"] == "buy"
            and price > Decimal(order["limit_price"])
        ):
            if order["time_in_force"] != "day":
                order["status"] = "canceled"  # an on-close limit that the close missed
            return
        qty, sign = Decimal(order["qty"]), 1 if order["side"] == "buy" else -1
        self.positions[order["symbol"]] = self.positions.get(order["symbol"], 0) + sign * qty
        self.cash -= sign * qty * price
        order.update(status="filled", filled_qty=order["qty"], filled_avg_price=str(price))
        order["filled_at"] = "2025-01-06T21:00:00Z"

    def _due(self, body):
        now = self.clock()
        if body["time_in_force"] == "cls":
            return next(w.close_ns for w in (DAY1, DAY2) if w.close_ns > now)
        if body["time_in_force"] == "opg":
            return next(w.open_ns for w in (DAY1, DAY2) if w.open_ns > now)
        window = next((w for w in (DAY1, DAY2) if w.open_ns <= now < w.close_ns), None)
        if window:
            return now
        return next(w.open_ns for w in (DAY1, DAY2) if w.open_ns > now)

    def request(self, method, path, body=None):
        for order in self.orders.values():
            self._settle(order)
        if path.startswith("/v2/account/activities/"):
            return []
        if path == "/v2/account":
            return {"id": "paper-1", "status": "ACTIVE", "cash": str(self.cash)}
        if path == "/v2/clock":
            now = self.clock()
            return {"is_open": any(w.open_ns <= now < w.close_ns for w in (DAY1, DAY2))}
        if path.startswith("/v2/assets/"):
            return {"tradable": True, "fractionable": True, "class": "us_equity"}
        if path == "/v2/positions":
            return [{"symbol": s, "qty": str(q)} for s, q in self.positions.items() if q]
        if path.startswith("/v2/orders?"):
            return [self._public(o) for o in self.orders.values() if o["status"] == "new"]
        if method == "POST":
            self.posts.append((body["symbol"], body["side"], body["time_in_force"], self.clock()))
            order = {
                **body,
                "id": f"o{len(self.orders)}",
                "status": "new",
                "filled_qty": "0",
                "_due": self._due(body),
            }
            self.orders[order["id"]] = order
            self._settle(order)
            return self._public(order)
        if method == "DELETE":
            order = self.orders[path.rsplit("/", 1)[1]]
            if order["status"] == "new":
                order["status"] = "canceled"
            return None
        if path.startswith("/v2/orders/"):
            return self._public(self.orders[path.rsplit("/", 1)[1]])
        raise AssertionError((method, path, body))

    @staticmethod
    def _public(order):
        return {k: v for k, v in order.items() if not k.startswith("_")}


def prices(symbol, at):
    base = {"BIG": Decimal(50), "SMALL": Decimal(400)}[symbol]
    return base * (Decimal("1.02") if at >= DAY2.open_ns else 1)  # +2% overnight


def _run(tmp_path, clock, exchange, budgets, until, guard=None, quotes=None):
    broker = PortfolioBroker(
        exchange,
        budgets,
        Costs(0, 0),
        lambda s: budget_gateway(budgets[s]),
        run_dir=tmp_path / "state",
        identity="run",
        clock=clock,
        book_class=OvernightBook,
    )
    config = OvernightConfig(
        "test",
        tuple(budgets),
        budgets,
        parse_strategy("overnight-drift"),
        log_dir=tmp_path / "logs",
        run_dir=tmp_path / "state",
    )
    runner = OvernightRunner(
        config,
        broker,
        [DAY1, DAY2],
        prices=quotes or (lambda symbols: {s: (prices(s, clock()), clock()) for s in symbols}),
        clock=clock,
        sleep=clock.sleep,
    )
    if guard:
        guard(runner)
    try:
        summary = asyncio.run(runner.run(until_ns=until))
    finally:
        broker.close()
    return summary


def _rows(tmp_path, symbol):
    path = next((tmp_path / "logs").glob(f"{symbol}-*.jsonl"))
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_buys_at_the_close_and_sells_at_the_next_open(tmp_path):
    clock = Clock(START)
    exchange = Exchange(clock, prices)
    summary = _run(tmp_path, clock, exchange, {"BIG": 1000, "SMALL": 100}, DAY2.open_ns + 1800 * NS)
    sides = [(s, side, tif) for s, side, tif, _ in exchange.posts]
    assert sides == [
        ("BIG", "buy", "cls"),  # 19 whole shares use 95% of the cash: closing auction
        ("SMALL", "buy", "day"),  # under one share: fractional day order before the close
        ("BIG", "sell", "opg"),
        ("SMALL", "sell", "day"),  # fractional: day market order sent before 09:28
    ]
    times = [t for *_, t in exchange.posts]
    assert DAY1.close_ns - 15 * 60 * NS <= times[0] < DAY1.close_ns - 10 * 60 * NS
    assert DAY1.close_ns - 3 * 60 * NS <= times[1] < DAY1.close_ns - 60 * NS
    assert DAY2.open_ns - 30 * 60 * NS <= times[2] < DAY2.open_ns - 150 * NS
    assert DAY2.open_ns - 30 * 60 * NS <= times[3] < DAY2.open_ns - 150 * NS
    assert exchange.positions == {"BIG": 0, "SMALL": 0}
    assert summary["BIG"]["trades"] == 1 and summary["SMALL"]["trades"] == 1
    big = Decimal(summary["BIG"]["cash"])
    assert big == Decimal(1000) + 19 * Decimal(50) * Decimal("0.02")  # +2% on 19 shares
    assert Decimal(summary["SMALL"]["cash"]) > Decimal(100)
    rows = _rows(tmp_path, "BIG")
    assert [r["event"] for r in rows].count("trade") == 1
    assert rows[-1]["event"] == "finish"


def test_drawdown_stops_entries_but_held_stock_still_exits(tmp_path):
    clock = Clock(START)
    exchange = Exchange(clock, prices)

    def latched(runner):
        runner.guard.latched = runner.guard.halted = "account_drawdown"

    _run(tmp_path, clock, exchange, {"BIG": 1000}, DAY2.open_ns + 1800 * NS, guard=latched)
    assert exchange.posts == []
    assert any(r.get("reason") == "account_drawdown" for r in _rows(tmp_path, "BIG"))


def test_a_restart_after_the_open_sells_the_overnight_position(tmp_path):
    clock = Clock(START)
    exchange = Exchange(clock, prices)
    budgets = {"BIG": 1000}
    _run(tmp_path, clock, exchange, budgets, DAY1.close_ns + 600 * NS)  # stops overnight
    assert exchange.positions == {"BIG": 19}
    clock.now = DAY2.open_ns + 300 * NS  # missed the pre-open exit
    summary = _run(tmp_path, clock, exchange, budgets, DAY2.open_ns + 900 * NS)
    assert exchange.posts[-1][:3] == ("BIG", "sell", "day")
    assert exchange.positions == {"BIG": 0}
    assert summary["BIG"]["position"] == "0"


def test_auction_orders_must_be_whole_shares(tmp_path):
    clock = Clock(DAY1.close_ns - 900 * NS)
    broker = PortfolioBroker(
        Exchange(clock, prices),
        {"SMALL": 100},
        Costs(0, 0),
        lambda s: budget_gateway(100),
        run_dir=tmp_path / "state",
        identity="run",
        clock=clock,
        book_class=OvernightBook,
    )
    try:
        book = broker.book("SMALL")
        decision = book.submit_order("buy", "0.2", "cls", 400, clock(), limit=400)
        assert decision.reason == "fractional_auction_order"
        assert book.submit_order("buy", 1, "cls", 400, clock()).reason == "buy_needs_limit"
        assert book.submit_order("buy", 1, "cls", 400, clock(), limit=400).reason == "cash_limit"
        assert book.submit_order("buy", "0.2", "day", 400, clock(), limit=400).allowed
    finally:
        broker.close()


def test_scheduled_strategy_is_parsed_and_refused_by_bar_replay():
    from types import SimpleNamespace

    from hft.trading.command import handle_replay

    strategy = parse_strategy("overnight-drift")
    assert strategy.is_scheduled and not strategy.is_rule
    args = SimpleNamespace(
        live=False, replay="2026-10-07", symbols=["NVDA=100"], strategy="overnight-drift"
    )
    with pytest.raises(ValueError, match="multiday_rules"):
        handle_replay(args)


def test_a_latched_drawdown_survives_a_restart(tmp_path):
    clock = Clock(START)
    exchange = Exchange(clock, prices)

    def latched(runner):
        runner.guard.peak = Decimal(10_000)  # equity far below the peak: drawdown latches

    _run(tmp_path, clock, exchange, {"BIG": 1000}, DAY1.close_ns - 600 * NS, guard=latched)
    assert exchange.posts == []
    clock.now = DAY2.close_ns - 3600 * NS
    _run(tmp_path, clock, exchange, {"BIG": 1000}, DAY2.close_ns)
    assert exchange.posts == []  # still halted after the restart


def test_price_failures_are_logged_and_skip_entries_without_stopping(tmp_path):
    clock = Clock(START)
    exchange = Exchange(clock, prices)

    def broken(symbols):
        raise RuntimeError("market data unavailable")

    summary = _run(
        tmp_path, clock, exchange, {"BIG": 1000}, DAY1.close_ns + 600 * NS, quotes=broken
    )
    assert exchange.posts == [] and summary["BIG"]["position"] == "0"
    reasons = [r.get("reason") or "" for r in _rows(tmp_path, "BIG")]
    assert any("market data unavailable" in r for r in reasons)


def test_the_cli_starts_an_overnight_run_on_the_paper_account(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from hft.trading import command

    clock = Clock(START)
    exchange = Exchange(clock, prices)
    started = {}

    async def run_once(runner):
        started["runner"] = runner
        return runner.summary()

    monkeypatch.setattr("hft.broker.AlpacaClient", lambda mode: exchange)
    monkeypatch.setattr("hft.cli._read_credentials", lambda: None)
    monkeypatch.setattr("hft.runtime.fetch_calendar", lambda client: [DAY1, DAY2])
    monkeypatch.setattr("hft.calendar.calendar_from_records", lambda records: list(records))
    monkeypatch.setattr(command, "_run_until_signal", run_once)
    monkeypatch.setattr(command, "ROOT", tmp_path)
    monkeypatch.setattr(
        command, "run_paths", lambda name: (tmp_path / "state" / name, tmp_path / "logs" / name)
    )
    args = SimpleNamespace(
        status=False,
        replay=None,
        live=False,
        paper=True,
        symbols=["BIG=1000", "SMALL=100"],
        strategy="overnight-drift",
        name="night",
        daily_loss=0.02,
        max_drawdown=0.05,
        max_spread_bps=10.0,
        engine="auto",
        frames="python",
    )
    summary = command.handle_trade(args)
    runner = started["runner"]
    assert type(runner).__name__ == "OvernightRunner"
    assert type(runner.broker.book("BIG")).__name__ == "OvernightBook"
    assert set(summary) == {"BIG", "SMALL"}


def test_an_on_close_fill_above_its_sizing_cap_is_kept_and_sold_at_the_open(tmp_path):
    def jumpy(symbol, at):
        if at >= DAY2.open_ns:
            return Decimal(57)
        return Decimal(56) if at >= DAY1.close_ns else Decimal(50)  # close 12% above 15:45

    clock = Clock(START)
    exchange = Exchange(clock, jumpy)
    budgets = {"BIG": 1000}
    summary = _run(
        tmp_path,
        clock,
        exchange,
        budgets,
        DAY2.open_ns + 1800 * NS,
        quotes=lambda symbols: {s: (jumpy(s, clock()), clock()) for s in symbols},
    )
    sides = [(s, side, tif) for s, side, tif, _ in exchange.posts]
    assert sides == [("BIG", "buy", "cls"), ("BIG", "sell", "opg")]
    assert exchange.positions == {"BIG": 0} and summary["BIG"]["position"] == "0"
    assert Decimal(summary["BIG"]["cash"]) == Decimal(1000) + 19 * Decimal(1)  # 56 -> 57
    reasons = [r.get("reason") or "" for r in _rows(tmp_path, "BIG")]
    assert any("more than the stock's cash" in r for r in reasons)


def test_a_daily_loss_halt_survives_a_restart_in_the_entry_window(tmp_path):
    clock = Clock(START)
    exchange = Exchange(clock, prices)

    def lost_today(runner):
        runner.guard.day_start = Decimal(10_000)  # equity is far below the day's start

    budgets = {"BIG": 1000}
    _run(tmp_path, clock, exchange, budgets, DAY1.close_ns - 12 * 60 * NS, guard=lost_today)
    assert exchange.posts == []
    # Restarted while the on-close window is still open: the halt must hold.
    _run(tmp_path, clock, exchange, budgets, DAY1.close_ns)
    assert exchange.posts == []


def test_a_contradicting_broker_report_stops_the_run(tmp_path, monkeypatch):
    from hft.broker import BrokerIntegrityError

    clock = Clock(START)
    exchange = Exchange(clock, prices)
    original = OvernightBook.poll

    def contradicted(self):
        if exchange.posts:
            raise BrokerIntegrityError("broker cumulative fill moved backward")
        return original(self)

    monkeypatch.setattr(OvernightBook, "poll", contradicted)
    with pytest.raises(BrokerIntegrityError):
        _run(tmp_path, clock, exchange, {"BIG": 1000}, DAY2.open_ns + 1800 * NS)


class FakePicker:
    def __init__(self):
        self.symbols = ["BIG", "SMALL"]
        self.refreshed, self.opens = [], []

    def refresh(self, today, calendar):
        self.refreshed.append(today)

    def pick(self, today, opens, calendar):
        self.opens.append(opens)
        return ["SMALL", "NOT-IN-RUN"]


def test_news_picks_buy_after_the_open_and_sell_before_the_close(tmp_path):
    from hft.trading.news_picks import NewsPicksRunner

    clock = Clock(START)
    exchange = Exchange(clock, prices)
    budgets = {"BIG": 1000, "SMALL": 100}
    broker = PortfolioBroker(
        exchange,
        budgets,
        Costs(0, 0),
        lambda s: budget_gateway(budgets[s]),
        run_dir=tmp_path / "state",
        identity="run",
        clock=clock,
        book_class=OvernightBook,
    )
    config = OvernightConfig(
        "test",
        tuple(budgets),
        budgets,
        parse_strategy("news-picks:/frozen"),
        log_dir=tmp_path / "logs",
        run_dir=tmp_path / "state",
    )
    picker = FakePicker()
    runner = NewsPicksRunner(
        config,
        broker,
        [DAY1, DAY2],
        picker=picker,
        clock=clock,
        sleep=clock.sleep,
        prices=lambda symbols: {s: (prices(s, clock()), clock()) for s in symbols},
    )
    try:
        summary = asyncio.run(runner.run(until_ns=DAY2.close_ns + 600 * NS))
    finally:
        broker.close()
    assert [d.isoformat() for d in picker.refreshed] == ["2025-01-07"]  # day 1's window had passed
    assert set(picker.opens[0]) >= {"BIG", "SMALL"}
    sides = [(s, side, tif) for s, side, tif, _ in exchange.posts]
    assert sides == [("SMALL", "buy", "day"), ("SMALL", "sell", "day")]
    bought, sold = exchange.posts[0][3], exchange.posts[1][3]
    assert DAY2.open_ns + 60 * NS <= bought < DAY2.open_ns + 600 * NS
    assert DAY2.close_ns - 300 * NS <= sold < DAY2.close_ns
    assert Decimal(summary["SMALL"]["position"]) == 0 and summary["SMALL"]["trades"] == 1
    rows = _rows(tmp_path, "SMALL")
    assert any(r["event"] == "picks" and r["picked"] for r in rows)


def test_last_trade_prices_batches_symbols(monkeypatch):
    from hft.trading import overnight

    calls = []

    class FakeClient:
        def get(self, url, params):
            calls.append((url, params))
            names = params["symbols"].split(",")
            return {"trades": {n: {"p": 10.5, "t": "2025-01-06T14:31:00Z"} for n in names}}

    monkeypatch.setattr(overnight, "_DATA_CLIENT", FakeClient())
    monkeypatch.setattr(overnight, "LATEST_BATCH", 2)
    prices = overnight.last_trade_prices(["AAPL", "MSFT", "KO"])
    assert [c[1]["symbols"] for c in calls] == ["AAPL,MSFT", "KO"]
    assert all(c[0].endswith("/v2/stocks/trades/latest") for c in calls)
    assert prices["KO"] == (Decimal("10.5"), 1_736_173_860 * NS)

    class Missing(FakeClient):
        def get(self, url, params):
            return {"trades": {}}

    monkeypatch.setattr(overnight, "_DATA_CLIENT", Missing())
    with pytest.raises(ValueError, match="no latest trade for AAPL"):
        overnight.last_trade_prices(["AAPL"])
