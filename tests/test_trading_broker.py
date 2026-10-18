from datetime import UTC, datetime
from decimal import Decimal
from zoneinfo import ZoneInfo

import pytest

from hft.account import Costs
from hft.broker import AlpacaBroker, BrokerHTTPError
from hft.calendar import SessionWindow
from hft.data import NS, Quote
from hft.risk import Snapshot
from hft.sizing import SizingConfig, make_budget_intent
from hft.trading.broker import PortfolioBroker
from hft.trading.budget import budget_gateway

NOW = 1_735_830_000 * NS
SESSION = SessionWindow("2025-01-02", NOW - NS, NOW + 1000 * NS)
LIMITS = SizingConfig(slippage_bps=0, fee_per_share=0)


class Exchange:
    mode = "broker-paper"

    def __init__(self, cash="1000"):
        self.cash = Decimal(cash)
        self.positions = {}
        self.orders = {}
        self.assets = {}
        self.posts = []
        self.reject_posts = False

    def fill(self, symbol, price="100"):
        for order in self.orders.values():
            if order["symbol"] == symbol and order["status"] == "new":
                qty = Decimal(order["qty"])
                order.update(status="filled", filled_qty=order["qty"], filled_avg_price=price)
                sign = 1 if order["side"] == "buy" else -1
                self.positions[symbol] = self.positions.get(symbol, Decimal(0)) + sign * qty
                self.cash -= sign * qty * Decimal(price)

    def request(self, method, path, body=None):
        if path.startswith("/v2/account/activities/"):
            kind = path.split("/")[4].split("?")[0]
            return [
                a for a in getattr(self, "activities", []) if a["activity_type"].startswith(kind)
            ]
        if path == "/v2/account":
            return {"id": "paper-1", "status": "ACTIVE", "cash": str(self.cash)}
        if path == "/v2/clock":
            return {"is_open": True}
        if path.startswith("/v2/assets/"):
            symbol = path.rsplit("/", 1)[1]
            return self.assets.get(
                symbol, {"tradable": True, "fractionable": True, "class": "us_equity"}
            )
        if path == "/v2/positions":
            return [{"symbol": s, "qty": str(q)} for s, q in self.positions.items() if q]
        if path.startswith("/v2/orders?"):
            return [o for o in self.orders.values() if o["status"] == "new"]
        if method == "POST":
            self.posts.append(body["symbol"])
            if self.reject_posts:
                raise BrokerHTTPError(403, "POST")
            order = {
                **body,
                "id": f"o{len(self.orders)}",
                "filled_qty": "0",
                "filled_avg_price": None,
                "status": "new",
            }
            self.orders[order["id"]] = order
            return order
        if ":by_client_order_id?" in path:
            if self.reject_posts:
                raise BrokerHTTPError(404, "GET")
            client_id = path.split("client_order_id=")[1]
            return next(o for o in self.orders.values() if o["client_order_id"] == client_id)
        if path.startswith("/v2/orders/"):
            return self.orders[path.rsplit("/", 1)[1]]
        raise AssertionError((method, path, body))


def portfolio(tmp_path, exchange, budgets=None, identity="run-a"):
    return PortfolioBroker(
        exchange,
        budgets or {"NVDA": 300, "AAPL": 200},
        Costs(0, 0),
        lambda symbol: budget_gateway((budgets or {"NVDA": 300, "AAPL": 200})[symbol]),
        run_dir=tmp_path,
        identity=identity,
        clock=lambda: NOW,
    )


def buy(broker, symbol, action=1):
    book = broker.book(symbol)
    quote = Quote("q", NOW, NOW, Decimal("99.99"), Decimal("100.01"), Decimal(500), Decimal(500))
    intent = make_budget_intent(
        action, book.account, quote, LIMITS, NOW, budget=book.account.initial_cash, symbol=symbol
    )

    def snapshot():
        return Snapshot(
            quote,
            SESSION,
            book.account.position,
            book.account.mark(quote.mid),
            book.account.initial_cash,
            book.account.cash,
            (),
        )

    return book.submit(intent, snapshot, NOW)


def test_two_stocks_trade_independently_and_reconcile(tmp_path):
    exchange = Exchange()
    broker = portfolio(tmp_path, exchange)
    try:
        assert buy(broker, "NVDA").allowed and buy(broker, "AAPL").allowed
        assert exchange.posts == ["NVDA", "AAPL"]  # concurrent pending orders per stock
        exchange.fill("NVDA")
        fills = broker.poll()
        assert [s for s, _ in fills] == ["NVDA"]
        assert broker.book("NVDA").account.position > 0
        assert broker.book("AAPL").account.position == 0
        assert broker.reconcile()
    finally:
        broker.close()


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda e: e.positions.update(MSFT=Decimal(1)), "flat"),
        (lambda e: setattr(e, "cash", Decimal(100)), "exceed"),
        (
            lambda e: e.assets.update(
                NVDA={"tradable": True, "fractionable": False, "class": "us_equity"}
            ),
            "fractionable",
        ),
    ],
)
def test_startup_refusals(tmp_path, change, message):
    exchange = Exchange()
    change(exchange)
    with pytest.raises(ValueError, match=message):
        portfolio(tmp_path, exchange)


def test_live_mode_is_refused(tmp_path):
    exchange = Exchange()
    exchange.mode = "live"
    with pytest.raises(ValueError, match="paper account only"):
        portfolio(tmp_path, exchange)


def test_foreign_position_and_cash_changes_fail_closed(tmp_path):
    exchange = Exchange()
    broker = portfolio(tmp_path, exchange)
    try:
        exchange.positions["MSFT"] = Decimal(1)
        with pytest.raises(RuntimeError, match="unexpected broker position: MSFT"):
            broker.reconcile()
        del exchange.positions["MSFT"]
        exchange.cash += 5
        with pytest.raises(RuntimeError, match="cash mismatch"):
            broker.reconcile()
    finally:
        broker.close()


def test_resume_restores_books_and_refuses_a_changed_identity(tmp_path):
    exchange = Exchange()
    broker = portfolio(tmp_path, exchange)
    buy(broker, "NVDA")
    exchange.fill("NVDA")
    broker.poll()
    position = broker.book("NVDA").account.position
    broker.close()
    resumed = portfolio(tmp_path, exchange)
    try:
        assert resumed.book("NVDA").account.position == position
    finally:
        resumed.close()
    with pytest.raises(ValueError, match="identity changed"):
        portfolio(tmp_path, exchange, identity="run-b")


def test_three_rejections_stop_a_stock_for_the_session(tmp_path):
    exchange = Exchange()
    broker = portfolio(tmp_path, exchange)
    try:
        exchange.reject_posts = True
        reasons = [buy(broker, "NVDA").reason for _ in range(4)]
        assert reasons == ["broker_rejected:403"] * 3 + ["rejection_limit"]
        assert exchange.posts == ["NVDA"] * 3
        broker.start_session()
        exchange.reject_posts = False
        assert buy(broker, "NVDA").allowed
    finally:
        broker.close()


def test_portfolio_and_single_stock_broker_share_the_account_lock(tmp_path):
    exchange = Exchange()
    broker = portfolio(tmp_path / "run", exchange)
    try:
        with pytest.raises(RuntimeError, match="already owned"):
            AlpacaBroker(
                exchange,
                "NVDA",
                budget_gateway(300),
                Costs(0, 0),
                state_path=tmp_path / "single",
                initial_cash=300,
                clock=lambda: NOW,
            )
    finally:
        broker.close()


def test_one_failing_book_does_not_hide_another_books_fills(tmp_path, monkeypatch):
    exchange = Exchange()
    broker = portfolio(tmp_path, exchange)
    try:
        assert buy(broker, "NVDA").allowed and buy(broker, "AAPL").allowed
        exchange.fill("AAPL")

        def broken():
            raise RuntimeError("lookup failed")

        monkeypatch.setattr(broker.book("NVDA"), "poll", broken)
        fills = broker.poll()
        assert [s for s, _ in fills] == ["AAPL"]
        assert broker.poll_errors == {"NVDA": "lookup failed"}
    finally:
        broker.close()


def test_reconcile_allows_the_brokers_per_fill_cent_rounding(tmp_path):
    """The first paper session stopped on a $0.012 drift after eight fills: the broker
    books each fill's cash to the cent, the ledger keeps exact decimals."""
    exchange = Exchange()
    broker = portfolio(tmp_path, exchange)
    try:
        account = broker.book("NVDA").account
        for _ in range(3):  # each round trip leaves the broker 0.8 cents above the ledger
            account.execute(1, Decimal("100.004"), 0, NOW)
            exchange.positions["NVDA"] = Decimal(1)
            exchange.cash -= Decimal("100.00")
            account.execute(-1, Decimal("100.006"), 0, NOW)
            exchange.positions["NVDA"] = Decimal(0)
            exchange.cash += Decimal("100.01")
            assert broker.reconcile()
        exchange.cash -= Decimal("0.05")  # an unexplained charge is still caught
        with pytest.raises(RuntimeError, match="cash mismatch"):
            broker.reconcile()
    finally:
        broker.close()


def test_dividends_on_held_stocks_are_booked_instead_of_stopping(tmp_path):
    exchange = Exchange()
    broker = portfolio(tmp_path, exchange)
    try:
        assert broker.reconcile()
        exchange.cash += Decimal("0.42")
        exchange.activities = [
            {"id": "d1", "activity_type": "DIV", "symbol": "NVDA", "net_amount": "0.42"}
        ]
        assert broker.reconcile()
        account = broker.book("NVDA").account
        assert account.cash == Decimal(300) + Decimal("0.42")
        assert account.income[0]["source"] == "DIV:d1"
        assert broker.reconcile()  # booked once, not again
        exchange.cash += Decimal("0.42")  # the same activity cannot explain new cash
        with pytest.raises(RuntimeError, match="cash mismatch"):
            broker.reconcile()
    finally:
        broker.close()


def test_integrity_errors_are_not_hidden_by_per_book_polling(tmp_path, monkeypatch):
    from hft.broker import BrokerIntegrityError

    exchange = Exchange()
    broker = portfolio(tmp_path, exchange)
    try:

        def contradicted():
            raise BrokerIntegrityError("broker overfill or invalid quantity")

        monkeypatch.setattr(broker.book("NVDA"), "poll", contradicted)
        with pytest.raises(BrokerIntegrityError):
            broker.poll()
    finally:
        broker.close()


def test_regulatory_fees_for_another_runs_trades_are_recorded_as_external(tmp_path):
    """The overnight run stopped at 03:46 when $0.40 of SEC and CAT fees for an
    earlier intraday run's sales were posted to the shared paper account."""
    exchange = Exchange()
    broker = portfolio(tmp_path, exchange)
    try:
        assert broker.reconcile()
        exchange.cash -= Decimal("0.40")
        exchange.activities = [
            {"id": "f1", "activity_type": "FEE", "date": "2026-10-08", "net_amount": "-0.39"},
            {"id": "f2", "activity_type": "FEE", "date": "2026-10-08", "net_amount": "-0.01"},
        ]
        assert broker.reconcile()
        assert [f["id"] for f in broker.extra["external_fees"]] == ["f1", "f2"]
        assert broker.book("NVDA").account.cash == Decimal(300)  # not charged to this run
        assert broker.reconcile()  # booked once
    finally:
        broker.close()


def test_fees_posted_before_a_run_started_are_not_counted_again(tmp_path):
    """news-picks stopped on 2026-10-12: the fresh run counted the 10-08 fees, already
    in its starting cash, together with the new 10-09 fees and found no match."""
    exchange = Exchange()
    exchange.activities = [
        {"id": "f1", "activity_type": "FEE", "date": "2026-10-08", "net_amount": "-0.39"},
        {"id": "f2", "activity_type": "FEE", "date": "2026-10-08", "net_amount": "-0.01"},
    ]
    broker = portfolio(tmp_path, exchange)
    try:
        assert broker.reconcile()
        exchange.cash -= Decimal("0.09")
        exchange.activities += [
            {"id": "f3", "activity_type": "FEE", "date": "2026-10-09", "net_amount": "-0.08"},
            {"id": "f4", "activity_type": "FEE", "date": "2026-10-09", "net_amount": "-0.01"},
        ]
        assert broker.reconcile()
        assert [f["id"] for f in broker.extra["external_fees"]] == ["f3", "f4"]
    finally:
        broker.close()


def test_fees_on_days_the_run_sold_are_split_by_sale_proceeds(tmp_path):
    exchange = Exchange()
    broker = portfolio(tmp_path, exchange)
    try:
        for symbol, price in (("NVDA", "100"), ("AAPL", "100")):
            assert buy(broker, symbol).allowed
            exchange.fill(symbol, price)
        broker.poll()
        for symbol in ("NVDA", "AAPL"):
            assert buy(broker, symbol, action=0).allowed
            exchange.fill(symbol, "100")
        broker.poll()
        assert broker.reconcile()
        day = datetime.fromtimestamp(NOW / 1e9, UTC).astimezone(ZoneInfo("America/New_York")).date()
        before = {s: broker.book(s).account.cash for s in ("NVDA", "AAPL")}
        exchange.cash -= Decimal("0.10")
        exchange.activities = [
            {"id": "f3", "activity_type": "FEE", "date": day.isoformat(), "net_amount": "-0.10"}
        ]
        assert broker.reconcile()
        charged = {s: before[s] - broker.book(s).account.cash for s in before}
        assert sum(charged.values()) == Decimal("0.10")
        assert charged["NVDA"] > charged["AAPL"] > 0  # NVDA sold more (300 vs 200 budget)
    finally:
        broker.close()
