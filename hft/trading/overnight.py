"""Overnight drift on the paper account: buy near the close, sell at the next open.

Stocks have earned most of their return between the close and the next open,
but on free data the spread paid to trade at those times eats the effect
(docs/multiday-results.md). Auction orders avoid the spread, so this runner trades
through the auctions wherever Alpaca allows it and is judged forward on paper:

- Entry, 15:45-15:49:30 ET: a market-on-close order (`time_in_force=cls`) for whole
  shares when they use at least 90% of the stock's cash. Limit-on-close would bound
  the price, but Alpaca paper expired all three sent on 2026-10-08 unfilled. Otherwise, 15:57-15:59 ET:
  a fractional marketable limit order. Alpaca accepts fractional quantities only
  for day orders, and refuses on-close orders after the 15:50 cutoff.
- Exit, 09:00-09:27:30 ET: a market-on-open order (`opg`) for whole-share
  positions, or a day market order for fractional ones (Alpaca fills market orders
  received before 09:28 at the official opening price). Any position still held
  after the open is sold with a day market order, retried each minute.

Day buys are limit orders sized so quantity x limit fits the stock's cash.
Market-on-close buys are sized as if filled 2% above the 15:45 trade; a larger rise
into the close still fills, and the ledger records it (its cash goes negative and
that stock's entries wait) so the holding is sold at the open as usual. There is no
spread guard: auctions trade at the official price and day buys are capped 1% above
the last trade. Positions are held overnight on purpose (owner decision,
2026-10-08). Each stock keeps its own ledger and order book on one paper
account, reusing the `PortfolioBroker` startup checks, durable order state and
reconciliation. A run that stops overnight keeps its positions; restarting it
sells them at the next open. Entries stop for the day after the account-wide daily
loss limit and for good after the drawdown limit; both decisions are saved, so a
restart cannot lift them. Exits are never blocked. Dividends paid on the stocks are
booked to their ledgers. Network and broker errors are logged and retried; a broker
report that contradicts the ledger, a reconciliation mismatch that survives a fresh
poll, or an order unresolved for three hours stops the run for review.
"""

import asyncio
import time
from dataclasses import dataclass
from decimal import ROUND_DOWN, Decimal
from pathlib import Path
from uuid import uuid4

from ..account import decimal
from ..broker import BrokerHTTPError, BrokerIntegrityError, BrokerOrder
from ..logs import EventLog
from ..risk import RiskDecision
from ..sizing import OrderIntent
from ..strategies import StrategySpec
from .broker import MAX_REJECTIONS_PER_SESSION, SymbolBook
from .budget import AccountGuard

NS = 1_000_000_000
MINUTE = 60 * NS
AUCTION_ENTRY = (15 * MINUTE, 10 * MINUTE + 30 * NS)  # before the close
FRACTIONAL_ENTRY = (3 * MINUTE, 1 * MINUTE)  # before the close
PRE_OPEN_EXIT = (30 * MINUTE, 2 * MINUTE + 30 * NS)  # before the open
WHOLE_SHARE_USE = Decimal("0.9")  # whole shares must use this much of the cash
AUCTION_CAP = Decimal("1.02")  # on-close buys are sized as if filled 2% above the last trade
DAY_LIMIT = Decimal("1.01")  # marketable day limit above the last trade
CASH_USE = Decimal("0.995")
MAX_PRICE_AGE = 10 * MINUTE  # entries need a recent trade
QUANTITY_STEP = Decimal("0.000001")
MINIMUM_NOTIONAL = Decimal(1)
ERROR_BACKOFF_SECONDS = 30.0
STUCK_ORDER_NS = 3 * 3600 * NS  # auction orders resolve within 45 minutes of submission


@dataclass(frozen=True)
class OvernightConfig:
    name: str
    symbols: tuple[str, ...]
    budgets: dict
    strategy: StrategySpec
    log_dir: Path
    run_dir: Path
    daily_loss: float = 0.02
    max_drawdown: float = 0.05


class OvernightBook(SymbolBook):
    """A stock's book that sends scheduled orders, including auction orders."""

    def __init__(self, portfolio, symbol, account, risk):
        super().__init__(portfolio, symbol, account, risk)
        # A market-on-close fill above its sizing cap is still a fact: record it, so the
        # holding is sold at the open, instead of losing track of broker shares.
        self.account.allow_overdraft = True

    def submit_order(
        self, side, quantity, time_in_force, reference_price, now_ns, *, limit=None, cap=None
    ):
        quantity, price = decimal(quantity), decimal(reference_price)
        limit = None if limit is None else decimal(limit)
        # Buys need a price ceiling for the cash checks: the limit, or for market-on-close
        # orders (Alpaca paper expires limit-on-close orders unfilled) a sizing cap.
        ceiling = limit if limit is not None else None if cap is None else decimal(cap)
        if self.pending:
            return RiskDecision(False, "order_outstanding")
        if self.rejections >= MAX_REJECTIONS_PER_SESSION and side == "buy":
            return RiskDecision(False, "rejection_limit")
        if now_ns - self.refreshed_ns > 5 * NS:
            return RiskDecision(False, "stale_broker_snapshot")
        if side not in ("buy", "sell") or time_in_force not in ("day", "cls", "opg"):
            return RiskDecision(False, "invalid_intent")
        if not quantity.is_finite() or quantity <= 0 or not price.is_finite() or price <= 0:
            return RiskDecision(False, "invalid_intent")
        if time_in_force != "day" and quantity != quantity.to_integral_value():
            return RiskDecision(False, "fractional_auction_order")
        if side == "sell" and quantity > self.account.position:
            return RiskDecision(False, "inventory_limit")
        if side == "buy":
            if ceiling is None or not ceiling.is_finite() or ceiling <= 0:
                return RiskDecision(False, "buy_needs_limit")
            if self.risk.permanent_halt or self.risk.halted:
                return RiskDecision(False, self.risk.halted or "permanent_halt")
            if self.account.position > 0:
                return RiskDecision(False, "already_long")
            if quantity * ceiling > self.account.cash:
                return RiskDecision(False, "cash_limit")
            snapshot = self.account_snapshot
            free = min(
                decimal(snapshot["cash"]),
                decimal(snapshot.get("non_marginable_buying_power", snapshot["cash"])),
            )
            if quantity * ceiling > free:
                return RiskDecision(False, "broker_cash_limit")
        intent = OrderIntent(
            "hft-" + uuid4().hex, self.symbol, side, quantity, ceiling or price, now_ns
        )
        self.pending = BrokerOrder(intent)
        if side == "buy":
            self.risk.record_order(now_ns)
        self._save()  # persist intent and unique ID BEFORE any network mutation
        body = {
            "symbol": self.symbol,
            "qty": str(quantity),
            "side": side,
            "type": "market" if limit is None else "limit",
            "time_in_force": time_in_force,
            "client_order_id": intent.client_order_id,
            "extended_hours": False,
        }
        if limit is not None:
            body["limit_price"] = str(limit)
        return self._send(intent, body)

    def poll(self):
        """Like `AlpacaBroker.poll`, but a submission that never reached the broker is
        cleared once a lookup by its client id confirms it does not exist."""
        try:
            return super().poll()
        except BrokerHTTPError as error:
            order = self.pending
            stale = order and self.clock() - order.intent.created_ns > MINUTE
            if error.status == 404 and order and not order.id and stale:
                self.pending = None
                self._save()
                return ()
            raise

    def cancel(self):
        """A cancel refused because the order is already final is confirmed by `poll`."""
        try:
            super().cancel()
        except BrokerHTTPError as error:
            if error.status not in (404, 422):
                raise


_DATA_CLIENT = None


LATEST_BATCH = 100  # symbols per latest-trades request


def last_trade_prices(symbols):
    """{symbol: (price, trade time in UTC ns)} from the latest IEX trades (read-only)."""
    from ..feed import parse_timestamp
    from ..history import DATA_BASE, ReadOnlyClient

    global _DATA_CLIENT
    if _DATA_CLIENT is None:
        _DATA_CLIENT = ReadOnlyClient()  # one client, so its request budget holds
    client = _DATA_CLIENT
    prices = {}
    for start in range(0, len(symbols), LATEST_BATCH):
        batch = symbols[start : start + LATEST_BATCH]
        payload = client.get(
            f"{DATA_BASE}/v2/stocks/trades/latest", {"symbols": ",".join(batch), "feed": "iex"}
        )
        trades = payload.get("trades") if isinstance(payload, dict) else None
        for symbol in batch:
            trade = trades.get(symbol) if isinstance(trades, dict) else None
            if not isinstance(trade, dict) or "p" not in trade or "t" not in trade:
                raise ValueError(f"no latest trade for {symbol}")
            price = decimal(str(trade["p"]))
            if not price.is_finite() or price <= 0:
                raise ValueError(f"invalid latest trade for {symbol}")
            prices[symbol] = (price, parse_timestamp(trade["t"]))
    return prices


class OvernightRunner:
    def __init__(
        self,
        config,
        broker,
        windows,
        *,
        prices=last_trade_prices,
        clock=time.time_ns,
        sleep=asyncio.sleep,
        calendar_source=None,
    ):
        if not config.strategy.is_scheduled:
            raise ValueError("the overnight runner trades the overnight-drift strategy")
        self.config, self.broker, self.windows = config, broker, windows
        self.prices, self.clock, self.sleep = prices, clock, sleep
        self.calendar_source = calendar_source
        self.started_ns = clock()
        self.guard = AccountGuard(
            sum((decimal(b) for b in config.budgets.values()), Decimal(0)),
            config.daily_loss,
            config.max_drawdown,
        )
        if "guard" in broker.extra:
            self.guard.restore(broker.extra["guard"])
        self.attempted = set()  # (symbol, session_id, step) sent to the broker
        self.retry_after = {}  # (symbol, session_id, step) -> ns after a local refusal
        self.session_id = None
        self.last_prices = {}  # symbol -> (price, trade ns)
        self.logs = {}
        for symbol in config.symbols:
            log = EventLog(config.log_dir / f"{symbol}-{self.started_ns}.jsonl", clock=clock)
            book = broker.book(symbol)
            log.write(
                "start",
                event_ns=self.started_ns,
                symbol=symbol,
                source="trade-paper",
                synthetic=False,
                initial_cash=book.account.initial_cash,
                position=book.account.position,
                metadata={
                    "run": config.name,
                    "strategy": config.strategy.name,
                    "strategy_version": config.strategy.version,
                    "account_id": broker.account_id,
                },
            )
            self.logs[symbol] = log
        self.trades_seen = {s: len(broker.book(s).account.trades) for s in config.symbols}

    # Helpers --------------------------------------------------------------
    def _window(self, now):
        """The next session that has not closed yet."""
        return next((w for w in self.windows if w.close_ns > now), None)

    @staticmethod
    def _near(window, now):
        """Within the order windows, plus margin for fills and retries after the open."""
        return (
            window.open_ns - PRE_OPEN_EXIT[0] - MINUTE <= now < window.open_ns + 20 * MINUTE
            or window.close_ns - AUCTION_ENTRY[0] - MINUTE <= now < window.close_ns + 5 * MINUTE
        )

    def _quality(self, reason, symbols=None):
        known = [s for s in symbols or () if s in self.logs] or list(self.logs)
        for symbol in known:  # symbols outside the run (e.g. SPY for features) log run-wide
            self.logs[symbol].write("quality", event_ns=self.clock(), reason=reason)

    def _fetch(self, symbols):
        """Update `last_prices`; returns False (and logs) when the fetch fails."""
        try:
            self.last_prices.update(self.prices(list(symbols)))
            return True
        except (RuntimeError, OSError, KeyError, TypeError, ValueError) as error:
            self._quality(f"price: {error}", symbols)
            return False

    def _fresh_price(self, symbol, now):
        price, stamp = self.last_prices.get(symbol, (None, 0))
        return price if price is not None and now - stamp <= MAX_PRICE_AGE else None

    def _record(self, symbol, execution):
        book, log = self.broker.book(symbol), self.logs[symbol]
        log.write("fill", event_ns=execution.timestamp, execution=execution)
        trades = book.account.trades
        for trade in trades[self.trades_seen[symbol] :]:
            log.write(
                "trade",
                event_ns=trade.closed,
                entry_ns=trade.opened,
                exit_ns=trade.closed,
                pnl=trade.pnl,
                opening_notional=trade.opening_notional,
            )
        self.trades_seen[symbol] = len(trades)
        if book.account.cash < 0:
            self._quality("fill cost more than the stock's cash; entries wait", [symbol])
        log.write(
            "equity",
            event_ns=execution.timestamp,
            equity=book.account.mark(execution.price),
            cash=book.account.cash,
            position=book.account.position,
        )

    def _poll(self):
        """Poll each book separately so one failure cannot hide another book's fills."""
        for symbol in self.config.symbols:
            try:
                fills = self.broker.book(symbol).poll()
            except ValueError as error:  # the ledger refused a confirmed fill: stop for review
                raise BrokerIntegrityError(f"{symbol} fill mismatch: {error}") from error
            except BrokerIntegrityError:
                raise
            except (RuntimeError, OSError) as error:
                self._quality(f"poll: {error}", [symbol])
                continue
            for execution in fills:
                self._record(symbol, execution)

    def _reconcile(self):
        """Reconcile when no order is open; a mismatch must survive a fresh poll."""
        if any(self.broker.book(s).pending for s in self.config.symbols):
            return
        try:
            self.broker.reconcile()
        except RuntimeError as first:
            self._quality(f"reconcile retry: {first}")
            self._poll()
            self.broker.refresh()
            self.broker.reconcile()  # a second mismatch stops the run
        self.broker.save()

    def _submit(self, symbol, side, quantity, tif, price, now, step, window, limit=None, cap=None):
        key = (symbol, window.session_id, step)
        if self.clock() < self.retry_after.get(key, 0):
            return None
        if self.clock() - self.broker.refreshed_ns > 4 * NS:
            self.broker.refresh()
        book = self.broker.book(symbol)
        decision = book.submit_order(side, quantity, tif, price, self.clock(), limit=limit, cap=cap)
        sent = decision.allowed or (decision.reason or "").startswith("broker_rejected")
        if sent:
            self.attempted.add(key)
        else:
            self.retry_after[key] = self.clock() + MINUTE  # local refusal: retry next minute
        self.logs[symbol].write(
            "order",
            event_ns=now,
            side=side,
            quantity=quantity,
            time_in_force=tif,
            reference_price=price,
            limit_price=limit,
            allowed=decision.allowed,
            reason=decision.reason,
            step=step,
        )
        return decision

    # Schedule ---------------------------------------------------------------
    def _exits(self, window, now):
        pre_open = window.open_ns - PRE_OPEN_EXIT[0] <= now < window.open_ns - PRE_OPEN_EXIT[1]
        session = window.open_ns <= now < window.close_ns - AUCTION_ENTRY[0]
        if not (pre_open or session):
            return
        for symbol in self.config.symbols:
            book = self.broker.book(symbol)
            position = book.account.position
            if position <= 0 or book.pending:
                continue
            price = self.last_prices.get(symbol, (book.account.entry_price, 0))[0]
            if pre_open:
                if (symbol, window.session_id, "exit") in self.attempted:
                    continue
                whole = position == position.to_integral_value()
                tif = "opg" if whole else "day"
                self._submit(symbol, "sell", position, tif, price, now, "exit", window)
            else:
                # Retry each minute: an unfilled opening order, or a run restarted late.
                step = f"exit-{(now - window.open_ns) // MINUTE}"
                if (symbol, window.session_id, step) not in self.attempted:
                    self._submit(symbol, "sell", position, "day", price, now, step, window)

    def _cancel_late_entries(self, window, now):
        """A day limit the price ran away from would stay open all day."""
        for symbol in self.config.symbols:
            book = self.broker.book(symbol)
            order = book.pending
            if order and order.intent.side == "buy" and not order.cancel_requested:
                self._quality("entry not filled in the entry window; cancelled", [symbol])
                book.cancel()

    def _check_guard(self, window, now):
        """Once per session before entries: one overnight cycle since the last check."""
        key = (None, window.session_id, "guard")
        if key in self.attempted:
            return True
        saved = self.broker.extra.get("guard_session")
        if saved and saved["session"] == window.session_id:
            # Already checked this session before a restart: keep its decision rather than
            # re-observing against the new baseline, which would lift a daily-loss halt.
            self.attempted.add(key)
            self._apply_guard(saved["reason"], now, log=False)
            return True
        if self.clock() < self.retry_after.get(key, 0):
            return False
        if not self._fetch(self.config.symbols):
            self.retry_after[key] = self.clock() + 10 * NS
            return False
        equity = Decimal(0)
        for symbol in self.config.symbols:
            account = self.broker.book(symbol).account
            price = self._fresh_price(symbol, now)
            if account.position > 0 and price is None:
                self._quality("guard: no recent price for a held stock", [symbol])
                self.retry_after[key] = self.clock() + 10 * NS
                return False  # never mark a holding at zero; retry shortly
            equity += account.mark(price) if price is not None else account.cash
        self.attempted.add(key)
        reason = self.guard.observe(equity)
        self.guard.start_session(equity)
        self.broker.extra["guard"] = self.guard.to_state()
        self.broker.extra["guard_session"] = {"session": window.session_id, "reason": reason}
        self._apply_guard(reason, now, log=True)
        self.broker.save()
        return True

    def _apply_guard(self, reason, now, *, log):
        for symbol in self.config.symbols:
            book = self.broker.book(symbol)
            book.risk.halted = reason or self.guard.latched
            book.risk.permanent_halt = book.risk.permanent_halt or self.guard.latched is not None
            if reason and log:
                self.logs[symbol].write("halt", event_ns=now, reason=reason)
            if not log:
                continue
            price = self._fresh_price(symbol, now)
            self.logs[symbol].write(
                "equity",
                event_ns=now,
                equity=book.account.mark(price) if price else book.account.cash,
                cash=book.account.cash,
                position=book.account.position,
            )

    def _entries(self, window, now):
        auction = window.close_ns - AUCTION_ENTRY[0] <= now < window.close_ns - AUCTION_ENTRY[1]
        fractional = (
            window.close_ns - FRACTIONAL_ENTRY[0] <= now < window.close_ns - FRACTIONAL_ENTRY[1]
        )
        if not (auction or fractional) or not self._check_guard(window, now):
            return
        for symbol in self.config.symbols:
            book = self.broker.book(symbol)
            if book.account.position > 0 or book.pending or book.risk.halted:
                continue
            key = (symbol, window.session_id, "entry-late")
            if fractional and (
                key in self.attempted or self.clock() < self.retry_after.get(key, 0)
            ):
                continue
            if fractional and not self._fetch([symbol]):
                self.retry_after[key] = self.clock() + 10 * NS  # respect the data rate limit
                continue
            price = self._fresh_price(symbol, now)
            if price is None:
                stale = (symbol, window.session_id, "stale")
                if stale not in self.attempted:
                    self.attempted.add(stale)
                    self._quality("entry skipped: no trade in the last 10 minutes", [symbol])
                if fractional:
                    self.retry_after[key] = self.clock() + 10 * NS
                continue
            cash = book.account.cash * CASH_USE
            auction_cap = price * AUCTION_CAP
            whole = (cash / auction_cap).to_integral_value(rounding=ROUND_DOWN)
            use_auction = whole >= 1 and whole * price >= book.account.cash * WHOLE_SHARE_USE
            if auction:
                if use_auction and (symbol, window.session_id, "entry") not in self.attempted:
                    self._submit(
                        symbol, "buy", whole, "cls", price, now, "entry", window, cap=auction_cap
                    )
                continue
            # Fractional stocks enter here; so does a whole-share stock whose on-close
            # order failed or was never sent (a pending on-close order is skipped above).
            limit = (price * DAY_LIMIT).quantize(Decimal("0.01"), ROUND_DOWN)
            quantity = (cash / limit).quantize(QUANTITY_STEP, rounding=ROUND_DOWN)
            if quantity * limit >= MINIMUM_NOTIONAL:
                self._submit(
                    symbol, "buy", quantity, "day", price, now, "entry-late", window, limit
                )
            else:
                self.attempted.add(key)  # too little cash for an order today

    # Main loop --------------------------------------------------------------
    def _tick(self, state):
        now = self.clock()
        for symbol in self.config.symbols:
            order = self.broker.book(symbol).pending
            if order and now - order.intent.created_ns > STUCK_ORDER_NS:
                raise BrokerIntegrityError(f"{symbol} order unresolved for 3 hours")
        busy = any(self.broker.book(s).pending for s in self.config.symbols)
        if now - state["poll"] >= (5 if busy else 60) * NS:
            self._poll()
            state["poll"] = now
        if now - state["reconcile"] >= MINUTE:
            self.broker.refresh()
            self._reconcile()
            state["reconcile"] = now
        window = self._window(now)
        if window and window.session_id != self.session_id:
            self.session_id = window.session_id
            self.broker.start_session()
        if window:
            self._exits(window, self.clock())
            self._entries(window, self.clock())
        if (
            self.calendar_source
            and (window is None or now < window.open_ns - PRE_OPEN_EXIT[0] - MINUTE)
            and now >= state["calendar"]
        ):
            try:
                self.windows = self.calendar_source()
                state["calendar"] = now + 6 * 3600 * NS
            except (RuntimeError, OSError, KeyError, ValueError) as error:
                self._quality(f"calendar: {error}")
                state["calendar"] = now + MINUTE  # keep the old calendar; retry soon
        return window

    async def run(self, *, until_ns=None):
        for symbol, execution in self.broker.recovered_fills:
            self._record(symbol, execution)
        state = {"poll": 0, "reconcile": 0, "calendar": self.clock() + 6 * 3600 * NS}
        try:
            while until_ns is None or self.clock() < until_ns:
                now = self.clock()
                # Broker calls run on this thread: cancelling the task can never leave a
                # request mutating the ledger while shutdown runs.
                try:
                    window = self._tick(state)
                except BrokerHTTPError as error:
                    self._quality(f"broker HTTP {error.status}; retrying")
                    await self.sleep(ERROR_BACKOFF_SECONDS)
                    continue
                except (OSError, ValueError) as error:
                    self._quality(f"error: {error}; retrying")
                    await self.sleep(ERROR_BACKOFF_SECONDS)
                    continue
                except RuntimeError as error:
                    stop = isinstance(error, BrokerIntegrityError) or any(
                        text in str(error) for text in ("mismatch", "unexpected broker")
                    )
                    if stop:
                        raise  # the broker contradicts the ledger: stop for a human
                    self._quality(f"error: {error}; retrying")
                    await self.sleep(ERROR_BACKOFF_SECONDS)
                    continue
                await self.sleep(1.0 if window and self._near(window, now) else 10.0)
        finally:
            await self.shutdown()
        return self.summary()

    async def shutdown(self):
        """Cancel open orders; positions stay and are sold at the next open after a restart."""
        try:
            for symbol in self.config.symbols:
                try:
                    self.broker.book(symbol).cancel()
                except (RuntimeError, OSError) as error:
                    self._quality(f"cancel: {error}", [symbol])
            for _ in range(10):
                self._poll()
                if not any(self.broker.book(s).pending for s in self.config.symbols):
                    break
                await self.sleep(0.5)
        finally:
            self.broker.save()
            for symbol, log in self.logs.items():
                log.write(
                    "finish",
                    event_ns=self.clock(),
                    position=self.broker.book(symbol).account.position,
                )
                log.close()

    def summary(self):
        return {
            symbol: {
                "position": str(self.broker.book(symbol).account.position),
                "cash": str(self.broker.book(symbol).account.cash),
                "trades": len(self.broker.book(symbol).account.trades),
                "halted": self.broker.book(symbol).risk.halted,
            }
            for symbol in self.config.symbols
        }
