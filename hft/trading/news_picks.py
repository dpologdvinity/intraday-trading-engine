"""Day-trade the stocks a frozen news-aware daily model picks, on the paper account.

The forward test of the best ML day-trader variant: every session is data the model
never saw. Schedule (ET, relative to the exchange calendar):

- 08:55: refresh daily bars (through yesterday), news (through now) and the calendar.
- 09:30:30 to 09:35: read the first IEX trades as today's opening prices and pick
  stocks with the model (`hft.ml.live.NewsPicker`); the picks are journaled.
- 09:31 to 09:40: buy each pick with a marketable day limit 1% above its last trade,
  sized from the stock's cash.
- Five minutes before the close: sell every holding with a day market order, retried
  each minute until flat. A holding found at any other time (for example after a
  restart) that was not bought today is sold at once.

It reuses the overnight runner's paper broker, durable orders, reconciliation,
account guard and error handling; only the schedule differs. Paper only.
"""

from datetime import date
from decimal import ROUND_DOWN, Decimal

from .overnight import (
    CASH_USE,
    DAY_LIMIT,
    MINIMUM_NOTIONAL,
    MINUTE,
    NS,
    QUANTITY_STEP,
    OvernightRunner,
)

REFRESH_BEFORE_OPEN = 35 * MINUTE
PICK_WINDOW = (30 * NS, 5 * MINUTE)  # after the open
ENTRY_WINDOW = (1 * MINUTE, 10 * MINUTE)  # after the open
EXIT_BEFORE_CLOSE = 5 * MINUTE


class NewsPicksRunner(OvernightRunner):
    def __init__(self, config, broker, windows, *, picker, **kwargs):
        super().__init__(config, broker, windows, **kwargs)
        self.picker = picker
        self.picks = {}  # session_id -> list of symbols
        self.refreshed = set()

    @staticmethod
    def _near(window, now):
        return (
            window.open_ns - REFRESH_BEFORE_OPEN - MINUTE <= now < window.open_ns + 15 * MINUTE
            or window.close_ns - EXIT_BEFORE_CLOSE - MINUTE <= now < window.close_ns + 5 * MINUTE
        )

    def _refresh(self, window, now):
        if window.session_id in self.refreshed:
            return
        if not window.open_ns - REFRESH_BEFORE_OPEN <= now < window.open_ns:
            return
        self.refreshed.add(window.session_id)
        try:
            self.picker.refresh(date.fromisoformat(window.session_id), self.windows)
            self._quality("data refreshed")
        except (RuntimeError, OSError, KeyError, TypeError, ValueError) as error:
            self._quality(f"refresh failed: {error}; picks use the data on disk")

    def _pick(self, window, now):
        if window.session_id in self.picks:
            return
        if not window.open_ns + PICK_WINDOW[0] <= now < window.open_ns + PICK_WINDOW[1]:
            return
        key = (None, window.session_id, "pick")
        if self.clock() < self.retry_after.get(key, 0):
            return
        if not self._fetch(self.picker.symbols):
            self.retry_after[key] = self.clock() + 10 * NS
            return
        opens = {
            s: float(price)
            for s, (price, stamp) in self.last_prices.items()
            if stamp >= window.open_ns  # a trade from today's session, not yesterday's
        }
        for name in ("SPY", "QQQ"):
            if name not in opens and self._fetch([name]):
                price, stamp = self.last_prices[name]
                if stamp >= window.open_ns:
                    opens[name] = float(price)
        try:
            picks = self.picker.pick(date.fromisoformat(window.session_id), opens, self.windows)
        except (RuntimeError, OSError, KeyError, TypeError, ValueError) as error:
            self._quality(f"pick failed: {error}")
            self.retry_after[key] = self.clock() + 30 * NS
            return
        picks = [s for s in picks if s in self.config.symbols]
        self.picks[window.session_id] = picks
        for symbol in self.config.symbols:
            self.logs[symbol].write(
                "picks",
                event_ns=now,
                session=window.session_id,
                picks=picks,
                picked=symbol in picks,
            )

    def _bought_today(self, symbol, window):
        """From the ledger, so a restart during the session still knows."""
        fills = self.broker.book(symbol).account.fills
        return any(f.signed_quantity > 0 and f.timestamp_ns >= window.open_ns for f in fills)

    def _entries(self, window, now):
        if window.open_ns + ENTRY_WINDOW[1] <= now < window.close_ns:
            self._cancel_late_entries(window, now)
        if not window.open_ns + ENTRY_WINDOW[0] <= now < window.open_ns + ENTRY_WINDOW[1]:
            return
        if not self._check_guard(window, now):
            return
        for symbol in self.picks.get(window.session_id, []):
            book = self.broker.book(symbol)
            key = (symbol, window.session_id, "entry")
            if book.account.position > 0 or book.pending or book.risk.halted:
                continue
            if key in self.attempted or self.clock() < self.retry_after.get(key, 0):
                continue
            price = self._fresh_price(symbol, now)
            if price is None:
                self.retry_after[key] = self.clock() + 10 * NS
                continue
            limit = (price * DAY_LIMIT).quantize(Decimal("0.01"), ROUND_DOWN)
            quantity = (book.account.cash * CASH_USE / limit).quantize(QUANTITY_STEP, ROUND_DOWN)
            if quantity * limit < MINIMUM_NOTIONAL:
                self.attempted.add(key)
                continue
            self._submit(symbol, "buy", quantity, "day", price, now, "entry", window, limit)

    def _exits(self, window, now):
        closing = window.close_ns - EXIT_BEFORE_CLOSE <= now < window.close_ns
        in_session = window.open_ns <= now < window.close_ns
        if not in_session:
            return
        for symbol in self.config.symbols:
            book = self.broker.book(symbol)
            position = book.account.position
            if position <= 0 or book.pending:
                continue
            if not closing and self._bought_today(symbol, window):
                continue  # today's pick: hold until five minutes before the close
            step = f"exit-{(now - window.open_ns) // MINUTE}"  # retry each minute
            if (symbol, window.session_id, step) in self.attempted:
                continue
            price = self.last_prices.get(symbol, (book.account.entry_price, 0))[0]
            self._submit(symbol, "sell", position, "day", price, now, step, window)

    def _tick(self, state):
        window = self._window(self.clock())
        if window:
            self._refresh(window, self.clock())
            self._pick(window, self.clock())
        return super()._tick(state)
