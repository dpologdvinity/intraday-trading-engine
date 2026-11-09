"""Hold the ten strongest stocks by twelve-month momentum, rebalanced monthly, on paper.

The forward test of the pre-registered momentum rule
(docs/specs/2026-10-25-longer-horizons-preregistration.md): for month M it holds the
top 10 by the return from the last close of month M-13 to the last close of month M-2,
the same choice the backtest makes for a holding month. Each stock keeps its own
budget (the paper runner's per-stock ledgers), so a chosen stock invests its whole
budget and the others stay in cash.

Each session from one minute after the open until an hour after it, the runner sells
holdings outside this month's choice and buys chosen stocks it does not hold, so a
day missed (machine asleep, broker errors) is made up on the next session it runs.
Daily bars come from `data/ml/1Day`, which the news-picks run refreshes each morning;
this runner only reads them. Paper only.
"""

from decimal import ROUND_DOWN, Decimal
from pathlib import Path

import numpy as np

from .overnight import (
    CASH_USE,
    DAY_LIMIT,
    MINIMUM_NOTIONAL,
    MINUTE,
    NS,
    QUANTITY_STEP,
    OvernightRunner,
)

TOP = 10
TRADE_WINDOW = (1 * MINUTE, 60 * MINUTE)  # after the open


def choose(dates, closes, month, top=TOP):
    """Indices of the `top` symbols to hold in `month` (numpy datetime64[M]), or None
    when the bars do not yet reach the end of the month before it."""
    months = np.asarray(dates).astype("datetime64[M]")
    month = np.datetime64(month, "M")
    one = np.timedelta64(1, "M")
    late_month, early_month = month - 2 * one, month - 13 * one
    if not (months == month - one).any():
        return None  # the decision needs the last close of month M-1 on record
    late_rows = np.flatnonzero(months == late_month)
    early_rows = np.flatnonzero(months == early_month)
    if not len(late_rows) or not len(early_rows):
        return None
    late, early = closes[late_rows[-1]], closes[early_rows[-1]]
    scored = np.isfinite(late) & np.isfinite(early)
    if scored.sum() < top:
        return None
    score = np.where(scored, late / early, -np.inf)
    return [int(s) for s in np.argsort(-score, kind="stable")[:top]]


def daily_closes(root, symbols):
    """(dates, closes[D, S]) from the saved daily bars, NaN where a symbol has none."""
    from ..ml.download import read_bars
    from ..ml.features import _aligned, _day

    days = set()
    for symbol in symbols:
        days |= {_day(t) for t in read_bars(root, "1Day", symbol).column("t").to_numpy()}
    dates = np.array(sorted(days), dtype="datetime64[D]")
    closes = np.stack([_aligned(root, s, dates)[3] for s in symbols], axis=1)
    return dates, closes


class MomentumRunner(OvernightRunner):
    def __init__(
        self, config, broker, windows, *, data_root, closes=daily_closes, top=TOP, **kwargs
    ):
        super().__init__(config, broker, windows, **kwargs)
        self.data_root, self.closes, self.top = Path(data_root), closes, top

    @staticmethod
    def _near(window, now):
        return window.open_ns - MINUTE <= now < window.open_ns + TRADE_WINDOW[1] + MINUTE

    def _target(self, window, now):
        """This month's chosen symbols, computed once a month and saved with the run."""
        month = window.session_id[:7]
        saved = self.broker.extra.get("momentum")
        if saved and saved["month"] == month:
            return saved["symbols"]
        key = (None, window.session_id, "target")
        if self.clock() < self.retry_after.get(key, 0):
            return None
        symbols = list(self.config.symbols)
        try:
            dates, closes = self.closes(self.data_root, symbols)
            chosen = choose(dates, closes, month, self.top)
        except (OSError, KeyError, ValueError) as error:
            chosen, reason = None, f"momentum: {error}"
        else:
            reason = "momentum: daily bars do not reach the end of last month"
        if chosen is None:
            self._quality(reason)
            self.retry_after[key] = self.clock() + 10 * MINUTE
            return None
        target = [symbols[s] for s in chosen]
        self.broker.extra["momentum"] = {"month": month, "symbols": target}
        self.broker.save()
        for symbol in symbols:
            self.logs[symbol].write(
                "picks",
                event_ns=now,
                session=window.session_id,
                picks=target,
                picked=symbol in target,
            )
        return target

    def _in_window(self, window, now):
        return window.open_ns + TRADE_WINDOW[0] <= now < window.open_ns + TRADE_WINDOW[1]

    def _exits(self, window, now):
        if not self._in_window(window, now):
            return
        target = self._target(window, now)
        if target is None:
            return
        for symbol in self.config.symbols:
            book = self.broker.book(symbol)
            position = book.account.position
            if symbol in target or position <= 0 or book.pending:
                continue
            step = f"exit-{(now - window.open_ns) // MINUTE}"  # retry each minute
            if (symbol, window.session_id, step) in self.attempted:
                continue
            price = self.last_prices.get(symbol, (book.account.entry_price, 0))[0]
            self._submit(symbol, "sell", position, "day", price, now, step, window)

    def _entries(self, window, now):
        if window.open_ns + TRADE_WINDOW[1] <= now < window.close_ns:
            self._cancel_late_entries(window, now)
        if not self._in_window(window, now) or not self._check_guard(window, now):
            return
        target = self._target(window, now)
        for symbol in target or []:
            book = self.broker.book(symbol)
            key = (symbol, window.session_id, "entry")
            if book.account.position > 0 or book.pending or book.risk.halted:
                continue
            if key in self.attempted or self.clock() < self.retry_after.get(key, 0):
                continue
            price = self._fresh_price(symbol, now)
            if price is None and self._fetch([symbol]):
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
