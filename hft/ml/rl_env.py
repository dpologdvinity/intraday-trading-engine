"""Tensor trading environments for the reinforcement-learning search.

Implements docs/specs/2026-11-11-rl-daytrader-design.md. Thousands of episodes step
together as PyTorch tensors (on a GPU on Kaggle, on the CPU in tests).

Track A (intraday): one symbol on one session. Decisions at the end of minutes 4, 9,
14, ... fill at the next minute with a real bar, exactly as `hft.ml.fills`; the last
decision, at `last_minute - 1`, always sells. One round trip a day, a 15-minute
minimum hold, no entries in the last 15 minutes.

Track B (daily): one symbol over consecutive sessions. A decision at a session's open
(from bars up to the previous close plus today's opening gap) fills at that open;
positions may be held for 2 to 20 sessions.

Rewards: the step's change in the slot's log wealth, entry and exit costs included,
so a day's (or episode's) rewards sum to its net log return; or the differential
Sharpe ratio of those changes (Moody and Saffell 1998).
"""

from dataclasses import dataclass

import numpy as np
import torch

from .fills import SLIPPAGE_BPS, half_spread_bps

STEP_MINUTES = 5
FIRST_DECISION = 4  # end of the fifth minute
LOOKBACK = 60
MIN_HOLD_MINUTES = 15
NO_ENTRY_MINUTES = 15  # before last_minute
MAX_STEPS_A = 78
MIN_HOLD_DAYS, MAX_HOLD_DAYS = 2, 20
EPISODE_DAYS = 60
REWARD_SCALE = 100.0  # log returns in percent
DSR_ETA = 0.05
CHANNEL_SCALE = np.array([1e3, 1.0, 1e2, 1.0, 1e2], dtype=np.float32)  # minute channels
STATE_A, STATE_B = 5, 3


def _finite(x, clip=50.0):
    """Non-finite values (a log of zero volume, a missing bar) become 0; extremes are
    clipped so one bad bar cannot swamp the network."""
    x = np.asarray(x, dtype=np.float32)
    return np.clip(np.where(np.isfinite(x), x, 0.0), -clip, clip).astype(np.float32)


def _fill_index(valid):
    """fill[m]: first minute after m with a real bar; with none, the last real minute
    at or before m (the price a position is valued and sold at)."""
    n, width = valid.shape
    minutes = np.arange(width)
    after = np.where(valid, minutes, width)
    nxt = np.minimum.accumulate(after[:, ::-1], axis=1)[:, ::-1]  # first valid at or after m
    nxt = np.concatenate([nxt[:, 1:], np.full((n, 1), width)], axis=1)  # strictly after m
    before = np.where(valid, minutes, -1)
    last = np.maximum.accumulate(before, axis=1)
    return np.where(nxt < width, nxt, last)


def decision_minutes(last_minute):
    """(N, MAX_STEPS_A) decision minutes and step counts: 4, 9, ... then last_minute - 1."""
    last_minute = np.asarray(last_minute, dtype=np.int64)
    grid = FIRST_DECISION + STEP_MINUTES * np.arange(MAX_STEPS_A)
    regular = grid[None, :] < (last_minute - 1)[:, None]
    count = regular.sum(axis=1)  # decisions before the forced sale
    minutes = np.where(regular, grid[None, :], (last_minute - 1)[:, None])
    return minutes, count + 1


@dataclass
class IntradayData:
    """Track A episodes (symbol-days) as numpy arrays; `to(device)` makes tensors."""

    symbol: np.ndarray  # (N,) symbol index
    date: np.ndarray  # (N,) date index into `dates`
    dates: np.ndarray  # (D,) datetime64[D]
    channels: np.ndarray  # (N, 390, 5) scaled minute channels
    market: np.ndarray  # (D, 390, 10) SPY and QQQ channels
    fill_open: np.ndarray  # (N, 390) price a decision at minute m fills at
    fill_minute: np.ndarray  # (N, 390)
    known_open: np.ndarray  # (N, 390) latest open printed by the end of minute m
    last_minute: np.ndarray  # (N,)
    cost_bps: np.ndarray  # (N,) half-spread with the tick floor at the day's first price
    extras: np.ndarray  # (N, E) standardized daily and news features (E may be 0)

    def to(self, device):
        return {
            k: torch.as_tensor(v, device=device) for k, v in self.__dict__.items() if k != "dates"
        }


def build_intraday(inputs, days, *, rich, market_symbols=("SPY", "QQQ"), stats=None):
    """Episodes for every (symbol, session) with bars on the frame rows `days` (bool).

    `rich` adds the 23 daily and 9 news features, standardized with `stats` (computed
    from these episodes when None, so pass the training split's stats elsewhere)."""
    frame, store = inputs.frame, inputs.store
    rows = np.flatnonzero(days)
    dates = frame.dates[rows]
    date_of = {d: i for i, d in enumerate(dates)}
    market = np.zeros((len(dates), 390, 5 * len(market_symbols)), np.float32)
    for k, name in enumerate(market_symbols):
        data = store[name]
        for r, d in enumerate(data.dates):
            i = date_of.get(d)
            if i is not None:
                market[i, :, 5 * k : 5 * k + 5] = _finite(data.channels[r] * CHANNEL_SCALE)
    picks = []
    for s, symbol in enumerate(inputs.symbols):
        data = store[symbol]
        for r, d in enumerate(data.dates):
            i = date_of.get(d)
            if i is not None and data.valid[r].any() and inputs.trainable[s]:
                picks.append((s, i, r))
    picks = np.array(picks, dtype=np.int64).reshape(-1, 3)
    n = len(picks)
    channels = np.zeros((n, 390, 5), np.float32)
    opens = np.zeros((n, 390), np.float32)
    valid = np.zeros((n, 390), bool)
    last = np.zeros(n, np.int64)
    cost = np.zeros(n, np.float32)
    for j, (s, _, r) in enumerate(picks):
        data = store[inputs.symbols[s]]
        channels[j] = _finite(data.channels[r] * CHANNEL_SCALE)
        valid[j] = data.valid[r]
        opens[j] = np.nan_to_num(data.open[r])
        last[j] = data.last_minute[r]
        prices = data.open[r][data.valid[r]]
        # As the ML day trader: the tick floor at the day's first price, both sides.
        cost[j] = half_spread_bps(inputs.costs_bp[s], float(prices[0]))
    fill = _fill_index(valid)
    fill_open = np.take_along_axis(opens, fill, axis=1)
    seen = np.maximum.accumulate(np.where(valid, np.arange(390), 0), axis=1)
    known_open = np.take_along_axis(opens, seen, axis=1)
    extras = np.zeros((n, 0), np.float32)
    if rich:
        frame_rows = rows[picks[:, 1]]
        parts = [frame.x[frame_rows, picks[:, 0]]]
        if inputs.news_x is not None:
            parts.append(inputs.news_x[frame_rows, picks[:, 0]])
        extras = _finite(np.concatenate(parts, axis=-1), clip=1e6)
        if stats is None:
            stats = (extras.mean(axis=0), extras.std(axis=0) + 1e-6)
        extras = _finite((extras - stats[0]) / stats[1], clip=10.0)
    data = IntradayData(
        picks[:, 0],
        picks[:, 1],
        dates,
        channels,
        market,
        fill_open,
        fill,
        known_open,
        last,
        cost,
        extras,
    )
    return data, stats


class IntradayEnv:
    """Track A: `reset(items, cost_multiple)` then `step(action)` MAX_STEPS_A times."""

    def __init__(self, tensors, *, reward="log", device="cpu"):
        self.t, self.reward_kind, self.device = tensors, reward, device
        self.extra_dim = tensors["extras"].shape[1]
        self.obs_window = (LOOKBACK, 15)
        self.obs_extra = STATE_A + self.extra_dim

    def reset(self, items, cost_multiple):
        t, n = self.t, len(items)
        self.items = items
        minutes, steps = decision_minutes(t["last_minute"][items].cpu().numpy())
        self.minutes = torch.as_tensor(minutes, device=self.device)
        self.steps = torch.as_tensor(steps, device=self.device)
        self.k = 0
        self.position = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.used = torch.zeros_like(self.position)
        self.entry_minute = torch.zeros(n, dtype=torch.long, device=self.device)
        self.entry_price = torch.ones(n, device=self.device)
        self.cost_multiple = cost_multiple
        self.log_value = torch.zeros(n, device=self.device)  # this day's net log return
        self.trades = torch.zeros(n, dtype=torch.long, device=self.device)
        self.entry_k = torch.full((n,), -1, dtype=torch.long, device=self.device)
        self.exit_k = torch.full((n,), -1, dtype=torch.long, device=self.device)
        self.dsr = (torch.zeros(n, device=self.device), torch.full((n,), 1e-6, device=self.device))
        return self.observe()

    def _price(self, minute):
        return self.t["fill_open"][self.items].gather(1, minute[:, None]).squeeze(1)

    def _cost(self):
        return self.cost_multiple * (self.t["cost_bps"][self.items] + SLIPPAGE_BPS) / 1e4

    def observe(self):
        t, items = self.t, self.items
        m = self.minutes[:, min(self.k, MAX_STEPS_A - 1)]
        offsets = torch.arange(-LOOKBACK + 1, 1, device=self.device)
        index = m[:, None] + offsets[None, :]
        inside = (index >= 0).float()[..., None]
        index = index.clamp(min=0)
        own = t["channels"][items[:, None], index]
        market = t["market"][t["date"][items][:, None], index]
        window = torch.cat([own, market], dim=-1) * inside
        last = t["last_minute"][items]
        held = torch.where(
            self.position, (m - self.entry_minute).float(), torch.zeros_like(m).float()
        )
        # Only prices printed by the end of minute m: the fill price lies in the future.
        price = t["known_open"][items].gather(1, m[:, None]).squeeze(1).clamp(min=1e-6)
        unrealized = torch.where(
            self.position,
            torch.log(price / self.entry_price) * REWARD_SCALE,
            torch.zeros_like(price),
        )
        state = torch.stack(
            [
                self.position.float(),
                held / 390,
                unrealized,
                (last - m).float() / 390,
                self.used.float(),
            ],
            dim=1,
        )
        extra = torch.cat([state, t["extras"][items]], dim=1)
        return window, extra, self.mask()

    def mask(self):
        """(N, 2) allowed actions: 0 flat, 1 long."""
        k = min(self.k, MAX_STEPS_A - 1)
        m = self.minutes[:, k]
        last = self.t["last_minute"][self.items]
        final = self.k >= self.steps - 1
        fill = self.t["fill_minute"][self.items].gather(1, m[:, None]).squeeze(1)
        can_enter = (
            ~self.used & (m < last - NO_ENTRY_MINUTES) & (fill > m) & (fill <= last - 1) & ~final
        )
        must_hold = self.position & (m - self.entry_minute < MIN_HOLD_MINUTES) & ~final
        flat_ok = ~must_hold
        long_ok = torch.where(self.position, ~final, can_enter)
        return torch.stack([flat_ok, long_ok], dim=1)

    def step(self, action):
        """Apply actions (N,) in {0, 1}; returns (reward, done) with done after the
        day's forced sale. Disallowed actions are replaced by the allowed one."""
        allowed = self.mask()
        action = action.bool()
        action = torch.where(allowed[torch.arange(len(action)), action.long()], action, ~action)
        active = self.k < self.steps
        m = self.minutes[:, min(self.k, MAX_STEPS_A - 1)]
        price = self._price(m)
        change = torch.zeros_like(price)
        enter = active & ~self.position & action
        leave = active & self.position & ~action
        cost = self._cost()
        change = change - torch.where(enter, torch.log1p(cost), torch.zeros_like(cost))
        change = change + torch.where(leave, torch.log1p(-cost), torch.zeros_like(cost))
        self.entry_minute = torch.where(enter, m, self.entry_minute)
        self.entry_k = torch.where(enter, torch.full_like(self.entry_k, self.k), self.entry_k)
        self.exit_k = torch.where(leave, torch.full_like(self.exit_k, self.k), self.exit_k)
        self.entry_price = torch.where(enter, price, self.entry_price)
        self.used = self.used | enter
        self.trades = self.trades + enter.long()
        self.position = (self.position | enter) & ~leave
        # Holding through to the next decision: the move between the two fill prices.
        k_next = min(self.k + 1, MAX_STEPS_A - 1)
        next_price = self._price(self.minutes[:, k_next])
        hold = self.position & active & (self.k + 1 < self.steps)
        change = change + torch.where(hold, torch.log(next_price / price), torch.zeros_like(price))
        self.log_value = self.log_value + torch.where(active, change, torch.zeros_like(change))
        reward = self._reward(torch.where(active, change, torch.zeros_like(change)), active)
        self.k += 1
        done = self.k >= self.steps
        return reward, done, active

    def _reward(self, change, active):
        if self.reward_kind == "log":
            return change * REWARD_SCALE
        return differential_sharpe(self, change, active, prior=1e-6)


def differential_sharpe(env, change, active, prior):
    """Moody-Saffell differential Sharpe ratio of the step returns, per episode."""
    a, b = env.dsr
    delta_a, delta_b = change - a, change**2 - b
    denominator = (b - a**2).clamp(min=prior) ** 1.5
    d = (b * delta_a - 0.5 * a * delta_b) / denominator
    env.dsr = (
        torch.where(active, a + DSR_ETA * delta_a, a),
        torch.where(active, b + DSR_ETA * delta_b, b),
    )
    return torch.where(active, d.clamp(-5, 5) * 0.1, torch.zeros_like(d))


# Track B ----------------------------------------------------------------------


@dataclass
class DailyData:
    """Track B arrays on the frame's dates: features per (day, symbol) and opens."""

    dates: np.ndarray  # (D,)
    features: np.ndarray  # (D, S, 3) causal per day: return, range, relative volume
    gap: np.ndarray  # (D, S) today's opening gap (known at the open)
    open: np.ndarray  # (D, S)
    valid: np.ndarray  # (D, S) tradable at this open with a full lookback
    cost_bps: np.ndarray  # (S,)
    news: np.ndarray  # (D, S, F) standardized pre-open news (F may be 0)
    market: tuple  # symbol indices of SPY and QQQ

    def to(self, device):
        out = {
            k: torch.as_tensor(v, device=device)
            for k, v in self.__dict__.items()
            if k not in ("dates", "market")
        }
        out["market"] = torch.as_tensor(self.market, device=device)
        return out


def build_daily(inputs, last_day, *, rich, stats=None):
    """Arrays for every frame row up to `last_day` (datetime64[D]); later rows are never
    read. Features for day d use bars up to d - 1 only."""
    from .features import _aligned

    frame = inputs.frame
    keep = frame.dates <= np.datetime64(last_day)
    dates = frame.dates[keep]
    bars = [_aligned(inputs.root, s, frame.dates)[:, keep] for s in inputs.symbols]
    o = np.stack([b[0] for b in bars], 1)
    h = np.stack([b[1] for b in bars], 1)
    low = np.stack([b[2] for b in bars], 1)
    c = np.stack([b[3] for b in bars], 1)
    v = np.stack([b[4] for b in bars], 1)
    ret = np.log(c[1:] / c[:-1])
    rng = np.log(h / low)
    mean_v = np.full_like(v, np.nan)
    for d in range(20, len(v)):
        mean_v[d] = np.nanmean(v[d - 20 : d], axis=0)
    rel = np.log(v / mean_v)
    day_feature = np.stack(
        [np.r_[np.full((1, ret.shape[1]), np.nan), ret] * 1e2, rng * 1e2, rel], -1
    )
    # Shift by one day: the row for day d holds day d - 1's bar.
    features = np.zeros_like(day_feature)
    features[1:] = day_feature[:-1]
    gap = np.zeros_like(o)
    gap[1:] = np.log(o[1:] / c[:-1]) * 1e2
    finite = np.isfinite(features).all(-1)
    lookback_ok = np.zeros_like(finite)
    run = np.zeros(finite.shape[1], np.int64)
    for d in range(len(finite)):
        run = np.where(finite[d], run + 1, 0)
        lookback_ok[d] = run >= LOOKBACK
    valid = lookback_ok & np.isfinite(o) & np.isfinite(gap) & inputs.trainable[None, :]
    news = np.zeros((len(dates), len(inputs.symbols), 0), np.float32)
    if rich and inputs.news_x is not None:
        news = _finite(inputs.news_x[keep], clip=1e6)
        if stats is None:
            flat = news.reshape(-1, news.shape[-1])
            stats = (flat.mean(0), flat.std(0) + 1e-6)
        news = _finite((news - stats[0]) / stats[1], clip=10.0)
    data = DailyData(
        dates,
        _finite(features),
        _finite(gap),
        np.nan_to_num(o).astype(np.float32),
        valid,
        np.asarray(inputs.costs_bp, np.float32),
        news,
        (inputs.symbols.index("SPY"), inputs.symbols.index("QQQ")),
    )
    return data, stats


class DailyEnv:
    """Track B: each episode is one symbol from a start day for up to `length` sessions;
    the episode sells at its last step (or earlier where the symbol stops trading)."""

    def __init__(self, tensors, *, reward="log", device="cpu"):
        self.t, self.reward_kind, self.device = tensors, reward, device
        self.obs_window = (LOOKBACK, 9)
        self.obs_extra = STATE_B + 1 + tensors["news"].shape[-1]

    def reset(self, symbol, start, length, cost_multiple, *, record=False):
        n = len(symbol)
        self.symbol, self.start, self.k = symbol, start, 0
        self.length = length
        self.position = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.held = torch.zeros(n, dtype=torch.long, device=self.device)
        self.entry_price = torch.ones(n, device=self.device)
        self.cost_multiple = cost_multiple
        self.log_value = torch.zeros(n, device=self.device)
        self.trades = torch.zeros(n, dtype=torch.long, device=self.device)
        self.daily = []  # per-step slot log returns, for portfolio scoring
        self.alive = torch.ones(n, dtype=torch.bool, device=self.device)
        self.trade_value = torch.zeros(n, device=self.device)
        self.record = record
        self.completed = []  # (env index, start step, length, net log return) per sale
        self.entry_k = torch.zeros(n, dtype=torch.long, device=self.device)
        self.dsr = (torch.zeros(n, device=self.device), torch.full((n,), 4e-4, device=self.device))
        return self.observe()

    def _day(self, k=None):
        return self.start + (self.k if k is None else k)

    def observe(self):
        t, d = self.t, self._day()
        d = d.clamp(max=t["open"].shape[0] - 1)
        offsets = torch.arange(-LOOKBACK + 1, 1, device=self.device)
        index = (d[:, None] + offsets[None, :]).clamp(min=0)
        own = t["features"][index, self.symbol[:, None]]
        spy = t["features"][index, t["market"][0]]
        qqq = t["features"][index, t["market"][1]]
        window = torch.cat([own, spy, qqq], dim=-1)
        price = t["open"][d, self.symbol]
        unrealized = torch.where(
            self.position,
            torch.log(price / self.entry_price) * REWARD_SCALE,
            torch.zeros_like(price),
        )
        state = torch.stack(
            [
                self.position.float(),
                self.held.float() / MAX_HOLD_DAYS,
                unrealized,
                t["gap"][d, self.symbol],
            ],
            dim=1,
        )
        extra = torch.cat([state, t["news"][d, self.symbol]], dim=1)
        return window, extra, self.mask()

    def _final(self):
        t = self.t
        d = self._day()
        last_row = t["open"].shape[0] - 1
        nxt = (d + 1).clamp(max=last_row)
        tradable_next = t["valid"][nxt, self.symbol] & (d + 1 <= last_row)
        return (self.k >= self.length - 1) | ~tradable_next

    def mask(self):
        final = self._final()
        tradable = self.t["valid"][self._day().clamp(max=self.t["open"].shape[0] - 1), self.symbol]
        must_hold = self.position & (self.held < MIN_HOLD_DAYS) & ~final
        must_sell = self.position & ((self.held >= MAX_HOLD_DAYS) | final)
        flat_ok = ~must_hold
        long_ok = torch.where(self.position, ~must_sell, tradable & ~final)
        return torch.stack([flat_ok, long_ok], dim=1)

    def step(self, action):
        allowed = self.mask()
        action = action.bool()
        action = torch.where(allowed[torch.arange(len(action)), action.long()], action, ~action)
        active = self.alive.clone()
        d = self._day()
        price = self.t["open"][d, self.symbol]
        floor = 0.005 / price.clamp(min=1e-6) * 1e4
        cost = (
            self.cost_multiple
            * (torch.maximum(self.t["cost_bps"][self.symbol], floor) + SLIPPAGE_BPS)
            / 1e4
        )
        enter = active & ~self.position & action
        leave = active & self.position & ~action
        change = -torch.where(enter, torch.log1p(cost), torch.zeros_like(cost))
        change = change + torch.where(leave, torch.log1p(-cost), torch.zeros_like(cost))
        self.entry_price = torch.where(enter, price, self.entry_price)
        self.trades = self.trades + enter.long()
        self.position = (self.position | enter) & ~leave
        final = self._final()
        nxt = (d + 1).clamp(max=self.t["open"].shape[0] - 1)
        hold = self.position & active & ~final
        change = change + torch.where(
            hold, torch.log(self.t["open"][nxt, self.symbol] / price), torch.zeros_like(price)
        )
        self.held = torch.where(self.position, self.held + 1, torch.zeros_like(self.held))
        change = torch.where(active, change, torch.zeros_like(change))
        self.entry_k = torch.where(enter, torch.full_like(self.entry_k, self.k), self.entry_k)
        self.trade_value = torch.where(
            enter, torch.zeros_like(change), self.trade_value
        ) + torch.where(enter | self.position | leave, change, torch.zeros_like(change))
        for i in torch.nonzero(leave).flatten().tolist() if self.record else ():
            k0 = int(self.entry_k[i])
            self.completed.append((i, k0, self.k - k0, float(self.trade_value[i])))
        self.log_value = self.log_value + change
        if self.record:
            self.daily.append(change)
        if self.reward_kind == "log":
            reward = change * REWARD_SCALE
        else:
            reward = differential_sharpe(self, change, active, prior=4e-4)
        self.alive = active & ~final
        self.k += 1
        done = ~self.alive
        return reward, done, active
