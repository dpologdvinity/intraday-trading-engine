# Pre-registration: intraday momentum rule

Written and committed before any result was computed. The evaluation script
must implement exactly this; changes after the development report are recorded
as amendments with their reason and do not touch the holdout.

## Why this rule

Replays showed the 5/20-bar EMA crossover flipping about every 50 seconds and
losing 3.7-5.5 bp per round trip to the spread on flat days
([replay results](https://github.com/dpologdvinity/intraday-trading-engine/blob/master/replay-results.md)). The cost hurdle says spread-crossing
strategies need far fewer trades or much longer holds. Market intraday momentum
(Gao, Han, Li and Zhou, "Market intraday momentum", Journal of Financial
Economics, 2018) is a published, parameter-free effect with one trade per day:
the first half hour's return predicts the last half hour's. It was documented
for the S&P 500 ETF; whether it holds for single stocks net of costs is the
question. No parameter is tuned here.

## Rule `intraday-momentum` (long only)

- Signal: `r_first = close(09:59 bar) / open(09:30 bar) - 1` (09:30-10:00 ET).
  The engine version uses the first and last 5-second bar closes in that window.
  The paper's overnight component (previous close) is excluded because the live
  engine starts each session without the prior close; it is reported only as a
  secondary reference.
- Entry: if `r_first > 0`, buy at the 15:30 bar open. Otherwise stay flat.
- Exit: the 15:58 bar close, matching the runner's flatten 60 seconds before the
  close.
- At most one round trip per stock per day.

## Data

- Two years of Alpaca 1-minute bars, 2024-10-03 to 2026-10-02, for the 50 symbols
  cached by candlebench. The primary universe is the 46 stocks; SPY, QQQ, IWM and
  DIA are reported separately as the literature's reference, not used for the
  decision.
- A day is used only when the 09:30, 09:59, 15:30 and 15:58 bars exist and the
  session runs to 15:59 (half-days are skipped).
- Sessions returned by `hft.training.reserved_final_sessions()` are excluded.

## Costs per round trip

The symbol's median NBBO half-spread for the close bucket, measured by
candlebench (`quoted_spreads.json`, basis points per side), paid on both sides, plus 1 bp slippage per side and $0.005 per share per side
at the entry price. Results are also reported at 3x this cost. Alpaca paper fills
reference the NBBO; the IEX quote used for live signals is wider, which is why
the stricter 3x figure is shown.

## Split and statistics

- Development: sessions before 2025-10-03. Holdout: 2025-10-03 onward, evaluated
  once, after the development report is committed.
- Each date's result is the equal-weight mean net return of the stocks traded that
  day; the headline is the mean across dates, in basis points per trade. 95%
  confidence intervals come from 10,000 bootstrap resamples of dates (seed 0),
  because stocks on the same day are not independent.
- Reported alongside: the unconditional control (buy every day at 15:30), the
  days with `r_first <= 0`, per-stock means, and hit rate.

## Decision

Ship `intraday-momentum` as a paper-trading rule only if, on the holdout, the
mean net return per trade is above zero with a 95% interval lower bound above
zero. Whether it beats the unconditional control is reported but does not
decide. A pass makes it a paper candidate only; real money still requires the
existing research qualification and 30-session paper graduation. A failure is
recorded as a negative result and the rule is not added to the engine.
