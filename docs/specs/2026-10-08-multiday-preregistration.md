# Pre-registration: overnight and multi-day rules

Written and committed before any result was computed. The evaluation script
must implement exactly this. Any change made after the development report is
recorded as an amendment with its reason.

## Why these rules

Same-day strategies do not survive costs on this data. The 5-second EMA crossover
lost 3.7-5.5 bp per round trip ([replay results](https://github.com/dpologdvinity/intraday-trading-engine/blob/master/replay-results.md)). Intraday
momentum lost 7.50 bp per trade on the holdout
([intraday momentum results](https://github.com/dpologdvinity/intraday-trading-engine/blob/master/intraday-momentum-results.md)). A rule that holds
for a night or a week spreads the same round-trip cost over a larger expected move.
Both rules below are published, have no tuned parameters, and trade long only.

1. **Overnight drift.** Most of the US equity premium is earned between the close
   and the next open (Cliff, Cooper and Gulen, 2008; Lou, Polk and Skouras,
   "A tug of war: Overnight versus intraday expected returns", Journal of Financial
   Economics, 2019). Rule: buy every stock at the 15:58 bar close and sell at the
   next session's 09:30 bar open.
2. **Weekly short-term reversal.** Stocks that fell most over the past week tend to
   rebound the following week (Jegadeesh, 1990; Lehmann, 1990). Rule: every fifth
   session, after the 15:58 bar, rank the stocks by return over the previous five
   sessions (15:58 close to 15:58 close). Buy the 5 worst (about the bottom decile
   of 46) at the 15:58 close and sell them at the 15:58 close five sessions later.
   Holding periods do not overlap.

Neither rule fits the current paper runner, which goes flat 60 seconds before
every close. Deploying either one would need an owner decision to allow overnight
positions, plus a new versioned strategy contract. This test only decides whether
that is worth asking for.

## Data

- Alpaca 1-minute bars, split-adjusted, 2024-10-03 to 2026-10-02, cached by
  candlebench, for the 46 stocks. SPY, QQQ, IWM and DIA are reported separately
  and are not used in the decision.
- Bars are split-adjusted but not dividend-adjusted. A long position therefore
  shows the ex-dividend price drop without the dividend received. This biases
  both rules slightly against passing. The bias is reported, not corrected.
- A session is used only when it has the 09:30, 15:58 and 15:59 bars, so half-days
  are skipped. A holding period is used only when both ends are usable sessions.
- Sessions returned by `hft.training.reserved_final_sessions()` are excluded.

## Costs per round trip

Each side pays the symbol's median NBBO half-spread for the time bucket of that
side, from candlebench's `quoted_spreads.json` in basis points. A 15:58 trade uses
the close bucket. A 09:30 trade uses the open bucket. Each side also pays 1 bp
slippage and $0.005 per share at the trade price. Results are also reported at 3x
this cost.

## Split and statistics

- Development: holding periods that start before 2025-10-03. Holdout: periods that
  start on or after 2025-10-03, evaluated once, after the development report is
  committed.
- The holdout year was already used once, for the intraday momentum test. The
  results of that test do not inform these rules, but the period is no longer
  completely unseen. This is disclosed in the results.
- The unit is a holding period. Its value is the equal-weight mean net return of
  the stocks held, in basis points. Intervals come from 10,000 bootstrap
  resamples of holding periods (seed 0). Two rules are tested, so the decision
  uses Bonferroni-adjusted 97.5% intervals (1.25th and 98.75th percentiles).
- Controls, reported for both rules:
  - Overnight drift: the same stocks held from the 09:30 open to the 15:58 close.
  - Weekly reversal: all 46 stocks held over the same five-session periods, with
    the same costs (equal-weight buy and hold of the universe).

## Decision

- Overnight drift passes if the holdout mean net return per night has a 97.5%
  interval lower bound above zero.
- Weekly reversal passes if the holdout mean net return in excess of the
  equal-weight universe control has a 97.5% interval lower bound above zero. A
  long-only rule that only earns the market's return is not an edge.
- A pass makes the rule a candidate for an owner decision on overnight holds and
  then paper trading. It does not unlock real money; the existing research
  qualification and 30-session paper graduation still apply. A failure is
  recorded as a negative result.
