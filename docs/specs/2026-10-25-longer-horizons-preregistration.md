# Pre-registration: longer holding periods

Written and committed before any result of these two rules was computed.

## Why

Every same-day and overnight rule tested so far earned about its trading cost or
less ([project history](../project-history.md)): a round trip costs roughly 3 to 10 bp,
and the predictable part of a day's move is about that size. Holding for weeks or
months spreads the same cost over a much larger move. Two rules with long published
records are tested, each fixed in advance, with no parameter search.

## Data

- The 68 tradable symbols of the ML universe (`hft/ml/universe.py`, SPCX excluded),
  daily SIP bars adjusted for splits and dividends (`data/ml/1Day`), and the
  pre-open news features from 335,339 headlines
  ([news design](2026-10-09-news-features-design.md)).
- **Development:** 2016-01-01 to 2022-12-31. **Holdout:** 2023-01-01 to 2024-12-31.
  The holdout years were already used to validate the ML day trader and the
  same-day analyst rule; neither of these rules was run on them.
- **Test:** 2025-01-01 onward stays untouched. A rule that passes is run once on it
  and recorded either way; no rule is run on it otherwise.
- Hindsight note: many of the 68 symbols were chosen in 2026 partly because they had
  risen. Every comparison is therefore against the same symbols, never against cash.

## Costs

Each buy and sell pays the symbol's half-spread from the universe file (1.5 bp for
the original stocks and ETFs, 4.0 bp for the added stocks) plus 1 bp slippage, at the
daily open. Every result is also reported at three times these costs.

## Rule 1: twelve-month momentum, monthly

- On the last session of each month, score every symbol with daily closes for the
  whole window by its return from the last close of the month twelve months earlier
  to the last close of the previous month (the "12-1" return; the latest month is
  skipped, as in Jegadeesh and Titman 1993).
- Hold the top 10 scores in equal weights from the open of the next month's first
  session to the open of the following month's first session, then rebalance.
- **Benchmark:** every scored symbol in equal weights, rebalanced the same way, with
  the same costs on the weight it trades.
- **Statistic:** the monthly excess return (rule minus benchmark, both net).

## Rule 2: analyst upgrade drift, twenty sessions

- A symbol is bought at the open of session d when its pre-open news holds net
  analyst-up events (`news_analyst_net > 0`, focused articles only), the same signal
  as the failed same-day rule, and sold at the open of session d+20. Post-revision
  drift over weeks is documented (Womack 1996).
- **Benchmark per trade:** the equal-weight average return of every symbol valid at
  both opens, over the same window, without costs.
- **Statistic:** the excess return per trade (trade net of costs minus benchmark).

## Pass rule

Two rules are tested, so each interval is at the Bonferroni-corrected 98.75% level
(two-sided). A rule passes only if all of these hold:

1. On the holdout, the mean of its statistic is above zero and the lower bound of the
   98.75% bootstrap interval is above zero. Rule 1 resamples blocks of three months;
   rule 2 resamples entry months (all trades entered in a month together, because
   their windows overlap). 10,000 resamples, seed 0.
2. On development, the mean is above zero.
3. On the holdout at three times the costs, the mean is above zero.

Also reported, not part of the pass rule: annual returns of the rule, the benchmark
and buy-and-hold of the same symbols; trades or turnover per year; and, for rule 2,
an equal-slot portfolio (twenty slots of 5%, cash otherwise).

Two years of monthly data give 24 observations, so rule 1 can only pass on a large
effect. A failure means "not shown", which is recorded as such.
