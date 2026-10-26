# Longer holding periods: results

Evaluation of the [pre-registered rules](specs/2026-10-25-longer-horizons-preregistration.md)
on the 68 tradable symbols, at daily opens with each symbol's half-spread plus 1 bp
slippage per side. Intervals are two-sided 98.75% (Bonferroni over two rules).

```bash
.venv/bin/python scripts/longer_horizons.py --output artifacts/longer-horizons/holdout.json
```

## Rule 1: twelve-month momentum, top 10, monthly

| | Months | Mean excess per month | 98.75% interval | Months ahead | Rule a year | Benchmark a year |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| Development 2017-2022 | 71 | +61 bp | -48 to +166 bp | 58% | +24.7% | +16.5% |
| Holdout 2023-2024 | 23 | +75 bp | -100 to +277 bp | 61% | +47.2% | +37.0% |
| Holdout at 3x costs | 23 | +73 bp | -103 to +274 bp | | +46.6% | +36.9% |

Turnover is about 6 portfolio weights a year, so costs barely matter: at three times
the costs the excess falls by only 3 bp a month. Buy-and-hold of the same symbols
made +18.7% a year in development and +40.6% in the holdout.

## Rule 2: analyst upgrade drift, twenty sessions

| | Trades | Mean excess per trade | 98.75% interval | Beat benchmark | Slot portfolio a year |
| --- | ---: | ---: | --- | ---: | ---: |
| Development 2016-2022 | 3,360 | +24 bp | -29 to +82 bp | 50% | +20.7% |
| Holdout 2023-2024 | 1,480 | +52 bp | -28 to +140 bp | 50% | +34.5% |
| Holdout at 3x costs | 1,480 | +41 bp | -39 to +129 bp | | +32.7% |

The slot portfolio (twenty slots of 5%, cash otherwise) reports daily open-to-open
returns with slots reset to equal weight daily, an approximation.

## Decision

Both rules fail the pre-registered bar: their holdout intervals include zero. The
test years (2025 onward) stay unused.

Both, unlike every same-day rule before them, point the right way in both periods
and survive three times the costs. Momentum beat its benchmark by about 8% a year in
development and 10% a year in the holdout, with little trading. Two years of months
are too few to tell an effect of that size from luck: the interval spans about
±190 bp a month. The momentum result is also flattered by the universe: several
symbols were added in 2026 because they had risen, and a stock that later became a
big winner tends to show momentum on the way. These are leads for forward testing,
not results.

Found on the way: the news-picks bot's morning refresh had replaced the saved
2016-2026 market calendar with its 84 most recent sessions, so the first run of this
evaluation found no news before mid-2026. The refresh now merges sessions
(ceeb30e), the calendar was restored from Alpaca, and both rules were run on the
restored data.
