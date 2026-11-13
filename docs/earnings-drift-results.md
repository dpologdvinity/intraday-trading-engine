# Post-earnings drift: results

Evaluation of the [pre-registered rules](specs/2026-11-12-earnings-drift-preregistration.md)
(experiment 22) on the 60 stocks of the universe. Earnings announcements are SEC
EDGAR 8-K Item 2.02 filings; consensus beats come from Benzinga EPS headlines. Every
trade is held five sessions, open to open, and pays the stock's half-spread plus 1 bp
slippage per side. Intervals are two-sided 98.75% (Bonferroni over two rules) from
10,000 resamples of entry months.

```bash
.venv/bin/python -m hft.ml.earnings      # one-time EDGAR download (SEC_EDGAR_USER_AGENT)
.venv/bin/python scripts/earnings_drift.py --output artifacts/earnings-drift/holdout.json
```

The run read no bar after 2024-12-31; the last exit was 2024-12-30. Events came from
55 stocks: 2,072 filings, of which 1,682 had a matched headline.

## Rule 1: consensus beat, five sessions

Buy at the first open after a headline that reports EPS above the consensus, labelled
by the latest headline out before that open.

| | Trades | Mean excess per trade | 98.75% interval | Beat benchmark |
| --- | ---: | ---: | --- | ---: |
| Development 2016-2022 | 1,067 | -19.1 bp | -57.7 to +18.2 bp | 45% |
| Holdout 2023-2024 | 346 | -0.3 bp | -61.0 to +66.5 bp | 48% |
| Holdout at 3x costs | 346 | -12.0 bp | -72.7 to +55.0 bp | 47% |

By year (bp per trade): 2016 -49, 2017 -51, 2018 -29, 2019 +13, 2020 -81, 2021 -9,
2022 +53, 2023 +33, 2024 -30.

## Rule 2: announcement-return proxy, five sessions

Buy at the day-1 open when the stock beat the equal-weight universe from the close
before the announcement to day 0's close.

| | Trades | Mean excess per trade | 98.75% interval | Beat benchmark |
| --- | ---: | ---: | --- | ---: |
| Development 2016-2022 | 777 | +14.9 bp | -16.3 to +48.6 bp | 51% |
| Holdout 2023-2024 | 247 | +43.5 bp | -18.0 to +104.1 bp | 50% |
| Holdout at 3x costs | 247 | +31.7 bp | -30.0 to +92.6 bp | 49% |

By year (bp per trade): 2016 +3, 2017 +14, 2018 +1, 2019 +19, 2020 +6, 2021 +26,
2022 +34, 2023 +36, 2024 +52. About 110 trades a year.

## Reported only

| | Development | | Holdout | |
| --- | ---: | --- | ---: | --- |
| | Trades, mean | 98.75% interval | Trades, mean | 98.75% interval |
| Every event, day-1 entry (no-surprise control) | 1,579, -3.4 bp | -28.3 to +23.2 | 473, +7.7 bp | -46.4 to +63.8 |
| Rule 1 misses (EPS at or below consensus) | 210, -108.6 bp | -195.9 to -19.3 | 56, -50.8 bp | -199.6 to +157.7 |
| Announcement return, lowest third | 526, -22.7 bp | -66.5 to +19.3 | 157, -42.4 bp | -139.8 to +57.1 |
| Announcement return, middle third | 526, -27.6 bp | -74.7 to +20.2 | 158, +5.8 bp | -77.9 to +92.6 |
| Announcement return, top third | 527, +40.1 bp | -7.6 to +85.9 | 158, +59.4 bp | -20.0 to +143.2 |

The tercile edges were about ±1.9% in development and -1.7% / +2.3% in the
holdout.

## Decision

Both rules fail the pre-registered bar, because their holdout intervals include zero.
The test years (2025 onward) stay unused.

**Rule 1 found no drift after consensus beats.** It lost 19 bp a trade in
development and broke even in the holdout. That is the result the prior predicted:
Martineau (2022) finds that large stocks absorb an earnings surprise on the day it is
announced. A beat is also the common case (84% of headlines), so it carries little
news.

**Rule 2 points the right way in both periods** (+15 and +44 bp a trade) and
survives three times the costs. The pattern matches experiment 19: plausible, but
two years with about 120 trades each are too few to separate it from luck. Two
reported-only checks support reading it as a weak signal, not proving one:
- The returns rise from the lowest third of announcement returns to the top third in
  both periods.
- Buying every event regardless of surprise earned about zero (-3 and +8 bp), so the
  benchmark is not flattering the rule.

The clearest effect is on the side this system cannot trade. After a reported miss,
the stock trailed its benchmark by 109 bp over five sessions in development, and the
interval excludes zero. Long-only fractional trading cannot short it, and a miss
rule was not pre-registered. So this is a note, not a result. The holdout has only 56
misses, too few to confirm it.

## Found on the way

- **EDGAR timestamps are UTC.** The submissions API's `acceptanceDateTime` is true
  UTC. An early version of the code assumed New York time and was corrected before
  any evaluation. Apple's 16:30 New York releases are accepted at 20:30Z in summer
  and 21:30Z in winter.
- **The pre-registration was amended before the run.** A code review found that
  taking the earliest headline mislabelled events whose correction was public before
  the entry, such as Intel's 2024-10-31 "beat" that was a miss. The amendment was
  committed, with the code fix, before any price was computed.
- **Finnhub's free tier has no usable earnings history:** about a month of calendar
  and four quarters of EPS. EDGAR plus the existing news store cost nothing.
