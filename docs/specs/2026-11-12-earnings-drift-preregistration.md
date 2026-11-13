# Pre-registration: post-earnings drift

Written and committed before any price around an earnings announcement was looked
at. Experiment 22 in the [project history](../project-history.md). An
[amendment](#amendment-2026-11-12-before-any-evaluation), also made before any
evaluation, changes how rule 1 labels an event and what the parser rejects.

## Why

Every same-day rule lost about its round-trip cost. Holding for weeks (experiment
19) was the first idea that pointed the right way in both periods, but it failed the
bar. Post-earnings announcement drift (PEAD) is one of the most replicated
anomalies: after a positive earnings surprise, prices keep rising for weeks (Ball and
Brown 1968; Bernard and Thomas 1989). A five-session hold keeps costs small.

The prior is weak, and it is stated up front: Martineau (2022, "Rest in Peace
Post-Earnings Announcement Drift", *Critical Finance Review* 11:613-646) finds the
drift essentially gone in large stocks since 2006, as prices now absorb the news on
the announcement day. These are large stocks. A failure here is the expected,
documented outcome, and it would be recorded as such.

## Data

- **Universe:** the 60 stocks of `hft/ml/universe.py` (kind `stock`, SPCX
  excluded); ETFs have no earnings. Daily SIP bars adjusted for splits and
  dividends (`data/ml/1Day`), sessions from `data/ml/calendar.json`. Prices before
  a stock's first date in the universe file are ignored (NBIS's earlier bars belong
  to Yandex).
- **Announcement times:** SEC EDGAR 8-K filings with Item 2.02 ("Results of
  Operations and Financial Condition"), from the free submissions API
  (`hft/ml/earnings.py`, cached once in `data/earnings/`). Each filing carries its
  acceptance time in UTC. Checked against news times before this was written:
  the median headline appears 3 minutes before the filing is accepted, and events
  cluster at 06:00-08:00 and 16:00 New York time. Disney's filings before 2019 sit
  under its former registrant (CIK 1001039) and are included. 8-K/A amendments are
  not events.
- **Consensus beats:** the news store (`data/ml/news`, Benzinga headlines via
  Alpaca) holds results headlines such as "Oracle Q2 EPS $1.21 Beats $1.18 Estimate"
  with their first publication time. `parse_eps_headline` reads the reported EPS and
  the consensus. It accepts "Beats", "Misses", "vs" and "Inline" forms. It rejects
  guidance ("Sees", "Raises", ...), "May Not Compare" (a GAAP figure against an
  adjusted consensus), prose, and a verb that contradicts its numbers. A headline
  matches a filing when it is tagged with that stock and no other universe stock and
  was first published within one day of the acceptance. The earliest match is used.
- **Not used:** Finnhub's free tier, because its calendar covers about a month and
  its EPS history the last four quarters, without dates for older quarters.
- **Event time t:** the earlier of the acceptance and the matched headline's
  publication.
- **Splits by entry date:** development 2016-2022, holdout 2023-2024. The evaluation
  reads no bar after 2024-12-31, so trades that would exit in 2025 are dropped.
  **Test:** 2025 onward stays untouched. A rule that passes is run once on it, after
  the owner agrees, and recorded either way.
- Hindsight note, as in experiment 19: many added stocks were chosen in 2026 because
  they had risen. Every comparison is therefore against the same stocks, never cash.

Coverage, counted before this was written (counts only, no prices):

| Year | 8-K 2.02 events | With a headline | Beats |
| --- | ---: | ---: | ---: |
| 2016 | 222 | 121 (55%) | 102 |
| 2017 | 220 | 171 (78%) | 145 |
| 2018 | 232 | 194 (84%) | 170 |
| 2019 | 230 | 196 (85%) | 153 |
| 2020 | 228 | 190 (83%) | 152 |
| 2021 | 227 | 207 (91%) | 182 |
| 2022 | 235 | 207 (88%) | 166 |
| 2023 | 238 | 198 (83%) | 164 |
| 2024 | 236 | 205 (87%) | 180 |
| 2016-2024 | 2,068 | 1,689 (82%) | 1,414 |

Events come from 55 of the 60 stocks. Excluded, with no 8-K Item 2.02 filing in
2016-2024: TSM and NBIS (foreign filers, which use 6-K), XOM (its two Item 2.02 filings are from
2025 on), and SNDK and CRWV (listed in 2025). META matches few headlines
(9 of 36), because its older headlines were tagged FB. TSLA has 75 events, because
it also files Item 2.02 for delivery reports. Those filings rarely match an EPS
headline, but rule 2 includes them.

## Costs

Each buy and sell pays the stock's half-spread from the universe file (1.5 bp for
the original stocks, 4.0 bp for the added ones) plus 1 bp slippage, at the daily
open (`hft.ml.rule_tests.side_costs`). Every result is also reported at three times
these costs.

## Rule 1: consensus beat, five sessions

- For every 8-K Item 2.02 event with a matched headline whose reported EPS is above
  the consensus, buy at the first regular-session open at or after the headline's
  publication, which is when the beat is known. A pre-open report enters that
  morning; a report after the close enters the next morning.
- Sell at the open five sessions later.
- Two filings that give the same stock and the same entry session make one trade.

## Rule 2: announcement-return proxy, five sessions

No estimates needed (Chan, Jegadeesh and Lakonishok 1996).

- For every 8-K Item 2.02 event, day 0 is the first session whose close is after t.
- The announcement return is the stock's return from the close of the session
  before day 0 to the close of day 0, minus the equal-weight average of the same
  return over every stock with both closes.
- If it is above zero, buy at the open of day 1 and sell at the open of day 6. The
  signal is known at day 0's close, before the entry.
- Same-stock, same-entry duplicates make one trade, as in rule 1.

## Benchmark and statistic (both rules)

- **Benchmark per trade:** the equal-weight return of every universe stock with
  both opens, over the same entry and exit opens, without costs.
- **Statistic:** the excess return per trade (trade net of costs minus benchmark).
  Implemented by `hft.ml.rule_tests.drift`, which experiment 19's rule 2 also used.

## Pass rule

The pass rule is unchanged from experiment 19. Two rules are tested, so each
interval is at the Bonferroni-corrected 98.75% level (two-sided). A rule passes only
if all of these hold:

1. On the holdout, the mean excess per trade is above zero and the lower bound of the
   98.75% bootstrap interval is above zero. The bootstrap resamples entry months
   (all trades entered in a month together), with 10,000 resamples and seed 0.
2. On development, the mean is above zero.
3. On the holdout at three times the costs, the mean is above zero.

## Reported only

These are not part of the pass rule:

- Per-year means and trades per year.
- Headline coverage, as above.
- For rule 1, the misses: the same trade after a headline at or below the consensus.
- For rule 2, terciles of the announcement return over all events in each split,
  with the mean excess per trade of buying each tercile at day 1.
- A no-surprise control: every 8-K Item 2.02 event bought at the rule-2 entry,
  whatever the announcement return.

With roughly 340 beats and 240 rule-2 trades in the holdout, the interval can only
exclude zero for an effect of tens of basis points per trade. A failure means "not
shown", which is recorded as such.

## Amendment (2026-11-12, before any evaluation)

A code review of the committed evaluation found that "the earliest match is used"
ignores information that is public before the entry. No price had been computed
when this amendment was written, and none has been computed since the original
pre-registration. Changes:

1. **Rule 1's label is the latest matched headline published before the entry
   open.** The entry stays the first open after the first matched headline. A
   correction or the adjusted figure, when it is out before that open, supersedes
   the first report. Examples: Intel 2024-10-31 was first reported as a beat
   ("$0.12 Beats $(0.02)") and corrected at 17:51 to a miss ("$(0.46)"); Disney
   2024-02-07 was "Inline", then a beat ten minutes later; for Citigroup 2024-01-12
   and Morgan Stanley 2024-01-16 the GAAP miss came minutes before the adjusted
   beat. This changes 8 labels in 2016-2024, 4 of them in the holdout.
2. **Strictly after.** The entry is the first open *strictly* after the headline, so
   news published at 09:30:00 is not traded in that morning's opening auction. No
   such case occurs in the data.
3. **The parser also rejects** any headline containing "Not Compare" anywhere (not
   only in place of the verb) and any rumor headline containing "Hearing". This
   removes 7 matched events.
4. **Stale headlines.** A headline more than four days older than its entry open is
   not traded (only possible before the first stored session; no case occurs).
5. **The test-year run** (`--final`) takes one named rule that passed the holdout
   and needs the results doc in the HEAD commit (staged is not enough). The
   controls are not run on the test years.
6. **No bar after 2024-12-31 is read.** The cutoff is applied as the bar files are
   read, rather than after loading.

Coverage under these rules (2016-2024): 2,068 events, 1,682 with a headline (81%),
1,415 beats at entry. The other rules, costs and the pass rule are unchanged.

Known limits, recorded rather than fixed:

- The news store keeps a headline's latest text with its first publication time, so
  an in-place edit could change a label. 5 matched headlines were updated more than
  an hour after publication.
- Some Item 2.02 filings are not quarterly results: pre-announcements and segment
  recasts (for example BAC 2024-01-08, C 2024-01-10, GOOGL 2023-04-20, CVX
  2023-07-24, GE 2020-04-09), as well as TSLA's delivery reports. They rarely match an
  EPS headline, but rule 2 trades them.
