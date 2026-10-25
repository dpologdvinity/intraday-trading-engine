# Paper acceptance results

The first sessions of `hft trade --paper` against a real Alpaca paper account. They
test the plumbing: market data, decisions, orders, fills, ledgers, reconciliation,
shutdown, and the overnight schedule. Paper fills are simulated by Alpaca. These are
rule-strategy runs, which never count as graduation evidence.

## 2026-10-08: intraday runs (ema-crossover, NVDA and AAPL, $100 each)

| Run | Time (ET) | Round trips | Net | Ended by |
| --- | --- | ---: | ---: | --- |
| `acceptance` | 09:30-09:35 | 1 | +$0.03 | stopped to deploy fix 1 |
| `acceptance-2` | 09:36-09:47 | 4 | -$0.01 | bug 2 (cash mismatch) |
| `acceptance-3` | 09:52-15:40 | 180 | +$0.90 | planned stop; flattened on SIGTERM |

`acceptance-3` made 90 round trips per stock, about +0.26 bp (AAPL) and +0.75 bp
(NVDA) per trade, with 27 and 30 winners. The spread guard refused 32 entries, and
the volatility breaker 43. One day of rule trading says nothing about an edge. The
quote-level replay of this strategy lost about 4-5 bp per round trip
([replay results](replay-results.md)), so Alpaca's simulated paper fills look
more generous than the project's simulator; paper profits should not be read as
achievable live.

## Bugs found and fixed

Each fix has a regression test that fails on the old code.

1. **One stall halted a stock for the day.** At 09:31:07, during the opening burst,
   130 NVDA events arrived more than 1 s late. The backlog rule from the original
   design halted NVDA for the session. Now stale events are dropped and the stock
   warms up again from fresh bars (15fd6b2).
2. **Cent rounding stopped the run.** Alpaca books each fill's cash to the cent; the
   ledger keeps exact decimals. After eight fractional fills they differed by
   $0.012 against a fixed $0.01 tolerance. Reconciliation now allows half a cent per
   fill since its last clean check and re-anchors (d1e1a0b).
3. **REST latency aged market events.** `acceptance-3` restarted warmup 37 times in
   six hours at low machine load: events waited in the queue while the loop made
   about 0.5 s of sequential REST calls. Queued events are now routed after every
   broker call (7d6e255). The run itself kept the old code.
4. **Paper expires limit-on-close orders.** The overnight run's three limit-on-close
   entries (BAC, NVDA, PFE) were accepted and then expired unfilled at the close,
   although the close was below each limit. Whole-share entries now use
   market-on-close orders sized as if filled 2% above the 15:45 price (b82d767).
5. **Status after a restart.** The newest journal of a restarted run has no equity
   row yet, so status showed holdings as near-total losses. It now reads older
   journals and never shows a holding at zero (1438cc7).
6. **Overnight regulatory fees stopped the overnight run.** At 03:46 ET Alpaca posted
   $0.40 of SEC and CAT fees for the day's intraday sales to the shared account, and
   reconciliation stopped the run on the unexplained cash change. Fee activities are
   now booked: split over the run's stocks by sale proceeds when the run sold that day,
   recorded as external otherwise (954fe7c). The run was restarted at 06:10 with its
   seven positions intact, before the opening exits.

## 2026-10-08: first overnight entry (overnight-drift, 10 stocks, $500 each)

Started 15:41 ET. Seven fractional marketable limit buys at 15:57 filled within
cents of the reference trade (AAPL 340.55, AMZN 254.22, GOOGL 348.09, JPM 331.64,
KO 87.74, MSFT 522.85, XOM 168.38). The three on-close orders expired (bug 4), so
BAC, NVDA and PFE hold nothing tonight. The run was restarted with the fixes and
resumed its saved state and positions. The opening exits are next.

## 2026-10-09: first overnight exits, run stopped

The seven queued day market sells filled between 09:30:01 and 09:33 ET (KO and XOM in
several partial fills). The fractional positions could not use opening-auction orders,
so they sold in the first minutes of continuous trading.

| Stock | Bought (10-08 15:57) | Sold (10-09 open) | Net |
| --- | ---: | ---: | ---: |
| AAPL | 340.55 | 332.80 | -$11.21 |
| AMZN | 254.22 | 256.42 | +$4.26 |
| GOOGL | 348.09 | 350.71 | +$3.71 |
| JPM | 331.64 | 331.01 | -$0.94 |
| KO | 87.74 | 87.66-87.70 | -$0.27 |
| MSFT | 522.85 | 528.18 | +$5.02 |
| XOM | 168.38 | 167.30-168.13 | -$1.56 |

Total -$0.98 over seven trades (-2.9 bp per trade), one night: no information about
an edge. The overnight run was stopped at 09:39 ET once flat, so the paper account
could move to the news-picks forward test (below).

## 2026-10-09: news-picks forward test started

Run `news-picks` started 09:43 ET on the frozen news variant 7f7b87ac489e
(experiment 16 in the [project history](project-history.md)), all 68 trainable
symbols at $1,000 each, top three picks a day. It failed validation against holding,
so this is a forward record on unseen days, not a candidate for real money. Its first
session is Monday 2026-10-12: data refresh 08:55 ET, picks after 09:30, sells at 15:55.
Starting it needed two fixes (7da1a49): scheduled runs poll prices over REST, so the
30-symbol stream limit no longer applies to them (up to 100), and one latest-trades
request now covers up to 100 symbols.

## 2026-10-12: news-picks stopped before its first session

The host computer slept from Saturday 10-10 until Monday 10-12 19:55 ET, so the run
skipped Monday's refresh, picks and trades. On waking it reconciled the account and
stopped on a cash mismatch: Alpaca had posted $0.09 of fees for the overnight run's
10-09 sales, but the fresh run also counted the 10-08 fees ($0.40) that were already
in its starting cash, so the activities did not explain the difference. A new run now
marks the account's existing dividend and fee activities as booked (b4fb454). The run
had no trades; it was restarted fresh on 2026-10-18 and first trades Monday 10-19.
The bot cannot trade while the computer sleeps or after a reboot until restarted.

## 2026-10-19: first news-picks session

The run refreshed its data at 08:55, picked INTC, NBIS and CRWV at 09:30, bought from
09:31 and sold at 15:55: -$83.66 (-2.8% of the $2,950 invested). Each session's
results are now recorded in the [forward record](news-picks-forward.md).

## 2026-10-20 to 10-23: sessions missed

The news-picks process stopped some time after the 10-19 close and left no error;
its journals end at 15:55:09. The most likely cause is WSL shutting the Linux
instance down after the last terminal closed (its default idle timeout is 15 s),
which kills background processes without a trace. Four sessions were missed. The
idle shutdown is now disabled (`instanceIdleTimeout=-1` in `.wslconfig`) and the run
was restarted on 2026-10-25 from its saved state.
