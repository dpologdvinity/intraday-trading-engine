# Operations guide

Commands and operating contracts for acquiring data, training, replay, dry runs,
broker paper trading and live activation. Start with the [README](https://github.com/dpologdvinity/intraday-trading-engine/blob/master/README.md)
for installation and the offline smoke check, and the
[trading specification](trading-spec.md) for the full execution and risk contracts.

## Get free stock data

Create an Alpaca paper account yourself and provide market-data credentials in
local environment variables. Keep keys out of Git and logs:

```bash
export ALPACA_API_KEY='your-market-data-key'
export ALPACA_SECRET_KEY='your-market-data-secret'
python -m hft data-probe --symbol AAPL --session 2026-09-28
python -m hft download --symbol AAPL --start 2026-03-01 --end 2026-09-30 --output data/aapl
python -m hft prepare --input data/aapl/manifest.json
```

These commands only read data/calendar endpoints. No paid feed fallback exists.
Free IEX is one exchange's feed; the software does not pretend it is consolidated
NBBO or a Level 2 order book. The probe distinguishes holidays, empty coverage
and denied historical access. Historical entitlement must be checked on your own
account. Downloads paginate, checkpoint checksummed session partitions and resume
by rerunning the same command. They preserve a 5GB disk reserve.

If past quote/trade data is unavailable, collect real events locally instead:

```bash
python -m hft record --symbol AAPL --output data/recordings/aapl --duration 23400
python -m hft prepare --input data/recordings/aapl
```

Start before the actual exchange opening and keep the computer awake. Recording
preserves arrival timestamps. Partial sessions remain labelled partial for
eligibility. A recording directory holds one stock. Missing history produces an
insufficient-data report, never invented historical quotes. Calendar records
handle holidays, DST and early closes.

## Train and compare

```bash
python -m hft experiment --dataset data/aapl/manifest.json --config config/research.toml --output artifacts/aapl/experiment.json
python -m hft data-quality --experiment artifacts/aapl/experiment.json
python -m hft data-quality --experiment artifacts/aapl/experiment.json --all-development
python -m hft train --experiment artifacts/aapl/experiment.json
python -m hft train --experiment artifacts/aapl/experiment.json --resume
python -m hft report --experiment artifacts/aapl/experiment.json
```

Edit `config/research.toml` **before freezing an experiment**, including
`initial_cash` if your strategy allocation will differ from $500. The search
compares three PPO configurations, two seeds and three expanding validation
folds. It refits the selected configuration on development data and consumes the
reserved final test once. Real final-test dates are registered in `.state/` so
another output filename does not silently allow retuning on the same dates.
After a failed final test, obtain later unseen sessions for another final test.
Interrupted final evaluation may require fresh test dates; it fails conservatively.

The development-only data-quality check uses the shared simulation to count
warmup coverage and gaps. Training runs this check automatically before fitting.
Any event gap over the runtime's five-second limit blocks the experiment; fewer
than 61 completed bars or unverified data provenance also fail. The check stops
at the first unusable session and records the number checked in `diagnostics.json`.
Pass `--all-development` to inspect every development session despite ordinary
coverage/provenance failures. Complete coverage can still be insufficient data;
per-session reasons and eligible fractions explain the result. Integrity, loader
memory limits and training deadlines remain enforced. Neither mode fits a model,
reads reserved final-test prices or establishes paper eligibility. Both modes
atomically write `diagnostics.json` beside the frozen experiment after successful
traversal; exceptions preserve the previous report. A later default preflight can
replace a full report with a correctly labelled prefix.

Each session reports numeric and metadata bytes, metadata column sizes and retained
total using the loader's existing buffer accounting. Research loads the
execution metadata view (identities, arrival times, trade conditions/tape), so
these counts cover what is held in memory, not every archived column; raw JSON
remains in the Parquet archive. Diagnostics recorded before October 7, 2026
counted every column. These counts exclude Python
object overhead, simulation timeline indices and transient copies. Process peak
RSS is reported separately in bytes: it is the cumulative process lifetime high
water mark, not that session's allocation. Full inspection releases each session
before loading the next. Setup, preflight and fitting all count toward the cumulative
training time budget. Resuming a completed final report preserves it.

The current local MCD dataset has 82 real IEX sessions, with 52 development and
30 reserved final-test sessions. Full inspection of
`artifacts/mcd-v3/experiment.json` found gaps in all 52 development sessions:
2,054 eligible and 238,106 ineligible decisions (0.855% eligible), 57,049 gaps.
Diagnostic peak RSS was 295.16 MiB; no model was fitted or qualified. Training
remains blocked before fitting. See [measured results and next project](research-readiness-results.md)
for identities, memory details and limitations. Extra training cannot fix missing
market coverage.

Reports include cash, risk-matched intraday long, EMA 5/20, twenty random controls,
transaction costs, completed flat-to-flat trades, quote-level drawdown, stress
execution and a corrected paired block bootstrap. A separate full-stock return
is context rather than a risk-matched control. Failure to match actual random
execution frequency prevents eligibility. Too little history is an explicit
`insufficient-data` result. Passing engineering tests does not establish an edge.

The resulting `artifacts/aapl/search/bundle/` contains `model.onnx` and a hashed
manifest with the feature, execution, dataset and research contracts. Old single
ONNX files and the previous three-action schema are rejected.
Feature-v2 bundles and older frozen experiments are also incompatible with the
current feature-v3 and causal boundary-timer contract. Freeze a new experiment
and retrain; do not relabel an old model or delete final-test reservations.

## Replay and dry runs

```bash
python -m hft replay --model artifacts/aapl/search/bundle --dataset data/aapl/manifest.json
python -m hft dry-run --model artifacts/aapl/search/bundle --duration 3600
python -m hft status --logs logs --state .state
```

Replay and dry-run use the same long-only fractional sizing, ledger, features,
risk gateway and displayed-liquidity execution simulator as training. Fills need
a genuine eligible future quote after 75ms latency; they cannot use a bar close or
invent liquidity. Defaults allocate 10% of session-start strategy equity, reserve
fees, cap maintenance exposure at 15%, enforce daily loss at 1% and persistent
peak drawdown at 5%, and limit entries to five per minute. Repeated long signals
hold the existing quantity. The strategy can grow its allocation with earnings
within its frozen hard notional ceiling.

The last minute of the actual session is reserved for closing. Warmup requires
61 completed bars each session. Stale/future quotes, excessive volatility, sleep,
backlogs and unresolved orders prevent entries. A stop without a subsequent
executable local quote records remaining inventory; it cannot fabricate a fill.
Local account and risk state survives restart. Simulated outstanding orders are
cancelled on restart; broker orders are reconciled.

Replay publishes bars at causal five-second timer boundaries. Training and
runtime keep the same bounded 61-bar history and reset warmup on feed gaps.
Truncated training episodes use genuine subsequent quotes for bounded liquidation;
unresolved positions block reset rather than disappearing. Incomplete normal,
stress or primary-control rollouts cannot qualify. Historical data without arrival
timestamps still cannot establish actual network latency, and boundary-timer
replay idealizes the streaming loop's polling delay.

## Broker paper trading and graduation

Broker paper trading submits real requests to the **paper endpoint only**. Use a
dedicated account with no unrelated positions/orders, with strategy capital no
larger than verified non-borrowed cash:

```bash
export ALPACA_PAPER_API_KEY='your-paper-trading-key'
export ALPACA_PAPER_SECRET_KEY='your-paper-trading-secret'
python -m hft broker-paper --model artifacts/aapl/search/bundle --capital 500
python -m hft graduate --model artifacts/aapl/search/bundle --logs logs/broker-paper --output artifacts/aapl/graduation.json
```

Keep market-data credentials configured too; paper credentials are accepted as
their fallback. Runtime credentials for paper and live order endpoints are
separate. No account is created, funded or reset by this program.

Quote intake uses a bounded independent task while REST order operations run
serially in a worker thread. Trade updates provide primary notifications and REST
polling recovers missed updates. An acknowledgement is not a fill. Client order
IDs and pending state are durable before POST; a timeout triggers lookup using
the same ID, never a second blind submission. A cancellation acknowledgement does
not establish a terminal status. Fresh broker inventory permits an emergency
market reduction when local data has failed. Unexpected positions, orders, cash,
fees or funding changes stop trading and disqualify that session.

The gate recomputes results from hash-checked journals and their executions,
checks the frozen research result, and requires at least 30 consecutive complete
real broker-paper sessions, 100 completed trades, positive profit/expectancy,
profit factor >=1.3 and drawdown <5%. It replays recorded events under worse costs
and latency, and checks the stricter live canary separately. Evidence expires
five actual exchange sessions later. Replay, dry runs, synthetic sessions,
incomplete coverage, changed settings and a caller-supplied `passed` value cannot
satisfy the live constructor. Hash chains detect corruption, not a malicious
owner rewriting the whole application or its local files.

## Multi-stock paper runs (`hft trade`)

`hft trade --paper` trades several stocks on one paper account, each with its own
dollar budget, ledger and order book. Only one run can use a paper account at a
time: every run takes the same account lock as `broker-paper`, and a new run needs
a flat account with no open orders. A stopped run resumes from its saved state
when restarted with the same name and settings; changed settings need a new name.

```bash
python -m hft trade --paper --symbols NVDA=100 AAPL=100 --strategy ema-crossover --name day
python -m hft trade --paper --symbols KO=500 BAC=500 --strategy overnight-drift --name night
python -m hft trade --status
```

Bar strategies decide on 5-second bars and are flat at every close. Stopping one
(Ctrl-C or SIGTERM) cancels orders and sells holdings before exiting.

`overnight-drift` acts on a clock instead, using the exchange calendar:

| Time (ET) | Action |
| --- | --- |
| 09:00-09:27:30 | Sell held stock: `opg` market order for whole shares, day market order otherwise |
| After 09:30 | Sell anything still held with a day market order, retried each minute |
| 15:45-15:49:30 | Buy with a `cls` market-on-close order, sized as if filled 2% above the last trade, when whole shares use at least 90% of the stock's cash (Alpaca paper expires limit-on-close orders unfilled) |
| 15:57-15:59 | Buy the other stocks, or any whose on-close order failed, with fractional marketable day limits (1% above) |

The times move with early closes. Before each entry the account-wide guard marks
holdings at the latest IEX trade: losing the daily limit since the previous entry
skips that day's entries, and the drawdown limit stops entries for good. Exits are
never blocked, and both
guard decisions survive restarts. A market-on-close fill above its sizing cap is still
recorded (the stock's cash goes negative and its entries wait) and sold at the open.
Overnight entries have no spread guard: auctions trade at the official price and day
buys are capped 1% above the last trade. Dividends on the stocks are booked to their
ledgers during reconciliation. Broker, network and market-data errors are logged as
`quality` rows and retried; a broker report that contradicts the ledger, a
reconciliation mismatch that survives a fresh poll, or an order unresolved for three
hours stops the run. Stopping the run
cancels open orders but keeps positions, because the market is usually closed;
restarting it sells them at the next open.

`news-picks:<variant folder>` day-trades the picks of a frozen ML day-trader variant
(a folder with `config.json` and saved daily weights, from `hft.ml.search`). At 08:55
ET it refreshes the daily bars, news and calendar; between 09:30:30 and 09:35 it reads
the first IEX trades as today's opening prices and picks with the model; from 09:31
it buys each pick with a marketable day limit 1% above its last trade; five minutes
before the close it sells everything. Holdings it did not buy today are sold at once.

```bash
python -m hft trade --paper --strategy news-picks:artifacts/ml-news-search/7f7b87ac489e \
    --symbols AAPL=1000 MSFT=1000 ... --name news-picks
```

Journals go to `logs/trade/<name>/<SYMBOL>-<start_ns>.jsonl` and state to
`.state/trade/<name>/`. Rule-strategy runs are engineering evidence only and never
count toward graduation.

Both are Git-ignored: they hold the paper account ID. `scripts/record_paper_day.sh
<name>` writes the run's daily forward record (picks, average fill prices, results)
to `docs/<name>-forward.md`, commits and pushes it, and copies all journals and run
state to a private backup repository (`HFT_RECORDS`, default
`~/git/hft-paper-records`). Schedule it after the close, for example with cron:

```bash
30 16 * * 1-5 cd ~/git/hft && scripts/record_paper_day.sh news-picks >> tmp/record.log 2>&1
```

## Live activation and operations

The build and tests never submit live orders. The operator must separately
configure `ALPACA_LIVE_API_KEY` and `ALPACA_LIVE_SECRET_KEY`, provide a frozen real
bundle, and explicitly activate the command. After passing graduation:

```bash
python -m hft live --model artifacts/aapl/search/bundle --logs logs/broker-paper --enable-live --capital 500 --max-live-notional 10
```

This path rechecks evidence and the current broker calendar before trading. The
first live allocation is capped at $10 per entry and stops after ten completed
live sessions for operator review. Raising the ceiling is not implemented as an
automatic response to profits. Changing capital, feed, features, costs or risk
settings requires a newly evaluated bundle; the only supported execution override
is a stricter canary cap with its own replay.

Use Ctrl-C for controlled shutdown. Remote shutdown cancels outstanding orders,
waits for final fills, closes verified remaining inventory while the market is
open, and reconciles. Unresolved shutdown returns an error and leaves durable IDs
and truthful logs. Inspect the broker's positions/open orders before restarting.
Do not delete `.state/` to bypass an unknown order, a drawdown latch or a failed
reconciliation. A second process cannot own the same account by choosing a new
log or state path.

Actual broker regulatory fees not exposed on an order, external transfers and
corporate actions currently require operator reconciliation: cash differences
beyond one cent halt entries rather than being silently treated as trading PnL.
Fractional limit order acceptance and fee/settlement behavior still need a real
paper-account integration run. Broker rules and entitlements are checked through
account/API responses; the code does not embed an obsolete universal $25,000
intraday-trading assumption.
