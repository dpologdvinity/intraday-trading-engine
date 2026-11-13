# Session loading performance: measured results

October 7, 2026. Two changes to `hft/data.py` made research sessions cheaper to load
and hold without changing simulated behavior:

1. **One bounded preflight per session** (`869e997`). `load_dataset` ran the
   memory-budget preflight twice per session, and its metadata sizing scan read
   64-row Arrow batches. Profiling a 5.1M-quote session showed that scan was about
   95% of load time. Each session is now preflighted once, scanning 8,192-row batches;
   the budget is still checked after every batch, before any market array is allocated.
2. **Execution metadata view** (`e318ca1`). Research loads now keep only the metadata
   the simulation reads: record identities (de-duplication and quote matching),
   arrival times (causal ordering) and trade conditions/tape (bar filtering). Raw
   JSON, raw lot sizes and exchange codes stay in the Parquet archive. Quote
   exchange/condition fields are retained whenever any quote lacks an identity,
   because they feed the fallback identity hash. Replay and recording still load
   every column (replay journals log whole events), and a projected session refuses
   to be re-saved.

Archives, checksums and session identity hashes are unchanged: identity hashes
exclude loaded metadata, and partition checksums cover the file bytes.

## Results

Measured with [`benchmarks/session_pipeline.py`](https://github.com/dpologdvinity/intraday-trading-engine/blob/master/benchmarks/session_pipeline.py),
which runs each repetition in a fresh process and fingerprints the bars, gaps,
decision eligibility and (with `--rollout`) the full always-long rollout result.
"Before" is commit `d67bb47`; "after" is the execution view at `e318ca1`. Raw
outputs are in [benchmarks/results/2026-10-07](https://github.com/dpologdvinity/intraday-trading-engine/blob/master/benchmarks/results/2026-10-07/).

### NVDA 2026-06-05: 5,103,471 quotes, 76,525 trades

| Metric | Before | After | Change |
| --- | ---: | ---: | ---: |
| Load (s) | 54.97 | 5.52 | **10.0× faster** |
| Simulation build (s) | 80.98 | 47.68 | −41% |
| Diagnostic traversal (s) | 121.03 | 117.84 | −3% (noise range) |
| Load + build + traversal (s) | 256.98 | 171.04 | −33% |
| Retained session buffers (MiB) | 1,420.3 | 529.3 | **−62.7%** |
| RSS after load (MiB) | 1,686.3 | 698.5 | −58.6% |
| Process peak RSS (MiB) | 2,922.0 | 1,941.8 | **−33.5%** |

Fingerprints are identical: 4,680 bars, 0 gaps, 4,619 eligible decisions, same bar
digest. The intermediate single-preflight build (load 4.77 s, retained 1,420.3 MiB,
peak RSS 2,911.1 MiB) isolates the first change; the execution view's identity
completeness scan adds about 0.75 s of the final load time.

### MCD 2026-06-04: 59,084 quotes, 3,541 trades (median of 3)

| Metric | Before | After | Change |
| --- | ---: | ---: | ---: |
| Load (s) | 0.40 | 0.08 | 5× faster |
| Retained session buffers (MiB) | 16.8 | 6.3 | −62.5% |
| Process peak RSS (MiB) | 152.6 | 137.4 | −10% |

Bars, 936 gaps, 42/4,577 eligible/ineligible decisions **and the complete always-long
rollout result** (fills, costs, account and risk outcomes) are identical. Simulation,
diagnostic and rollout times for this small session varied 1.1–2.9 s between
repetitions on both sides; the differences are within that noise. Peak RSS here is
dominated by the interpreter and libraries rather than session data.

### All 52 MCD development sessions (retained bytes)

| | Retained | Of which metadata |
| --- | ---: | ---: |
| Every column | 659.0 MiB | 567.6 MiB |
| Execution view | 245.1 MiB | 153.7 MiB |
| Change | **−62.8%** | −72.9% |

The every-column metadata total (595,218,153 bytes) matches the figure previously
published in [development readiness results](research-readiness-results.md).

## Equivalence evidence

- `tests/test_projection.py` loads real-archive-schema fixtures both ways and
  requires identical event streams (minus dropped keys), bars, gaps, per-decision
  eligibility, rollout results with repeated entries/exits, and observations, with
  and without recorded arrival times. It also covers the identity-hash fallback,
  the re-save guard, checksum failures and the smaller preflight budget.
- Phase-isolation tests now forbid the preflight and reader stages directly, so a
  reserved final-test partition cannot be checksummed, scanned or read.
- Full suite: 218 Python tests, Ruff lint/format.

## Environment and limitations

Python 3.12.3, Linux x86_64 under WSL2, 12 logical CPUs, about 15 GiB visible RAM.
NVDA timings are single uncontended runs per configuration; memory figures are
deterministic (retained bytes) or stable across runs (two baseline-code NVDA runs
peaked within 0.4% of each other). A three-repetition NVDA rerun was abandoned
when an unrelated CPU-heavy workload started on the same machine (load average
above 17), which inflated both wall and CPU time several-fold; those runs are not
reported. The harness now also records per-phase CPU time.

NVDA dates are development-only probe sessions; MCD figures use development
sessions only. Diagnostics recorded before this change counted every metadata
column; new diagnostics report the execution view and say so.

## Next measured opportunities

Profiling the execution view on NVDA shows the remaining time is in per-event work:

- Simulation build: per-element Arrow scalar conversion in `SessionData.iter_events`,
  `BarAggregator.add` and Decimal `Quote` construction (5.1M events).
- Diagnostic traversal: per-quote Decimal conversion, risk observation and risk
  snapshot construction (about 1.3M quotes per 1,000 decisions).

Both are on the causal execution and accounting path, so any change needs the same
fingerprint and rollout equivalence proof before adoption.

## C++ bar aggregator (preliminary, October 7, 2026)

`hftcore` (C++20, pybind11) ports `hft.feed.BarAggregator` line for line. Parity
tests (`tests/test_engine_parity.py`) require every bar field and quality counter
to equal the Python reference on shared edge-case vectors and full synthetic
sessions; matching CPython 3.12's compensated float `sum()` was needed for
bit-identical VWAPs.

| Implementation | Events/s | ns per event |
| --- | ---: | ---: |
| Python reference (`hft.feed.BarAggregator.add`, 299,000 events) | 35,611 | 28,081 |
| C++ (`cpp/bench/bench_aggregator.cpp`, 6,337,500 events) | 1,404,946 | 712 |

Synthetic NVDA-like stream: one quote every 4 ms with provider ids and a trade every
48 ms over a 6.5-hour session. Single runs on the same machine while an unrelated
CPU-heavy job was running (load average about 16-22 on 12 logical CPUs), so both
figures are conservative; the ratio (about 39x) is the comparable quantity. This
measures the aggregator alone, not a full simulation; the first C++ version still
builds a string identity key per event, which the planned hashed-key change removes.

## C++ session replay on real data (October 7, 2026)

`hftcore.replay` reproduces the bar, publication-time and feed-gap construction of
`hft.execution.Simulation` (`hft.feed.historical_bars`). It reads identities from
the Arrow string buffers zero-copy and releases the GIL while running.
`tests/test_replay_parity.py` requires identical bars, ready times and gap times on
archive-schema fixtures and on real development sessions (MCD and NVDA 2026-06-04).

| NVDA 2026-06-05 (5,103,471 quotes, 76,525 trades) | Seconds |
| --- | ---: |
| Python reference (`historical_bars(implementation="python")`) | 111.51 |
| C++ (`historical_bars(implementation="cpp")`, including Python-side setup) | 3.14 |
| Speedup | **35.5x** |

Single run on a shared machine (load average about 15 on 12 logical CPUs), both
paths in the same process; outputs compared equal (4,680 bars). `Simulation` uses
the C++ path automatically when `hftcore` is installed (`HFT_ENGINE=python` forces
the reference).

## Tick-to-decision latency and allocation-free deduplication (October 7, 2026)

`cpp/bench/bench_engine.cpp` drives `hftcore::MarketEngine` (EMA-crossover) through
a synthetic NVDA-like day: 6,337,500 quote and trade events and 4,619 decisions.
"Decision" latency is the time of the event call that completed a bar and produced
a decision (bar publication, history, 10 features, strategy).

Profiling pointed at per-bar duplicate tracking: a node-based
`unordered_set<std::string>` allocated a key per event and freed ~1,350 nodes at
every bar boundary. It was replaced by a flat open-addressing set of 128-bit
identity hashes cleared by a generation counter (no allocation, O(1) clear; a false
duplicate would need a 128-bit collision). All parity suites still pass.

| Same machine, load average ~11 | Before | After |
| --- | ---: | ---: |
| Event latency p50 / p99 | 155 ns / 1,467 ns | **71 ns / 354 ns** |
| Decision latency p50 / p99 | 72.1 µs / 796.9 µs | **12.7 µs / 101.4 µs** |
| Aggregator throughput (synthetic stream) | 1.40M events/s | **9.69M events/s** |
| Real NVDA 2026-06-05 replay (Python 111.51 s) | 3.14 s (35.5x) | **1.29 s (86x)** |

Single runs (replay: best of three) on a shared machine; tail percentiles include
scheduler preemption from other workloads.

## Native stream parsing (October 7, 2026)

`hftcore.FrameRouter` parses Alpaca market-data frames with simdjson (on-demand)
and routes each subscribed symbol's messages into its C++ market engine. Parity
tests feed the same frames to the router and to the Python path (`json.loads`,
stream filtering, `PyMarketEngine.on_event`) and require identical updates,
parsed quotes (including Decimal-exact lot-to-share scaling) and malformed counts.

`cpp/bench/bench_frames.cpp` (5,850,000 single-message frames, two symbols, 9,238
decisions): **parse-to-decision p50 998 ns**, p99 10.0 µs, p99.9 53.4 µs on the
shared machine. The live runner still uses the Python stream path; switching it is
a follow-up once a paper session has run on the current path.

## Nasdaq ITCH 5.0 order book (November 13, 2026)

`hftcore` also contains a Nasdaq TotalView-ITCH 5.0 feed handler and full-depth limit
order book (`itch`, `order_book`, `itch_book`), with a plain Python reference
(`hft/itch.py`) that it must match exactly. It is a market-data engineering exercise
that no strategy or paper bot uses; the [design](specs/2026-11-13-itch-order-book-design.md)
records the goals before any measurement. This section reports what was measured on a
real trading day, and one of the three goals was missed.

### Data

- Day: 2019-01-30, `01302019.NASDAQ_ITCH50` from Nasdaq's free ITCH samples (BinaryFILE:
  2-byte big-endian length before each message). 4,764,426,091 bytes compressed,
  11,245,883,092 bytes decompressed (10,724.9 MiB), **368,366,634 messages**, which equals
  the count Nasdaq publishes for the day.
- Messages by type: add `A` 162,970,455 and `F` 1,725,898; delete `D` 158,273,361;
  replace `U` 27,222,746; execute `E` 8,096,995; cancel `X` 4,669,874; the other types
  total 5,407,305 (trades, cross trades, directory, halts, imbalance and so on).
- License: Nasdaq publishes no terms for the samples, so the files are used locally
  only, in the git-ignored `data/itch/`. Raw or reconstructable data (messages, order
  books, per-order output) is never committed or published; this page contains only
  counters, timings and four closing prices.

### Method

- `cpp/bench/bench_itch.cpp` (`hftcore_bench_itch FILE --repeat 3`) maps the decompressed
  file read-only and consumes it in 256 MiB windows. Each window is pre-faulted before
  the clock starts and released afterwards, so page-in is outside the timed spans
  (the tool prints the page faults that still land inside them).
- Pass (a) parses every message and folds every decoded field into a checksum. Pass (b)
  parses and applies every message to the full book. Both report the CPU-bound sum of
  the timed spans (wall ns per message), three runs each, median of the three.
- Pass (c) times one `dispatch()` call per message with `rdtscp` after `lfence`
  (2.673 GHz, calibrated against `steady_clock`). The timer overhead (29.2 ns in run 1,
  23.9 ns in run 2) is measured as the median of 1,000,000 empty regions and subtracted.
  Samples go into a log-linear histogram (64 buckets per power of two); percentiles are
  bucket midpoints within 0.8% of the recorded value, and the maximum is exact.
- The tool checks itself: pass (c) counters equal pass (b), pass (a) frame counters
  equal pass (b), and the book validates after pass (c). All three held in both runs.
- Pinned with `taskset -c 2`, one run at a time. `rdtscp` saw no CPU migration in
  either run. Build: Release (`-O3 -DNDEBUG`), g++ 13.3.0, warnings as errors.
- The bench was run twice in full (6 runs of each of passes (a) and (b)); the second
  run was made because the first reported major page faults in timed spans.

### Results

| Pass | Run 1 | Run 2 |
| --- | ---: | ---: |
| (a) parse only, median of 3 | 21.5 ns/msg (46.5 Mmsg/s) | 18.0 ns/msg (55.7 Mmsg/s) |
| (a) per-run values | 22.4, 21.5, 20.9 | 20.3, 18.0, 17.1 |
| (b) parse plus full book, median of 3 | 444.4 ns/msg (2.25 Mmsg/s) | 320.7 ns/msg (3.12 Mmsg/s) |
| (b) per-run values | 444.4, 422.3, 457.8 | 296.7, 509.9, 320.7 |
| (c) latency, all messages: p50 / p99 / p99.9 | 445.7 / 3,231.7 / 7,421.3 ns | 315.6 / 1,322.7 / 4,477.3 ns |
| (c) max, all messages | 135.8 ms | 113.1 ms |
| (c) `A` p50 / p99 / p99.9 | 363.4 / 2,669.1 / 5,290.6 ns | 270.7 / 951.6 / 3,759.0 ns |
| (c) `D` p50 / p99 / p99.9 | 601.3 / 3,423.3 / 10,006.8 ns | 421.8 / 1,334.7 / 4,477.3 ns |
| (c) `U` p50 / p99 / p99.9 | 727.0 / 4,955.4 / 18,673.1 ns | 529.6 / 1,807.6 / 8,092.8 ns |
| (c) `E` p50 / p99 / p99.9 | 481.6 / 3,519.0 / 7,229.7 ns | 351.5 / 1,382.6 / 4,716.7 ns |
| (c) `X` p50 / p99 / p99.9 | 391.8 / 2,908.5 / 5,099.1 ns | 249.7 / 1,191.0 / 3,759.0 ns |

Across the two bench runs, pass (a) medians are 18.0-21.5 ns per message and pass (b)
medians are 320.7-444.4 ns per message (all six per-run values: 296.7-509.9 ns). The
two bench runs are not interchangeable repeats of one measurement; their spread is the
result, and no quiet-machine measurement exists. As a cross-check, the unfiltered
`hftcore_itch_replay` (the pass (b) code path, including its own file reads) used
156.72 s of user time for 368,366,634 messages, about 425 ns per message.

### Goals

Set in the design before measuring; the mean was the stated statistic, and the table
uses the medians of three runs.

| Goal (one core, 2019-01-30 day) | Result | Status |
| --- | --- | --- |
| Parse only at or below 25 ns per message | 18.0 and 21.5 ns (every per-run value 17.1-22.4 ns) | Met |
| Parse plus full book at or below 120 ns per message | 320.7 and 444.4 ns (per-run 296.7-509.9 ns), 2.7 to 3.7 times the goal | **Not met** |
| Peak RSS below 1 GiB | 173,616 kB (0.17 GiB) | Met |

The full-book goal was missed in every one of the six runs, including the fastest
(296.7 ns). The measurements say how far, not why. Where the time goes (the order map
lookups, the sorted level vectors, cache misses on a 1.7M-order working set, or
something else) has not been profiled; that is the next measured opportunity, and no
optimization was attempted before reporting.

Memory is the unfiltered `hftcore_itch_replay` maximum resident set under
`/usr/bin/time -v`: 173,616 kB (175,544 kB with `--validate-every 10000000`). The
bench's own `ru_maxrss` (709 and 710 MiB) is not the memory figure: it includes the
mmapped input pages of about one 256 MiB window. The bench's sampled peak anonymous RSS
(books and buffers, sampled between timed spans, so it can slightly underestimate an
instantaneous peak) was 195 MiB in both runs.

### Verification

- The message count equals the published 368,366,634. Unknown order references,
  duplicate references, overfills and invalid (zero-share) orders are all 0. Unknown
  message types, malformed lengths, trailing bytes and truncated bytes are all 0.
- The file has no end-of-session (zero-length) marker; the input ends on a frame
  boundary.
- Peak live orders 1,742,866; peak price levels 707,130. At the end of input there are
  0 live orders and 0 levels (every order was executed or deleted during the day).
  377 successful add or replace messages left the touched stock's best bid at or above
  its best ask ("crossed"); these are counted and not interpreted here.
- A full replay with `--validate-every 10000000` ran `validate()` 37 times (36 chunk
  boundaries and one at the end); all passed.
- `scripts/itch_parity.py` feeds identical bytes to the Python reference and the C++
  book and compares counters, every locate's stock summary, a full-depth snapshot and
  the FIFO queue of every level. On the first 5,000,000 messages (4,302 locates and
  17,212 price levels with live orders) there were 0 mismatches; on the first
  50,000,000 messages (8,694 locates, 661,217 levels) also 0. Those runs took 7.7 and 9.0 microseconds of user
  CPU per message (Python reference, C++ book and comparison together), so the full
  day was not compared; the books are known identical only on those prefixes (and on the seeded
  random streams in `tests/test_itch_parity.py`).
- Closing cross prices from `Q` messages in a run restricted to four symbols (63 s,
  peak 22,068 kB): AAPL $165.25, MSFT $106.38, SPY $267.44, QQQ $165.68. The end-of-input
  book is empty, so no comparison of the closing cross against a book mid is possible
  from this run and none is made.

### Environment and limitations

- Intel Core i7-1265U (12 logical CPUs; a hybrid CPU, and WSL2 does not show whether
  core 2 is a performance or efficiency core), 15 GiB RAM, Linux 6.18.58.1 under WSL2,
  on a laptop that was also running the two paper-trading bots (pinned to cores 4-5,
  not touched). Measurements were pinned to core 2; whether the host shares physical
  resources with the bots is not visible from inside WSL2.
- The 10.7 GiB input is larger than the page cache WSL2 keeps free (about 10 GiB,
  shared with the bots and Windows), so some pre-faulted pages were evicted again
  before their timed span. Major page faults inside timed spans, totals over the three
  runs of a pass: pass (a) 230 (run 1) and 15 (run 2); pass (b) 40 (run 1) and 140
  (run 2). A repeat run did not remove them. Each is a disk read, which matters more
  for pass (a) (8 s in total) than for pass (b) (minutes), but no rate here is free
  of them.
- Run-to-run variance is large (pass (b) per-run 296.7-509.9 ns; pass (c) p50 445.7 ns
  against 315.6 ns). Pass (c) times only the dispatch into the builder, after the timer
  overhead is subtracted.
- Calibration figures quoted in the design (parse-only 5-25 ns, a tuned full book
  40-120 ns, a naive map-based book 250-300 ns per message) are self-reported by other
  authors on other machines and workloads. They are not comparable measurements and
  are not a benchmark of this implementation.
- One day, one machine, one build. The book is not used by any strategy, no live ITCH
  feed is involved, and no real data is used in CI.

### Reproduce

Needs a local copy of a Nasdaq ITCH sample in git-ignored `data/itch/`.

```bash
gzip -dc data/itch/01302019.NASDAQ_ITCH50.gz > /path/with/12GB/free/day.itch
taskset -c 2 cpp/build/hftcore_bench_itch /path/with/12GB/free/day.itch --repeat 3
taskset -c 2 /usr/bin/time -v cpp/build/hftcore_itch_replay /path/with/12GB/free/day.itch
python -m pip install ./cpp
python scripts/itch_parity.py /path/with/12GB/free/day.itch --messages 5000000
```
