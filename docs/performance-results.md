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
