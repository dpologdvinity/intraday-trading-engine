# hftcore

C++20 market-data and decision engine for the trading system, exposed to Python
with pybind11. It reproduces the Python reference engine bit for bit (bars, gaps,
warmup, features and rule decisions) and powers both backtests and the paper bot;
see the [design](../docs/specs/2026-10-07-paper-trading-runner-design.md) and
[measured results](../docs/performance-results.md). It also holds a separate Nasdaq
ITCH 5.0 parser and order book (`itch`, `order_book` and `itch_book` below), which no
strategy uses; see its [design](../docs/specs/2026-11-13-itch-order-book-design.md) and
[results](../docs/performance-results.md#nasdaq-itch-50-order-book-november-13-2026).

| Component | Purpose |
| --- | --- |
| `bar_aggregator` | Causal 5-second bars with boundary timers, late/future/duplicate rules |
| `id_set` | Allocation-free 128-bit identity set with O(1) per-bar clearing |
| `features`, `market_engine` | Decision history, gap/warmup rules, 10 market features, EMA crossover / hold-day |
| `replay` | Whole-session replay for backtests (zero-copy Arrow identity buffers) |
| `frame_router` | simdjson parsing of Alpaca stream frames into per-symbol engines |
| `itch` | Nasdaq ITCH 5.0 BinaryFILE parser: one struct and decoder per message type, framing, unknown/malformed/truncated accounting |
| `order_book` | Full-depth limit order book: order pool, open-addressing ref map, sorted price levels with FIFO queues, `validate()` |
| `itch_book` | Feed handler: stock directory, one book per stock locate, optional symbol filter, trade prints and counters |
| `timestamp`, `decimal`, `numeric` | RFC3339 parsing, exact decimal scaling, CPython-compatible float sums |

## Build, test, benchmark

```bash
python -m pip install cmake ninja
cmake -S cpp -B cpp/build -G Ninja -DHFTCORE_BUILD_TESTS=ON
cmake --build cpp/build -j2
ctest --test-dir cpp/build --output-on-failure
cpp/build/hftcore_bench_engine      # tick-to-decision latency percentiles
cpp/build/hftcore_bench_frames      # JSON frame parse-to-decision latency
cpp/build/hftcore_bench_aggregator  # aggregator throughput

# ITCH order book (needs a Nasdaq ITCH sample in the git-ignored data/itch/)
gzip -dc data/itch/01302019.NASDAQ_ITCH50.gz | cpp/build/hftcore_itch_replay -
cpp/build/hftcore_itch_replay --symbols AAPL,MSFT --validate-every 10000000 DAY.itch
taskset -c 2 cpp/build/hftcore_bench_itch DAY.itch --repeat 3   # DAY.itch: decompressed file
python scripts/itch_parity.py DAY.itch --messages 5000000       # C++ vs Python book

python -m pip install ./cpp           # Python module; used automatically when installed
HFT_REQUIRE_HFTCORE=1 python -m pytest -q tests/test_engine_parity.py tests/test_replay_parity.py
```

`-DHFTCORE_SANITIZE=ON` builds with AddressSanitizer and UndefinedBehaviorSanitizer
(also run in CI). Warnings are errors, floating-point contraction is disabled for
reproducible results, and third-party code (simdjson, Catch2) is fetched at pinned
versions with verified SHA-256 hashes.

`hftcore_itch_replay` prints counters and per-symbol aggregates only. The bench
needs a decompressed file and one pinned core (`taskset -c N`) because it reads the
cycle counter; it also prints page faults that land inside timed spans. The ctest
`itch_replay_cli` checks the replay tool on a small synthetic file; real data is never
used in tests or CI, and no ITCH data is committed.
