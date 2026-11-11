# Paper Trading Runner with C++ Decision Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run Alpaca paper trading on user-chosen stocks with per-stock budgets and selectable strategies every trading day, with market data and decisions computed by a C++ engine that also powers backtests.

**Architecture:** Task 0 extracts the market side of `PaperEngine` (bars, history, gaps, market features, rule actions) into a Python reference `MarketEngine` with a frozen interface. Track A ports that engine to C++ (`hftcore`, pybind11) and proves bit-identical output; Track B builds the multi-stock paper runner (stream, budgets, portfolio broker, supervisor, CLI, dashboard) against the Python reference. Integration selects the C++ engine when installed.

**Tech Stack:** Python 3.12, NumPy, websockets, existing hft modules; C++20 (g++ 13), CMake, pybind11, scikit-build-core, simdjson, Catch2; pytest, Ruff, GitHub Actions.

**Spec:** `docs/specs/2026-10-07-paper-trading-runner-design.md`

## Global Constraints

- Paper only. The `live` command, `evidence.graduate`, the $10 canary and ten-session stop are not modified. Orders go only to the Alpaca paper endpoint, and only during the authorized acceptance task (I2) or under explicit owner request.
- Decision cadence 5 s; v3 contract unchanged: 17 float32 features, actions 0 flat / 1 long, 61-bar warmup, 5 s gap rule, 250 ms future-event tolerance, 75 ms latency.
- The C++ engine must equal the Python reference exactly: bars field-equal (floats compared with `==`), quality counters equal, market features bitwise equal (`np.array_equal(a.view(np.uint32), b.view(np.uint32))`), identical actions and gap events. A difference is a bug, never a contract relabel. The Python reference stays and is the fallback.
- Fraction-based `RiskConfig`/`SizingConfig` defaults and validation (10% entry, 15% maintenance) stay unchanged; budget runs use the separate budget path in `hft/trading/budget.py`.
- Real-data parity and benchmarks use development sessions only and must call `hft.training.reserved_final_sessions()` and refuse matches before opening data.
- No AI-tool names in tracked files or commit messages (local hooks enforce). This plan file stays untracked.
- Resource hygiene: remove worktrees after merge, delete temporary files and build directories you create, leave no background processes running.
- Keep green after every task: `.venv/bin/python -m pytest -q`, `.venv/bin/ruff check hft tests benchmarks scripts`, `.venv/bin/ruff format --check hft tests benchmarks scripts`; frontend tasks also `npm --prefix frontend test`, `npm --prefix frontend run format:check`, `npm --prefix frontend run build`.
- Pinned C++ dependencies: pybind11 2.13.6, scikit-build-core 0.10.7, simdjson v3.10.1, Catch2 v3.7.1 (FetchContent by URL with `URL_HASH SHA256=` recorded at first download).

## Review Focus

- Run started mid-session (after the open): no immediate fatal gap; warmup begins from the first bar boundary after start. Test in B6.
- Stream delivers an unsubscribed symbol, a `subscription`/`success` control message, or a malformed frame: ignored (unsubscribed/control) or counted and skipped (malformed), never crashes the run. Tests in B3 and A5.
- Paper account not flat, or holding a foreign position/order at start: refused with a clear message before any order. Test in B4.
- Sum of budgets above non-borrowed cash, or a symbol that is inactive/not fractionable: refused before trading. Test in B4.
- Machine sleeps mid-session (runtime gap) or a session has an early close / holiday: entries halt for that session; the supervisor resumes at the next calendar session. Tests in B6 and B7.

---

## Task 0: Extract the Python reference market engine and freeze the interface

Done first, sequentially, on `master` before the tracks split.

**Files:**
- Create: `hft/market_engine.py`, `hft/strategies.py`, `tests/engine_vectors.py`, `tests/test_market_engine.py`
- Modify: `hft/features.py` (split `observation`), `hft/paper.py` (delegate market logic)

**Interfaces:**
- Produces (`hft/features.py`): `market_features(history: Sequence[Bar]) -> np.ndarray` (float32[10], features 0-9) and `state_features(history, account, risk_state, session_window, *, now_ns=None, quote=None) -> np.ndarray` (float32[7], features 10-16). `observation(...)` returns `np.concatenate([market, state])` and keeps its exact validation errors.
- Produces (`hft/strategies.py`): `RULE_STRATEGIES = {"ema-crossover", "hold-day"}`; `rule_action(name: str, history: Sequence[Bar]) -> int` (ema-crossover = `int(ema(closes,5)[-1] > ema(closes,20)[-1])` using `hft.research.ema`; hold-day = 1); `StrategySpec(name: str, version: str, bundle: Path | None)`; `parse_strategy(text: str) -> StrategySpec` accepting `ema-crossover`, `hold-day`, `model:<path>`; version `"1"`.
- Produces (`hft/market_engine.py`):
  ```python
  @dataclass(frozen=True)
  class GapEvent:
      symbol: str; now_ns: int; reason: str          # "feed_gap" | "runtime_sleep"
  @dataclass(frozen=True)
  class BarUpdate:
      symbol: str; bar: Bar; now_ns: int
      accepted: bool          # False: bar starts before the warmup boundary (ignored)
      reset: str | None       # "bar_gap" when non-contiguous history was cleared
      ready: bool             # history holds 61 bars
      market: np.ndarray | None   # market_features(history) when ready
      action: int | None      # rule_action when ready and strategy is a rule
  class PyMarketEngine:
      implementation = "python"; version = "1"
      def __init__(self, symbol: str, strategy: str = "model", bar_seconds: int = 5)
      def start_session(self, session: SessionWindow, now_ns: int | None = None) -> None
      def on_event(self, event: dict, arrival_ns: int) -> list[GapEvent | BarUpdate]
      def advance_to(self, now_ns: int) -> list[GapEvent | BarUpdate]
      history: tuple[Bar, ...]   # property
      quality: Counter           # property (aggregator counters)
      last_event_ns: int | None  # property
  def make_market_engine(symbol: str, strategy: str, *, implementation: str = "auto", bar_seconds: int = 5)
  ```
  Semantics are exactly today's `PaperEngine` rules: `start_session` aligns the aggregator start like `PaperEngine.start_session`; `on_event` emits `GapEvent("feed_gap")` and clears history when `arrival_ns - last_event_ns > 5 s`; `advance_to` emits `GapEvent("runtime_sleep")` when `now_ns - last_tick_ns > 5 s`; bars before the warmup boundary are `accepted=False`; non-contiguous bars reset history with `reset="bar_gap"`. `make_market_engine(..., "auto")` returns `PyMarketEngine` until Integration I1.
- Produces (`tests/engine_vectors.py`): `engine_vectors() -> list[tuple[str, SessionWindow, str, list[tuple[dict | None, int]]]]` (name, session, symbol, timeline of `(event or None for timer, now_ns)`), covering: normal quotes/trades, same-time timer and arrival, duplicates by `i`, duplicates without `i`, late-closed, late-tolerance, future event (>250 ms), outside session, corrections `c` and cancellations `x`, clock reversal, filtered trade conditions (`W` with `z=C`, `B` otherwise, each `CLOSE_EXCLUDED` code), missing context, 6 s feed gap, mid-session start.

- [ ] **Step 1: Write failing tests** in `tests/test_market_engine.py`:
  - `test_observation_equals_market_plus_state`: for a synthetic `TradingEnv` history, `observation(...)` is bitwise equal to `np.concatenate([market_features(h), state_features(...)])`.
  - `test_engine_bars_match_aggregator`: over `synthetic_sessions(1, 200)[0]` via `historical_timeline`, the accepted bars from `PyMarketEngine` equal `BarAggregator` bars.
  - `test_feed_gap_resets_history`: events at t and t+6 s produce one `GapEvent("feed_gap")` and `history == ()` afterwards.
  - `test_rule_actions`: rising closes give ema-crossover 1, falling give 0; hold-day always 1; `parse_strategy("model:x")` gives bundle `Path("x")`; unknown names raise `ValueError`.
  - `test_vectors_run`: every `engine_vectors()` case runs through `PyMarketEngine` without exceptions and is deterministic across two runs.
- [ ] **Step 2: Run** `.venv/bin/python -m pytest -q tests/test_market_engine.py` — expect failures (modules missing).
- [ ] **Step 3: Implement** the interfaces above; make `PaperEngine` own a `PyMarketEngine` and react to its updates (gap logging, `risk.halted = "runtime_gap"` on `runtime_sleep`, policy call only when `ready`, policy receives `observation` built from `update.market` + `state_features`). Journal rows (`gap` reasons, order of writes) must stay identical.
- [ ] **Step 4: Run** the new tests and the full suite; all pass with no changes to existing tests.
- [ ] **Step 5: Commit** `refactor: extract the market engine behind a frozen interface`.

---

## Track A: C++ engine (worktree `../hft-cpp`, branch `feat/hftcore`)

### Task A1: hftcore project scaffold and CI

**Files:**
- Create: `cpp/pyproject.toml` (scikit-build-core 0.10.7, pybind11 2.13.6, name `hftcore`, version `0.1.0`), `cpp/CMakeLists.txt`, `cpp/src/bindings.cpp`, `cpp/include/hftcore/version.hpp`, `cpp/tests/test_version.cpp`, `cpp/README.md`, `tests/test_hftcore.py`
- Modify: `.github/workflows/ci.yml` (new `cpp` job), `.gitignore` (`cpp/build/`)

**Interfaces:**
- Produces: `hftcore.version() -> str` (`"0.1.0"`); CMake targets `hftcore_core` (static library), `hftcore` (pybind11 module), `hftcore_tests` (Catch2, registered with CTest); options `HFTCORE_SANITIZE` (address+undefined); compile flags `-Wall -Wextra -Werror`, Release `-O2`.
- Produces (`tests/test_hftcore.py`): `hftcore = pytest.importorskip("hftcore")` unless env `HFT_REQUIRE_HFTCORE=1`, which turns a missing module into a failure.

- [ ] **Step 1:** Write `cpp/tests/test_version.cpp` (`REQUIRE(hftcore::version() == "0.1.0")`) and `tests/test_hftcore.py::test_version`.
- [ ] **Step 2:** Run `cmake -S cpp -B cpp/build -G Ninja` (cmake/ninja via `.venv/bin/python -m pip install cmake ninja` if absent) — expect failure (no CMakeLists).
- [ ] **Step 3:** Implement the scaffold; Catch2 and pybind11 via FetchContent/`find_package(pybind11 CONFIG)` per the pinned versions.
- [ ] **Step 4:** Verify `cmake --build cpp/build && ctest --test-dir cpp/build` passes; `.venv/bin/python -m pip install ./cpp` then `HFT_REQUIRE_HFTCORE=1 .venv/bin/python -m pytest -q tests/test_hftcore.py` passes. CI `cpp` job: install pinned Python stack, `pip install ./cpp`, ctest, `HFT_REQUIRE_HFTCORE=1 pytest -q`.
- [ ] **Step 5:** Delete `cpp/build`; commit `feat(cpp): scaffold the hftcore engine with CMake, pybind11 and CI`.

### Task A2: Bar aggregator port

**Files:**
- Create: `cpp/include/hftcore/event.hpp`, `cpp/include/hftcore/bar_aggregator.hpp`, `cpp/src/bar_aggregator.cpp`, `cpp/tests/test_bar_aggregator.cpp`, `tests/test_engine_parity.py`
- Modify: `cpp/src/bindings.cpp`, `cpp/CMakeLists.txt`

**Interfaces:**
- Produces (C++): `struct Quote {int64_t event_ns, arrival_ns; double bid, ask, bid_size, ask_size; Identity id;}`, `struct Trade {int64_t event_ns, arrival_ns; double price, size; bool eligible; Identity id;}` where `Identity` is a 128-bit hash of the identity string, or of the numeric fallback tuple when absent (quotes: event_ns, bid, ask, bid_size, ask_size; trades: event_ns, price, size); `struct Bar` mirroring `hft.data.Bar` fields; `class BarAggregator { BarAggregator(std::string symbol, int bar_seconds, Session); std::vector<Bar> add_quote(const Quote&); add_trade(const Trade&); add_correction(int64_t arrival_ns, bool cancel); advance_to(int64_t now_ns); std::map<std::string,int64_t> quality() const; }`.
- Produces (Python): `hftcore.BarAggregator(symbol, bar_seconds, session_id, open_ns, close_ns)` with `add(event: dict, arrival_ns: int) -> list[tuple]`, `advance_to(now_ns) -> list[tuple]`, `quality() -> dict`; the dict path performs the same validation and condition filtering as `hft.feed.BarAggregator.add` (raising `ValueError` with the same messages).
- Sizes in shares: the dict path multiplies quote sizes by 100 unless `sizes_in_shares` is true, as `quote_from_event` does.

- [ ] **Step 1:** Catch2 tests: boundary publication at `open + k*5 s`, timer before same-time arrival, latest-quote selection, VWAP, `tradable` rule (quote age <= 2 s, positive sizes), dedup cleared each bar. Pytest `test_aggregator_parity[vector]` over `engine_vectors()` plus `synthetic_sessions(2, 200)`: tuples equal `hft.feed.BarAggregator` bars field by field; `quality()` equals the Python `Counter`.
- [ ] **Step 2:** Run both suites — fail.
- [ ] **Step 3:** Implement the exact port of `hft/feed.py` `BarAggregator.add/advance_to` (`CLOSE_EXCLUDED = "479CGHIMNPQRTUVZ"`, `W`/`B` tape rule, 250 ms tolerances, `max_events = 100000`); per-bar buffers are `std::vector` reused across bars (no per-event allocation after warmup).
- [ ] **Step 4:** Both suites pass; full Python suite green.
- [ ] **Step 5:** Commit `feat(cpp): port the causal 5-second bar aggregator`.

### Task A3: Market engine port (history, gaps, features, strategies)

**Files:**
- Create: `cpp/include/hftcore/market_engine.hpp`, `cpp/src/market_engine.cpp`, `cpp/include/hftcore/features.hpp`, `cpp/src/features.cpp`, `cpp/tests/test_market_engine.cpp`, `hft/market_engine_cpp.py`
- Modify: `cpp/src/bindings.cpp`, `tests/test_engine_parity.py`

**Interfaces:**
- Consumes: Task 0 `BarUpdate`, `GapEvent`, `rule_action` semantics; A2 `BarAggregator`.
- Produces (C++): `std::array<float,10> market_features(const std::deque<Bar>&)` computing each value in `double` in the same operation order as `hft/features.py` (`std::log`, sequential sums) and casting to `float` last; `int rule_action(Strategy, const std::deque<Bar>&)` with the EMA seeded at the first close; `class MarketEngine` with `start_session`, `on_quote`, `on_trade`, `on_event(dict)`, `advance_to` returning updates.
- Produces (Python): `class CppMarketEngine` in `hft/market_engine_cpp.py` with `implementation = "cpp"`, `version = hftcore.version()`, and the exact `PyMarketEngine` method set, converting native results to `BarUpdate`/`GapEvent` with `hft.data.Bar` objects.

- [ ] **Step 1:** Catch2: EMA crossover on rising/falling closes, feature values on a hand-computed 61-bar history, history reset on 6 s gap and on bar gap. Pytest `test_market_engine_parity[vector]` over `engine_vectors()` and synthetic sessions for strategies `model`, `ema-crossover`, `hold-day`: update lists equal, `market` bitwise equal, actions equal, gap events equal.
- [ ] **Step 2:** Run — fail.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Pass; full suite green.
- [ ] **Step 5:** Commit `feat(cpp): port the market engine, features and rule strategies`.

### Task A4: Batch replay for backtests and real-session parity

**Files:**
- Create: `cpp/include/hftcore/replay.hpp`, `cpp/src/replay.cpp`, `tests/test_replay_parity.py`
- Modify: `hft/feed.py` (add `historical_bars`), `hft/execution.py` (`Simulation.__init__` calls `historical_bars`), `cpp/src/bindings.cpp`, `benchmarks/session_pipeline.py` (`--engine python|cpp`)

**Interfaces:**
- Produces: `hft.feed.historical_bars(session, bar_seconds=5, *, implementation="python") -> tuple[tuple[Bar, ...], np.ndarray, np.ndarray]` (bars, ready_ns int64, gaps_ns int64) — the Python implementation is the loop moved verbatim out of `Simulation.__init__`; `hftcore.replay(quote_ns, bid, ask, bid_size, ask_size, quote_arrival_ns | None, quote_ids, trade_ns, price, size, trade_arrival_ns | None, trade_eligible, trade_ids, session_id, open_ns, close_ns, bar_seconds) -> (bar_rows, ready_ns, gaps_ns, quality)`. Python precomputes `quote_ids`/`trade_ids` (Arrow `i`/`quote_id`/`trade_id` strings, else `None` for numeric fallback) and `trade_eligible` from conditions/tape with the `CLOSE_EXCLUDED` rule.
- `Simulation.__init__` uses `implementation="python"` in this task (Integration I1 switches the default).

- [ ] **Step 1:** Tests: `test_historical_bars_python_unchanged` (bars/ready/gaps equal the pre-refactor `Simulation` values on synthetic sessions, with and without arrival times); `test_replay_parity_synthetic`; `test_replay_parity_real` that, for available development sessions (MCD 2026-06-04; NVDA 2026-06-04/05 under `data/free-data-probe-2026-10-04`), refuses `reserved_final_sessions()` matches, skips when data is absent, and requires equal bars, ready times, gap times and quality counters.
- [ ] **Step 2:** Run — fail.
- [ ] **Step 3:** Implement; `replay` iterates arrays in C++ with the `historical_timeline` ordering (stable arrival order, timer before same-time arrival).
- [ ] **Step 4:** Pass. Run `benchmarks/session_pipeline.py --engine python` and `--engine cpp --rollout` on MCD 2026-06-04 and `--engine cpp` on NVDA 2026-06-05: identical fingerprints; save outputs under `benchmarks/results/<date>/`.
- [ ] **Step 5:** Commit `feat(cpp): replay sessions natively for backtests`.

### Task A5: Native stream frame router and latency benchmark

**Files:**
- Create: `cpp/include/hftcore/frame_router.hpp`, `cpp/src/frame_router.cpp`, `cpp/bench/bench_engine.cpp`, `cpp/tests/test_frame_router.cpp`, `tests/test_frame_parity.py`
- Modify: `cpp/CMakeLists.txt` (simdjson, `hftcore_bench` target), `cpp/src/bindings.cpp`

**Interfaces:**
- Produces: `hftcore.FrameRouter(engines: dict[str, CppMarketEngine-native])` with `on_frame(raw: bytes, arrival_ns: int) -> tuple[list[update], list[quote_tuple], int]` (updates per symbol, parsed quotes `(symbol, event_ns, bid, ask, bid_size, ask_size, identity)` for Python's Decimal quote path, malformed-message count). Control messages (`success`, `subscription`) and unsubscribed symbols are ignored; `error` messages raise `RuntimeError("market data stream error: <code>")`; malformed entries increment the count and are skipped.
- `hftcore_bench` prints p50/p99 tick-to-decision nanoseconds and events/second for a synthetic multi-symbol frame stream.

- [ ] **Step 1:** Tests: parity of updates between `FrameRouter.on_frame` and the Python path (`json.loads` + per-event `PyMarketEngine.on_event`) on synthetic frames; control/unsubscribed/malformed handling; error raises.
- [ ] **Step 2:** Run — fail.
- [ ] **Step 3:** Implement with simdjson ondemand; timestamps parsed exactly as `hft.feed.parse_timestamp` (RFC3339 with up to 9 fraction digits and offsets).
- [ ] **Step 4:** Pass; run `hftcore_bench` and record the numbers.
- [ ] **Step 5:** Commit `feat(cpp): parse stream frames natively and benchmark tick-to-decision latency`.

### Task A6: Sanitizer CI and performance write-up

**Files:**
- Modify: `.github/workflows/ci.yml` (`cpp-sanitize` job: `HFTCORE_SANITIZE=ON`, ctest, parity tests with `LD_PRELOAD` of libasan as required), `docs/performance-results.md` (C++ section with A4/A5 numbers, method, environment, limitations), `cpp/README.md` (build, test, bench)

- [ ] **Step 1:** Add the sanitizer job; run it locally (`cmake -S cpp -B /tmp/hftcore-asan -DHFTCORE_SANITIZE=ON && ctest`) — must pass with zero sanitizer reports; delete `/tmp/hftcore-asan`.
- [ ] **Step 2:** Write the performance section from the recorded benchmark outputs only.
- [ ] **Step 3:** Commit `ci: run hftcore under address and undefined-behavior sanitizers` and `docs: publish C++ engine benchmarks`.

---

## Track B: Python runner (worktree `../hft-runner`, branch `feat/trade-runner`)

### Task B1: Clear rejected orders

**Files:** Modify `hft/broker.py`; Test `tests/test_broker.py`

**Interfaces:**
- Produces: `AlpacaBroker` treats a submission error whose follow-up `GET /v2/orders:by_client_order_id` returns 404 as rejected: clears `pending`, saves state, journals via the existing callback an `order_rejected` row with `client_order_id` and HTTP status, and returns `False` from `submit`. Timeouts and 5xx keep the existing lookup-and-wait path.

- [ ] **Step 1:** Tests: `test_rejected_submission_clears_pending` (fake client raises 422 on POST, 404 on lookup → `pending is None`, state saved, next `submit` allowed); `test_timeout_still_looks_up_same_id` (existing behavior preserved).
- [ ] **Step 2:** Run — first fails.
- [ ] **Step 3:** Implement in `AlpacaBroker.submit`/`poll` (`hft/broker.py:294-349, 404-410`).
- [ ] **Step 4:** Pass; full suite green.
- [ ] **Step 5:** Commit `fix(broker): clear orders the broker rejected instead of blocking forever`.

### Task B2: Budget sizing and account guard

**Files:** Create `hft/trading/__init__.py`, `hft/trading/budget.py`; Modify `hft/risk.py` (optional `entry_cap`), `hft/paper.py` (`intent_factory` parameter); Test `tests/test_trading_budget.py`

**Interfaces:**
- Produces: `RiskGateway(config=None, *, entry_cap: Decimal | None = None)`; when set, the entry exposure check compares `q * p` to `entry_cap - position value` instead of `day_start * entry_allocation` (default path unchanged).
- Produces: `budget_risk_config(daily_loss: float = 0.02, max_drawdown: float = 0.05) -> RiskConfig` (other fields default, `maintenance_exposure=1.0`).
- Produces: `make_budget_intent(action: int, snapshot: Snapshot, budget: Decimal, sizing: SizingConfig) -> OrderIntent | None` reusing `hft.sizing` pricing/precision helpers; buy quantity = floor((min(budget - position value, available cash)) / limit price) at `quantity_precision`; `None` below `minimum_notional`.
- Produces: `class AccountGuard(total_budget: Decimal, daily_loss: float, max_drawdown: float)` with `start_session(equity: Decimal)`, `observe(equity: Decimal) -> str | None` returning `"account_daily_loss"` or latched `"account_drawdown"`.
- Produces: `PaperEngine(..., intent_factory=make_intent)`.

- [ ] **Step 1:** Tests: entry capped at remaining budget; cash-limited; minimum notional; `entry_cap` default path identical (existing risk tests untouched); guard daily-loss resets next session; drawdown latches.
- [ ] **Step 2:** Run — fail. **Step 3:** Implement. **Step 4:** Pass; full suite green.
- [ ] **Step 5:** Commit `feat(trading): size entries from per-stock budgets with an account guard`.

### Task B3: Multi-symbol stream with reconnect

**Files:** Create `hft/trading/stream.py`; Test `tests/test_trading_stream.py`

**Interfaces:**
- Produces: `async def stream_events(symbols: Sequence[str], *, feed: str = "iex", connect=websockets connect, clock=time.time_ns, sleep=asyncio.sleep, backoff=(1, 2, 4, 8, 16, 32, 60)) -> AsyncIterator[tuple[str, dict | None, int]]` yielding `("event", message, arrival_ns)`, `("connected", None, ns)` and `("disconnected", None, ns)`. One subscription for quotes and trades of all symbols (raise `ValueError` above 30). Auth failure raises `RuntimeError` (no retry); other socket errors reconnect with the backoff sequence (last value repeats); control messages and unsubscribed symbols are skipped; malformed JSON frames are counted (`stream_events.malformed` attribute on the generator's state object) and skipped. Credentials from `ALPACA_API_KEY`/`ALPACA_SECRET_KEY`, never logged.

- [ ] **Step 1:** Tests with a fake connect: subscription payload lists all symbols; reconnect after a dropped socket with delays 1, 2; auth error raises; unsubscribed symbol skipped; malformed frame skipped and counted; 31 symbols rejected.
- [ ] **Step 2-4:** fail → implement → pass.
- [ ] **Step 5:** Commit `feat(trading): stream several symbols with automatic reconnect`.

### Task B4: Portfolio broker startup checks and reconciliation

**Files:** Create `hft/trading/broker.py`; Test `tests/test_trading_broker.py`

**Interfaces:**
- Produces: `class PortfolioBroker(client: AlpacaClient, symbols: Sequence[str], budgets: Mapping[str, Decimal], run_dir: Path)` using `AccountStateStore(account_id, mode="paper", root=run_dir)` (shares the paper account lock with `broker-paper`).
- `start() -> None` refuses with `RuntimeError` messages: `"paper account is not active"`, `"symbol <S> is not active, tradable and fractionable"`, `"paper account must be flat with no open orders to start a new run"`, `"budgets exceed non-borrowed cash"`; on resume, refuses a changed run identity (`"run identity changed"`).
- `reconcile(ledgers: Mapping[str, Account]) -> None`: per symbol, broker position == ledger position and open orders == that symbol's pending; account cash within $0.01 of `initial_remote_cash + Σ(ledger.cash - ledger.initial_cash)`; any other position or order raises `RuntimeError("unexpected broker position/order: <S>")`.

- [ ] **Step 1:** Tests with a fake client for each refusal, clean reconcile, foreign position fail-closed, cash mismatch fail-closed, resume identity check.
- [ ] **Step 2-4:** fail → implement → pass.
- [ ] **Step 5:** Commit `feat(trading): verify and reconcile one paper account across several stocks`.

### Task B5: Portfolio broker orders, fills and rejections

**Files:** Modify `hft/trading/broker.py`; Test `tests/test_trading_broker.py`

**Interfaces:**
- Produces: `submit(symbol: str, intent: OrderIntent, now_ns: int) -> bool` (per-symbol pending map; durable intent + client order id before POST; limit/day/fractional; emergency sells as market), `poll(now_ns) -> list[tuple[str, Execution]]`, `apply_update(update: dict)`, `cancel(symbol)`, `cancel_all()`, `flatten(symbol, now_ns)`, `rejections(symbol) -> int` reset per session; three rejections for one symbol in a session make `submit` return `False` for that symbol until `start_session()`.
- Reuses B1's rejection semantics (404 on lookup → rejected).

- [ ] **Step 1:** Tests: concurrent pending orders on two symbols; fill routed to the right ledger; rejection cleared; third rejection blocks the symbol; timeout looks up the same id; flatten sends a market sell for the full position.
- [ ] **Step 2-4:** fail → implement → pass.
- [ ] **Step 5:** Commit `feat(trading): route orders and fills per stock on one paper account`.

### Task B6: Trade runner (one session, many stocks)

**Files:** Create `hft/trading/runner.py`; Modify `hft/paper.py` only if a hook is missing; Test `tests/test_trading_runner.py`

**Interfaces:**
- Consumes: `make_market_engine`, `parse_strategy`, `make_budget_intent`, `AccountGuard`, `stream_events`, `PortfolioBroker`.
- Produces: `class TradeRunner(config: TradeConfig, broker, stream_factory, clock=time.time_ns)` and `@dataclass(frozen=True) class TradeConfig(name: str, symbols: tuple[str, ...], budgets: Mapping[str, Decimal], strategy: StrategySpec, daily_loss: float = 0.02, max_drawdown: float = 0.05, log_dir: Path, run_dir: Path)`; `async run_session(session: SessionWindow) -> dict` returns per-symbol summary `{symbol: {"trades", "net", "position", "halted"}}`.
- One `PaperEngine` per symbol (`initial_cash=budget`, `risk=RiskGateway(budget_risk_config(...), entry_cap=budget)`, `intent_factory=make_budget_intent bound to the budget`, `submit_callback` → `broker.submit(symbol, ...)`, `source="trade-paper"`, log `log_dir/session-<date>.jsonl` with per-row symbol), policy: rule strategies use `update.action`; `model` uses `OnnxPolicy` on the full observation.
- Feed silence > 5 s for a symbol: `GapEvent` handling only (no exception). While disconnected: entries blocked on all symbols. At the pre-close window: `broker.flatten` for each held symbol (emergency market reduction if still disconnected). `AccountGuard` halts entries on all engines.
- Shutdown (`cancel()` of the task or SIGINT/SIGTERM): stop entries, `cancel_all`, wait up to 10 s for fills, flatten if the market is open, reconcile, write `finish`.

- [ ] **Step 1:** Tests with fake stream/broker/clock: two symbols trade independently within budgets; silence on one symbol does not stop the other or raise; mid-session start (`now = open + 1 h`) produces no fatal gap and warms up from the next boundary; disconnect blocks entries and pre-close flattens via broker positions; runtime sleep halts entries for the session; account daily loss halts all symbols; shutdown sequence order (`cancel_all` → flatten → reconcile → finish row).
- [ ] **Step 2-4:** fail → implement → pass.
- [ ] **Step 5:** Commit `feat(trading): run a paper session across several stocks`.

### Task B7: Daily supervisor, run identity and the `trade` command

**Files:** Create `hft/trading/supervisor.py`, `hft/trading/identity.py`, `hft/trading/status.py`; Modify `hft/cli.py`; Test `tests/test_trading_cli.py`, `tests/test_trading_supervisor.py`

**Interfaces:**
- Produces: `run_identity(config: TradeConfig, engine_impl: str, engine_version: str) -> str` (sha256 of canonical JSON of symbols, budgets, strategy name/version/bundle sha, engine impl/version, risk/sizing/cost settings, latency 75, bar 5, feed `iex`); `run_paths(name) -> (Path(".state/trade/<name>"), Path("logs/trade/<name>"))`.
- Produces: `async run_daily(runner: TradeRunner, calendar: Callable[[date], SessionWindow | None], clock, sleep) -> None` looping: next session from calendar (skip holidays, honor early close), sleep until open, `run_session`, write session row, repeat until cancelled.
- Produces CLI: `hft trade --paper --symbols S=USD [...] --strategy NAME [--name RUN] [--daily-loss F] [--max-drawdown F]` and `hft trade --status [--name RUN]`. Errors (exit 1, message on stderr): missing `--paper` or `--live` given → `"real money unlocks after research qualification and 30-session paper graduation"`; bad `S=USD`, duplicate symbol, non-positive budget, more than 30 symbols, `model:` with more than one symbol or a symbol different from the bundle's.
- `status.py`: `format_status(run_dir, log_dir) -> str` table per stock (status, position, avg price, last price, unrealized/realized today/total, open order, halt) from state and journals only.

- [ ] **Step 1:** Tests: argument validation and live refusal; identity changes when any identity input changes; supervisor over a fake calendar with a normal day, an early close and a holiday runs exactly two sessions and sleeps correctly; status output for a fixture state/journal.
- [ ] **Step 2-4:** fail → implement → pass.
- [ ] **Step 5:** Commit `feat(trading): add the daily trade command with status`.

### Task B8: Dashboard view of running bots

**Files:** Modify `hft/dashboard.py`, `frontend/src/` (Paper trading view panel), `tests/test_dashboard.py`, `frontend/tests/`

**Interfaces:**
- Produces: `GET /api/trading` → `{"runs": [{"name", "updated_ns", "stocks": [{"symbol", "status", "position", "avg_price", "last_price", "unrealized", "realized_today", "realized_total", "open_order", "halt"}]}]}` built from `.state/trade/*` and `logs/trade/*` only (reuse `status.py` parsing); existing Host/Origin/path validation; empty `runs` when none.
- Frontend panel follows `DESIGN.md`: per-run table, status chips, empty and stale states.

- [ ] **Step 1:** Tests: endpoint shape on a fixture run; rejects bad Host/Origin as other endpoints do; empty state; frontend test renders rows and empty state.
- [ ] **Step 2-4:** fail → implement → pass (Python suite and all frontend checks).
- [ ] **Step 5:** Commit `feat(dashboard): show paper trading bots per stock`.

---

## Integration (on `master` after both tracks merge)

### Task I1: Select the C++ engine and use native replay

**Files:** Modify `hft/market_engine.py` (`make_market_engine` auto selection), `hft/execution.py` (`historical_bars(..., implementation="auto")`), `hft/trading/runner.py` (use `FrameRouter` when the engine is native), `README.md`, `.github/workflows/ci.yml`; Test `tests/test_engine_selection.py`

**Interfaces:**
- `implementation="auto"` → C++ when `hftcore` imports and `hftcore.version() == "0.1.0"`, else Python; env `HFT_ENGINE=python|cpp` overrides (`cpp` without the module raises `ImportError`). Run identity records the implementation and version.

- [ ] **Step 1:** Tests: auto selection with and without the module (monkeypatched import); `HFT_ENGINE` override; full suite passes in both modes (CI runs the Python suite once with `HFT_ENGINE=python` and once with `cpp`).
- [ ] **Step 2-4:** fail → implement → pass; rerun the A4 benchmarks through the default path.
- [ ] **Step 5:** Commit `feat: use the C++ engine automatically when installed`.

### Task I2: Paper acceptance and publication

- [ ] **Step 1:** During market hours (owner-authorized), run `hft trade --paper --symbols <3+ liquid stocks>=<budgets within paper cash> --strategy ema-crossover --name acceptance` for at least one hour including the pre-close window, with the dashboard open; then Ctrl-C after the close.
- [ ] **Step 2:** Verify: journals contain decisions for every stock, orders when signaled, flat before close, clean reconciliation, `hft trade --status` and the dashboard agree.
- [ ] **Step 3:** Record a dated results note in `docs/paper-acceptance-results.md` (what ran, counts, issues; no claims of profitability); update README (C++ engine, `trade` command, benchmarks).
- [ ] **Step 4:** Remove worktrees and temporary files; confirm no background processes remain; full checks green; commit `docs: record paper trading acceptance`.
