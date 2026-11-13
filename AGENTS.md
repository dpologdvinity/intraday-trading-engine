# Agent instructions

Canonical instructions for coding agents working in this repository. `CLAUDE.md`
only imports this file, so every agent reads the same rules.

## Project

Intraday Trading Engine (Python package `hft`, C++ engine `cpp/` module `hftcore`):
a local stock-trading system that learns from historical simulations, proves results
through honest out-of-sample evaluation, and runs on paper before any real money. It
is also the owner's portfolio project, so docs must stay accurate and measured.

Constraints: stocks only, long-only fractional, $0 recurring cost, local CPU,
eventual $500-$5,000 capital, 5 s decisions (changing the cadence needs evidence and
a new contract), process RSS below 8 GiB. No strategy has beaten holding after costs
yet; see [project history](docs/project-history.md).

## Read before changing behavior

- [Trading specification](docs/trading-spec.md): execution, risk and graduation contracts.
- [Runner and C++ engine design](docs/specs/2026-10-07-paper-trading-runner-design.md),
  [architecture](docs/architecture.md), [operations guide](docs/operations.md).
- Experiment specs and pre-registrations in [docs/specs/](docs/specs/); results in
  `docs/*-results.md`.
- [DESIGN.md](DESIGN.md) before modifying the frontend.

Check code and artifacts before making claims; plans describe proposals, not
implemented features.

## Setup

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-train.txt   # pinned CPU-only stack
.venv/bin/python -m pip install --no-deps -e .
npm --prefix frontend ci
lefthook install    # pre-commit: gitleaks, ruff, typos (lefthook.yml)
```

Tests need no credentials or market data. Environment variables (export them or keep
them in the git-ignored `.env`; never print their values):

| Variable | Used for |
| --- | --- |
| `ALPACA_API_KEY`, `ALPACA_SECRET_KEY` | Free read-only market data (`hft/history.py`, `hft/feed.py`, `hft/trading/stream.py`). |
| `ALPACA_PAPER_API_KEY`, `ALPACA_PAPER_SECRET_KEY` | Paper account orders (`hft/broker.py`); also accepted as market-data keys. |
| `ALPACA_LIVE_API_KEY`, `ALPACA_LIVE_SECRET_KEY` | Live account; only after activation (see operations guide). |
| `HFT_ENGINE` | `auto` (default), `python` or `cpp` market engine. |
| `HFT_REQUIRE_HFTCORE` | `1` makes tests fail instead of skipping when `hftcore` is missing. |
| `HFT_RECORDS` | Private backup clone for `scripts/record_paper_day.sh`. |

## Commands

```bash
.venv/bin/python -m pytest -q tests/<file>.py   # focused locally; CI runs everything
.venv/bin/ruff check hft tests benchmarks scripts
.venv/bin/ruff format --check hft tests benchmarks scripts
.venv/bin/python -m pip check
npm --prefix frontend test && npm --prefix frontend run build
npm --prefix frontend run format:check
# C++ engine (CI installs the build tools with `pip install cmake ninja`)
export PATH="$PWD/.venv/bin:$PATH"
cmake -S cpp -B cpp/build -G Ninja && cmake --build cpp/build
ctest --test-dir cpp/build
.venv/bin/python -m pip install -q ./cpp
HFT_REQUIRE_HFTCORE=1 .venv/bin/python -m pytest -q tests/test_engine_parity.py
# ITCH order book (tests use synthetic streams; real sample days stay local in data/itch/)
HFT_REQUIRE_HFTCORE=1 .venv/bin/python -m pytest -q tests/test_itch.py tests/test_itch_parity.py tests/test_itch_parity_script.py
gzip -dc data/itch/<day>.NASDAQ_ITCH50.gz | cpp/build/hftcore_itch_replay -
.venv/bin/python scripts/itch_parity.py <decompressed day> --messages 5000000
taskset -c N cpp/build/hftcore_bench_itch <decompressed day> --repeat 3   # one core, never 4-5 (paper bots)
```

Dashboard: build the frontend, then `.venv/bin/python -m hft dashboard --port 8765`
(check for an existing server first). CI (`.github/workflows/ci.yml`) runs Python
lint and tests, the offline smoke run, the C++ build with the full suite on the C++
engine, and C++ tests under ASan/UBSan.

## Code ownership

| Files | Responsibility |
| --- | --- |
| `hft/data.py`, `hft/history.py` | Validated historical data, checksums, provenance, memory limits, read-only acquisition. |
| `hft/feed.py`, `hft/market_engine.py`, `hft/strategies.py` | Causal bars, market engine interface (Python reference), rule strategies. |
| `cpp/` | `hftcore` C++20 engine; must match the Python reference exactly. |
| `hft/itch.py`, `cpp/` ITCH units (`itch`, `order_book`, `itch_book`), `cpp/bench/{bench_itch,itch_replay}.cpp`, `scripts/itch_parity.py` | Nasdaq ITCH 5.0 parser and full-depth order book; Python reference must match the C++ book exactly. Separate from the trading engine; samples stay local and are never committed. |
| `hft/execution.py` | Quote-event execution simulator and shared timing. |
| `hft/features.py`, `hft/env.py`, `hft/training.py` | Observations (market 0-9 / state 10-16), environment, CPU PPO, budgeted search. |
| `hft/research.py`, `hft/evidence.py` | Frozen experiments, controls, evaluation, qualification evidence. |
| `hft/ml/` | ML day trader: universe, data download, features, training, search, Kaggle packaging. |
| `hft/account.py`, `hft/risk.py`, `hft/sizing.py`, `hft/configuration.py` | Decimal accounting, risk limits, runtime constraints. |
| `hft/runtime.py`, `hft/paper.py`, `hft/broker.py`, `hft/trading/` | Streaming, paper engine, broker integration, multi-stock runner. |
| `hft/dashboard.py`, `frontend/src/` | Loopback read-only API and React/Vite dashboard. |

## Non-negotiable invariants

- Diagnostics and development selection never read reserved final-test prices; data
  tools call `hft.training.reserved_final_sessions()` first. Preserve `.state/` and
  reservations; never reset them or tune against held-out prices.
- Preserve causal boundary publication, arrival ordering, gap/warmup behavior and
  training/runtime parity. The C++ engine must be bit-identical to the Python
  reference; a difference is a bug, never a relabeled contract.
- Preserve fresh-quote checks, future-quote rejection, bid/ask and depth matching,
  fees, latency and partial fills. Carried bar prices are not executable quotes.
- Liquidation uses real subsequent quotes; unresolved inventory cannot disappear.
- Keep loss, drawdown, exposure, rate, reconciliation and activation guards. Budget
  paper runs use a separate budget path; fraction-based limits stay unchanged.
- Feature schema v3: 17 float32 features, actions 0 flat / 1 long. Contract changes
  need versioning and fresh experiments; never relabel artifacts.
- Separate synthetic, replay, dry-run, broker-paper and live evidence. Rule-strategy
  paper runs never count as graduation evidence.
- Real money stays behind research qualification and 30-session paper graduation.
  Place paper orders only when authorized. Preserve live activation and the $10
  canary/10-session review.
- Dashboard: keep Host/Origin/path validation, reserved-data isolation, read-only
  endpoints, accessibility and truthful empty/error/stale states.
- Never print credentials or environment values, or commit secrets.

## Working practices

- `git status` first; preserve unrelated changes and stage specific files.
- Conventional Commits (`feat(trading): ...`, `fix: ...`, `docs: ...`), small and
  scoped. Naming the tools used is fine; never add `Co-Authored-By` trailers.
- Trace behavior before editing; extend the responsible layer; prefer the standard
  library and installed dependencies. For bugs, write a failing regression test first.
- Run focused test files locally and let CI run the full suite. Kill processes and
  delete temporary files you created.
- Paper bots may be running (`pgrep -af "[h]ft trade"`); leave them alone unless the
  task is about them. Check them with `scripts/paper_report.py --run <name>`.
- Pre-register new strategy tests in `docs/specs/` (rule, data split, costs, pass
  criteria) and commit them before running validation.
- Record every experiment, idea and goal change in `docs/project-history.md` as it
  happens (date, idea, method, result, decision). Keep README.md current.
- UI changes need browser verification.

## Style and tests

- Python 3.12+, four-space indent, `snake_case` functions/modules, `PascalCase`
  classes, ruff 100-column limit. Frontend: Prettier, two-space indent, `PascalCase`
  React components.
- pytest files `tests/test_*.py`; frontend tests use Node's built-in test runner. Add
  focused regressions for changed behavior (causality, quote matching, accounting,
  risk limits, held-out isolation).
- PRs explain the problem, resulting behavior, validation and limitations; include
  screenshots for visible UI changes.

## Agent workflow

- **One instruction file**: this file. `CLAUDE.md` is the one-line import `@AGENTS.md`;
  personal, machine-specific notes go in git-ignored `*.local.*` files.
- **Skills** live in `.agents/skills/` (read by Codex); `.claude/skills/` symlinks
  to them for Claude Code. See [.agents/skills/README.md](.agents/skills/README.md).
- **Design flow**: brainstorm a feature into an approved spec in `docs/specs/`, turn
  it into a task-by-task plan in `docs/plans/`, then execute it with one subagent per
  task and a review after each. Per-run progress ledgers go to `.superpowers/`
  (git-ignored).
- **Guardrails**: `.claude/settings.json` denies reading `.env`; `lefthook.yml` runs
  gitleaks, ruff and typos before each commit.
