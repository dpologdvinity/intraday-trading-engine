# Contributing

Thanks for looking. This is a personal research project, so issues and small pull
requests are welcome, but larger changes should start as an issue.

## Setup

Python 3.12 and, for the dashboard, Node.js 22. Tests need no credentials or market data.

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-train.txt
.venv/bin/python -m pip install --no-deps -e .
npm --prefix frontend ci
lefthook install   # optional: gitleaks, ruff and typos before each commit
```

## Checks

Run what CI runs (`.github/workflows/ci.yml`) before opening a pull request:

```bash
.venv/bin/python -m pytest -q
.venv/bin/ruff check hft tests benchmarks scripts
.venv/bin/ruff format --check hft tests benchmarks scripts
npm --prefix frontend test && npm --prefix frontend run build && npm --prefix frontend run format:check
```

Changes under `cpp/` also need the C++ build, `ctest` and the parity tests; see
[cpp/README.md](cpp/README.md) and [AGENTS.md](AGENTS.md#commands). Changes to the ITCH order book (`hft/itch.py`, the
`itch`, `order_book` and `itch_book` units) must keep the C++ and Python books
identical (`tests/test_itch_parity.py`, `scripts/itch_parity.py`); timing claims need
the pinned-core bench described in [cpp/README.md](cpp/README.md). Nasdaq sample days
are used locally only: never commit them or anything reconstructable from them.

## Conventions

- Branch from `master`; one logical change per pull request.
- [Conventional Commits](https://www.conventionalcommits.org/): `feat(trading): ...`,
  `fix: ...`, `docs: ...`, `test: ...`, `chore: ...`.
- Bug fixes come with a regression test. Behavior changes must keep the invariants in
  [AGENTS.md](AGENTS.md#non-negotiable-invariants) and the contracts in
  [docs/trading-spec.md](docs/trading-spec.md).
- New strategies are pre-registered in `docs/specs/` before validation, and every
  experiment is recorded in [docs/project-history.md](docs/project-history.md).
- Never commit credentials, `.env` files, market data or generated artifacts.
- Pull requests describe the problem, the resulting behavior, how it was validated and
  its limitations; include screenshots for UI changes.
