# Research Readiness Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans to implement this plan
> task by task if that skill is available. Steps use checkbox syntax. Sequential
> execution is sufficient; this plan does not require additional agents.

**Status:** Implemented and measured October 3, 2026 (UTC). See
[results](../research-readiness-results.md); complete coverage remains insufficient data.

**Goal:** Measure every development session's decision coverage and memory without
changing trading behavior, fitting a model or inspecting final-test prices.

**Architecture:** Extend the existing preflight and CLI with an explicit full
inspection mode. Keep the existing simulator and validation as authorities;
consolidate metadata byte accounting in the data layer. Publish additive diagnostics
through the existing atomic writer, then record real measurements in a short report.

**Tech Stack:** Existing Python 3.12+, NumPy, PyArrow, pytest, stdlib resource/argparse;
no additional dependencies or frontend work.

**Spec:** [Research-readiness design](../specs/2026-10-02-research-readiness-design.md).
Read the spec before implementing.

## Global Constraints

- Stocks only; zero paid services; local CPU execution; no new dependencies.
- Inspect development sessions only; never read final-test session manifests or partitions.
- Preserve existing checksum, provenance, experiment identity and path validation.
- Preserve default training early stop and cumulative budget/deadline enforcement.
- Preserve execution, features, risk limits and compatibility versions.
- `paper_eligible` remains false in every diagnostic result.
- Target process RSS below 8 GiB; report actual memory separately from retained-buffer totals.
- No broker orders, credential output or changes to unrelated user files.

## Review Focus

1. Ordinary failure followed by a good day: full mode continues, default stops,
   and the good row does not inherit the earlier row's reasons (Task 1).
2. Corruption on a later day: full mode raises and preserves a previous published
   report instead of claiming complete coverage (Task 1).
3. Deadline expires between sessions or during decisions: no subsequent load or
   complete report publication; final data remains unopened (Task 1).
4. Empty/zero-decision history and mixed metadata storage: ratios avoid division
   by zero; exact existing byte-accounting semantics remain intact (Tasks 1–2).
5. CLI opt-in is omitted or supplied: only the supplied flag changes inspection;
   training still blocks before fit on bad data (Task 3).

## File ownership

| File | Change |
| --- | --- |
| `hft/research.py` | Inspection mode, per-row reasons/ratios, memory fields and additive diagnostics. |
| `hft/data.py` | Metadata-column byte helper; existing totals reuse it. |
| `hft/cli.py` | Explicit `--all-development` opt-in. |
| `tests/test_quality.py` | Coverage, isolation, integrity, deadline and reporting regressions. |
| `tests/test_data.py` | Metadata sizing consistency and unchanged event data. |
| `tests/test_cli.py` | Help and dispatch/default tests. |
| `README.md` | Usage, limits and distinction between complete coverage and passing data. |
| `docs/research-readiness-results.md` | New measured real-data report at the end. |
| `HANDOFF.md`, `docs/progress.md` | Update current evidence after actual verification. |

Do not create a second diagnostics service, edit frontend components or restructure
unrelated modules. Start from the existing `frozen(root, *, gap=False)` test helper,
which produces two development sessions and one final-test session.

### Task 1: Complete inspection without weakening default preflight

**Files:** `hft/research.py`, `tests/test_quality.py`.

**Interfaces:** Extend
`preflight_experiment(path, *, deadline=float("inf"), inspect_all=False) -> dict`.
Existing callers remain valid. Add row `reasons`/`eligible_fraction`, top-level
`inspection_mode`/`eligible_fraction`; other fields retain their meanings.

- [x] Add `test_all_development_continues_after_gap_without_reading_final_test`.
  Build `frozen(tmp_path, gap=True)`; corrupt only the reserved session manifest.
  Spy on `hft.data.load_session`. Default call must yield coverage 1/2 and
  `coverage_complete is False`; full call must yield coverage 2/2 and `True`.
  Full report must still have `status == "insufficient-data"`,
  `live_comparable is False`, `paper_eligible is False`. First row includes
  `runtime-feed-gap`, second row has `reasons == []`; no reserved manifest load.
  Check aggregate counts equal row sums and final-test-consumed remains false.
- [x] Add `test_full_inspection_preserves_published_report_on_later_corruption`.
  Publish a valid default report, save its bytes, corrupt the second development
  quote partition, then assert full mode raises a checksum error and the saved
  `diagnostics.json` bytes remain unchanged. Never catch corruption as coverage failure.
- [x] Add `test_full_inspection_still_validates_partition_identity` for checksum-valid
  experiment fixtures with duplicate development IDs or development/final overlap.
  Recompute fixture hashes correctly before calling; assert rejection before tick
  loads. Also retain existing changed-identity rejection tests in full mode.
- [x] Add `test_full_inspection_deadline_blocks_later_load_and_publication` using
  a controlled monotonic clock that expires after the first day. Assert only one
  development session load, `TimeoutError`, no report replacement. Add a clock
  expiry during decision iteration to prove the existing per-decision check remains.
- [x] Add `test_quality_ratios_and_empty_development`. All-good fixtures remain
  `data-ready` and never paper eligible. Empty development yields existing
  `insufficient-history` and aggregate `eligible_fraction is None`. A checksum-valid
  short-session fixture records `insufficient-decision-warmup`, zero decisions and
  row fraction `None`. For positive totals assert the fraction equals eligible/total.
- [x] Run `.venv/bin/python -m pytest tests/test_quality.py -q`. New tests should fail
  on the missing parameter/report fields; existing tests should still pass.
- [x] Implement using a fresh reasons list for each row. Append only that row's
  reasons to the aggregate. Continue ordinary coverage/provenance failures only
  when `inspect_all` is true. Preserve the current narrow insufficient-warmup
  exception handling and all integrity/deadline checks. Keep one session loaded,
  the existing `Simulation` decision loop and atomic report publication.
- [x] Run `.venv/bin/python -m pytest tests/test_quality.py -q`, then Ruff on changed
  Python files. Expect all pass, including the existing before-fit/held-out and
  completed-report/exhausted-budget tests. Commit only these task files with
  `feat: inspect complete development coverage without fitting`.

### Task 2: Account for memory at its existing owner

**Files:** `hft/data.py`, `hft/research.py`, `tests/test_data.py`, `tests/test_quality.py`.

**Interfaces:** Add
`metadata_column_bytes(manifest) -> dict[str, dict[str, int]]` with fixed outer
keys `quote_metadata` and `trade_metadata`. Refactor `metadata_bytes(manifest)`
to sum that helper. Preserve `session_bytes(session)` semantics. Add each row's
`memory` object exactly as defined in the spec; no memory projection in this task.

- [x] Add `test_metadata_column_bytes_preserves_total_accounting`. Include Arrow
  arrays/chunked arrays, NumPy arrays, the existing small Python-list fixture
  representation, empty metadata and a large raw JSON column. Assert per-column
  counts use `.nbytes` or the existing `pa.array(values).nbytes` fallback; nested
  sum equals `metadata_bytes`, and `session_bytes` equals numeric plus metadata.
- [x] Add `test_quality_reports_memory_without_changing_events`. Use checksum-valid
  fixture metadata and capture the loaded session's emitted events before/after
  measurement. Assert events unchanged, integer byte fields nonnegative, nested
  metadata sum equals reported metadata bytes, and numeric + metadata == retained.
  Patch the resource sample to a known KiB value to assert correct byte conversion;
  restore it for real inspection. Assert a usable day and a gap day both report memory.
- [x] Run focused new tests and confirm they fail because the helper/fields are absent.
- [x] Implement the helper in `hft/data.py` using existing sizing semantics. In
  preflight use `session_bytes`, `metadata_bytes` and the helper; derive numeric
  bytes from the existing total minus metadata bytes. Sample
  `resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024` after inspection on
  Linux. Do not create a second data loader, deep-copy events or serialize raw JSON.
- [x] Run `.venv/bin/python -m pytest tests/test_data.py tests/test_quality.py -q`.
  Run Ruff on changed files. Expect exact byte invariants and all prior guards
  to pass. Commit `feat: report retained session metadata and peak memory`.

### Task 3: Explicit CLI opt-in and operator documentation

**Files:** `hft/cli.py`, `tests/test_cli.py`, `README.md`.

**Interfaces:** `data-quality --experiment PATH [--all-development]`.
Argparse stores `args.all_development` as bool, default false. Dispatch to
`preflight_experiment(args.experiment, inspect_all=args.all_development)`.
No change to `train`, no deadline override flag, no extra model evaluation.

- [x] Add `test_data_quality_help_explains_all_development`, asserting command
  help exits successfully and lists the flag as inspecting every development
  session while preserving failures. Add in-process dispatch tests named
  `test_data_quality_defaults_to_early_stop` and
  `test_data_quality_explicitly_requests_all_development`; monkeypatch the preflight
  to capture the exact boolean passed through the existing CLI parser/dispatch.
  Strip credentials as the existing subprocess tests do; no API calls are needed.
- [x] Run `.venv/bin/python -m pytest tests/test_cli.py -q` and confirm new cases fail.
- [x] Implement the flag on the existing `data-quality` parser. Help text:
  `inspect every development session; report failures without fitting or reading final-test data`.
  Dispatch through the existing handler. Document both modes, output path,
  memory accounting limitations and that complete coverage can still fail.
- [x] Run `.venv/bin/python -m pytest tests/test_cli.py tests/test_quality.py -q`.
  Confirm `test_gap_stops_preflight_and_training_before_fit_or_held_out_reads`
  still passes; inspect `hft/training.py` call sites to confirm no opt-in was added.
  Commit `feat: expose complete development inspection in the cli`.

### Task 4: Measure the real development set and choose the next project

**Files:** Create `docs/research-readiness-results.md`; update `HANDOFF.md` and
`docs/progress.md`. Generated `artifacts/mcd-v3/diagnostics.json` remains ignored.

**Interfaces:** Task 3's CLI and the additive diagnostics schema. The result
document contains measurements and a recommendation, not a fitted model.

- [x] Check `git status --short`; preserve unrelated changes. Verify local
  `data/mcd/manifest.json` and `artifacts/mcd-v3/experiment.json` exist and match
  the frozen identity. If absent, record that limitation rather than fabricate
  measurements or silently download a new dataset with different final dates.
- [x] Run the full Python suite once for these cross-layer changes:
  `.venv/bin/python -m pytest -q`,
  `.venv/bin/ruff check hft tests`,
  `.venv/bin/ruff format --check hft tests`, and
  `.venv/bin/python -m pip check`. Expect all pass; identify existing exporter
  warnings separately from new failures. No frontend checks needed if unchanged.
- [x] Save the prior diagnostic JSON to a separate ignored local file for comparison.
  Record final-test-consumed and existing reservation-state file identity without
  opening reserved prices or printing secrets. Run:
  `.venv/bin/python -m hft data-quality --experiment artifacts/mcd-v3/experiment.json --all-development`.
  Measure monotonic elapsed time externally. Do not run `train` as part of this task.
- [x] Inspect the generated report: expected coverage is 52/52 if traversal succeeds;
  all rows must belong to development, aggregate counts must match, and any row
  failure must keep data insufficient and paper eligibility false. Compare
  reservation state afterwards. If a load/resource/integrity failure stops the
  run, report it precisely and preserve the last successfully published report.
- [x] Write the result document with experiment hash, contract identity, inspected
  dates/count, per-reason session counts, eligible totals/fraction, largest retained
  sessions, leading metadata columns, process peak RSS and elapsed time. Explain
  cumulative RSS versus retained bytes; no claim about an unmeasured full search.
  Include a clear next choice using the roadmap's data/memory/feed-health decision
  criteria, or a concrete blocking condition if the inspection could not finish.
- [x] Update handoff/progress from actual evidence, mark this plan's completed steps,
  check `git diff --check` and commit specific documentation files with
  `docs: record development readiness measurements and next research step`.

## Stop condition

Stop this project after verified diagnostics and a measured recommendation.
Do not automatically implement loader projection, relax feed gaps, change cadence,
start a long search, reserve final tests or submit orders. Develop the selected
follow-up as its own small, testable project using the evidence just collected.
