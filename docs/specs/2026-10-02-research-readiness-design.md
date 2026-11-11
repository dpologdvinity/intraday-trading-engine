# Comprehensive development diagnostics and memory measurements

Status: implemented and measured October 3, 2026 (UTC). Original design October 2.
See [measured results](../research-readiness-results.md); no model has qualified.
Priority selected by the user: training and evaluation first.

## Purpose and success

The current MCD v3 preflight stops after the first unusable development session.
Its 1/52 coverage is correctly labeled partial, but cannot tell us whether other
development days are usable or how much of session memory is unnecessary metadata.
Add an explicit inspection mode that supplies those measurements without changing
the trading contract, fitting a model or reading reserved final-test prices.

Success is a complete development diagnostic report, or a clear error if integrity
or resource checks prevent completing it, followed by a measured next-project
recommendation. Passing coverage alone must never imply passing data or eligibility.

## Constraints

- Stocks only; zero paid services; local CPU execution; no new dependencies.
- Inspect development sessions only; never read final-test session manifests or partitions.
- Preserve existing checksum, provenance, experiment identity and path validation.
- Preserve default training early stop and cumulative budget/deadline enforcement.
- Preserve execution, features, risk limits and compatibility versions.
- `paper_eligible` remains false in every diagnostic result.
- Target process RSS below 8 GiB; report actual memory separately from retained-buffer totals.
- No broker orders, credential output or changes to unrelated user files.

## Flow and ownership

Extend `hft.research.preflight_experiment(path, *, deadline=float("inf"), inspect_all=False)`.
The default remains the current behavior. `inspect_all=True` walks each allowed
development session one at a time despite ordinary coverage/provenance failures.
It still aborts on corrupt data, changed identity, escaped paths, duplicate/overlapping
partitions, expired deadlines, and existing loader memory-limit failures.

The existing `Simulation` remains the single authority for decision eligibility
and event gaps. Do not implement a second classifier. The `train` path continues
calling the default mode. `data-quality --all-development` opts into full inspection.

Release each session/simulation before loading the next one. Counting retained
buffers must not create extra event arrays or serialize raw metadata. Add a small
metadata-column sizing helper in `hft/data.py` and make the existing total helper
reuse it, so byte accounting has one owner.

## Additive report contract

Keep existing top-level keys and `diagnostics.json` atomic writes. For each session,
preserve date and eligible/ineligible decision and gap counts, then add:

```text
reasons: list[str]                 # this session's reasons only
eligible_fraction: float | null    # eligible / (eligible + ineligible); null if zero
memory:
  numeric_bytes: int
  metadata_bytes: int
  retained_bytes: int              # numeric + metadata, using existing accounting
  metadata_columns_bytes:
    quote_metadata: {column: int}
    trade_metadata: {column: int}
  process_peak_rss_bytes: int      # process lifetime high-water mark, sampled after inspection
```

At top level add `inspection_mode` (`"all-development"` or `"stop-first-failure"`)
and aggregate `eligible_fraction` with the same zero-denominator rule. Existing
aggregate counts are sums over inspected rows; `reasons` is their stable unique union,
plus the existing empty-development reason where applicable.

`coverage_complete` means all declared development sessions were inspected; it
does not mean they passed. Any session reason produces `status="insufficient-data"`
and `live_comparable=false`, even when coverage is complete. With no reasons,
retain `status="data-ready"` and existing comparison semantics. Diagnostics never
establish profit or paper eligibility. Do not claim that old one-day reports
represent full coverage.

RSS is measured with the existing Unix `resource` pattern: on this Linux platform,
`ru_maxrss` is KiB and must be converted to bytes. Label it as a cumulative process
peak; it is not a per-session allocation total. Buffer byte counts exclude Python
object overhead, timeline indices and transient copies. Report that distinction.

## Failures and persistence

An inspection run may overwrite `diagnostics.json` for the same frozen experiment
only after a successful diagnostic traversal, using existing atomic publication.
An integrity/resource/deadline exception must not replace a previous report with
a misleading complete result. Do not alter `experiment.json`, search reports,
budget files or final-test reservations during a standalone data-quality run.
Do not catch arbitrary `ValueError` and relabel corrupt input as ordinary quiet data.
Both inspection modes write the same diagnostic path. A later default preflight
can replace a full report with a correctly labeled prefix; preserve the measured
full-run findings in the result document rather than assuming the JSON is permanent.

## Tests and acceptance

Use tiny checksum-valid fixture datasets and the existing `tests/test_quality.py`
helper. First development day has a real event gap; second is usable; final-test
manifest is deliberately unreadable. Default inspection stops after day one;
full inspection records both days and never opens the final-test manifest.
Per-row reasons must not leak from the bad day into the usable day's row.

Test no decisions, all usable days, empty development, corrupt second-day data,
expired deadline before a later load and during decisions, and identity/partition
validation in the new mode. Test byte totals against Arrow metadata including a
large raw JSON column and a NumPy metadata fixture, without changing event data.
CLI tests prove opt-in dispatch, default dispatch and actionable help without keys.
Existing training-before-fit and completed-report/budget regressions must still pass.

Finally run full inspection on the real 52-session development set, preserve the
30-session final reservation state, and record actual coverage, memory and time
in `docs/research-readiness-results.md`. If it cannot complete, record the exact
failure and partial measurement provenance without pretending the set passed.

## Scope boundary and next choice

No loader projection, new feed-health semantics, new models, extra training,
cadence change, job controller or frontend feature in this project. Those are
conditional follow-ups (now tracked in the [research roadmap](../research-improvement-roadmap.md)).
This project ends with truthful measurements and a next-project recommendation.
