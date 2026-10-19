# Research Improvement Roadmap

**Date:** October 4, 2026

**Reviewed revision:** `283eadc`

**Status:** Proposed work; implementation has not started.

## Purpose

This document records the prioritized improvements identified during the current
repository assessment. The objective is a reproducible, resource-bounded research
workflow that can evaluate real stock data without compromising chronological
isolation, execution fidelity, or risk controls.

The existing framework supports historical acquisition, causal simulation, PPO
training, evaluation, immutable model export, and gated trading operations.
However, no profitable model has qualified. Additional functionality should be
selected according to its ability to resolve a demonstrated research limitation.

## Evidence and Constraints

- All 52 inspected MCD development sessions fail the current continuity contract.
- Both sampled NVDA sessions pass continuity; one of two AAPL sessions passes,
  and neither sampled MSFT session passes. Two dates do not establish broader
  data readiness or predictive edge.
- NVDA on June 5 retains approximately 1.39 GiB of session buffers. Metadata
  accounts for 86.17% of those buffers, including approximately 674 MiB of raw
  JSON. The metadata percentage is not an estimate of removable memory.
- Experiment freezing and ordinary search setup now respect phase isolation.
  The optional model-export observation fallback was removed on October 7, 2026;
  export now requires caller-supplied observations.
- The modern ONNX exporter passes the six policy tests without warnings.

Measurements and limitations are recorded in
[free-data suitability results](free-data-suitability-results.md) and
[development readiness results](research-readiness-results.md).

All proposed work must preserve stocks-only operation, local CPU execution,
zero recurring service costs, archive integrity, final-test reservations, and
existing execution and risk contracts. Broker order submission remains subject
to explicit authorization and the established evidence gates.

## Priorities

| Order | Work item | Expected result | Dependency |
| --- | --- | --- | --- |
| 1 | ~~Close the optional export isolation gap~~ (done 2026-10-07) | Every price-loading path enforces its research phase | None |
| 2 | ~~Project execution metadata~~ (done 2026-10-07) | Reduce retained buffers and per-event conversion work | Consumer and equivalence verification |
| 3 | Bound simulation ownership and preparation | Keep training memory bounded while preserving throughput | Memory measurements and phase ownership |
| 4 | Expand declared NVDA development coverage | Establish suitability across a broader sample | A resource-bounded diagnostic workload |
| 5 | Evaluate a simple causal baseline | Determine whether predictive information survives execution costs | Suitable development data |
| 6 | Resume interrupted final evaluation safely | Recover the identical selected evaluation without reopening selection | Separate persistence and recovery design |

### 1. Close the Optional Export Isolation Gap (Completed October 7, 2026)

**Resolution:** the fallback had no callers, so it was removed. `observations` is
now a required keyword argument, validated before the checkpoint or any data is
read. A regression in `tests/test_policy.py` proves omitted, empty, or malformed
observations are rejected without calling the dataset loader or rollout, and the
explicit-observation ONNX parity test still passes.

Original finding: in [policy.py](https://github.com/dpologdvinity/intraday-trading-engine/blob/master/hft/policy.py), `export_bundle` can generate observations when
none are supplied. That fallback loads the entire dataset before selecting
final-test sessions and does not check reservation state. Ordinary search passes
observations explicitly and therefore does not use this branch.

Remove the fallback if it is unnecessary, or route it through the established
phase loader with validated experiment identity and final-test authorization.
Choose the smallest change that serves an actual caller.

Acceptance requires rejection before any unauthorized market partition is opened,
loading only the permitted sessions, and preserving explicit-observation export
and ONNX action parity.

### 2. Project Execution Metadata (Completed October 7, 2026)

**Resolution:** research loads now use an execution metadata view, and each
session is preflighted once. On NVDA 2026-06-05 retained buffers fell 63%, peak
RSS 33% and load time 10×; across the 52 MCD development sessions retained bytes
fell 62.8%. Bars, gaps, eligibility and rollout results are identical. See
[performance results](performance-results.md).

Original proposal: in [data.py](https://github.com/dpologdvinity/intraday-trading-engine/blob/master/hft/data.py), `load_session` retains every nonnumeric archive
column, and `iter_events` converts the retained fields into Python values for
each event. Raw JSON and original quote-size columns have no execution consumers;
normalized quote sizes already provide executable quantities.

Introduce a clearly owned execution view that loads only verified required
columns. Preserve the original Parquet files and checksum verification. Retain
the identity, condition, exchange, tape, and arrival fields needed by current
consumers and validation.

Acceptance requires equivalent event identities, arrival ordering, bars,
features, feed gaps, fills, rewards, and accounting on fixed fixtures. Duplicate
and corruption rejection must remain intact. Measure retained bytes, process RSS,
and preparation time on the declared real development sample before reporting
the improvement.

### 3. Bound Simulation Ownership and Preparation

In [env.py](https://github.com/dpologdvinity/intraday-trading-engine/blob/master/hft/env.py), prepared simulations accumulate in a per-environment
cache. Closing an environment clears the cache but retains its active simulation
and session references. The default training configuration creates four
environments, each with independently prepared simulations and quote-order arrays.
In [training.py](https://github.com/dpologdvinity/intraday-trading-engine/blob/master/hft/training.py), a previous model can remain referenced while
the next fit is created.

Define explicit ownership and release points, bound cached preparation, and
discard completed model references when they are no longer required. Consider
sharing immutable prepared bars and ordering arrays only if measurement justifies
the change; account, risk, and execution state must remain independent.

The loader also performs redundant preflight work: `load_dataset` calls
`_session_preflight`, followed by the same check inside `load_session`. Consolidate
this ownership while retaining resident accounting and load-time integrity checks.
Repeated preparation across evaluation controls is another benchmark candidate.

Acceptance requires bounded references and caches, preserved seeded behavior,
identical evaluation metrics, and unchanged unresolved-inventory protections.
Measure peak RSS and repeated-episode throughput against the existing workload.
Retain the process memory ceiling; cache eviction must not make the workload
impractically slow.

### 4. Expand Declared NVDA Development Coverage

The current NVDA result covers only June 4 and June 5. Declare additional
development dates before inspecting their outcomes, account for every selected
date, and retain failures in the report. Check the selected identities against
frozen final-test sets and the reservation registry before acquisition.

Use the existing read-only acquisition and diagnostic modules. Report continuity,
feature warmup, quote freshness, memory, and provenance limitations separately.
Passing feature warmup does not establish entry eligibility, executable profit,
or live arrival behavior.

Acceptance requires complete accounting for the declared sample, preserved
held-out isolation, and an evidence-based decision about whether broader
acquisition and training are practical. Do not choose only passing dates or
relax the continuity rule to manufacture readiness.

### 5. Evaluate a Simple Causal Baseline

Once development data is suitable, evaluate one inexpensive predictive baseline
through the same quote-level execution, latency, fees, sizing, and risk engine
used for PPO. Existing cash, intraday-long, EMA, and matched-random controls remain
part of the comparison.

A pinned causal pattern rule or regularized linear model is a candidate, not a
preselected winner. Fit parameters and preprocessing on training folds only;
select thresholds using development validation. The
[stock-analyzer assessment](stock-analyzer-integration.md) identifies a possible
pattern subset and its integration boundaries.

Acceptance requires reproducible chronological folds, causal inputs and labels,
cost-adjusted results, execution stress checks, and truthful reporting of a
no-trade or failed-edge outcome. Increase PPO work only when development evidence
supports the additional cost.

### 6. Resume Interrupted Final Evaluation Safely

Final-test identities are reserved before evaluation. An interruption can leave
those dates consumed without a completed report. Preserve that protection while
designing recovery for the identical selected evaluation.

Persist the selected checkpoint identity, dataset and execution identities,
controls, seeds, per-session progress, and carried account/risk state. Define
recovery during a session, atomic publication, and concurrent-resume exclusion
before implementation. Resume must reject policy substitution or changed data
and must never retrain against the reserved dates.

Acceptance requires interruption and concurrency regressions, identity-tampering
rejection, preserved reservations, and equivalence to uninterrupted evaluation.
Use synthetic fixtures for recovery development; existing final-test prices must
not become a tuning dataset.

## Recommended Execution Sequence

The export isolation correction and metadata projection are complete. Next,
bound simulation ownership: reference release and measured preparation costs,
including the per-event conversion hotspots recorded in the performance results. Expand the declared NVDA development sample after that workload fits the
available resources. Proceed to baseline comparison and bounded PPO experiments
only when the data evidence supports them.

Treat final-evaluation recovery as a separate reliability project. Each work item
should record its scope, acceptance evidence, measured limitations, and stop
condition before the next project begins.
