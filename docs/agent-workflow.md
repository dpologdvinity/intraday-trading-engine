# How this project was built with Claude Code

Most of the code in this repository was written by Claude Code (Anthropic's coding
agent) under my direction. This page describes how I ran that collaboration so the
results can be trusted: what the agent did, what I decided, and the guardrails that
kept a fast-moving agent from fooling either of us.

## Division of labour

| I decided | The agent did |
| --- | --- |
| Goals and success criteria (beat holding the same stocks, after costs, on unseen data) | Turned goals into design specs and task plans in [`docs/specs/`](specs/) and [`docs/plans/`](plans/) |
| Which experiments to run, and approving each design | Implementation, tests, data pipelines, Kaggle GPU runs, result write-ups |
| Real-money rules: paper only until research qualification and 30 paper sessions | Paper-trading bots, monitoring, incident fixes with regression tests |
| Risk to accept (overnight holds, second paper account, GPU budget) | Research summaries with sources, options with trade-offs |

## Guardrails

- **Instructions in one file.** [`AGENTS.md`](../AGENTS.md) holds the project rules
  every session reads: invariants (causal data, held-out isolation, C++/Python parity,
  no real money), commands, code ownership. `CLAUDE.md` only imports it.
- **Secrets stay out of reach.** `.claude/settings.json` denies the agent reading
  `.env`; it checks that a key is set without seeing its value. `lefthook.yml` runs
  gitleaks, ruff and typos before every commit.
- **Permission checks on persistence.** Cron jobs, Windows start-up tasks and
  bot-check workarounds needed my explicit approval; the agent stopped and asked.
- **Pre-registration.** Every strategy test was written down and committed (rule, data
  split, costs, pass criteria) before any result was computed: ten specs, from
  intraday momentum to a 16-configuration reinforcement-learning search. Results are
  recorded even when they fail, in [`project-history.md`](project-history.md).
- **One-time final tests.** The 2025-2026 test years are guarded by a registry that
  refuses a second run; no strategy has earned the run yet.

## Verification instead of trust

- **Parity tests.** The C++20 engine must be bit-identical to the Python reference;
  the reinforcement-learning environment must reproduce the existing fill simulator's
  returns exactly. A difference is a bug, never a relabeled result.
- **Independent reviews by subagents.** Reviews before deployment found, among others:
  six critical issues in the overnight runner (a fill above available cash, a
  reconciliation race, transient errors killing the run), a market-on-close fill that
  could strand a position, a bypass of the one-time final-test guard, and a data
  fingerprint that was never checked. All were fixed before the code traded.
- **Self-review against look-ahead.** The agent caught that its own first
  reinforcement-learning observation used the next minute's price, and a causality
  test now fails if any observation reads a later bar.
- **Live incidents became tests.** Paper trading surfaced bugs the tests missed
  (broker cent rounding, overnight regulatory fees, an opening-burst stall, a calendar
  file overwritten by a morning refresh, unfilled limit orders stopping a run); each
  fix shipped with a test that fails on the old code
  ([paper results](paper-acceptance-results.md)).

## Techniques used

- **Subagents for research and exploration**, keeping the main session's context for
  decisions: literature reviews of reinforcement learning for trading and of HFT
  systems in C++, each with sources.
- **Skills** shared at project scope in [`.agents/skills/`](../.agents/skills/)
  (investigate-first, surgical-patch, safe-refactor, lean-build, verify-and-stop) and
  process skills for brainstorming, planning, test-driven development and code review.
- **MCP servers** in [`.mcp.json`](../.mcp.json): Firecrawl for web research, Playwright
  for browser checks.
- **Long-running work outside the session:** Kaggle GPU kernels pushed and monitored
  from the CLI (about 26 GPU hours for the reinforcement-learning search), paper bots
  under cron watchdogs, a GitHub Actions workflow that alerts when a trading day is
  missed.

## What came out of it

A C++20 market engine 86x faster than its Python reference and bit-identical to it;
multi-stock paper trading on two Alpaca paper accounts; and 21 recorded experiments,
including 124 machine-learning and reinforcement-learning configurations, none of
which beat holding the same stocks after costs. The honest negative result is the
point: the workflow is built so that a positive one would be believable.
