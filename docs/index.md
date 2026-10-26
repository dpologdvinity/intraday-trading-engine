# Intraday Trading Engine

A local stock-trading system: a C++20 engine ingests market data and decides, Python
runs paper trading on Alpaca, and every strategy is judged on data it never saw,
against simply holding the same stocks after costs.
[Source code on GitHub](https://github.com/dpologdvinity/intraday-trading-engine).

![Local dashboard](images/dashboard-overview.png)

## Engineering

- **C++20 decision engine** (`hftcore`, pybind11, simdjson, Catch2, sanitizer CI):
  replays a real 5.1M-quote NVDA session in 1.3 s instead of 111.5 s in Python, with
  bit-identical parity tests; tick-to-decision 12.7 µs p50.
  [Performance results](performance-results.md) ·
  [Design](specs/2026-10-07-paper-trading-runner-design.md)
- **Paper trading** on one Alpaca paper account: per-stock budgets, Decimal ledgers,
  loss and drawdown guards, durable orders, broker reconciliation, auction orders.
  The first live sessions found bugs the tests missed, each fixed with a regression
  test. [Paper sessions](paper-acceptance-results.md) · [Operations](operations.md)
- **Causal research pipeline**: quote-level fill simulation, reserved final-test
  data, pre-registration, dated bootstrap intervals and multiple-testing corrections.
  [Architecture](architecture.md) · [Trading specification](trading-spec.md)

## Research results

No strategy has beaten holding the same stocks after costs on unseen data, so real
money stays locked behind research qualification and 30 paper sessions.
[Full project history](project-history.md) records every idea, method and result.

| Idea | Result |
| --- | --- |
| [EMA crossover replays](replay-results.md) | Churns and pays the spread: about -2% a day |
| [Intraday momentum](intraday-momentum-results.md) | Holdout -7.5 bp per trade after costs |
| [Overnight drift and weekly reversal](multiday-results.md) | Overnight +9.1 bp before costs, +0.4 after |
| [Overnight through auction orders](overnight-auction-results.md) | +5.9 bp per night before costs (interval above zero), +1.4 after |
| [ML day trader, 84 variants](ml-daytrader-results.md) | Best +13.6% against +97.2% for holding (2023-2024) |
| [Pre-open analyst upgrades](analyst-upgrade-results.md) | +4.7 bp per trade, interval includes zero |
| [News and FinBERT features](ml-daytrader-results.md) | Best +47.8% against +97.2% for holding |
| [Momentum and analyst drift over weeks](longer-horizons-results.md) | +75 bp/month and +52 bp/trade above the same stocks (holdout), intervals include zero |

The best news model now runs forward on the paper account to build a record on new
days. Paper results are simulated money and do not establish an edge.

## More

- [Demo walkthrough](demo.md)
- [Research readiness](research-readiness-results.md) and
  [free-data suitability](free-data-suitability-results.md)
- [Research roadmap](research-improvement-roadmap.md)
