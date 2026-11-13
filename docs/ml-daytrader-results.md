# ML day trader results

Validation of the [ML day trader](specs/2026-10-08-ml-daytrader-design.md): 84 variants
trained on 2016-2022 and scored on 2023-2024 by simulated trading with costs, against
holding the same 68 symbols (equal slots, late listings bought on their first
tradable day). Run on a Kaggle T4 GPU in about 20 minutes (code b8f72b6, data
fingerprint 54cce01d9385). The test years (2025-2026) were not touched.

## Result: no variant beats holding

Holding the 68 symbols returned **+97.2%** over the two validation years. The best
variants:

| Daily model | Target | Picks | Minute model | Total | Per trade | 3x costs | Edge vs holding (bp/day, 95% interval) |
| --- | --- | ---: | --- | ---: | ---: | ---: | --- |
| Trees | range | 3 | hold from open | +8.6% | +4.1 bp | -48.8% | -10.0 (-26.0 to +6.0) |
| Network | return, gated | 5 | hold from open | +13.6% | +4.1 bp | -33.5% | -10.9 (-20.3 to -1.4) |
| Network | return | 5 | hold from open | +12.6% | +3.1 bp | -42.5% | -11.0 (-20.6 to -1.5) |
| Network | range | 5 | hold from open | +5.7% | +2.7 bp | -49.3% | -11.4 (-23.9 to +1.1) |
| Network | range | 3 | hold from open | +1.2% | +2.6 bp | -52.3% | -11.6 (-27.7 to +4.5) |

The best variant's deflated Sharpe ratio (the probability that its true Sharpe is
above zero after 84 tries) is 0.03.

What the 84 variants show:

- **Daily picks held through the day earn about the trading cost.** The top
  variants made +3 to +4 bp per trade at the assumed costs, roughly the stocks'
  average intraday drift in a rising market. At three times the costs every one of
  them loses 33-53%. None is close to holding, which also earns the overnight
  moves the day trader gives up.
- **The minute models mostly learned to do nothing.** The GRU and transformer
  networks at a 15-minute horizon, and the transformers at 30 minutes, never
  predicted a gain larger than the round-trip cost, so 36 variants made no trades
  at all. The GRU at 30 minutes traded rarely (1 to 161 trades in two years).
- **When the minute models traded often, they lost.** The convolutional networks
  traded up to about 1,300 times; most lost 1-8 bp per trade, and the few with a
  small gain traded rarely. On days picked for their expected range they lost the
  most: -12% to -22%.

## Decision

No variant is frozen for the final test. All five candidates trail holding on
average (two with intervals entirely below zero), so spending the one-time test on
them would prove nothing. The test years stay unseen for a better idea.

This repeats the lesson of every earlier experiment ([project history](project-history.md)):
on these liquid stocks, the move a model can predict from prices alone at minute and
daily horizons is smaller than the cost of trading it.

## News features: 16 more variants (2026-10-09)

The daily models also received nine pre-open news features from 335,339 headlines
([design](specs/2026-10-09-news-features-design.md)); same years, costs and benchmark,
counted as trials 85 to 100 (Kaggle T4, code 3ff3dc5).

| Daily model | Picks | Minute model | Total | Per trade | 3x costs | Sharpe | Edge vs holding (bp/day, 95% interval) |
| --- | ---: | --- | ---: | ---: | ---: | ---: | --- |
| Network, gated, with news | 3 | hold from open | +47.8% | +9.9 bp | -19.1% | 0.97 | -5.3 (-17.0 to +6.2) |
| Network, with news | 3 | hold from open | +45.8% | +8.6 bp | -25.1% | | -5.5 (-17.0 to +6.0) |
| Network, with news | 5 | hold from open | +19.3% | +4.2 bp | -37.9% | | -9.9 (-19.0 to -0.9) |

Holding the same symbols: +97.2%, Sharpe 2.04. The best news variant's deflated
Sharpe ratio after 100 trials is 0.12.

News roughly tripled the best daily-pick result (+47.8% against +13.6% without
news, +9.9 against +4.1 bp per trade), the first sign in this project that a
model learned something from data beyond prices. It still trails holding by about
5 bp a day with an interval that includes zero, loses at three times the costs, and
after 100 trials its Sharpe ratio is not distinguishable from luck. The 30-minute
GRU again traded rarely and lost. No variant is frozen; the test years stay unseen.

## FinBERT sentiment: 8 more variants (2026-10-09)

All 335,339 headlines were scored with ProsusAI/finbert (revision 4556d13, trained in
2019 on older text) on a Kaggle GPU. Its pre-open sentiment (mean, maximum and minimum
over a symbol's focused articles, and the market mean) was added to the news features.
In the training years, strongly positive pre-open sentiment preceded +4.6 bp from open
to close and strongly negative -7.2 bp. Trials 101 to 108, held from the open:

| Daily model | Picks | Total | Per trade | 3x costs | Sharpe | Edge vs holding (bp/day, 95% interval) |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Trees, news + sentiment | 3 | +42.6% | +7.9 bp | -18.8% | 1.00 | -6.3 (-16.7 to +3.9) |
| Network, news + sentiment | 5 | +39.9% | +7.5 bp | -28.8% | 0.93 | -6.6 (-16.2 to +2.9) |
| Trees, gated, news + sentiment | 5 | +12.7% | +55.7 bp | +9.8% | 0.88 | -11.7 (-21.6 to -1.9) |

Sentiment did not improve on the plain news features (+47.8%). Across the 24 news
variants the pattern is consistent: daily picks informed by pre-open news earn about
+6 to +10 bp per trade, roughly 20% a year with a Sharpe ratio near 1, yet trail
holding the same symbols (+97.2%, Sharpe 2.04 in a strong market), lose at three times
the costs, and after 108 trials are not distinguishable from luck. The gated trees
traded only 112 times at +55.7 bp, the only variant still positive at 3x costs; with
so few trades, picked out of 108 tries, it is a lead to test on fresh data, not a
result. No variant is frozen; the test years stay unseen.

## Reinforcement learning: 16 more configurations (2026-11-12)

The [pre-registered search](specs/2026-11-11-rl-daytrader-design.md): PPO on
GPU-vectorized environments, three seeds per configuration ensembled, early stopping on
2022, scored on 2023-2024 (trials 109 to 124; code 0efba1a, Kaggle T4, about 26 GPU
hours). Track A trades intraday (decisions every 5 minutes, flat every night); track B
decides at each open and holds 2 to 20 sessions. Holding the same 68 symbols: +97.2%,
Sharpe 2.04.

| Configuration | Total | Sharpe | Trades | Per trade | 3x costs |
| --- | ---: | ---: | ---: | ---: | ---: |
| B, convolution, rich, differential Sharpe | +51.9% | 1.93 | 3,346 | +84 bp | +43.1% |
| B, GRU, rich, differential Sharpe | +50.4% | 1.82 | 3,200 | +86 bp | +42.0% |
| B, GRU, base, differential Sharpe | +52.3% | 1.80 | 3,044 | +93 bp | +44.2% |
| B, other five | +37.0% to +47.7% | 1.37-1.70 | 1,724-3,324 | +80 to +154 bp | +32% to +42% |
| A, GRU, rich, log return | +4.5% | 1.76 | 948 | +31 bp | +2.4% |
| A, log-return configurations | -0.5% to +3.9% | -0.07 to 0.77 | 431-4,493 | -1 to +18 bp | -8.9% to +0.9% |
| A, differential-Sharpe configurations | -16.8% to -19.5% | about -1.5 | 22,380-28,623 | -5 bp | about -50% |

Best (by Sharpe): B, convolution, rich inputs, differential Sharpe reward. Edge over
holding -5.6 bp a day (95% interval -13.6 to +2.4); deflated Sharpe ratio over 124
trials 0.54; probability of backtest overfitting 0.23; +43.1% at 3x costs; it beat the
random trades with its own holding times (+34.8%) and the news model (+47.8%) but not
holding every session through the environment (+53.5%).

**Decision: fails** the pre-registered bar (interval includes zero, deflated Sharpe
below 0.95, PBO above 0.2, below one baseline). No configuration is frozen; the test
years stay unseen.

The daily track is the strongest result of the project: all eight configurations made
money, held up at three times the costs, agreed across seeds, and beat random trades
with the same timing, so the agent learned something. But it stays long only about
half the time and gives up more of the market's rise than its timing earns back. The
intraday track learned to stay selective with the plain return reward and churned with
the differential Sharpe reward, whose per-step signal rewarded frequent small trades.
