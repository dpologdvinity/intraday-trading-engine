# News features for the ML day trader: design

Status: decided 2026-10-09 under the owner's delegation ("you make all the
decisions"). Builds on the [ML day trader](2026-10-08-ml-daytrader-design.md), whose
price-only search found no variant close to holding
([results](https://github.com/dpologdvinity/intraday-trading-engine/blob/master/ml-daytrader-results.md)).

## Hypothesis

News that arrives before the open (analyst actions, earnings, guidance, deals,
lawsuits, offerings) carries information that prices before the open have not fully
absorbed, so stocks with strong overnight news drift in that direction during the
day. If true, adding news to the daily model should lift the daily picks above
holding the same symbols.

## Data

- Alpaca's free news API (Benzinga), every article tagged with any of the 69 symbols,
  2016-01-01 to the latest complete day. Headline, summary, symbols, source and the
  `created_at` time; article bodies are not downloaded. Stored as Parquet per month
  under `data/ml/news/`, resumable, read-only.
- Only `created_at` is used: an article counts from the moment it was first
  published, never from later edits.

## Reading the news

Phase 1 uses no language model, so no model knowledge from after the event can leak
in. Each headline is matched against fixed patterns for events with a direction:

| Event | Direction |
| --- | --- |
| analyst upgrade, price target raised, initiated at buy | up |
| analyst downgrade, price target cut, initiated at sell | down |
| earnings or revenue beat, guidance raised | up |
| earnings or revenue miss, guidance cut | down |
| FDA approval / rejection | up / down |
| acquisition target, buyback, dividend raise | up |
| stock offering, lawsuit or investigation, recall | down |

Phase 2 (later, on a Kaggle GPU): FinBERT sentiment, a finance model trained in
2019 on earlier text, as an extra score.

## Features

For session d, from articles created between the previous session's close and 09:25
ET on d (and for the market as a whole):

- article count; count of focused articles (at most three symbols tagged);
- counts of up events and down events, and their difference, from focused articles;
- separate counts for analyst actions, earnings and guidance;
- article count over the previous five sessions (attention);
- market-wide up minus down events over all symbols.

Features join the 23 price features of the daily model when a variant sets
`daily.news = true`.

## Evaluation

- Same harness, splits, costs and benchmark as the ML day trader.
- Variants: the 8 return-target daily settings with news, each with no minute model
  and with the 30-minute GRU (16 variants). Trials are counted on top of the 84
  already tried on the same validation years (100 in total).
- Pass bar unchanged: beat holding on validation with the daily difference's 95%
  interval above zero. Only then is the one-time final test spent.
