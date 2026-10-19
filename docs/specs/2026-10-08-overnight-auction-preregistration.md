# Pre-registration: overnight drift with auction execution, 2016-2024

Written and committed before any result was computed. Only a five-day AAPL request
was made beforehand, to confirm that the endpoint serves 2016 bars.

## Why

The paper strategy `overnight-drift` now buys in the closing auction and sells in
the opening auction, which avoids the bid-ask spread that made the earlier
backtest break even ([multiday results](https://github.com/dpologdvinity/intraday-trading-engine/blob/master/multiday-results.md)). That backtest
used 2024-10-03 to 2026-10-02, which has now been seen. Alpaca's free historical
SIP daily bars go back to 2016, so the years before October 2024 give a fresh test
of the strategy as deployed.

## Rule

Hold every stock from one session's close to the next session's open, every
night. Long only, equal weight.

## Data

- Alpaca daily SIP bars (`timeframe=1Day`, `adjustment=all`, so splits and
  dividends are included and the series is a total return) for the same 46 stocks
  and, separately, SPY, QQQ, IWM and DIA.
- Period: nights starting 2016-01-04 through 2024-10-01 (the last night ending
  before 2024-10-03, where the previously seen data begins).
- A daily bar's open and close are used as the auction prices. They are close to,
  but not guaranteed to equal, the official opening and closing auction prices.
- A night is used for a stock when both bars exist and are consecutive sessions in
  that stock's bars.
- Sessions returned by `hft.training.reserved_final_sessions()` are excluded.

## Costs per round trip

No spread is charged, because auction orders trade at the single auction price.
Each side pays 1 bp as an allowance for auction price impact and for the gap
between bar and official prices, and $0.005 per share at the trade price (Alpaca
charges no commission; this is the project's standard cost model). Results are
also reported at 3x these costs and before costs.

## Statistics and decision

- The unit is a night: the equal-weight mean net return of the stocks with data,
  in basis points. Intervals come from 10,000 bootstrap resamples of nights
  (seed 0). One test is pre-registered, so the decision uses the 95% interval.
- Reported alongside: the same stocks held from open to close (intraday control),
  each calendar year separately, and the ETFs.
- **Pass:** the stocks' mean net return per night has a 95% interval lower bound
  above zero.
- **Caveat recorded in advance:** the 46 stocks were chosen in 2026 as large,
  liquid companies, so they survived and grew. That survivorship lifts returns in
  both the overnight and the intraday legs. The ETFs, which have no survivorship
  bias, are reported as the check on this; if the stocks pass but the ETFs' net
  overnight mean is not positive, the result is recorded as likely inflated.
- A pass supports continuing the forward paper evaluation. It does not unlock real
  money; research qualification and 30-session paper graduation still apply.
