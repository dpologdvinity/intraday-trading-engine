"""Pick today's stocks with a frozen daily model, from data refreshed this morning.

`NewsPicker.refresh` brings the daily bars, news headlines and exchange calendar up to
date (read-only, before the open). `NewsPicker.pick` builds today's row of daily
features from yesterday's bars and today's opening prices (`daily_frame(live=...)`),
adds the pre-open news features, scores every symbol with the saved model and applies
the same selection as the backtest. The opening prices come from the first IEX trades
after the open, a stand-in for the official open the backtest used.
"""

import json
from datetime import date
from pathlib import Path

import numpy as np

from .datasets import write_data_manifest
from .download import download_bars, update_calendar
from .features import daily_frame
from .news import download_news, read_news
from .news_features import news_features
from .train import DEFAULT, DailyModel, _costs, _merge, _select
from .universe import UNIVERSE


class NewsPicker:
    def __init__(self, variant_dir, data_root):
        self.variant_dir, self.root = Path(variant_dir), Path(data_root)
        self.config = _merge(DEFAULT, json.loads((self.variant_dir / "config.json").read_text()))
        self.config["device"] = "cpu"
        settings = self.config["daily"]
        if settings["sentiment"] or self.config["minute"]["model"] != "none":
            raise ValueError("live picks support daily models held from the open, without FinBERT")
        self.model = DailyModel.load(self.variant_dir, self.config)
        trade_only = {s.ticker for s in UNIVERSE if s.trade_only}
        self.symbols = list(
            self.config["symbols"] or [s.ticker for s in UNIVERSE if s.ticker not in trade_only]
        )

    def refresh(self, today: date, calendar, client=None):
        """Bring bars (through yesterday), news (through now) and the calendar up to date."""
        update_calendar(self.root, [w for w in calendar if w.session_id <= today.isoformat()])
        yesterday = date.fromordinal(today.toordinal() - 1)
        for symbol in {*self.symbols, "SPY", "QQQ"}:
            first = next((s.first_date for s in UNIVERSE if s.ticker == symbol), yesterday)
            download_bars(
                symbol, "1Day", first, yesterday, self.root, client=client, windows=calendar
            )
        if self.config["daily"]["news"]:
            download_news(UNIVERSE_TICKERS, today.replace(day=1), today, self.root, client=client)
        if (self.root / "manifest.json").exists():
            write_data_manifest(self.root)  # the data changed on purpose: record the version

    def pick(self, today: date, opens: dict, calendar) -> list[str]:
        frame = daily_frame(self.root, self.symbols, live=(today, opens))
        x = frame.x
        if self.config["daily"]["news"]:
            extra = news_features(read_news(self.root), self.symbols, frame.dates, calendar)
            x = np.concatenate([x, extra], axis=-1)
        predicted = self.model.predict(x[-1]).reshape(1, -1)
        settings = self.config["daily"]
        chosen = _select(
            predicted,
            frame.valid[-1:],
            _costs(self.symbols),
            settings["k"],
            np.ones(len(self.symbols), bool),
            gate=settings["gate"] and settings["target"] == "return",
            threshold_bp=settings["threshold_bp"],
        )[0]
        return [self.symbols[s] for s in chosen]


UNIVERSE_TICKERS = [s.ticker for s in UNIVERSE]
