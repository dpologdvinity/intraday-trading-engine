"""`hft trade`: paper-trade chosen stocks every trading day until stopped."""

import asyncio
import hashlib
import json
import re
import signal
import time
from collections import deque
from dataclasses import asdict
from decimal import Decimal, InvalidOperation
from pathlib import Path

from ..account import Costs
from ..strategies import STRATEGY_VERSION, parse_strategy
from .budget import budget_gateway, budget_risk_config
from .stream import MAX_SYMBOLS

SCHEDULED_SYMBOLS = 100

SYMBOL = re.compile(r"[A-Z][A-Z0-9.-]{0,14}")
RUN_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")
LIVE_REFUSAL = "real money unlocks after research qualification and 30-session paper graduation"
ROOT = Path(__file__).resolve().parents[2]


def parse_symbols(items, limit=MAX_SYMBOLS) -> dict[str, Decimal]:
    budgets = {}
    for item in items or ():
        symbol, sep, amount = item.partition("=")
        symbol = symbol.strip().upper()
        if not sep or not SYMBOL.fullmatch(symbol):
            raise ValueError(f"expected SYMBOL=DOLLARS, got {item!r}")
        try:
            budget = Decimal(amount)
        except InvalidOperation:
            raise ValueError(f"invalid budget in {item!r}") from None
        if not budget.is_finite() or budget <= 0:
            raise ValueError(f"budget for {symbol} must be a positive dollar amount")
        if symbol in budgets:
            raise ValueError(f"{symbol} is listed twice")
        budgets[symbol] = budget
    if not 1 <= len(budgets) <= limit:
        raise ValueError(f"choose 1-{limit} stocks with --symbols SYMBOL=DOLLARS")
    return budgets


def run_paths(name, root=ROOT):
    if not RUN_NAME.fullmatch(name):
        raise ValueError("run name may use letters, digits, '.', '_' and '-' (max 64)")
    return root / ".state" / "trade" / name, root / "logs" / "trade" / name


def default_name(budgets, strategy):
    return "-".join(budgets)[:40] + "-" + strategy.name.replace(":", "")


def run_identity(
    budgets, strategy, *, daily_loss, max_drawdown, max_spread_bps, engine, engine_version, feed
):
    bundle = None
    if strategy.name == "news-picks":
        folder = Path(strategy.bundle)
        weights = sorted(p for p in folder.iterdir() if p.name.startswith("daily"))
        digest = hashlib.sha256((folder / "config.json").read_bytes())
        for path in weights:
            digest.update(path.read_bytes())
        bundle = digest.hexdigest()
    elif strategy.bundle is not None:
        bundle = hashlib.sha256((Path(strategy.bundle) / "model.onnx").read_bytes()).hexdigest()
    value = {
        "budgets": {s: str(b) for s, b in budgets.items()},
        "strategy": [strategy.name, strategy.version, bundle],
        "engine": [engine, engine_version],
        "risk": asdict(budget_risk_config(daily_loss, max_drawdown)),
        "max_entry_spread_bps": str(max_spread_bps),
        "costs": asdict(Costs()),
        "latency_ms": 75,
        "bar_seconds": 5,
        "feed": feed,
    }
    text = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()


def handle_trade(args):
    if args.status:
        from .status import format_status

        return format_status(ROOT, args.name)
    if args.replay:
        return handle_replay(args)
    if args.live or not args.paper:
        raise ValueError(LIVE_REFUSAL)
    strategy = parse_strategy(args.strategy)
    # Scheduled strategies poll prices over REST, so the stream's symbol limit does not apply.
    budgets = parse_symbols(
        args.symbols, SCHEDULED_SYMBOLS if strategy.is_scheduled else MAX_SYMBOLS
    )
    if strategy.version != STRATEGY_VERSION:
        raise ValueError("unsupported strategy version")
    if strategy.bundle is not None and not strategy.is_scheduled:
        from ..policy import OnnxPolicy

        if len(budgets) != 1:
            raise ValueError("a trained model trades exactly one stock")
        if OnnxPolicy(strategy.bundle).metadata["symbol"] != next(iter(budgets)):
            raise ValueError("the model was trained for a different stock")
    name = args.name or default_name(budgets, strategy)
    state_dir, log_dir = run_paths(name)

    from ..broker import AlpacaClient
    from ..calendar import calendar_from_records
    from ..cli import _read_credentials
    from ..market_engine import make_market_engine
    from ..runtime import fetch_calendar
    from .broker import PortfolioBroker
    from .runner import TradeConfig, TradeRunner

    _read_credentials()
    if strategy.is_scheduled:
        engine, engine_version = "schedule", STRATEGY_VERSION
    else:
        probe = make_market_engine(next(iter(budgets)), implementation=args.engine)
        engine, engine_version = probe.implementation, probe.version
    identity = run_identity(
        budgets,
        strategy,
        daily_loss=args.daily_loss,
        max_drawdown=args.max_drawdown,
        max_spread_bps=args.max_spread_bps,
        engine=engine,
        engine_version=engine_version,
        feed="iex",
    )
    config = TradeConfig(
        name,
        tuple(budgets),
        budgets,
        strategy,
        log_dir=log_dir,
        run_dir=state_dir,
        daily_loss=args.daily_loss,
        max_drawdown=args.max_drawdown,
        engine=args.engine,
        frames=args.frames,
    )
    client = AlpacaClient("broker-paper")

    def calendar():
        return calendar_from_records(fetch_calendar(client))

    if strategy.is_scheduled:
        return _run_overnight(args, name, budgets, strategy, identity, client, calendar)
    broker = PortfolioBroker(
        client,
        budgets,
        Costs(),
        lambda s: budget_gateway(
            budgets[s], args.daily_loss, args.max_drawdown, args.max_spread_bps
        ),
        run_dir=state_dir,
        identity=identity,
        clock=time.time_ns,
    )
    try:
        runner = TradeRunner(config, broker, calendar(), calendar_source=calendar)
        print(f"paper trading {', '.join(budgets)} as run '{name}'; Ctrl-C to stop", flush=True)
        return asyncio.run(_run_until_signal(runner))
    finally:
        broker.close()


def _run_overnight(args, name, budgets, strategy, identity, client, calendar):
    from .broker import PortfolioBroker
    from .overnight import OvernightBook, OvernightConfig, OvernightRunner

    state_dir, log_dir = run_paths(name)
    broker = PortfolioBroker(
        client,
        budgets,
        Costs(),
        lambda s: budget_gateway(budgets[s], args.daily_loss, args.max_drawdown),
        run_dir=state_dir,
        identity=identity,
        clock=time.time_ns,
        book_class=OvernightBook,
    )
    try:
        config = OvernightConfig(
            name,
            tuple(budgets),
            budgets,
            strategy,
            log_dir=log_dir,
            run_dir=state_dir,
            daily_loss=args.daily_loss,
            max_drawdown=args.max_drawdown,
        )
        if strategy.name == "news-picks":
            from ..ml.live import NewsPicker
            from .news_picks import NewsPicksRunner

            picker = NewsPicker(strategy.bundle, ROOT / "data" / "ml")
            runner = NewsPicksRunner(
                config, broker, calendar(), calendar_source=calendar, picker=picker
            )
            print(
                f"news-picks paper trading as run '{name}': picks at the open from "
                f"{strategy.bundle}, sells five minutes before the close; Ctrl-C cancels orders",
                flush=True,
            )
        elif strategy.name == "momentum":
            from .momentum import MomentumRunner

            runner = MomentumRunner(
                config, broker, calendar(), calendar_source=calendar, data_root=ROOT / "data" / "ml"
            )
            print(
                f"momentum paper trading as run '{name}': holds the top 10 of {len(budgets)} "
                "stocks by 12-1 momentum, rebalanced monthly; Ctrl-C cancels orders",
                flush=True,
            )
        else:
            runner = OvernightRunner(config, broker, calendar(), calendar_source=calendar)
            print(
                f"overnight paper trading {', '.join(budgets)} as run '{name}': buys near the "
                "close, sells at the next open; Ctrl-C cancels open orders and keeps positions",
                flush=True,
            )
        return asyncio.run(_run_until_signal(runner))
    finally:
        broker.close()


async def _run_until_signal(runner):
    task = asyncio.create_task(runner.run())
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, task.cancel)
    try:
        return await task
    except asyncio.CancelledError:
        return {"stopped": True, **runner.summary()}


def tail_rows(path, limit=200):
    """Last journal rows without reading a whole (possibly large) file."""
    with open(path, "rb") as f:
        f.seek(0, 2)
        size = f.tell()
        f.seek(max(0, size - 256 * 1024))
        lines = f.read().splitlines()[-limit:]
    rows = deque()
    for line in lines:
        try:
            rows.append(json.loads(line))
        except ValueError:
            continue  # first line may be a partial row
    return list(rows)


def handle_replay(args):
    """`hft trade --replay DAY`: what the bot would have done on a past session."""
    from datetime import date

    from .replay import ensure_session, format_replay, replay_stock

    if args.live:
        raise ValueError(LIVE_REFUSAL)
    day = date.fromisoformat(args.replay).isoformat()
    budgets = parse_symbols(args.symbols)
    strategy = parse_strategy(args.strategy)
    if strategy.is_scheduled:
        raise ValueError(
            "replay runs bar strategies; overnight-drift is evaluated by scripts/multiday_rules.py"
        )
    name = args.name or f"replay-{day}-" + default_name(budgets, strategy)
    _, log_dir = run_paths(name)
    rows = []
    for symbol, budget in budgets.items():
        manifest = ensure_session(symbol, day, download=_read_only_download)
        print(f"replaying {symbol} {day} ...", flush=True)
        risk = budget_gateway(budget, args.daily_loss, args.max_drawdown, args.max_spread_bps)
        rows.append(
            replay_stock(manifest, budget, strategy, log_dir, engine=args.engine, risk=risk)
        )
    return format_replay(rows)


def _read_only_download(symbol, start, end, cache_dir):
    from ..cli import _read_credentials
    from ..history import download_sessions

    _read_credentials()
    print(f"downloading {symbol} {start} (read-only IEX history; first time only) ...", flush=True)
    return download_sessions(symbol, start, end, cache_dir)
