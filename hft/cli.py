"""Operator CLI. Import training and mutable broker code only in explicit commands."""

import argparse
import asyncio
import json
import os
import sys
import time
import tomllib
from pathlib import Path


def _parser():
    parser = argparse.ArgumentParser(description="Historical learning and risk-gated stock trading")
    commands = parser.add_subparsers(dest="command", required=True)
    smoke = commands.add_parser(
        "smoke", help="offline synthetic engineering proof; never live eligible"
    )
    smoke.add_argument("--output", type=Path, default=Path("artifacts/smoke"))
    smoke.add_argument("--timesteps", type=int, default=2048)
    for name in ("data-probe", "download", "prepare", "record"):
        command = commands.add_parser(name)
        command.add_argument("--symbol", default="AAPL")
        command.add_argument("--feed", choices=["iex"], default="iex")
        if name == "data-probe":
            command.add_argument("--session", required=True)
        elif name == "download":
            command.add_argument("--start", required=True)
            command.add_argument("--end", required=True)
            command.add_argument("--output", type=Path, default=Path("data/cache"))
        elif name == "prepare":
            command.add_argument("--input", type=Path)
            command.add_argument("--bar-seconds", type=int, choices=[5], default=5)
        else:
            command.add_argument("--output", type=Path, default=Path("data/recordings"))
            command.add_argument("--duration", type=float)
    experiment = commands.add_parser("experiment")
    experiment.add_argument("--config", type=Path, default=Path("config/research.toml"))
    experiment.add_argument("--dataset", type=Path)
    experiment.add_argument("--output", type=Path, required=True)
    train = commands.add_parser("train")
    train.add_argument("--experiment", type=Path, required=True)
    train.add_argument("--resume", action="store_true")
    quality = commands.add_parser(
        "data-quality", help="check development coverage without training or test evaluation"
    )
    quality.add_argument("--experiment", type=Path, required=True)
    quality.add_argument(
        "--all-development",
        action="store_true",
        help="inspect every development session; report failures without fitting or reading final-test data",
    )
    report = commands.add_parser("report")
    report.add_argument("--experiment", type=Path, required=True)
    dashboard = commands.add_parser("dashboard", help="local read-only research dashboard")
    dashboard.add_argument("--port", type=int, default=8765)
    for name in ("replay", "dry-run", "broker-paper", "live"):
        command = commands.add_parser(name)
        command.add_argument("--model", type=Path)
        command.add_argument("--config", type=Path)
        command.add_argument("--capital", type=float)
        command.add_argument("--output", type=Path)
        if name == "replay":
            command.add_argument("--dataset", type=Path, required=True)
        else:
            command.add_argument("--duration", type=float)
        if name == "live":
            command.add_argument("--enable-live", action="store_true")
            command.add_argument("--max-live-notional", type=float)
            command.add_argument("--logs", type=Path, default=Path("logs/broker-paper"))
    trade = commands.add_parser(
        "trade", help="paper-trade chosen stocks every trading day until stopped"
    )
    trade.add_argument("--paper", action="store_true", help="trade on the Alpaca paper account")
    trade.add_argument("--live", action="store_true", help=argparse.SUPPRESS)
    trade.add_argument("--symbols", nargs="+", metavar="SYMBOL=DOLLARS")
    trade.add_argument(
        "--strategy",
        default="ema-crossover",
        help="ema-crossover, hold-day, overnight-drift, momentum, news-picks:<variant> or model:<bundle>",
    )
    trade.add_argument("--name", help="run name (state and journals are kept per run)")
    trade.add_argument("--daily-loss", type=float, default=0.02)
    trade.add_argument("--max-drawdown", type=float, default=0.05)
    trade.add_argument(
        "--max-spread-bps",
        type=float,
        default=10.0,
        help="skip entries while the bid-ask spread is wider than this (exits are never blocked)",
    )
    trade.add_argument("--engine", choices=["auto", "python", "cpp"], default="auto")
    trade.add_argument(
        "--frames",
        choices=["python", "native"],
        default="python",
        help="native: parse market-data frames in C++ (needs the C++ engine)",
    )
    trade.add_argument("--status", action="store_true", help="show runs without trading")
    trade.add_argument(
        "--replay", metavar="YYYY-MM-DD", help="simulate a past day instead of live paper trading"
    )
    status = commands.add_parser("status")
    status.add_argument("--state", type=Path, default=Path(".state"))
    status.add_argument("--logs", type=Path, default=Path("logs"))
    graduate = commands.add_parser("graduate")
    graduate.add_argument("--model", type=Path, required=True)
    graduate.add_argument("--logs", type=Path, required=True)
    graduate.add_argument("--calendar", type=Path)
    graduate.add_argument("--output", type=Path)
    return parser


def _read_credentials():
    key = os.environ.get("ALPACA_API_KEY") or os.environ.get("ALPACA_PAPER_API_KEY")
    secret = os.environ.get("ALPACA_SECRET_KEY") or os.environ.get("ALPACA_PAPER_SECRET_KEY")
    if key and secret:
        os.environ.setdefault("ALPACA_API_KEY", key)
        os.environ.setdefault("ALPACA_SECRET_KEY", secret)
    if not key or not secret:
        raise ValueError("set ALPACA_API_KEY and ALPACA_SECRET_KEY for free read-only market data")


def _handle(args):
    if args.command == "trade":
        from .trading.command import handle_trade

        return handle_trade(args)
    if args.command == "dashboard":
        from .dashboard import serve

        if not 0 < args.port <= 65535:
            raise ValueError("dashboard port must be between 1 and 65535")
        return serve(Path(__file__).resolve().parents[1], args.port)
    if args.command == "live":
        if not args.enable_live:
            raise ValueError("live requires explicit --enable-live and validated paper evidence")
        if args.capital is None:
            raise ValueError("live requires explicit --capital matching the evaluated allocation")
        if args.max_live_notional is None or not 0 < args.max_live_notional <= 10:
            raise ValueError("live requires --max-live-notional in (0,10] for the canary")
    if args.command == "data-probe":
        _read_credentials()
        from .history import probe_access

        return probe_access(args.symbol, args.session)
    if args.command == "download":
        _read_credentials()
        from .history import download_sessions

        return {"dataset": str(download_sessions(args.symbol, args.start, args.end, args.output))}
    if args.command == "smoke":
        from .data import save_dataset, synthetic_sessions
        from .paper import PaperEngine, replay_sessions
        from .policy import OnnxPolicy
        from .training import train

        if args.output.exists():
            raise ValueError("smoke output already exists; choose a fresh directory")
        sessions = synthetic_sessions(days=4, bars_per_day=100)
        save_dataset(sessions, args.output / "dataset")
        result = train(sessions, args.output / "search", timesteps=args.timesteps, smoke=True)
        policy = OnnxPolicy(args.output / "search" / "bundle")
        replay = replay_sessions(
            PaperEngine(
                policy,
                log_path=args.output / "replay.jsonl",
                symbol=policy.metadata["symbol"],
                synthetic=True,
                metadata=policy.metadata,
            ),
            sessions,
        )
        return {
            "status": result["status"],
            "paper_eligible": False,
            "bundle": str(args.output / "search" / "bundle"),
            "replay": replay,
        }
    if args.command == "prepare":
        from .recording import prepare_recordings

        if not args.input:
            raise ValueError("prepare requires --input recording directory or manifest")
        return prepare_recordings(args.input)
    if args.command == "record":
        _read_credentials()
        from .recording import record

        return asyncio.run(record(args.symbol, args.output, args.duration))
    if args.command == "experiment":
        from .research import freeze_experiment

        if not args.dataset:
            raise ValueError("experiment requires --dataset manifest")
        return freeze_experiment(args.dataset, tomllib.loads(args.config.read_text()), args.output)
    if args.command == "train":
        from .training import run_search

        return run_search(args.experiment, resume=args.resume)
    if args.command == "data-quality":
        from .research import preflight_experiment

        return preflight_experiment(args.experiment, inspect_all=args.all_development)
    if args.command == "report":
        path = (
            args.experiment
            if args.experiment.name == "research.json"
            else args.experiment.parent / "search" / "research.json"
        )
        if not path.exists() and args.experiment.exists():
            frozen = json.loads(args.experiment.read_text())
            if frozen.get("status") == "insufficient-data":
                return frozen
            raise ValueError("research report not available; run train for this frozen experiment")
        return json.loads(path.read_text())
    if args.command == "status":
        from .logs import iter_log

        journals = []
        for path in sorted(args.logs.rglob("*.jsonl")):
            try:
                first = last = None
                for row in iter_log(path):
                    first = first or row
                    last = row
                journals.append({"path": str(path), "source": first.get("source"), "last": last})
            except (ValueError, OSError) as error:
                journals.append({"path": str(path), "error": str(error)})
        return {"journals": journals, "state_files": [str(p) for p in args.state.glob("*.json")]}
    if args.command == "graduate":
        from .data import atomic_json
        from .evidence import graduate

        result = graduate(args.model, args.logs, calendar_path=args.calendar)
        if args.output:
            atomic_json(args.output, result)
        return result
    if args.command in ("replay", "dry-run", "broker-paper", "live"):
        if not args.model:
            raise ValueError("--model immutable bundle directory is required")
        from .configuration import runtime_config
        from .policy import OnnxPolicy

        policy = OnnxPolicy(args.model)
        config = runtime_config(
            policy.metadata,
            args.config,
            args.capital,
            args.max_live_notional if args.command == "live" else None,
        )
        output = args.output or Path("logs") / args.command / f"run-{time.time_ns()}.jsonl"
        if args.command == "replay":
            from .data import load_dataset
            from .paper import PaperEngine, replay_sessions
            from .risk import RiskGateway

            sessions = load_dataset(args.dataset)
            if any(
                s.symbol != config.symbol or s.manifest.get("feed") != config.feed for s in sessions
            ):
                raise ValueError("dataset/model market contract mismatch")
            return replay_sessions(
                PaperEngine(
                    policy,
                    log_path=output,
                    initial_cash=config.capital,
                    costs=config.costs,
                    risk=RiskGateway(config.risk),
                    latency_ms=config.latency_ms,
                    bar_seconds=config.bar_seconds,
                    sizing=config.sizing,
                    symbol=config.symbol,
                    synthetic=any(s.synthetic for s in sessions),
                    metadata=policy.metadata,
                ),
                sessions,
            )
        if policy.metadata.get("synthetic_training") or config.feed != "iex":
            raise ValueError("streaming trading requires a model trained on real IEX data")
        _read_credentials()
        broker = None
        if args.command in ("broker-paper", "live"):
            from .broker import AlpacaBroker, AlpacaClient
            from .risk import RiskGateway

            broker = AlpacaBroker(
                AlpacaClient(args.command),
                config.symbol,
                RiskGateway(config.risk),
                config.costs,
                initial_cash=config.capital,
                enable_live=args.command == "live",
                model_path=args.model,
                logs_path=args.logs if args.command == "live" else None,
                max_live_notional=args.max_live_notional if args.command == "live" else None,
                contract_hash=policy.manifest["contract_hash"] + ":" + config.execution_hash,
            )
        from .runtime import run_stream

        try:
            return asyncio.run(
                run_stream(
                    policy,
                    config,
                    mode=args.command,
                    log_path=output,
                    duration=args.duration,
                    broker=broker,
                )
            )
        finally:
            if broker:
                broker.close()
    raise ValueError("unknown command")


def main(argv=None):
    args = _parser().parse_args(argv)
    try:
        result = _handle(args)
        if result is not None:
            print(
                result
                if isinstance(result, str)
                else json.dumps(result, indent=2, allow_nan=False, default=str)
            )
        return 0
    except (ValueError, RuntimeError, OSError, ImportError, KeyError, TypeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
