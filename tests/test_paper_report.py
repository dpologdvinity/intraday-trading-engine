import importlib.util
from pathlib import Path

from hft.logs import EventLog

NS = 1_000_000_000
SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "paper_report.py"


def _module():
    spec = importlib.util.spec_from_file_location("paper_report", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_report_counts_each_trade_once_across_restarts(tmp_path):
    entry, exit_ = 1_759_867_200 * NS, 1_759_930_200 * NS  # 2025-10-07 20:00, 13:30 next day UTC
    for start in (1, 2):  # a restart journals the same trade again only if it re-reads it
        log = EventLog(tmp_path / f"KO-{start}.jsonl")
        log.write("start", event_ns=start)
        log.write(
            "trade",
            event_ns=exit_,
            entry_ns=entry,
            exit_ns=exit_,
            pnl="1.50",
            opening_notional="500",
        )
        log.close()
    log = EventLog(tmp_path / "BAC-1.jsonl")
    log.write("start", event_ns=1)
    log.write(
        "trade", event_ns=exit_, entry_ns=entry, exit_ns=exit_, pnl="-0.50", opening_notional="500"
    )
    log.close()
    module = _module()
    trades = module.load_trades(tmp_path)
    assert len(trades) == 2
    text = module.report(trades)
    assert "KO" in text and "+30.00" in text  # 1.50 on 500 = 30 bp
    assert "2025-10-08" in text
    assert "total +1.00 over 2 trades on 1 exit days" in text


def test_markdown_lists_picks_fill_prices_and_cumulative_results(tmp_path):
    entry, exit_ = 1_759_930_260 * NS, 1_759_952_100 * NS  # 2025-10-08 09:31 and 15:35 ET
    log = EventLog(tmp_path / "KO-1.jsonl")
    log.write("start", event_ns=1)
    log.write("picks", event_ns=entry, session="2025-10-08", picks=["KO", "BAC"], picked=True)
    for execution_id, quantity, price, stamp in (
        ("b1", "3", "60", entry),
        ("b2", "1", "64", entry),
        ("b2", "1", "64", entry),  # journaled twice after a restart: counted once
        ("s1", "-4", "62", exit_),
    ):
        log.write(
            "fill",
            event_ns=stamp,
            execution={
                "execution_id": execution_id,
                "signed_quantity": quantity,
                "price": price,
                "timestamp_ns": stamp,
            },
        )
    log.write(
        "trade", event_ns=exit_, entry_ns=entry, exit_ns=exit_, pnl="4", opening_notional="244"
    )
    log.close()
    module = _module()
    text = module.markdown("demo", module.load_trades(tmp_path), module.load_sessions(tmp_path))
    assert "| 2025-10-08 | KO, BAC | KO | 61.00 | 62.00 | +4.00 | +164 |" in text
    assert "| 2025-10-08 | +4.00 | +1.64% | +4.00 |" in text
