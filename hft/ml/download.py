"""Download Alpaca SIP daily and minute bars for the ML universe, resumably.

Bars are split- and dividend-adjusted (`adjustment=all`). Minute bars keep only
regular-session minutes, cut at each session's real close from the exchange
calendar (13:00 on half days). Each year is written atomically once all its pages
arrived, so an interrupted run resumes at the first missing year; the latest year is
always refreshed. Sessions reserved for the final test of the original research are
never stored.

    .venv/bin/python -m hft.ml.download --timeframe 1Day
    .venv/bin/python -m hft.ml.download --timeframe 1Min
"""

import argparse
import json
import sys
import time as time_module
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from ..feed import parse_timestamp
from ..history import DATA_BASE, HistoryError, ReadOnlyClient, _calendar
from .universe import UNIVERSE

NEW_YORK = ZoneInfo("America/New_York")
SCHEMA = pa.schema(
    [
        ("t", pa.int64()),
        ("o", pa.float64()),
        ("h", pa.float64()),
        ("l", pa.float64()),
        ("c", pa.float64()),
        ("v", pa.float64()),
        ("vw", pa.float64()),
        ("n", pa.int64()),
    ]
)
MAX_PAGES = 10_000


def _utc(day, at=time(0)):
    return datetime.combine(day, at, NEW_YORK).astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _end(day):
    # Free SIP data excludes the latest 15 minutes; asking for them is refused (403).
    latest = datetime.now(UTC) - timedelta(minutes=16)
    wanted = datetime.combine(day, time(23, 59), NEW_YORK).astimezone(UTC)
    return min(wanted, latest).strftime("%Y-%m-%dT%H:%M:%SZ")


def _session_day(ns):
    return datetime.fromtimestamp(ns / 1e9, UTC).astimezone(NEW_YORK).date().isoformat()


def _fetch(client, symbol, timeframe, start, end):
    params = {
        "timeframe": timeframe,
        "start": _utc(start),
        "end": _end(end),
        "adjustment": "all",
        "feed": "sip",
        "sort": "asc",
        "limit": 10000,
    }
    rows, seen = [], set()
    for _ in range(MAX_PAGES):
        payload = client.get(f"{DATA_BASE}/v2/stocks/{symbol}/bars", params)
        if not isinstance(payload, dict):
            raise TypeError("invalid bars response")
        rows += payload.get("bars") or []
        token = payload.get("next_page_token")
        if not token:
            return rows
        if token in seen:
            raise RuntimeError("pagination token loop")
        seen.add(token)
        params = {**params, "page_token": token}
    raise RuntimeError("pagination page budget exhausted")


def _table(rows, timeframe, windows, reserved):
    t = np.array([parse_timestamp(r["t"]) for r in rows], dtype=np.int64)
    keep = np.ones(len(t), dtype=bool)
    if timeframe == "1Min" and len(t):
        opens = np.array([w.open_ns for w in windows], dtype=np.int64)
        closes = np.array([w.close_ns for w in windows], dtype=np.int64)
        index = np.searchsorted(opens, t, side="right") - 1
        keep = (index >= 0) & (t < closes[np.clip(index, 0, None)])
    if reserved:
        keep &= np.array([_session_day(x) not in reserved for x in t], dtype=bool)
    kept = [r for r, k in zip(rows, keep, strict=True) if k]
    columns = {
        "t": t[keep],
        **{k: np.array([float(r[k]) for r in kept], dtype=np.float64) for k in "ohlcv"},
        "vw": np.array([float(r.get("vw") or r["c"]) for r in kept], dtype=np.float64),
        "n": np.array([int(r.get("n") or 0) for r in kept], dtype=np.int64),
    }
    order = np.argsort(columns["t"], kind="stable")
    table = pa.table({k: v[order] for k, v in columns.items()}, schema=SCHEMA)
    unique = np.concatenate([[True], np.diff(table.column("t").to_numpy()) > 0])[: len(table)]
    return table.filter(pa.array(unique))


def download_bars(symbol, timeframe, start, end, root, *, client=None, windows=None):
    """Write `root/<timeframe>/<SYMBOL>/<YEAR>.parquet`; returns the year files."""
    if timeframe not in ("1Day", "1Min"):
        raise ValueError("timeframe must be 1Day or 1Min")
    client = client or ReadOnlyClient()
    if windows is None:
        windows, _ = _calendar(client, start.isoformat(), end.isoformat())
    from ..training import reserved_final_sessions  # heavy; only downloads need it

    reserved = {d for s, d in reserved_final_sessions() if s == symbol}
    folder = Path(root) / timeframe / symbol
    folder.mkdir(parents=True, exist_ok=True)
    if _adjustments_changed(client, symbol, timeframe, folder, windows, reserved):
        for stale in folder.glob("*"):
            stale.unlink()  # history was re-adjusted (split or dividend): fetch it all again
    paths = []
    for year in range(start.year, end.year + 1):
        path = folder / f"{year}.parquet"
        paths.append(path)
        if year < end.year and _complete(path, year):
            continue
        first, last = max(start, date(year, 1, 1)), min(end, date(year, 12, 31))
        rows = _fetch(client, symbol, timeframe, first, last)
        temporary = path.with_suffix(".parquet.tmp")
        pq.write_table(_table(rows, timeframe, windows, reserved), temporary)
        temporary.replace(path)
        path.with_suffix(".json").write_text(json.dumps({"through": last.isoformat()}) + "\n")
    return paths


def _complete(path, year):
    """A year file counts as finished only once it was requested through December 31."""
    marker = path.with_suffix(".json")
    if path.exists() and marker.exists():
        return json.loads(marker.read_text())["through"] >= f"{year}-12-31"
    if path.exists():  # written before markers existed: trust it if it reaches year end
        times = pq.read_table(path, columns=["t"]).column("t").to_numpy()
        return len(times) > 0 and _session_day(int(times.max())) >= f"{year}-12-28"
    return False


def _adjustments_changed(client, symbol, timeframe, folder, windows, reserved):
    """Refetch the last stored week and compare: adjusted history moves after a split
    or dividend, and mixing old and new adjustments would fake a jump between files."""
    files = sorted(folder.glob("*.parquet"))
    if not files:
        return False
    stored = pq.read_table(files[-1])
    if not len(stored):
        return False
    last = date.fromisoformat(_session_day(int(stored.column("t").to_numpy().max())))
    fresh = _table(
        _fetch(client, symbol, timeframe, last - timedelta(days=7), last),
        timeframe,
        windows,
        reserved,
    )
    old = dict(zip(stored.column("t").to_pylist(), stored.column("c").to_pylist(), strict=True))
    pairs = [
        (old[t], c)
        for t, c in zip(fresh.column("t").to_pylist(), fresh.column("c").to_pylist(), strict=True)
        if t in old
    ]
    return any(abs(a / b - 1) > 1e-6 for a, b in pairs)


def save_calendar(root, windows):
    rows = [
        {"session_id": w.session_id, "open_ns": w.open_ns, "close_ns": w.close_ns} for w in windows
    ]
    path = Path(root) / "calendar.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows) + "\n")


def update_calendar(root, windows):
    """Add or replace sessions in the saved calendar, keeping every earlier session."""
    path = Path(root) / "calendar.json"
    sessions = {w.session_id: w for w in (load_calendar(root) if path.exists() else [])}
    sessions.update({w.session_id: w for w in windows})
    save_calendar(root, [sessions[k] for k in sorted(sessions)])


def load_calendar(root):
    """Exchange sessions saved by the download, as SessionWindow objects."""
    from ..calendar import SessionWindow

    rows = json.loads((Path(root) / "calendar.json").read_text())
    return [SessionWindow(r["session_id"], r["open_ns"], r["close_ns"]) for r in rows]


def read_bars(root, timeframe, symbol) -> pa.Table:
    files = sorted((Path(root) / timeframe / symbol).glob("*.parquet"))
    if not files:
        return SCHEMA.empty_table()
    table = pa.concat_tables([pq.read_table(f, schema=SCHEMA) for f in files])
    return table.sort_by("t")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--timeframe", choices=["1Day", "1Min"], required=True)
    parser.add_argument("--root", type=Path, default=Path("data/ml"))
    parser.add_argument("--symbols", nargs="*", help="default: the whole universe")
    parser.add_argument("--end", type=date.fromisoformat, default=datetime.now(NEW_YORK).date())
    args = parser.parse_args(argv)
    from ..cli import _read_credentials

    _read_credentials()
    client = ReadOnlyClient()
    windows, _ = _calendar(client, "2016-01-01", args.end.isoformat())
    save_calendar(args.root, [w for w in windows if w.session_id <= args.end.isoformat()])
    wanted = set(args.symbols or [])
    for symbol in UNIVERSE:
        if wanted and symbol.ticker not in wanted:
            continue
        started = time_module.monotonic()
        for attempt in range(6):
            try:
                paths = download_bars(
                    symbol.ticker,
                    args.timeframe,
                    symbol.first_date,
                    args.end,
                    args.root,
                    client=client,
                    windows=windows,
                )
                break
            except HistoryError as error:
                # Rate limits are shared with other programs on the same key: wait and resume.
                if "429" not in str(error) or attempt == 5:
                    raise
                print(f"{symbol.ticker:<6} rate limited; retrying in 60 s", flush=True)
                time_module.sleep(60)
        rows = sum(pq.read_metadata(p).num_rows for p in paths)
        seconds = time_module.monotonic() - started
        print(f"{symbol.ticker:<6} {args.timeframe} {rows:>9} bars {seconds:6.1f}s", flush=True)
    if (args.root / "manifest.json").exists():
        from .datasets import write_data_manifest

        write_data_manifest(args.root)  # the data changed on purpose: record the new version
    return 0


if __name__ == "__main__":
    sys.exit(main())
