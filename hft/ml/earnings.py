"""Earnings announcements from SEC EDGAR 8-K filings, matched to news headlines.

US companies file a Form 8-K with Item 2.02 ("Results of Operations and Financial
Condition") when they announce quarterly results. EDGAR's submissions API lists every
filing with its acceptance time, so these filings date each announcement to the
second without any paid calendar. Foreign filers (6-K) have no such item and are left
out. The download is free, needs no key, only a contact `User-Agent`
(`SEC_EDGAR_USER_AGENT`, never printed), and stays under EDGAR's limit of 10
requests a second. It runs once and is cached under `data/earnings/`, outside the ML
data root so that root's manifest is unchanged.

The news store (`hft.ml.news`) carries Benzinga headlines such as "Oracle Q2 EPS $1.21
Beats $1.18 Estimate"; `parse_eps_headline` reads the reported EPS and the consensus
from them, and `earnings_events` attaches every such headline to its filing.

    .venv/bin/python -m hft.ml.earnings
"""

import argparse
import gzip
import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from bisect import bisect_left, bisect_right
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from urllib.parse import urlsplit

import pyarrow as pa
import pyarrow.parquet as pq

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/{name}"
ALLOWED_HOSTS = frozenset({"www.sec.gov", "data.sec.gov"})
REQUESTS_PER_SECOND = 5  # half of EDGAR's published limit
FIRST_FILING = date(2015, 12, 1)  # a month before the first stored session
DAY_NS = 86_400 * 10**9
FILE_NAME = "edgar-8k.parquet"
# Companies whose earlier filings sit under another registrant: Disney's became
# "TWDC Enterprises 18" when the 2019 Fox deal created a new parent company.
FORMER_CIKS = {"DIS": (1001039,)}
SCHEMA = pa.schema(
    [
        ("symbol", pa.string()),
        ("cik", pa.int64()),
        ("accession", pa.string()),
        ("form", pa.string()),
        ("items", pa.string()),
        ("acceptance_ns", pa.int64()),
        ("filing_date", pa.string()),
    ]
)


class EdgarError(RuntimeError):
    """EDGAR refused a request or returned something unexpected."""


def _transport(url: str, headers: dict[str, str]) -> bytes:
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=30) as response:
        body = response.read()
        if response.headers.get("Content-Encoding") == "gzip":
            body = gzip.decompress(body)
        return body


class EdgarClient:
    """Rate-limited, GET-only JSON client restricted to EDGAR's hosts."""

    def __init__(
        self,
        user_agent: str,
        *,
        transport: Callable[[str, dict[str, str]], bytes] | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ):
        if not user_agent or "@" not in user_agent:
            raise EdgarError("EDGAR needs a User-Agent with a contact email address")
        self._headers = {"User-Agent": user_agent, "Accept-Encoding": "gzip"}
        self._transport = transport or _transport
        self._clock, self._sleep = clock, sleep
        self._last = -1e9

    def get(self, url: str):
        parts = urlsplit(url)
        if parts.scheme != "https" or parts.netloc not in ALLOWED_HOSTS:
            raise EdgarError(f"refusing a request outside EDGAR: {parts.netloc}")
        for attempt in range(5):
            wait = self._last + 1.0 / REQUESTS_PER_SECOND - self._clock()
            if wait > 0:
                self._sleep(wait)
            self._last = self._clock()
            try:
                return json.loads(self._transport(url, self._headers))
            except urllib.error.HTTPError as error:
                if error.code not in (429, 500, 502, 503) or attempt == 4:
                    raise EdgarError(f"EDGAR answered {error.code} for {url}") from error
                self._sleep(10.0 * (attempt + 1))  # EDGAR asks clients to back off
        raise EdgarError(f"no answer from {url}")  # pragma: no cover - loop always returns


def cik_map(payload: dict) -> dict[str, int]:
    """Ticker to CIK from EDGAR's company_tickers.json ({"0": {"cik_str", "ticker"}})."""
    return {row["ticker"].upper(): int(row["cik_str"]) for row in payload.values()}


def acceptance_ns(text: str) -> int:
    """EDGAR's `acceptanceDateTime` ("2024-08-01T20:30:41.000Z", UTC) as nanoseconds.

    Checked against headline times: Apple's results, released at 16:30 New York
    time, are accepted at 20:30Z in summer and 21:30Z in winter.
    """
    moment = datetime.fromisoformat(text)
    if moment.tzinfo is None:
        raise ValueError(f"acceptance time without a zone: {text}")
    return int(moment.timestamp()) * 10**9 + moment.microsecond * 1000


def earnings_filings(symbol: str, cik: int, blocks: Sequence[dict]) -> list[dict]:
    """8-K filings with Item 2.02 from columnar submission blocks (`filings.recent` and
    the older `filings.files` pages), from FIRST_FILING on, oldest first."""
    rows = {}
    for block in blocks:
        columns = zip(
            block["accessionNumber"],
            block["form"],
            block["items"],
            block["acceptanceDateTime"],
            block["filingDate"],
            strict=True,
        )
        for accession, form, items, accepted, filed in columns:
            if form != "8-K" or "2.02" not in [i.strip() for i in items.split(",")]:
                continue
            if date.fromisoformat(filed) < FIRST_FILING:
                continue
            rows[accession] = {
                "symbol": symbol,
                "cik": cik,
                "accession": accession,
                "form": form,
                "items": items,
                "acceptance_ns": acceptance_ns(accepted),
                "filing_date": filed,
            }
    return sorted(rows.values(), key=lambda r: (r["acceptance_ns"], r["accession"]))


def fetch_filings(client: EdgarClient, symbol: str, ciks: Sequence[int]) -> list[dict]:
    """Every 8-K Item 2.02 filing of one company under any of its registrants,
    following older pages as needed."""
    rows = []
    for cik in ciks:
        submissions = client.get(SUBMISSIONS_URL.format(name=f"CIK{cik:010d}.json"))
        filings = submissions["filings"]
        blocks = [filings["recent"]]
        for page in filings.get("files", []):
            if page["filingTo"] >= FIRST_FILING.isoformat():
                blocks.append(client.get(SUBMISSIONS_URL.format(name=page["name"])))
        rows += earnings_filings(symbol, cik, blocks)
    return sorted(rows, key=lambda r: (r["acceptance_ns"], r["accession"]))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def download_filings(symbols: Sequence[str], root, client: EdgarClient) -> Path:
    """Fetch and cache the filings once; an existing cache is kept as it is."""
    folder = Path(root)
    path = folder / FILE_NAME
    if path.exists():
        return path
    ciks = cik_map(client.get(TICKERS_URL))
    rows, counts, missing = [], {}, {}
    for symbol in symbols:
        if symbol not in ciks:
            missing[symbol] = "no CIK for this ticker"
            continue
        found = fetch_filings(client, symbol, (ciks[symbol], *FORMER_CIKS.get(symbol, ())))
        if not found:
            missing[symbol] = "no 8-K Item 2.02 filings"
        rows += found
        counts[symbol] = len(found)
        print(f"{symbol:<6} {len(found):>4} filings", flush=True)
    folder.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows, schema=SCHEMA)
    temporary = path.with_suffix(".parquet.tmp")
    pq.write_table(table, temporary)
    temporary.replace(path)
    manifest = {
        "downloaded_at": datetime.now(UTC).isoformat(),
        "source": [TICKERS_URL, SUBMISSIONS_URL.format(name="CIK##########.json")],
        "first_filing": FIRST_FILING.isoformat(),
        "filings": counts,
        "missing": missing,
        "sha256": _sha256(path),
    }
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    return path


def read_filings(root) -> pa.Table:
    """The cached filings, refusing a file that differs from its manifest."""
    folder = Path(root)
    manifest = json.loads((folder / "manifest.json").read_text())
    if _sha256(folder / FILE_NAME) != manifest["sha256"]:
        raise ValueError("earnings filings do not match their manifest")
    return pq.read_table(folder / FILE_NAME, schema=SCHEMA)


# Headlines ---------------------------------------------------------------------------

_VALUE = r"(?:-?\$\s?\(?-?\d+(?:\.\d+)?\)?|\(\$\d+(?:\.\d+)?\))"
_QUALIFIER = r"(?:(?:Adj(?:usted)?\.?|GAAP|Non-GAAP|Core|Operating) ?)*"
_HEADLINE = re.compile(
    rf"^(?P<company>[^;]+?) Q[1-4](?: (?:FY)?\d{{4}})? {_QUALIFIER}EPS(?: Adj\.)?"
    rf" (?P<loss>Loss (?:Of )?)?(?P<actual>{_VALUE})(?: Adj\.)?,?"
    rf" (?:(?P<verb>Beats|Beat|Misses|Miss|Vs\.?) (?P<estimate>{_VALUE}) Est(?:imate|\.|\b)"
    rf"|(?P<inline>Inline)\b)",
    re.IGNORECASE,
)
# Anywhere in a headline: a GAAP figure against an adjusted consensus, or a rumor.
_UNRELIABLE = re.compile(r"\bNot Compare\b|\bHearing\b", re.IGNORECASE)
_GUIDANCE = re.compile(r"\b(Sees|Raises|Lowers|Reaffirms|Guidance|Forecast|From)\b", re.IGNORECASE)


def _value(text: str, loss: bool) -> float:
    negative = "(" in text or "-" in text
    number = float(re.sub(r"[^\d.]", "", text))
    return -number if negative or loss else number


def parse_eps_headline(text: str) -> tuple[float, float] | None:
    """(reported EPS, consensus estimate) from a results headline, or None.

    Accepted: "<Company> Q<n> [year] [Adj./GAAP] EPS <value> Beats|Misses|vs <value>
    Estimate" ("Est." too) and "... EPS <value>, Inline" (the estimate equals the
    result). Losses may be written "$(0.12)", "-$0.12" or "Loss $0.12". Rejected as
    ambiguous: guidance ("Sees", "Raises", ...), any headline containing "Not Compare"
    (a GAAP figure against an adjusted consensus) or "Hearing" (a rumor), and any
    headline whose verb contradicts its numbers.
    """
    text = text.strip()
    if _UNRELIABLE.search(text):
        return None
    match = _HEADLINE.match(text)
    if match is None or _GUIDANCE.search(match.group("company")):
        return None
    loss = match.group("loss") is not None
    actual = _value(match.group("actual"), loss)
    if match.group("inline"):
        return actual, actual
    estimate = _value(match.group("estimate"), False)
    verb = match.group("verb").lower()
    if verb.startswith("vs"):
        return actual, estimate  # no verb to contradict: the numbers decide
    beat = verb.startswith("beat")
    if (actual > estimate) != beat or actual == estimate:
        return None
    return actual, estimate


# Events ------------------------------------------------------------------------------


Headline = tuple[int, str, float, float]  # (first published ns, text, actual, estimate)


@dataclass(frozen=True)
class EarningsEvent:
    """One 8-K Item 2.02 filing and the consensus headlines matched to it."""

    symbol: str
    accession: str
    acceptance_ns: int
    event_ns: int  # earliest of the acceptance and the first matched headline
    headlines: tuple[Headline, ...] = ()  # oldest first

    @property
    def headline_ns(self) -> int | None:
        return self.headlines[0][0] if self.headlines else None

    def known_at(self, ns: int) -> Headline | None:
        """The latest matched headline published before `ns`: corrections and the
        adjusted figure, when out by then, supersede an earlier headline."""
        known = [h for h in self.headlines if h[0] < ns]
        return known[-1] if known else None

    def beat_at(self, ns: int) -> bool | None:
        """Whether the EPS beat the consensus as known before `ns` (None: no headline)."""
        headline = self.known_at(ns)
        return None if headline is None else headline[2] > headline[3]


def earnings_events(filings: pa.Table, news: pa.Table, symbols: Sequence[str]):
    """One event per filing of `symbols`. A headline matches a filing when it is
    tagged with that stock and no other of `symbols`, was first published within a
    day of the acceptance, and parses."""
    universe = set(symbols)
    tagged: dict[str, list[tuple[int, int, str]]] = {s: [] for s in symbols}
    columns = (news.column(c).to_pylist() for c in ("created_ns", "id", "headline", "symbols"))
    for created, ident, headline, tags in zip(*columns, strict=True):
        mine = universe.intersection(tags or [])
        if len(mine) == 1 and parse_eps_headline(headline) is not None:
            tagged[mine.pop()].append((created, ident, headline))
    for rows in tagged.values():
        rows.sort()
    events = []
    for row in filings.to_pylist():
        if row["symbol"] not in universe:
            continue
        accepted = row["acceptance_ns"]
        rows = tagged[row["symbol"]]
        first = bisect_left(rows, (accepted - DAY_NS,))
        last = bisect_right(rows, (accepted + DAY_NS, sys.maxsize))
        matched = []
        for created, _, headline in rows[first:last]:
            parsed = parse_eps_headline(headline)
            assert parsed is not None  # only parsed headlines were kept
            matched.append((created, headline, *parsed))
        event_ns = min(accepted, matched[0][0]) if matched else accepted
        events.append(
            EarningsEvent(row["symbol"], row["accession"], accepted, event_ns, tuple(matched))
        )
    return sorted(events, key=lambda e: (e.event_ns, e.symbol, e.accession))


def first_open_after(ns: int, opens: Sequence[int]) -> int | None:
    """Index of the first session whose open is strictly after `ns`, so that news
    published at 09:30:00 is not traded in that morning's opening auction."""
    index = bisect_right(opens, ns)
    return index if index < len(opens) else None


def first_close_after(ns: int, closes: Sequence[int]) -> int | None:
    """Index of the first session whose close is strictly after `ns`."""
    index = bisect_right(closes, ns)
    return index if index < len(closes) else None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=Path("data/earnings"))
    args = parser.parse_args(argv)
    from .universe import UNIVERSE

    symbols = [s.ticker for s in UNIVERSE if s.kind == "stock" and not s.trade_only]
    client = EdgarClient(os.environ.get("SEC_EDGAR_USER_AGENT", ""))
    path = download_filings(symbols, args.root, client)
    table = read_filings(args.root)
    print(f"{table.num_rows} filings in {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
