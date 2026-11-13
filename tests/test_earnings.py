"""EDGAR 8-K Item 2.02 filings, EPS headline parsing and announcement timing."""

import json
import urllib.error
from datetime import UTC, datetime
from email.message import Message

import pyarrow as pa
import pytest

from hft.ml.earnings import (
    SCHEMA,
    EdgarClient,
    EdgarError,
    acceptance_ns,
    cik_map,
    download_filings,
    earnings_events,
    earnings_filings,
    first_close_after,
    first_open_after,
    parse_eps_headline,
    read_filings,
)
from hft.ml.news import SCHEMA as NEWS_SCHEMA

AGENT = "Research test@example.com"


def _ns(text):
    return int(datetime.fromisoformat(text).timestamp()) * 10**9


def _block(*rows):
    """Columnar submissions block from (accession, form, items, accepted, filed)."""
    keys = ("accessionNumber", "form", "items", "acceptanceDateTime", "filingDate")
    return {k: [r[i] for r in rows] for i, k in enumerate(keys)}


RECENT = _block(
    ("0001-24-1", "8-K", "2.02,9.01", "2024-08-01T20:30:41.000Z", "2024-08-01"),
    ("0001-24-2", "8-K", "5.07", "2024-03-01T21:00:00.000Z", "2024-03-01"),
    ("0001-24-3", "8-K/A", "2.02", "2024-08-05T21:00:00.000Z", "2024-08-05"),
    ("0001-24-4", "10-Q", "", "2024-08-02T10:00:00.000Z", "2024-08-02"),
)
OLDER = _block(
    ("0001-16-1", "8-K", "2.02, 9.01", "2016-01-26T21:31:00.000Z", "2016-01-26"),
    ("0001-15-1", "8-K", "2.02", "2015-10-27T20:30:00.000Z", "2015-10-27"),  # too early
)


class FakeEdgar:
    def __init__(self):
        self.calls = []
        self.payloads = {
            "https://www.sec.gov/files/company_tickers.json": {
                "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
            },
            "https://data.sec.gov/submissions/CIK0000320193.json": {
                "filings": {
                    "recent": RECENT,
                    "files": [
                        {"name": "CIK0000320193-submissions-001.json", "filingTo": "2016-12-31"},
                        {"name": "CIK0000320193-submissions-002.json", "filingTo": "2009-12-31"},
                    ],
                }
            },
            "https://data.sec.gov/submissions/CIK0000320193-submissions-001.json": OLDER,
        }

    def get(self, url):
        self.calls.append(url)
        return self.payloads[url]


def test_acceptance_times_are_utc():
    assert acceptance_ns("2024-08-01T20:30:41.000Z") == _ns("2024-08-01T20:30:41+00:00")
    with pytest.raises(ValueError):
        acceptance_ns("2024-08-01T20:30:41")


def test_only_8k_item_202_filings_from_the_first_filing_date_are_kept():
    rows = earnings_filings("AAPL", 320193, [RECENT, OLDER])
    assert [r["accession"] for r in rows] == ["0001-16-1", "0001-24-1"]
    assert rows[1]["acceptance_ns"] == _ns("2024-08-01T20:30:41+00:00")


def test_download_follows_pages_skips_old_ones_and_keeps_the_cache(tmp_path):
    client = FakeEdgar()
    path = download_filings(["AAPL", "ZZZZ"], tmp_path, client)
    assert "CIK0000320193-submissions-002.json" not in " ".join(client.calls)
    table = read_filings(tmp_path)
    assert table.column("accession").to_pylist() == ["0001-16-1", "0001-24-1"]
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["missing"] == {"ZZZZ": "no CIK for this ticker"}
    calls = len(client.calls)
    assert download_filings(["AAPL"], tmp_path, client) == path
    assert len(client.calls) == calls  # a single download
    path.write_bytes(path.read_bytes() + b"x")
    with pytest.raises(ValueError, match="manifest"):
        read_filings(tmp_path)


def test_client_refuses_other_hosts_needs_a_contact_and_paces_requests():
    with pytest.raises(EdgarError):
        EdgarClient("anonymous")
    now, slept = [0.0], []

    def sleep(seconds):
        slept.append(seconds)
        now[0] += seconds

    client = EdgarClient(AGENT, transport=lambda u, h: b"{}", clock=lambda: now[0], sleep=sleep)
    with pytest.raises(EdgarError):
        client.get("https://example.com/x.json")
    client.get("https://data.sec.gov/a.json")
    client.get("https://data.sec.gov/b.json")
    assert slept == [pytest.approx(0.2)]  # five requests a second


def test_client_backs_off_on_rate_limit_then_gives_up_on_other_errors():
    answers = [429, None]

    def transport(url, headers):
        assert headers["User-Agent"] == AGENT
        code = answers.pop(0)
        if code:
            raise urllib.error.HTTPError(url, code, "busy", Message(), None)
        return b'{"ok": 1}'

    client = EdgarClient(AGENT, transport=transport, sleep=lambda s: None)
    assert client.get("https://data.sec.gov/a.json") == {"ok": 1}
    answers[:] = [404]
    with pytest.raises(EdgarError, match="404"):
        client.get("https://data.sec.gov/a.json")


def test_cik_map_upper_cases_tickers():
    assert cik_map({"0": {"cik_str": 1, "ticker": "brk-b"}}) == {"BRK-B": 1}


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Oracle Q2 EPS $1.21 Beats $1.18 Estimate", (1.21, 1.18)),
        ("Apple Q2 2024 Adj. EPS $1.53 Beats $1.50 Estimate, Sales $90.75B Beat", (1.53, 1.5)),
        ("Boeing Q2 Core EPS $(0.37) Misses $(0.16) Estimate, Sales $16.68B", (-0.37, -0.16)),
        ("Intel Q2 Adj. EPS -$0.10 Misses $0.01 Estimate", (-0.1, 0.01)),
        ("Ford Q1 EPS Loss $0.23 Misses $0.12 Estimate", (-0.23, 0.12)),
        (
            "Coca-Cola Q1 Adj. EPS $0.47 Beats $0.46 Est., Sales $7.6B Beats $7.3B Est.",
            (0.47, 0.46),
        ),
        ("Netflix Q2 EPS $2.97 vs $3.15 Estimate, Sales $7.34B", (2.97, 3.15)),
        ("AMD Reports Q4 Adj. EPS $(0.10), Inline, Sales $958M vs $954.7M Est.", (-0.1, -0.1)),
        ("UPDATE: Occidental Petroleum Q1 EPS $(0.15) Beats $(0.33) Estimate", (-0.15, -0.33)),
        # Rejected: guidance, a GAAP figure against an adjusted consensus, prose, and a
        # verb that contradicts the numbers.
        ("Micron Raises Q2 Adj. EPS From $0.68-$0.82 To $0.93-$0.98 vs $0.79 Estimate", None),
        ("Qualcomm Sees Q3 Adj. EPS $2.15-$2.35 Vs $2.17 Estimate", None),
        ("Amazon.com Q2 EPS $5.22 May Not Compare To $5.58 Estimate", None),
        ("Tesla Q3 Adj. EPS $2.90 Does Not Compare To $(0.19) Estimate", None),
        ("Ford Motor Q3 Adj. EPS $0.65 Up From $0.34 YoY, Sales $37.50B Beat $32.86B", None),
        ("Adobe Q1 Earnings: Revenue Beat, EPS Beat, Soft Guidance", None),
        ("Acme Q1 EPS $1.00 Beats $1.10 Estimate", None),
        ("Acme Q1 EPS $1.00 Misses $1.00 Estimate", None),
        ("Morgan Stanley Q4 EPS $0.80 Misses $0.92 Estimate, Sales $8.5B, May Not Compare", None),
        ("Hearing Bank of America Q1 EPS $0.21 vs $0.21 est", None),
    ],
)
def test_parse_eps_headline(text, expected):
    assert parse_eps_headline(text) == expected


def _filings(*rows):
    return pa.Table.from_pylist(
        [
            {
                "symbol": s,
                "cik": 1,
                "accession": a,
                "form": "8-K",
                "items": "2.02",
                "acceptance_ns": ns,
                "filing_date": "",
            }
            for s, a, ns in rows
        ],
        schema=SCHEMA,
    )


def _news(*rows):
    return pa.Table.from_pylist(
        [
            {
                "id": i,
                "created_ns": ns,
                "updated_ns": ns,
                "headline": h,
                "summary": "",
                "source": "benzinga",
                "symbols": list(tags),
            }
            for i, (ns, h, tags) in enumerate(rows)
        ],
        schema=NEWS_SCHEMA,
    )


def test_events_keep_every_parsed_headline_of_that_stock_within_a_day():
    accepted = _ns("2024-08-01T20:31:00+00:00")
    filings = _filings(("AAPL", "a1", accepted), ("MSFT", "m1", accepted))
    news = _news(
        (accepted - 2 * 86_400 * 10**9, "Apple Q2 EPS $1.00 Beats $0.90 Estimate", ["AAPL"]),
        (accepted + 3 * 10**9, "Apple Q3 Adj. EPS $1.40 Misses $1.45 Estimate", ["AAPL"]),
        (accepted - 60 * 10**9, "Apple Q3 EPS $1.40 Beats $1.35 Estimate", ["AAPL"]),
        (accepted - 90 * 10**9, "Apple Q3 earnings are out", ["AAPL"]),  # not parsed
        (accepted, "Microsoft Q4 EPS $2.95 Beats $2.93 Estimate", ["MSFT", "AAPL"]),
    )
    events = earnings_events(filings, news, ["AAPL", "MSFT"])
    apple = next(e for e in events if e.symbol == "AAPL")
    assert [h[0] for h in apple.headlines] == [accepted - 60 * 10**9, accepted + 3 * 10**9]
    assert apple.headline_ns == accepted - 60 * 10**9
    assert apple.event_ns == apple.headline_ns  # the earlier of headline and filing
    microsoft = next(e for e in events if e.symbol == "MSFT")
    assert microsoft.headlines == ()  # tagged with two universe stocks: not matched
    assert microsoft.event_ns == accepted and microsoft.beat_at(accepted + 1) is None


def test_a_correction_published_before_the_entry_decides_beat_or_miss():
    # Intel, 2024-10-31: "Adj EPS $0.12 Beats $(0.02)" at 16:03, corrected at 17:51 to
    # "Adj EPS $(0.46) Misses $(0.02)". Disney, 2024-02-07: "Inline", then a beat.
    accepted = _ns("2024-10-31T20:05:00+00:00")
    first, correction = accepted - 2 * 60 * 10**9, accepted + 106 * 60 * 10**9
    news = _news(
        (first, "Intel Q3 Adj EPS $0.12 Beats $(0.02) Estimate", ["INTC"]),
        (correction, "CORRECTION: Intel Q3 Adj EPS $(0.46) Misses $(0.02) Estimate", ["INTC"]),
        (accepted, "Walt Disney Q1 Adj. EPS $1.22, Inline", ["DIS"]),
        (accepted + 600 * 10**9, "Walt Disney Q1 Adj. EPS $1.22 Beats $0.99 Estimate", ["DIS"]),
    )
    filings = _filings(("INTC", "i1", accepted), ("DIS", "d1", accepted))
    intel, disney = sorted(earnings_events(filings, news, ["INTC", "DIS"]), key=lambda e: e.symbol)[
        ::-1
    ]
    next_open = _ns("2024-11-01T13:30:00+00:00")
    assert intel.beat_at(first + 1) is True
    assert intel.beat_at(next_open) is False  # the correction was public by the open
    assert intel.known_at(next_open)[0] == correction
    assert intel.beat_at(correction) is True  # strictly before: not yet known
    assert disney.beat_at(accepted + 1) is False and disney.beat_at(next_open) is True


def test_entry_after_a_pre_open_report_is_that_morning_after_the_close_the_next():
    opens = [_ns("2024-08-01T13:30:00+00:00"), _ns("2024-08-02T13:30:00+00:00")]
    closes = [_ns("2024-08-01T20:00:00+00:00"), _ns("2024-08-02T20:00:00+00:00")]
    pre_open = _ns("2024-08-01T10:45:00+00:00")  # 06:45 New York
    after_close = _ns("2024-08-01T20:30:00+00:00")  # 16:30 New York
    assert first_open_after(pre_open, opens) == 0
    assert first_open_after(opens[0], opens) == 1  # at 09:30:00: too late for the auction
    assert first_open_after(after_close, opens) == 1
    assert first_open_after(closes[1], opens) is None
    # Day 0 is the first session whose close is after the announcement.
    assert first_close_after(pre_open, closes) == 0
    assert first_close_after(closes[0], closes) == 1  # at the close: the next session
    assert first_close_after(after_close, closes) == 1
    assert datetime.fromtimestamp(after_close / 1e9, UTC).hour == 20
