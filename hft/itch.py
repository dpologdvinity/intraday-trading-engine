"""Nasdaq TotalView-ITCH 5.0: message codec, BinaryFILE framing and a reference order book.

This module is the plain Python reference for the C++ feed handler in `cpp/`
(`hftcore.ItchBook`): both must return identical statistics, snapshots, queues and
stock summaries for any byte stream, so a difference is a bug in one of them. It favours
clarity over speed (dicts and sorted lists, no tricks) and is used by the tests to build
synthetic streams; the benchmark day is only ever processed by the C++ engine.

Wire format (docs/specs/2026-11-13-itch-order-book-design.md): every integer is
big-endian and unsigned; each message starts with an 11-byte header (type 1, stock
locate 2, tracking number 2, timestamp 6) followed by the fields in `MESSAGE_FIELDS`.
BinaryFILE prefixes each message with a 2-byte length; a zero length ends the session.
"""

from bisect import bisect_left, insort
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any, NamedTuple

HEADER_LENGTH = 11
_LOCATES = 1 << 16

# Body fields after the header, in wire order: (name, size in bytes, kind).
# Kinds: "u" unsigned big-endian integer, "a" ASCII text padded with spaces,
# "c" a single character. Names match the members of the C++ structs in itch.hpp.
MESSAGE_FIELDS: dict[str, tuple[tuple[str, int, str], ...]] = {
    "S": (("event_code", 1, "c"),),
    "R": (
        ("stock", 8, "a"),
        ("market_category", 1, "c"),
        ("financial_status", 1, "c"),
        ("round_lot_size", 4, "u"),
        ("round_lots_only", 1, "c"),
        ("issue_classification", 1, "c"),
        ("issue_subtype", 2, "a"),
        ("authenticity", 1, "c"),
        ("short_sale_threshold", 1, "c"),
        ("ipo_flag", 1, "c"),
        ("luld_tier", 1, "c"),
        ("etp_flag", 1, "c"),
        ("etp_leverage", 4, "u"),
        ("inverse", 1, "c"),
    ),
    "H": (("stock", 8, "a"), ("state", 1, "c"), ("reserved", 1, "c"), ("reason", 4, "a")),
    "Y": (("stock", 8, "a"), ("action", 1, "c")),
    "L": (
        ("mpid", 4, "a"),
        ("stock", 8, "a"),
        ("primary", 1, "c"),
        ("mode", 1, "c"),
        ("state", 1, "c"),
    ),
    "V": (("level1", 8, "u"), ("level2", 8, "u"), ("level3", 8, "u")),
    "W": (("level", 1, "c"),),
    "K": (
        ("stock", 8, "a"),
        ("release_time", 4, "u"),
        ("release_qualifier", 1, "c"),
        ("price", 4, "u"),
    ),
    "J": (
        ("stock", 8, "a"),
        ("reference", 4, "u"),
        ("upper", 4, "u"),
        ("lower", 4, "u"),
        ("extension", 4, "u"),
    ),
    "h": (("stock", 8, "a"), ("market_code", 1, "c"), ("action", 1, "c")),
    "A": (
        ("ref", 8, "u"),
        ("side", 1, "c"),
        ("shares", 4, "u"),
        ("stock", 8, "a"),
        ("price", 4, "u"),
    ),
    "F": (
        ("ref", 8, "u"),
        ("side", 1, "c"),
        ("shares", 4, "u"),
        ("stock", 8, "a"),
        ("price", 4, "u"),
        ("attribution", 4, "a"),
    ),
    "E": (("ref", 8, "u"), ("shares", 4, "u"), ("match", 8, "u")),
    "C": (
        ("ref", 8, "u"),
        ("shares", 4, "u"),
        ("match", 8, "u"),
        ("printable", 1, "c"),
        ("price", 4, "u"),
    ),
    "X": (("ref", 8, "u"), ("shares", 4, "u")),
    "D": (("ref", 8, "u"),),
    "U": (("ref", 8, "u"), ("new_ref", 8, "u"), ("shares", 4, "u"), ("price", 4, "u")),
    "P": (
        ("ref", 8, "u"),
        ("side", 1, "c"),
        ("shares", 4, "u"),
        ("stock", 8, "a"),
        ("price", 4, "u"),
        ("match", 8, "u"),
    ),
    "Q": (
        ("shares", 8, "u"),
        ("stock", 8, "a"),
        ("price", 4, "u"),
        ("match", 8, "u"),
        ("cross_type", 1, "c"),
    ),
    "B": (("match", 8, "u"),),
    "I": (
        ("paired", 8, "u"),
        ("imbalance", 8, "u"),
        ("direction", 1, "c"),
        ("stock", 8, "a"),
        ("far_price", 4, "u"),
        ("near_price", 4, "u"),
        ("reference", 4, "u"),
        ("cross_type", 1, "c"),
        ("price_variation", 1, "c"),
    ),
    "N": (("stock", 8, "a"), ("interest", 1, "c")),
    "O": (
        ("stock", 8, "a"),
        ("open_eligibility", 1, "c"),
        ("min_price", 4, "u"),
        ("max_price", 4, "u"),
        ("near_price", 4, "u"),
        ("near_time", 8, "u"),
        ("lower_collar", 4, "u"),
        ("upper_collar", 4, "u"),
    ),
}

# Whole-message length (header included) of every type, derived from the field table.
MESSAGE_LENGTHS: dict[str, int] = {
    kind: HEADER_LENGTH + sum(size for _, size, _ in fields)
    for kind, fields in MESSAGE_FIELDS.items()
}

# The length table published in the design spec; the derived table must agree.
_SPEC_LENGTHS = {
    "S": 12, "R": 39, "H": 25, "Y": 20, "L": 26, "V": 35, "W": 12, "K": 28,
    "J": 35, "h": 21, "A": 36, "F": 40, "E": 31, "C": 36, "X": 23, "D": 19,
    "U": 35, "P": 44, "Q": 40, "B": 19, "I": 50, "N": 20, "O": 48,
}  # fmt: skip
assert MESSAGE_LENGTHS == _SPEC_LENGTHS


def _uint(value: int, size: int, name: str) -> bytes:
    if not isinstance(value, int) or not 0 <= value < 1 << (8 * size):
        raise ValueError(f"{name} must be an unsigned integer of {size} bytes, got {value!r}")
    return value.to_bytes(size, "big")


def _text(value: str, size: int, name: str) -> bytes:
    if not isinstance(value, str) or len(value) > size:
        raise ValueError(f"{name} must be text of at most {size} characters, got {value!r}")
    try:
        return value.ljust(size).encode("latin-1")
    except UnicodeEncodeError as error:
        raise ValueError(f"{name} must be single-byte text, got {value!r}") from error


def encode(
    type: str, *, locate: int = 0, tracking: int = 0, timestamp: int = 0, **fields: Any
) -> bytes:
    """One message body (header and fields, no length prefix). Missing fields are 0 or spaces."""
    layout = MESSAGE_FIELDS.get(type)
    if layout is None:
        raise ValueError(f"unknown ITCH message type {type!r}")
    extra = set(fields) - {name for name, _, _ in layout}
    if extra:
        raise ValueError(f"message {type!r} has no field(s) {sorted(extra)}")
    parts = [
        type.encode("ascii"),
        _uint(locate, 2, "locate"),
        _uint(tracking, 2, "tracking"),
        _uint(timestamp, 6, "timestamp"),
    ]
    for name, size, kind in layout:
        if kind == "u":
            parts.append(_uint(fields.get(name, 0), size, name))
        else:
            value = fields.get(name, " " * size)
            if kind == "c" and (not isinstance(value, str) or len(value) != 1):
                raise ValueError(f"{name} must be a single character, got {value!r}")
            parts.append(_text(value, size, name))
    return b"".join(parts)


def frame(messages: Iterable[bytes], *, end: bool = True) -> bytes:
    """BinaryFILE bytes: each message behind its 2-byte length, then the zero-length marker."""
    parts = []
    for body in messages:
        parts.append(len(body).to_bytes(2, "big"))
        parts.append(body)
    if end:
        parts.append(b"\x00\x00")
    return b"".join(parts)


def decode(body: bytes) -> dict[str, Any]:
    """Decodes one message body into a dict keyed by `type`, `locate`, `tracking`,
    `timestamp` and the field names. Text fields keep their space padding; an `A`
    message reports `attribution` as four spaces, as the C++ decoder does."""
    if not body:
        raise ValueError("empty message")
    kind = chr(body[0])
    layout = MESSAGE_FIELDS.get(kind)
    if layout is None:
        raise ValueError(f"unknown ITCH message type {kind!r}")
    if len(body) != MESSAGE_LENGTHS[kind]:
        raise ValueError(f"message {kind!r} is {len(body)} bytes, expected {MESSAGE_LENGTHS[kind]}")
    out: dict[str, Any] = {
        "type": kind,
        "locate": int.from_bytes(body[1:3], "big"),
        "tracking": int.from_bytes(body[3:5], "big"),
        "timestamp": int.from_bytes(body[5:11], "big"),
    }
    pos = HEADER_LENGTH
    for name, size, field_kind in layout:
        chunk = body[pos : pos + size]
        out[name] = int.from_bytes(chunk, "big") if field_kind == "u" else chunk.decode("latin-1")
        pos += size
    if kind == "A":
        out["attribution"] = "    "
    return out


@dataclass
class ReadStats:
    """Frame-level counters, with the semantics of `hftcore::itch::ReadStats`."""

    messages: int = 0  # every frame with length > 0 (known or not)
    by_type: dict[int, int] = field(default_factory=dict)  # type byte -> frames
    unknown_types: int = 0  # type not in the table: skipped by length
    malformed: int = 0  # known type, length differs from the table: skipped
    trailing_bytes: int = 0  # bytes seen after the end-of-session marker
    end_of_session: bool = False  # zero-length frame seen


def parse_binary_file(
    data: bytes, visitor: Callable[[dict[str, Any]], None], stats: ReadStats
) -> int:
    """Walks BinaryFILE bytes and calls `visitor(decode(body))` for each well-formed known
    message. Stops before an incomplete frame and returns the bytes consumed so the
    caller can carry the tail. A zero-length frame sets `end_of_session`; it and every
    later byte (in this and later calls) is consumed, the bytes after it counted as
    `trailing_bytes`."""
    if stats.end_of_session:
        stats.trailing_bytes += len(data)
        return len(data)
    pos = 0
    while len(data) - pos >= 2:
        length = int.from_bytes(data[pos : pos + 2], "big")
        if length == 0:
            stats.end_of_session = True
            pos += 2
            stats.trailing_bytes += len(data) - pos
            return len(data)
        if len(data) - pos - 2 < length:
            break  # incomplete frame: leave it for the caller
        body = data[pos + 2 : pos + 2 + length]
        stats.messages += 1
        stats.by_type[body[0]] = stats.by_type.get(body[0], 0) + 1
        expected = MESSAGE_LENGTHS.get(chr(body[0]))
        if expected is None:
            stats.unknown_types += 1
        elif expected != length:
            stats.malformed += 1
        else:
            visitor(decode(body))
        pos += 2 + length
    return pos


class Fill(NamedTuple):
    """Result of a book operation. When `result` is not ok/overfill the other fields are
    meaningless (the C++ book fills them with zeros); `shares` is what left the book."""

    result: str  # ok | unknown_ref | duplicate_ref | overfill | invalid
    locate: int
    side: str
    price: int
    shares: int


def _rejected(result: str) -> Fill:
    return Fill(result, 0, "B", 0, 0)


class ReferenceBook:
    """Full-depth order books for all stocks (docs spec "Order book"), written for clarity.

    Orders live in a dict `ref -> [locate, side, price, shares]`. Each `(locate, side)`
    has a dict `price -> [refs in FIFO order]` and an ascending sorted price list kept
    with `bisect`, so the best bid is the last price and the best ask the first."""

    def __init__(self) -> None:
        self.orders: dict[int, list[Any]] = {}
        self.levels: dict[tuple[int, str], dict[int, list[int]]] = {}
        self.prices: dict[tuple[int, str], list[int]] = {}
        self.level_count = 0
        self.peak_live = 0
        self.peak_levels = 0

    def _insert(self, ref: int, locate: int, side: str, shares: int, price: int) -> None:
        key = (locate, side)
        level = self.levels.setdefault(key, {})
        if price not in level:
            level[price] = []
            insort(self.prices.setdefault(key, []), price)
            self.level_count += 1
            self.peak_levels = max(self.peak_levels, self.level_count)
        level[price].append(ref)
        self.orders[ref] = [locate, side, price, shares]
        self.peak_live = max(self.peak_live, len(self.orders))

    def _erase(self, ref: int) -> None:
        locate, side, price, _ = self.orders.pop(ref)
        key = (locate, side)
        queue = self.levels[key][price]
        queue.remove(ref)
        if not queue:  # a level that loses its last order is erased
            del self.levels[key][price]
            prices = self.prices[key]
            del prices[bisect_left(prices, price)]
            self.level_count -= 1

    def add(self, ref: int, locate: int, side: str, shares: int, price: int) -> str:
        if shares == 0:
            return "invalid"
        if ref in self.orders:
            return "duplicate_ref"
        self._insert(ref, locate, side, shares, price)
        return "ok"

    def _reduce(self, ref: int, shares: int) -> Fill:
        order = self.orders.get(ref)
        if order is None:
            return _rejected("unknown_ref")
        locate, side, price, remaining = order
        if shares == 0:
            return Fill("ok", locate, side, price, 0)
        if shares < remaining:
            order[3] -= shares
            return Fill("ok", locate, side, price, shares)
        self._erase(ref)
        return Fill("ok" if shares == remaining else "overfill", locate, side, price, remaining)

    def execute(self, ref: int, shares: int) -> Fill:
        return self._reduce(ref, shares)

    def cancel(self, ref: int, shares: int) -> Fill:
        return self._reduce(ref, shares)

    def remove(self, ref: int) -> Fill:
        order = self.orders.get(ref)
        if order is None:
            return _rejected("unknown_ref")
        locate, side, price, shares = order
        self._erase(ref)
        return Fill("ok", locate, side, price, shares)

    def replace(self, ref: int, new_ref: int, shares: int, price: int) -> Fill:
        order = self.orders.get(ref)
        if order is None:
            return _rejected("unknown_ref")
        if shares == 0:
            return _rejected("invalid")
        if new_ref != ref and new_ref in self.orders:
            return _rejected("duplicate_ref")
        locate, side = order[0], order[1]
        self._erase(ref)
        self._insert(new_ref, locate, side, shares, price)
        return Fill("ok", locate, side, price, shares)

    def depth(self, locate: int, side: str, n: int) -> list[tuple[int, int, int]]:
        """(price, shares, orders) for up to n levels, best first."""
        prices = self.prices.get((locate, side), [])
        best_first = reversed(prices) if side == "B" else iter(prices)
        out = []
        for price in best_first:
            if len(out) >= n:
                break
            refs = self.levels[(locate, side)][price]
            out.append((price, sum(self.orders[r][3] for r in refs), len(refs)))
        return out

    def queue(self, locate: int, side: str, price: int) -> list[tuple[int, int]]:
        refs = self.levels.get((locate, side), {}).get(price, [])
        return [(ref, self.orders[ref][3]) for ref in refs]

    def crossed(self, locate: int) -> bool:
        bids, asks = self.prices.get((locate, "B")), self.prices.get((locate, "S"))
        return bool(bids) and bool(asks) and bids[-1] >= asks[0]

    def locates(self) -> list[int]:
        return sorted({locate for (locate, _), level in self.levels.items() if level})

    def validate(self) -> str:
        """ "" when every structural invariant holds, else a description of the first break."""
        queued = 0
        for key, level in self.levels.items():
            if sorted(level) != self.prices.get(key, []):
                return f"price list differs from the levels at {key}"
            for price, refs in level.items():
                if not refs:
                    return f"empty level at {key}"
                if len(set(refs)) != len(refs):
                    return f"duplicate ref in a queue at {key}"
                for ref in refs:
                    order = self.orders.get(ref)
                    if order is None or order[:3] != [key[0], key[1], price] or order[3] <= 0:
                        return f"order {ref} filed under the wrong level"
                queued += len(refs)
        if queued != len(self.orders):
            return "live orders differ from the queued orders"
        if self.level_count != sum(len(level) for level in self.levels.values()):
            return "level count differs from the levels"
        return ""


class ItchBook:
    """Reference feed handler: the shared interface of `hftcore.ItchBook`.

    `symbols=None` or an empty collection books every stock; otherwise only stocks
    whose latest directory (`R`) symbol is listed. Counters and print rules follow the
    spec's feed-handler rules: executions and cancels act on the stored order's locate,
    the symbol filter always checks the message's header locate."""

    def __init__(self, symbols: Iterable[str] | None = None) -> None:
        if isinstance(symbols, str | bytes):
            raise TypeError("symbols must be a collection of symbols, not a string")
        names = frozenset(symbols or ())
        for name in names:
            _text(name, len(name), "symbol")  # must be single-byte text, like the wire
        self._filter = names or None
        self._allowed: dict[int, bool] = {}  # locate -> allowed by the latest R (default: all)
        self._stocks: dict[int, dict[str, Any]] = {}
        self._book = ReferenceBook()
        self._read = ReadStats()
        self._tail = b""
        self._truncated = 0
        self._counts = {
            "unknown_refs": 0,
            "duplicate_refs": 0,
            "overfills": 0,
            "invalid": 0,
            "crossed": 0,
            "filtered": 0,
        }
        self._handlers: dict[str, Callable[[dict[str, Any]], None]] = {
            "R": self._on_directory,
            "H": self._on_trading_action,
            "A": self._on_add,
            "F": self._on_add,
            "E": self._on_executed,
            "C": self._on_executed_with_price,
            "X": self._on_cancel,
            "D": self._on_delete,
            "U": self._on_replace,
            "P": self._on_trade,
            "Q": self._on_cross,
        }

    # --- input -------------------------------------------------------------

    def feed(self, data: bytes) -> None:
        """Consumes BinaryFILE bytes; an incomplete frame is carried to the next call."""
        buffer = self._tail + bytes(data)
        used = parse_binary_file(buffer, self._dispatch, self._read)
        self._tail = buffer[used:]

    def finish(self) -> None:
        """Marks the end of input: `truncated_bytes` is the carried tail (left in place)."""
        self._truncated = len(self._tail)

    def _dispatch(self, message: dict[str, Any]) -> None:
        handler = self._handlers.get(message["type"])
        if handler is not None:
            handler(message)

    # --- helpers -----------------------------------------------------------

    def _stock(self, locate: int) -> dict[str, Any]:
        info = self._stocks.get(locate)
        if info is None:
            info = self._stocks[locate] = {
                "symbol": "",
                "trading_state": " ",
                "trades": 0,
                "volume": 0,
                "last_price": 0,
                "open_cross": None,
                "close_cross": None,
            }
        return info

    def _admit(self, locate: int) -> bool:
        """The header-locate filter: counts and rejects messages for filtered stocks."""
        if self._allowed.get(locate, self._filter is None):
            return True
        self._counts["filtered"] += 1
        return False

    def _tally(self, result: str) -> bool:
        """Counts a rejected result; True when the book changed (ok or overfill)."""
        if result == "ok":
            return True
        if result == "overfill":
            self._counts["overfills"] += 1
            return True
        self._counts[
            {"unknown_ref": "unknown_refs", "duplicate_ref": "duplicate_refs"}.get(
                result, "invalid"
            )
        ] += 1
        return False

    def _print(self, locate: int, shares: int, price: int) -> None:
        info = self._stock(locate)
        info["trades"] += 1
        info["volume"] += shares
        info["last_price"] = price

    def _check_crossed(self, locate: int) -> None:
        if self._book.crossed(locate):
            self._counts["crossed"] += 1

    # --- message handlers ---------------------------------------------------

    def _on_directory(self, m: dict[str, Any]) -> None:
        symbol = m["stock"].rstrip(" ")
        self._stock(m["locate"])["symbol"] = symbol
        self._allowed[m["locate"]] = self._filter is None or symbol in self._filter

    def _on_trading_action(self, m: dict[str, Any]) -> None:
        self._stock(m["locate"])["trading_state"] = m["state"]

    def _on_add(self, m: dict[str, Any]) -> None:
        locate = m["locate"]
        if not self._admit(locate):
            return
        if m["side"] not in ("B", "S"):  # the book only knows buy and sell
            self._counts["invalid"] += 1
            return
        if self._tally(self._book.add(m["ref"], locate, m["side"], m["shares"], m["price"])):
            self._check_crossed(locate)

    def _on_executed(self, m: dict[str, Any]) -> None:
        if not self._admit(m["locate"]):
            return
        fill = self._book.execute(m["ref"], m["shares"])
        if self._tally(fill.result) and fill.shares > 0:
            self._print(fill.locate, fill.shares, fill.price)

    def _on_executed_with_price(self, m: dict[str, Any]) -> None:
        if not self._admit(m["locate"]):
            return
        fill = self._book.execute(m["ref"], m["shares"])
        if self._tally(fill.result) and m["printable"] == "Y" and fill.shares > 0:
            self._print(fill.locate, fill.shares, m["price"])

    def _on_cancel(self, m: dict[str, Any]) -> None:
        if self._admit(m["locate"]):
            self._tally(self._book.cancel(m["ref"], m["shares"]).result)

    def _on_delete(self, m: dict[str, Any]) -> None:
        if self._admit(m["locate"]):
            self._tally(self._book.remove(m["ref"]).result)

    def _on_replace(self, m: dict[str, Any]) -> None:
        if not self._admit(m["locate"]):
            return
        fill = self._book.replace(m["ref"], m["new_ref"], m["shares"], m["price"])
        if self._tally(fill.result):
            self._check_crossed(fill.locate)  # the stored locate, not the header's

    def _on_trade(self, m: dict[str, Any]) -> None:
        if self._admit(m["locate"]):
            self._print(m["locate"], m["shares"], m["price"])

    def _on_cross(self, m: dict[str, Any]) -> None:
        if not self._admit(m["locate"]) or m["shares"] == 0:
            return
        self._print(m["locate"], m["shares"], m["price"])
        info = self._stock(m["locate"])
        if m["cross_type"] == "O":
            info["open_cross"] = m["price"]
        elif m["cross_type"] == "C":
            info["close_cross"] = m["price"]

    # --- queries ------------------------------------------------------------

    @staticmethod
    def _int(value: int, what: str) -> int:
        if not isinstance(value, int):
            raise TypeError(f"{what} must be an integer")
        return value

    @classmethod
    def _locate(cls, locate: int) -> int:
        if not 0 <= cls._int(locate, "locate") < _LOCATES:
            raise ValueError(f"locate must be in 0..{_LOCATES - 1}, got {locate!r}")
        return locate

    def snapshot(self, locate: int, depth: int) -> dict[str, list[tuple[int, int, int]]]:
        """{"bids": [(price, shares, orders), ...], "asks": [...]}, best first."""
        self._locate(locate)
        if self._int(depth, "depth") < 0:
            raise ValueError("depth must not be negative")
        return {
            "bids": self._book.depth(locate, "B", depth),
            "asks": self._book.depth(locate, "S", depth),
        }

    def queue(self, locate: int, side: str, price: int) -> list[tuple[int, int]]:
        """[(ref, shares), ...] in FIFO order; side is "B" or "S"."""
        self._locate(locate)
        if side not in ("B", "S"):
            raise ValueError(f"side must be 'B' or 'S', got {side!r}")
        return self._book.queue(locate, side, self._int(price, "price"))

    def stock(self, locate: int) -> dict[str, Any]:
        """symbol, trading_state, trades, volume, last_price, open_cross, close_cross
        (None when unset). A locate never seen reports the defaults."""
        self._locate(locate)
        return dict(self._stock(locate))

    def locates(self) -> list[int]:
        """Sorted locates that have at least one live order."""
        return self._book.locates()

    def stats(self) -> dict[str, Any]:
        read = self._read
        return {
            "messages": read.messages,
            "by_type": {chr(code): count for code, count in read.by_type.items() if count},
            "unknown_types": read.unknown_types,
            "malformed": read.malformed,
            "trailing_bytes": read.trailing_bytes,
            "end_of_session": read.end_of_session,
            "truncated_bytes": self._truncated,
            **self._counts,
            "live_orders": len(self._book.orders),
            "peak_live_orders": self._book.peak_live,
            "levels": self._book.level_count,
            "peak_levels": self._book.peak_levels,
        }

    def validate(self) -> str:
        """ "" when the book is internally consistent."""
        return self._book.validate()
