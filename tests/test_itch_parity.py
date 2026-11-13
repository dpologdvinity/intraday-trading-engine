"""The C++ ITCH feed handler (`hftcore.ItchBook`) must equal the Python reference exactly.

Seeded random BinaryFILE streams exercise every message the book reacts to, including the
awkward cases (duplicate and unknown refs, overfills, zero shares, bad sides, replaces
that reuse or collide with refs, executions whose header locate differs from the stored
order's, directory messages that change the symbol filter, unknown and malformed frames,
an end-of-session marker and a truncated tail). Both implementations are fed the same
random chunks and must agree on statistics, snapshots, per-level FIFO queues and stock
summaries, with and without a symbol filter.
"""

import importlib
import os
import random
from functools import cache
from itertools import accumulate

import pytest

from hft import itch

LOCATES = (1, 2, 3, 4, 5, 300)
UNLISTED = 7000  # a locate that never gets a directory message
SYMBOLS = {1: "AAA", 2: "BBB", 3: "CCC", 4: "DDD", 5: "EEE", 300: "FFF"}
FILTER = ["AAA", "CCC", "FFF", "NOPE"]  # NOPE never appears in a directory message
MESSAGES = 16_000
SEEDS = range(20)
CHECK_LOCATES = (0, *LOCATES, UNLISTED, 65535)
UNKNOWN_TYPES = [chr(b) for b in range(256) if chr(b) not in itch.MESSAGE_LENGTHS]
OTHER_TYPES = [t for t in itch.MESSAGE_FIELDS if t not in "RHAFECXDUPQB"]
TICK = 100


@pytest.fixture(scope="module")
def hftcore():
    """The native module; skipped when missing or too old to have ItchBook, unless
    HFT_REQUIRE_HFTCORE=1 (CI), which turns the skip into a failure."""
    try:
        module = importlib.import_module("hftcore")
        if not hasattr(module, "ItchBook"):
            raise ImportError("the installed hftcore predates ItchBook")
        return module
    except ImportError:
        if os.environ.get("HFT_REQUIRE_HFTCORE") == "1":
            raise
        pytest.skip("hftcore with ItchBook is not installed")


class Generator:
    """Builds a random stream while tracking roughly which refs are live (so most
    targets are valid); the model may drift from the real book, which is fine: refs
    that are not live are just more unknown-ref cases."""

    def __init__(self, rng: random.Random) -> None:
        self.rng = rng
        self.live: dict[int, list[int]] = {}  # ref -> [locate, remaining shares]
        self.refs: list[int] = []  # live refs, for O(1) random choice
        self.position: dict[int, int] = {}
        self.next_ref = 1
        self.timestamp = 0

    # --- model of live orders -------------------------------------------------

    def _track(self, ref: int, locate: int, shares: int) -> None:
        if ref in self.live or shares == 0:
            return
        self.live[ref] = [locate, shares]
        self.position[ref] = len(self.refs)
        self.refs.append(ref)

    def _untrack(self, ref: int) -> None:
        if ref not in self.live:
            return
        index = self.position.pop(ref)
        last = self.refs.pop()
        if last != ref:
            self.refs[index] = last
            self.position[last] = index
        del self.live[ref]

    def _reduce(self, ref: int, shares: int) -> None:
        if ref in self.live:
            if shares >= self.live[ref][1]:
                self._untrack(ref)
            else:
                self.live[ref][1] -= shares

    # --- random pieces ----------------------------------------------------------

    def _fresh_ref(self) -> int:
        self.next_ref += self.rng.randint(1, 3)  # sparse, like the real feed
        return self.next_ref

    def _target(self) -> int:
        """A live ref most of the time; sometimes a dead or never-used one."""
        if self.refs and self.rng.random() < 0.9:
            return self.rng.choice(self.refs)
        return self.rng.randint(1, self.next_ref + 5)

    def _header_locate(self, ref: int) -> int:
        """Usually the stored locate; sometimes another one (listed, filtered or unlisted)."""
        if ref in self.live and self.rng.random() < 0.75:
            return self.live[ref][0]
        return self.rng.choice((*LOCATES, UNLISTED))

    def _quantity(self, ref: int) -> int:
        remaining = self.live[ref][1] if ref in self.live else self.rng.randint(1, 300)
        roll = self.rng.random()
        if roll < 0.03:
            return 0
        if roll < 0.33:
            return remaining
        if roll < 0.88:
            return self.rng.randint(1, max(1, remaining))
        return remaining + self.rng.randint(1, 50)  # overfill

    def _price(self, locate: int, side: str) -> int:
        mid = 10_000 + 500 * (locate % 7)
        offset = TICK * self.rng.randint(-2, 12)  # negative offsets cross the book
        return mid - offset if side == "B" else mid + offset

    def _stamp(self) -> int:
        self.timestamp += self.rng.randint(0, 5000)
        return self.timestamp

    def _msg(self, type_: str, locate: int, **fields) -> bytes:
        body = itch.encode(type_, locate=locate, timestamp=self._stamp(), **fields)
        return itch.frame([body], end=False)

    # --- message kinds -------------------------------------------------------------

    def directory(self) -> bytes:
        locate = self.rng.choice(LOCATES)
        symbol = SYMBOLS[locate] if self.rng.random() < 0.85 else self.rng.choice(("QQQ", "ZZ"))
        if self.rng.random() < 0.3:
            return self._msg("H", locate, stock=symbol, state=self.rng.choice("HPQT"))
        return self._msg("R", locate, stock=symbol, round_lot_size=100)

    def add(self) -> bytes:
        locate = self.rng.choice((*LOCATES, *LOCATES, UNLISTED))
        side = self.rng.choice("BS") if self.rng.random() < 0.993 else self.rng.choice("X ?")
        shares = 0 if self.rng.random() < 0.01 else self.rng.randint(1, 300)
        roll = self.rng.random()
        if roll < 0.02 and self.refs:
            ref = self.rng.choice(self.refs)  # duplicate of a live ref
        elif roll < 0.03:
            ref = self.rng.randint(1, self.next_ref)  # maybe a dead ref (legal reuse)
        else:
            ref = self._fresh_ref()
        price = self._price(locate, side if side in "BS" else "B")
        type_ = "F" if self.rng.random() < 0.25 else "A"
        extra = {"attribution": "MPID"} if type_ == "F" else {}
        message = self._msg(
            type_, locate, ref=ref, side=side, shares=shares, price=price, stock="XYZ", **extra
        )
        if side in "BS" and ref not in self.live:
            self._track(ref, locate, shares)
        return message

    def execute(self, with_price: bool) -> bytes:
        ref = self._target()
        locate = self._header_locate(ref)
        shares = self._quantity(ref)
        match = self.rng.randint(1, 10**9)
        if with_price:
            message = self._msg(
                "C",
                locate,
                ref=ref,
                shares=shares,
                match=match,
                printable=self.rng.choice("YYN"),
                price=self.rng.randint(1, 30_000),
            )
        else:
            message = self._msg("E", locate, ref=ref, shares=shares, match=match)
        self._reduce(ref, shares)
        return message

    def cancel(self) -> bytes:
        ref = self._target()
        shares = self._quantity(ref)
        message = self._msg("X", self._header_locate(ref), ref=ref, shares=shares)
        self._reduce(ref, shares)
        return message

    def delete(self) -> bytes:
        ref = self._target()
        message = self._msg("D", self._header_locate(ref), ref=ref)
        self._untrack(ref)
        return message

    def replace(self) -> bytes:
        ref = self._target()
        roll = self.rng.random()
        if roll < 0.03 and self.refs:
            new_ref = self.rng.choice(self.refs)  # collides with a live order
        elif roll < 0.05:
            new_ref = ref  # reuses its own ref
        else:
            new_ref = self._fresh_ref()
        shares = 0 if self.rng.random() < 0.02 else self.rng.randint(1, 300)
        stored = self.live.get(ref)
        side = self.rng.choice("BS")
        price = self._price(stored[0] if stored else 1, side)
        message = self._msg(
            "U", self._header_locate(ref), ref=ref, new_ref=new_ref, shares=shares, price=price
        )
        if stored and shares and (new_ref == ref or new_ref not in self.live):
            self._untrack(ref)
            self._track(new_ref, stored[0], shares)
        return message

    def trade(self) -> bytes:
        locate = self.rng.choice((*LOCATES, UNLISTED))
        return self._msg(
            "P",
            locate,
            ref=0,
            side=self.rng.choice("BS"),
            shares=self.rng.randint(0, 500),
            stock=SYMBOLS.get(locate, "UNL"),
            price=self.rng.randint(1, 30_000),
            match=self.rng.randint(1, 10**9),
        )

    def cross(self) -> bytes:
        locate = self.rng.choice((*LOCATES, UNLISTED))
        return self._msg(
            "Q",
            locate,
            shares=0 if self.rng.random() < 0.15 else self.rng.randint(1, 10**6),
            stock=SYMBOLS.get(locate, "UNL"),
            price=self.rng.randint(1, 30_000),
            match=self.rng.randint(1, 10**9),
            cross_type=self.rng.choice("OOCCHIE"),
        )

    def other(self) -> bytes:
        return self._msg(self.rng.choice(OTHER_TYPES + ["B"]), self.rng.choice(LOCATES))

    def unknown(self) -> bytes:
        body = self.rng.choice(UNKNOWN_TYPES).encode("latin-1") + self.rng.randbytes(
            self.rng.randint(0, 40)
        )
        return len(body).to_bytes(2, "big") + body

    def malformed(self) -> bytes:
        kind = self.rng.choice(list(itch.MESSAGE_LENGTHS))
        length = self.rng.choice([n for n in range(1, 60) if n != itch.MESSAGE_LENGTHS[kind]])
        body = kind.encode() + self.rng.randbytes(length - 1)
        return length.to_bytes(2, "big") + body

    def stream(self, count: int) -> list[bytes]:
        frames = [self._msg("R", locate, stock=SYMBOLS[locate]) for locate in LOCATES]
        frames.append(self._msg("H", 2, stock="BBB", state="T"))
        kinds = (
            (self.add, 34),
            (lambda: self.execute(False), 13),
            (lambda: self.execute(True), 6),
            (self.cancel, 7),
            (self.delete, 7),
            (self.replace, 8),
            (self.trade, 3),
            (self.cross, 2),
            (self.other, 3),
            (self.directory, 2),
            (self.unknown, 1),
            (self.malformed, 1),
        )
        makers = [maker for maker, _ in kinds]
        cumulative = list(accumulate(weight for _, weight in kinds))
        while len(frames) < count:
            frames.append(self.rng.choices(makers, cum_weights=cumulative)[0]())
        return frames


@cache
def build(seed: int) -> bytes:
    """The seed's byte stream. Every fourth seed ends the session with trailing junk and
    the next fourth ends in a truncated frame; the rest end cleanly."""
    rng = random.Random(seed)
    frames = Generator(rng).stream(MESSAGES)
    data = b"".join(frames)
    if seed % 4 == 1:
        data += b"\x00\x00" + rng.randbytes(rng.randint(0, 30))
    elif seed % 4 == 2:
        data += frames[-1][: rng.randint(1, len(frames[-1]) - 1)]
    return data


def chunks(data: bytes, rng: random.Random):
    pos = 0
    while pos < len(data):
        size = rng.choice((rng.randint(1, 5), rng.randint(6, 200), rng.randint(200, 6000)))
        yield data[pos : pos + size]
        pos += size


def assert_same(python, native) -> None:
    assert native.stats() == python.stats()
    assert native.locates() == python.locates()
    assert native.validate() == python.validate() == ""
    for locate in CHECK_LOCATES:
        expected = python.snapshot(locate, 1000)
        assert native.snapshot(locate, 1000) == expected, locate
        assert native.snapshot(locate, 2) == python.snapshot(locate, 2)
        for side, key in (("B", "bids"), ("S", "asks")):
            for price, _, _ in expected[key]:
                assert native.queue(locate, side, price) == python.queue(locate, side, price)
            assert native.queue(locate, side, 1) == python.queue(locate, side, 1) == []
        assert native.stock(locate) == python.stock(locate), locate


@pytest.mark.parametrize("seed", SEEDS)
def test_native_book_matches_the_python_reference(hftcore, seed):
    # Seeds 0-3 book every stock, 4-7 use the symbol filter, and so on; with the end of the
    # stream chosen by seed % 4, every ending is covered with and without the filter.
    filtered = seed // 4 % 2 == 1
    data = build(seed)
    symbols = FILTER if filtered else None
    python, native = itch.ItchBook(symbols), hftcore.ItchBook(symbols)
    rng = random.Random(seed * 2 + filtered)
    parts = list(chunks(data, rng))
    checkpoints = {len(parts) // 4, len(parts) // 2, 3 * len(parts) // 4}
    for index, part in enumerate(parts):
        python.feed(part)
        native.feed(part)
        if index in checkpoints:
            assert_same(python, native)
    python.finish()
    native.finish()
    assert_same(python, native)

    stats = python.stats()
    assert stats["messages"] > MESSAGES * 0.95
    assert stats["live_orders"] > 0 and stats["peak_levels"] > 3
    assert stats["unknown_types"] > 0 and stats["malformed"] > 0
    assert stats["unknown_refs"] > 0 and stats["overfills"] > 0 and stats["invalid"] > 0
    assert stats["duplicate_refs"] > 0 and stats["crossed"] > 0
    assert stats["filtered"] > 0 if filtered else stats["filtered"] == 0
    assert stats["end_of_session"] is (seed % 4 == 1)
    assert (stats["truncated_bytes"] > 0) is (seed % 4 == 2)


def test_native_one_shot_equals_chunked_feed(hftcore):
    data = build(3)
    whole, split = hftcore.ItchBook(), hftcore.ItchBook()
    whole.feed(data)
    for part in chunks(data, random.Random(99)):
        split.feed(part)
    whole.finish()
    split.finish()
    assert_same(whole, split)


def test_native_rejects_bad_arguments(hftcore):
    book = hftcore.ItchBook()
    with pytest.raises(ValueError):
        book.queue(1, "X", 0)
    with pytest.raises(ValueError):
        book.snapshot(70000, 1)
    with pytest.raises(ValueError):
        book.snapshot(1, -1)
    with pytest.raises(ValueError):
        book.stock(-1)
    with pytest.raises(TypeError):
        hftcore.ItchBook("AAPL")


def test_native_and_reference_accept_any_bytes_like_input(hftcore):
    stream = itch.frame([itch.encode("A", locate=1, ref=1, side="B", shares=5, price=100)])
    for make in (bytes, bytearray, memoryview):
        python, native = itch.ItchBook(), hftcore.ItchBook()
        for book in (python, native):
            book.feed(make(stream[:7]))
            book.feed(make(stream[7:]))
        assert_same(python, native)
        assert native.queue(1, "B", 100) == [(1, 5)]
    for book in (itch.ItchBook(), hftcore.ItchBook()):
        with pytest.raises(TypeError):
            book.feed("text")


def test_native_and_reference_validate_symbols_alike(hftcore):
    for factory in (itch.ItchBook, hftcore.ItchBook):
        with pytest.raises(TypeError):
            factory("AAPL")
        with pytest.raises(ValueError):
            factory(["€"])
        factory(iter(["AAPL"]))  # any iterable of symbols
        factory(("A", "B"))
