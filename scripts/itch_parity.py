"""Compare the C++ and Python ITCH order books on the first N messages of a real day.

Streams a decompressed Nasdaq ITCH 5.0 BinaryFILE (never loading it whole), cuts it at
frame boundaries after N frames (the end-of-session marker counts as a frame), feeds
identical bytes to `hft.itch.ItchBook` and `hftcore.ItchBook`, and compares everything
observable: `stats()`, `locates()`, `snapshot(locate, D)`, a full-depth snapshot per
locate, the FIFO `queue` of every level, and `stock` for every locate.

    PYTHONPATH=build/hftcore-py .venv/bin/python scripts/itch_parity.py day.itch --messages 5000000

Only counts are printed (no prices, refs or raw messages); the first mismatch is named
by locate and field. Exit status: 0 identical, 1 mismatch, 2 unusable input or setup.
"""

import argparse
import importlib
import sys
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hft.itch import ItchBook as ReferenceItchBook

CHUNK = 1 << 20
LOCATE_COUNT = 1 << 16
FULL_DEPTH = 1 << 31  # larger than any book: lists every level of a side
SIDES = (("bids", "B"), ("asks", "S"))


@dataclass
class Comparison:
    """What `compare_books` checked and found. `first` names the earliest mismatch."""

    messages: int = 0  # frames with length > 0 that the reference book read
    locates: int = 0  # locates with live orders in either book
    stocks: int = 0  # locates whose stock summary was compared
    levels: int = 0  # price levels whose snapshot and queue were compared
    mismatches: int = 0
    first: str | None = None

    def mismatch(self, where: str) -> None:
        self.mismatches += 1
        if self.first is None:
            self.first = where


def frames(path: Path, count: int, chunk: int = CHUNK) -> Iterator[bytes]:
    """Yields byte chunks that end on frame boundaries, covering the first `count`
    frames of the file; stops after the end-of-session marker (which counts as a
    frame). A file that ends early yields its incomplete tail last, so both books
    report it as truncated."""
    seen = 0
    carry = b""
    with path.open("rb") as handle:
        while seen < count:
            data = handle.read(chunk)
            if not data:
                break
            buffer = carry + data
            pos = 0
            ended = False
            while seen < count and len(buffer) - pos >= 2:
                length = int.from_bytes(buffer[pos : pos + 2], "big")
                if length == 0:
                    seen += 1
                    pos += 2
                    ended = True
                    break
                if len(buffer) - pos - 2 < length:
                    break  # incomplete frame: wait for the next chunk
                seen += 1
                pos += 2 + length
            if pos:
                yield buffer[:pos]
            if ended:
                return
            carry = buffer[pos:]
        if carry and seen < count:
            yield carry


def _same(comparison: Comparison, where: str, left: Any, right: Any) -> bool:
    if left != right:
        comparison.mismatch(where)
        return False
    return True


def compare_books(reference: Any, native: Any, depth: int) -> Comparison:
    """Compares every observable of two books that were fed the same bytes."""
    result = Comparison()
    ref_stats, nat_stats = reference.stats(), native.stats()
    result.messages = ref_stats["messages"]
    for name in sorted(set(ref_stats) | set(nat_stats)):
        _same(result, f"stats.{name}", ref_stats.get(name), nat_stats.get(name))
    ref_locates, nat_locates = reference.locates(), native.locates()
    _same(result, "locates", ref_locates, nat_locates)
    live = sorted(set(ref_locates) | set(nat_locates))
    result.locates = len(live)
    for locate in live:
        _same(
            result,
            f"locate {locate} snapshot depth {depth}",
            reference.snapshot(locate, depth),
            native.snapshot(locate, depth),
        )
        ref_full = reference.snapshot(locate, FULL_DEPTH)
        nat_full = native.snapshot(locate, FULL_DEPTH)
        if not _same(result, f"locate {locate} full snapshot", ref_full, nat_full):
            continue
        for key, side in SIDES:
            for price, _shares, _orders in ref_full[key]:
                result.levels += 1
                _same(
                    result,
                    f"locate {locate} {key} queue",
                    reference.queue(locate, side, price),
                    native.queue(locate, side, price),
                )
    for locate in range(LOCATE_COUNT):
        result.stocks += 1
        _same(result, f"locate {locate} stock", reference.stock(locate), native.stock(locate))
    return result


def load_native() -> Any:
    """The `hftcore` module, which must provide `ItchBook` (put it on PYTHONPATH)."""
    try:
        module = importlib.import_module("hftcore")
    except ImportError as error:
        print(
            f"itch_parity: cannot import hftcore ({error}); build it and put it on PYTHONPATH",
            file=sys.stderr,
        )
        raise SystemExit(2) from error
    if not hasattr(module, "ItchBook"):
        print("itch_parity: the installed hftcore predates ItchBook; rebuild it", file=sys.stderr)
        raise SystemExit(2)
    return module


def run(
    path: Path,
    messages: int,
    depth: int,
    native_book: Callable[[], Any],
    reference_book: Callable[[], Any] = ReferenceItchBook,
) -> Comparison:
    """Feeds the first `messages` frames to a fresh reference and native book and
    compares them."""
    reference, native = reference_book(), native_book()
    for data in frames(path, messages):
        reference.feed(data)
        native.feed(data)
    reference.finish()
    native.finish()
    return compare_books(reference, native, depth)


def report(comparison: Comparison) -> int:
    """Prints counts only and returns the exit status."""
    print(f"messages compared: {comparison.messages}")
    print(f"locates compared: {comparison.locates}")
    print(f"stocks compared: {comparison.stocks}")
    print(f"levels compared: {comparison.levels}")
    print(f"mismatches: {comparison.mismatches}")
    if comparison.mismatches:
        print(f"first mismatch: {comparison.first}")
        print("result: DIFFERENT")
        return 1
    print("result: identical")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("file", type=Path, help="decompressed ITCH BinaryFILE")
    parser.add_argument("--messages", type=int, required=True, help="frames to compare")
    parser.add_argument("--depth", type=int, default=10, help="snapshot depth (default 10)")
    args = parser.parse_args(argv)
    if args.messages < 1 or args.depth < 0:
        parser.error("--messages must be positive and --depth not negative")
    if not args.file.is_file():
        print(f"itch_parity: no such file: {args.file}", file=sys.stderr)
        return 2
    native = load_native()
    comparison = run(args.file, args.messages, args.depth, native.ItchBook)
    return report(comparison)


if __name__ == "__main__":
    sys.exit(main())
