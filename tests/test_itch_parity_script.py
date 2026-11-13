"""scripts/itch_parity.py: frame-boundary streaming, and detection of differing books.

The comparison runs on a small synthetic BinaryFILE. Tests that need the native module
skip without it unless HFT_REQUIRE_HFTCORE=1 (CI), like tests/test_itch_parity.py.
"""

import importlib
import importlib.util
import os
import random
from pathlib import Path
from types import SimpleNamespace

import pytest

from hft import itch

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "itch_parity.py"
LOCATES = (1, 2, 3)


def _script():
    spec = importlib.util.spec_from_file_location("itch_parity", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def script():
    return _script()


@pytest.fixture(scope="module")
def hftcore():
    try:
        module = importlib.import_module("hftcore")
        if not hasattr(module, "ItchBook"):
            raise ImportError("the installed hftcore predates ItchBook")
        return module
    except ImportError:
        if os.environ.get("HFT_REQUIRE_HFTCORE") == "1":
            raise
        pytest.skip("hftcore with ItchBook is not installed")


def _messages(count: int = 400, seed: int = 3) -> list[bytes]:
    """Directory messages, then random adds, executions, cancels, deletes, replaces and
    prints for a few stocks, with a few unknown refs mixed in."""
    rng = random.Random(seed)
    out = [itch.encode("R", locate=loc, stock=f"SYM{loc}", round_lot_size=100) for loc in LOCATES]
    live: dict[int, int] = {}
    next_ref = 1
    for _ in range(count):
        kind = rng.choice("AAAAEXDUP")
        locate = rng.choice(LOCATES)
        if kind == "A" or not live:
            live[next_ref] = rng.randint(1, 500)
            out.append(
                itch.encode(
                    "A",
                    locate=locate,
                    ref=next_ref,
                    side=rng.choice("BS"),
                    shares=live[next_ref],
                    stock=f"SYM{locate}",
                    price=1000 + rng.randint(-5, 5),
                )
            )
            next_ref += 1
            continue
        ref = rng.choice(list(live))
        if rng.random() < 0.05:
            ref += 10_000  # unknown ref
        if kind == "E":
            out.append(itch.encode("E", locate=locate, ref=ref, shares=1, match=ref))
        elif kind == "X":
            out.append(itch.encode("X", locate=locate, ref=ref, shares=1))
        elif kind == "D":
            out.append(itch.encode("D", locate=locate, ref=ref))
            live.pop(ref, None)
        elif kind == "U":
            live[next_ref] = 7
            out.append(
                itch.encode("U", locate=locate, ref=ref, new_ref=next_ref, shares=7, price=1003)
            )
            live.pop(ref, None)
            next_ref += 1
        else:
            out.append(
                itch.encode(
                    "P", locate=locate, ref=0, side="B", shares=5, stock=f"SYM{locate}", price=1001
                )
            )
    return out


@pytest.fixture
def day(tmp_path):
    path = tmp_path / "day.itch"
    path.write_bytes(itch.frame(_messages()))
    return path


def test_frames_stop_after_the_requested_count_on_a_frame_boundary(script, tmp_path):
    bodies = _messages(50)
    path = tmp_path / "f.itch"
    path.write_bytes(itch.frame(bodies, end=True))
    for chunk in (1, 7, 64, 1 << 20):
        got = b"".join(script.frames(path, 10, chunk))
        assert got == itch.frame(bodies[:10], end=False)


def test_frames_include_the_end_marker_and_stop_there(script, tmp_path):
    bodies = _messages(20)
    path = tmp_path / "f.itch"
    path.write_bytes(itch.frame(bodies) + b"trailing bytes after the marker")
    expected = itch.frame(bodies)
    for chunk in (3, 1 << 20):
        assert b"".join(script.frames(path, 10_000, chunk)) == expected
        # The marker is the Nth frame: N frames are the messages plus the marker.
        assert b"".join(script.frames(path, len(bodies) + 1, chunk)) == expected
        assert b"".join(script.frames(path, len(bodies), chunk)) == itch.frame(bodies, end=False)


def test_frames_yield_an_incomplete_tail_last(script, tmp_path):
    bodies = _messages(5)
    data = itch.frame(bodies, end=False) + b"\x00\x1f\x41"
    path = tmp_path / "f.itch"
    path.write_bytes(data)
    assert b"".join(script.frames(path, 100, 11)) == data


def test_identical_books_report_no_mismatches(script, hftcore, day):
    comparison = script.run(day, 10_000, 3, hftcore.ItchBook)
    assert comparison.mismatches == 0
    assert comparison.first is None
    assert comparison.messages == len(_messages())
    assert comparison.levels > 0
    assert comparison.locates > 0
    assert comparison.stocks == 1 << 16


def test_partial_prefix_compares_identical(script, hftcore, day):
    comparison = script.run(day, 40, 5, hftcore.ItchBook)
    assert comparison.mismatches == 0
    assert comparison.messages == 40


def test_a_book_fed_fewer_messages_is_detected(script, hftcore):
    messages = _messages()
    reference, native = itch.ItchBook(), hftcore.ItchBook()
    reference.feed(itch.frame(messages, end=False))
    native.feed(itch.frame(messages[:-20], end=False))
    for book in (reference, native):
        book.finish()
    comparison = script.compare_books(reference, native, 5)
    assert comparison.mismatches > 0
    assert comparison.first is not None
    assert comparison.first.startswith("stats.") or comparison.first.startswith("locate")


def test_main_exits_non_zero_when_the_books_differ(script, hftcore, day, monkeypatch, capsys):
    class Skewed:
        """A native book that saw one extra order before the file."""

        def __init__(self):
            self._book = hftcore.ItchBook()
            self._book.feed(
                itch.frame(
                    [itch.encode("A", locate=2, ref=999_999, side="B", shares=1, price=5)],
                    end=False,
                )
            )

        def __getattr__(self, name):
            return getattr(self._book, name)

    monkeypatch.setattr(script, "load_native", lambda: SimpleNamespace(ItchBook=Skewed))
    assert script.main([str(day), "--messages", "10000"]) == 1
    out = capsys.readouterr().out
    assert "result: DIFFERENT" in out
    assert "first mismatch:" in out
    assert "mismatches: 0" not in out


def test_main_exits_zero_for_identical_books(script, hftcore, day, monkeypatch, capsys):
    monkeypatch.setattr(script, "load_native", lambda: hftcore)
    assert script.main([str(day), "--messages", "10000"]) == 0
    out = capsys.readouterr().out
    assert "mismatches: 0" in out
    assert "result: identical" in out


def test_main_rejects_a_missing_file(script, tmp_path):
    assert script.main([str(tmp_path / "missing.itch"), "--messages", "5"]) == 2


def test_missing_native_module_is_a_clear_failure(script, monkeypatch, capsys):
    def broken(name):
        raise ImportError("no module named hftcore")

    monkeypatch.setattr(script.importlib, "import_module", broken)
    with pytest.raises(SystemExit) as raised:
        script.load_native()
    assert raised.value.code == 2
    assert "cannot import hftcore" in capsys.readouterr().err


def test_native_module_without_itchbook_is_a_clear_failure(script, monkeypatch, capsys):
    monkeypatch.setattr(script.importlib, "import_module", lambda name: SimpleNamespace())
    with pytest.raises(SystemExit) as raised:
        script.load_native()
    assert raised.value.code == 2
    assert "predates ItchBook" in capsys.readouterr().err
