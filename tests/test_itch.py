"""Reference ITCH 5.0 encoder/decoder and order-book rules (pure Python, always runs)."""

import pytest

from hft import itch
from hft.itch import ItchBook, ReadStats

# The message length table of docs/specs/2026-11-13-itch-order-book-design.md.
SPEC_LENGTHS = {
    "S": 12,
    "R": 39,
    "H": 25,
    "Y": 20,
    "L": 26,
    "V": 35,
    "W": 12,
    "K": 28,
    "J": 35,
    "h": 21,
    "A": 36,
    "F": 40,
    "E": 31,
    "C": 36,
    "X": 23,
    "D": 19,
    "U": 35,
    "P": 44,
    "Q": 40,
    "B": 19,
    "I": 50,
    "N": 20,
    "O": 48,
}


def make(type_: str, locate: int = 1, **fields) -> bytes:
    return itch.encode(type_, locate=locate, **fields)


def add(ref, side="B", shares=100, price=1000, locate=1, type_="A", stock="AAPL"):
    return make(type_, locate, ref=ref, side=side, shares=shares, price=price, stock=stock)


def book_of(*messages: bytes, symbols=None) -> ItchBook:
    book = ItchBook(symbols)
    book.feed(itch.frame(messages, end=False))
    return book


def test_lengths_match_the_spec_table():
    assert itch.MESSAGE_LENGTHS == SPEC_LENGTHS
    assert len(itch.MESSAGE_FIELDS) == 23


def test_encode_decode_round_trip_for_every_type():
    for type_, fields in itch.MESSAGE_FIELDS.items():
        values = {}
        for index, (name, size, kind) in enumerate(fields):
            if kind == "u":
                values[name] = (0x0102030405060708 >> (8 * (8 - size))) + index
            elif kind == "c":
                values[name] = chr(ord("a") + index % 26)
            else:
                values[name] = ("SYMBOLXY"[:size]).ljust(size)
        body = itch.encode(
            type_, locate=0x1234, tracking=0xBEEF, timestamp=0x0A0B0C0D0E0F, **values
        )
        assert len(body) == SPEC_LENGTHS[type_]
        decoded = itch.decode(body)
        assert decoded["type"] == type_
        assert decoded["locate"] == 0x1234
        assert decoded["tracking"] == 0xBEEF
        assert decoded["timestamp"] == 0x0A0B0C0D0E0F
        for name, value in values.items():
            assert decoded[name] == value, (type_, name)


def test_header_is_big_endian_and_timestamp_is_six_bytes():
    body = itch.encode("D", locate=0x0102, tracking=0x0304, timestamp=0x050607080910, ref=1)
    assert body[:11] == bytes([ord("D"), 1, 2, 3, 4, 5, 6, 7, 8, 9, 0x10])
    assert body[11:] == (1).to_bytes(8, "big")


def test_missing_fields_default_to_zero_and_spaces():
    decoded = itch.decode(itch.encode("A", ref=7))
    assert decoded["ref"] == 7
    assert decoded["shares"] == 0 and decoded["price"] == 0
    assert decoded["stock"] == "        " and decoded["side"] == " "
    assert decoded["attribution"] == "    "  # A has no attribution on the wire


def test_encode_rejects_bad_input():
    with pytest.raises(ValueError):
        itch.encode("?")
    with pytest.raises(ValueError):
        itch.encode("A", nonsense=1)
    with pytest.raises(ValueError):
        itch.encode("A", shares=2**32)
    with pytest.raises(ValueError):
        itch.encode("A", stock="TOOLONGSYM")
    with pytest.raises(ValueError):
        itch.encode("A", side="BS")
    with pytest.raises(ValueError):
        itch.encode("A", locate=65536)


def test_decode_rejects_wrong_lengths_and_unknown_types():
    with pytest.raises(ValueError):
        itch.decode(b"")
    with pytest.raises(ValueError):
        itch.decode(b"A" + bytes(10))
    with pytest.raises(ValueError):
        itch.decode(b"?" + bytes(10))


def test_frame_prefixes_lengths_and_appends_the_end_marker():
    one, two = make("D", ref=1), make("W")
    assert itch.frame([one, two], end=False) == (
        len(one).to_bytes(2, "big") + one + len(two).to_bytes(2, "big") + two
    )
    assert itch.frame([one]).endswith(b"\x00\x00")
    assert itch.frame([]) == b"\x00\x00"


def collect(data: bytes, stats: ReadStats | None = None):
    stats = stats or ReadStats()
    seen: list[dict] = []
    used = itch.parse_binary_file(data, seen.append, stats)
    return seen, used, stats


def test_parse_counts_messages_by_type_and_decodes_known_ones():
    data = itch.frame([make("S", event_code="O"), add(1), add(2)], end=False)
    seen, used, stats = collect(data)
    assert used == len(data)
    assert [m["type"] for m in seen] == ["S", "A", "A"]
    assert stats.messages == 3 and stats.by_type == {ord("S"): 1, ord("A"): 2}
    assert not stats.end_of_session and stats.trailing_bytes == 0


def test_parse_skips_unknown_and_malformed_frames_and_counts_them():
    unknown = b"\x00\x05z1234"
    short_add = b"\x00\x0aA" + bytes(9)
    data = unknown + short_add + itch.frame([add(1)], end=False)
    seen, used, stats = collect(data)
    assert used == len(data)
    assert [m["type"] for m in seen] == ["A"]
    assert (stats.messages, stats.unknown_types, stats.malformed) == (3, 1, 1)
    assert stats.by_type[ord("z")] == 1 and stats.by_type[ord("A")] == 2


def test_parse_stops_before_an_incomplete_frame():
    whole = itch.frame([add(1)], end=False)
    data = whole + itch.frame([add(2)], end=False)[:-3]
    seen, used, _ = collect(data)
    assert used == len(whole) and len(seen) == 1
    seen, used, _ = collect(whole + b"\x00")  # a lone length byte is incomplete too
    assert used == len(whole)


def test_parse_end_marker_consumes_and_counts_trailing_bytes():
    data = itch.frame([add(1)]) + b"junk!"
    stats = ReadStats()
    seen, used, _ = collect(data, stats)
    assert used == len(data) and len(seen) == 1
    assert stats.end_of_session and stats.trailing_bytes == 5
    later = add(2) + b"more"
    seen, used, _ = collect(later, stats)  # later calls are all trailing
    assert used == len(later) and seen == [] and stats.messages == 1
    assert stats.trailing_bytes == 5 + len(later)


# --- book rules -------------------------------------------------------------


def snap(book: ItchBook, locate: int = 1, depth: int = 10):
    return book.snapshot(locate, depth)


def test_orders_queue_fifo_within_a_level_and_levels_sort_best_first():
    book = book_of(
        add(1, "B", 100, 1000),
        add(2, "B", 50, 1010),
        add(3, "B", 25, 1000, type_="F"),
        add(4, "S", 10, 1020),
        add(5, "S", 20, 1030),
    )
    assert snap(book) == {
        "bids": [(1010, 50, 1), (1000, 125, 2)],
        "asks": [(1020, 10, 1), (1030, 20, 1)],
    }
    assert book.queue(1, "B", 1000) == [(1, 100), (3, 25)]
    assert book.queue(1, "B", 999) == []
    assert book.queue(1, "S", 1020) == [(4, 10)]
    assert book.locates() == [1]
    assert book.validate() == ""


def test_partial_execution_keeps_queue_position_and_full_execution_removes():
    book = book_of(
        add(1, "S", 100, 1000),
        add(2, "S", 40, 1000),
        make("E", 1, ref=1, shares=30, match=9),
        make("X", 1, ref=2, shares=15),
    )
    assert book.queue(1, "S", 1000) == [(1, 70), (2, 25)]
    book.feed(itch.frame([make("E", 1, ref=1, shares=70)], end=False))
    assert book.queue(1, "S", 1000) == [(2, 25)]
    book.feed(itch.frame([make("D", 1, ref=2)], end=False))
    assert snap(book) == {"bids": [], "asks": []}
    assert book.locates() == []
    stats = book.stats()
    assert stats["live_orders"] == 0 and stats["levels"] == 0
    assert stats["peak_live_orders"] == 2 and stats["peak_levels"] == 1
    assert stats["overfills"] == 0 and stats["unknown_refs"] == 0


def test_overfill_removes_the_order_and_prints_the_remaining_shares():
    book = book_of(add(1, "B", 10, 1000), make("E", 1, ref=1, shares=25))
    stats = book.stats()
    assert stats["overfills"] == 1 and stats["live_orders"] == 0
    assert book.stock(1)["volume"] == 10 and book.stock(1)["trades"] == 1
    book.feed(itch.frame([add(2, "B", 10, 1000), make("X", 1, ref=2, shares=11)], end=False))
    assert book.stats()["overfills"] == 2 and book.stats()["live_orders"] == 0
    assert book.stock(1)["trades"] == 1  # cancels never print


def test_zero_share_execute_and_cancel_change_nothing_and_do_not_print():
    book = book_of(
        add(1, "B", 10, 1000),
        make("E", 1, ref=1, shares=0),
        make("C", 1, ref=1, shares=0, printable="Y", price=5),
        make("X", 1, ref=1, shares=0),
    )
    assert book.queue(1, "B", 1000) == [(1, 10)]
    assert book.stock(1)["trades"] == 0
    stats = book.stats()
    assert stats["unknown_refs"] == stats["invalid"] == stats["overfills"] == 0


def test_unknown_duplicate_and_invalid_adds_are_counted_and_leave_the_book():
    book = book_of(
        add(1, "B", 10, 1000),
        add(1, "S", 99, 2000),  # duplicate ref
        add(2, "B", 0, 1000),  # zero shares
        add(3, "X", 5, 1000),  # unknown side
        add(4, " ", 5, 1000, type_="F"),
        make("E", 1, ref=77, shares=1),
        make("C", 1, ref=77, shares=1, printable="Y", price=1),
        make("X", 1, ref=77, shares=1),
        make("D", 1, ref=77),
        make("U", 1, ref=77, new_ref=78, shares=5, price=1),
    )
    stats = book.stats()
    assert stats["duplicate_refs"] == 1
    assert stats["invalid"] == 3
    assert stats["unknown_refs"] == 5
    assert stats["live_orders"] == 1 and stats["peak_live_orders"] == 1
    assert snap(book) == {"bids": [(1000, 10, 1)], "asks": []}
    assert book.stock(1)["trades"] == 0  # unknown refs never print


def test_replace_moves_to_the_back_of_the_level_and_keeps_side_and_locate():
    book = book_of(
        add(1, "B", 10, 1000),
        add(2, "B", 20, 1000),
        make("U", 1, ref=1, new_ref=3, shares=7, price=1000),
        make("U", 1, ref=2, new_ref=4, shares=8, price=990),
    )
    assert book.queue(1, "B", 1000) == [(3, 7)]
    assert book.queue(1, "B", 990) == [(4, 8)]
    stats = book.stats()
    assert stats["live_orders"] == 2 and stats["levels"] == 2
    assert book.validate() == ""


def test_replace_rejections_leave_the_original_untouched():
    book = book_of(
        add(1, "B", 10, 1000),
        add(2, "B", 20, 1000),
        make("U", 1, ref=1, new_ref=3, shares=0, price=1000),  # invalid
        make("U", 1, ref=1, new_ref=2, shares=5, price=1000),  # new ref is live
        make("U", 1, ref=9, new_ref=3, shares=5, price=1000),  # unknown original
    )
    assert book.queue(1, "B", 1000) == [(1, 10), (2, 20)]
    stats = book.stats()
    assert (stats["invalid"], stats["duplicate_refs"], stats["unknown_refs"]) == (1, 1, 1)
    book.feed(itch.frame([make("U", 1, ref=1, new_ref=1, shares=4, price=1000)], end=False))
    assert book.queue(1, "B", 1000) == [(2, 20), (1, 4)]  # same ref: still accepted


def test_crossed_counts_accepted_add_and_replace_only():
    book = book_of(
        add(1, "B", 10, 1000),
        add(2, "S", 10, 1000),  # crosses: bid >= ask
        add(3, "S", 10, 1000, type_="F"),  # still crossed
        add(2, "S", 10, 900),  # duplicate: rejected, not counted
        make("U", 1, ref=3, new_ref=4, shares=5, price=1100),  # still crossed through ref 2
    )
    assert book.stats()["crossed"] == 3
    book.feed(itch.frame([make("D", 1, ref=2)], end=False))
    assert book.stats()["crossed"] == 3  # deletes are not checked
    book.feed(itch.frame([make("U", 1, ref=4, new_ref=5, shares=5, price=1200)], end=False))
    assert book.stats()["crossed"] == 3  # no longer crossed


def test_crossed_after_replace_uses_the_stored_orders_locate():
    book = book_of(
        add(1, "B", 10, 1000, locate=1),
        add(2, "S", 10, 1100, locate=1),
        add(3, "S", 10, 1000, locate=2),
        make("U", 5, ref=2, new_ref=4, shares=10, price=1000),  # header locate 5, stored 1
    )
    assert book.stats()["crossed"] == 1
    assert book.queue(1, "S", 1000) == [(4, 10)]


def test_executions_use_the_stored_orders_locate_not_the_header():
    book = book_of(
        make("R", 1, stock="AAPL"),
        add(1, "B", 100, 1000, locate=1),
        make("E", 2, ref=1, shares=40, match=1),
        make("C", 3, ref=1, shares=10, match=2, printable="Y", price=1234),
        make("X", 4, ref=1, shares=5),
    )
    assert book.queue(1, "B", 1000) == [(1, 45)]
    assert book.stock(1)["trades"] == 2 and book.stock(1)["volume"] == 50
    assert book.stock(1)["last_price"] == 1234
    assert book.stock(2)["trades"] == 0 and book.stock(3)["trades"] == 0
    book.feed(itch.frame([make("D", 9, ref=1)], end=False))
    assert book.locates() == []


def test_c_prints_only_when_printable_and_at_the_message_price():
    book = book_of(
        add(1, "B", 100, 1000),
        make("C", 1, ref=1, shares=10, printable="N", price=2000),
        make("C", 1, ref=1, shares=20, printable="Y", price=3000),
        make("E", 1, ref=1, shares=5, match=3),
    )
    info = book.stock(1)
    assert (info["trades"], info["volume"], info["last_price"]) == (2, 25, 1000)
    assert book.queue(1, "B", 1000) == [(1, 65)]


def test_p_and_q_print_without_touching_the_book_and_set_cross_prices():
    book = book_of(
        make("P", 1, ref=0, side="B", shares=100, stock="AAPL", price=1111, match=1),
        make("Q", 1, shares=5000, stock="AAPL", price=1100, match=2, cross_type="O"),
        make("Q", 1, shares=7000, stock="AAPL", price=1200, match=3, cross_type="C"),
        make("Q", 1, shares=1, stock="AAPL", price=1300, match=4, cross_type="H"),
        make("Q", 1, shares=0, stock="AAPL", price=1400, match=5, cross_type="C"),
        make("B", 1, match=1),
    )
    info = book.stock(1)
    assert info["trades"] == 4 and info["volume"] == 100 + 5000 + 7000 + 1
    assert info["last_price"] == 1300  # the zero-share Q printed nothing
    assert (info["open_cross"], info["close_cross"]) == (1100, 1200)
    assert book.stock(2)["open_cross"] is None and book.stock(2)["close_cross"] is None
    assert book.locates() == [] and book.stats()["live_orders"] == 0


def test_directory_trading_state_and_default_stock():
    book = book_of(make("R", 7, stock="MSFT"), make("H", 7, stock="MSFT", state="T"))
    assert book.stock(7)["symbol"] == "MSFT" and book.stock(7)["trading_state"] == "T"
    fresh = book.stock(8)
    assert fresh == {
        "symbol": "",
        "trading_state": " ",
        "trades": 0,
        "volume": 0,
        "last_price": 0,
        "open_cross": None,
        "close_cross": None,
    }
    book.feed(itch.frame([make("H", 7, stock="MSFT", state="H")], end=False))
    assert book.stock(7)["trading_state"] == "H"


def test_symbol_filter_books_only_listed_stocks_and_counts_filtered():
    messages = [
        make("R", 1, stock="AAPL"),
        make("R", 2, stock="MSFT"),
        make("H", 2, stock="MSFT", state="Q"),
        add(1, "B", 10, 1000, locate=1),
        add(2, "B", 10, 1000, locate=2),
        make("E", 2, ref=1, shares=1),  # header filtered, even though ref 1 is live
        make("E", 1, ref=2, shares=1),  # ref 2 was never booked: unknown
        make("P", 2, ref=0, side="B", shares=1, stock="MSFT", price=1, match=1),
        make("Q", 2, shares=1, stock="MSFT", price=1, match=1, cross_type="O"),
    ]
    for symbols in (["AAPL"], ("AAPL",), {"AAPL"}):
        book = book_of(*messages, symbols=symbols)
        stats = book.stats()
        assert stats["filtered"] == 4  # the A, the E with header locate 2, the P and the Q
        assert stats["unknown_refs"] == 1
        assert book.locates() == [1]
        assert book.stock(2)["symbol"] == "MSFT" and book.stock(2)["trading_state"] == "Q"
        assert book.stock(2)["trades"] == 0
        assert book.queue(1, "B", 1000) == [(1, 10)]


def test_none_and_empty_symbols_both_mean_every_stock():
    messages = [make("R", 1, stock="AAPL"), add(1, locate=1)]
    for symbols in (None, []):
        assert book_of(*messages, symbols=symbols).locates() == [1]
    assert book_of(add(1, locate=1)).locates() == [1]  # no directory message needed


def test_a_later_directory_message_reevaluates_the_filter():
    book = book_of(
        make("R", 1, stock="AAPL"),
        add(1, locate=1),
        make("R", 1, stock="ZZZZ"),  # locate 1 is now a filtered stock
        add(2, locate=1),
        make("E", 1, ref=1, shares=1),  # filtered: ignored
        make("R", 1, stock="AAPL"),
        make("E", 1, ref=1, shares=1),
        symbols=["AAPL"],
    )
    assert book.stats()["filtered"] == 2
    assert book.queue(1, "B", 1000) == [(1, 99)]  # ref 1 stayed booked, ref 2 never was


def test_unlisted_locate_is_filtered_before_any_directory_message():
    book = book_of(add(1, locate=3), symbols=["AAPL"])
    assert book.stats()["filtered"] == 1 and book.locates() == []


def test_every_message_type_is_counted_including_unknown_ones():
    messages = [make(t, 1) for t in itch.MESSAGE_FIELDS if t not in "AFEXCDUPQ"]
    book = ItchBook()
    book.feed(itch.frame(messages, end=False) + b"\x00\x03\xc8ab")
    stats = book.stats()
    assert stats["messages"] == len(messages) + 1
    assert stats["unknown_types"] == 1 and stats["malformed"] == 0
    assert stats["by_type"]["S"] == 1 and stats["by_type"][chr(0xC8)] == 1
    assert all(count > 0 for count in stats["by_type"].values())
    assert sum(stats["by_type"].values()) == stats["messages"]


def test_chunked_feeding_equals_one_shot_and_finish_reports_the_tail():
    stream = itch.frame(
        [make("R", 1, stock="AAPL"), add(1, "B", 10, 1000), add(2, "S", 10, 1010)],
        end=False,
    )
    whole = ItchBook()
    whole.feed(stream)
    for size in (1, 2, 3, 7, 50):
        chunked = ItchBook()
        for start in range(0, len(stream), size):
            chunked.feed(stream[start : start + size])
        chunked.finish()
        assert chunked.stats()["truncated_bytes"] == 0
        assert chunked.snapshot(1, 5) == whole.snapshot(1, 5)
        assert chunked.stats()["messages"] == 3


def test_finish_reports_the_incomplete_tail_and_leaves_it_in_place():
    stream = itch.frame([add(1, "B", 10, 1000), add(2, "B", 10, 990)], end=False)
    book = ItchBook()
    book.feed(stream[:-5])
    book.finish()
    first = 2 + SPEC_LENGTHS["A"]
    assert book.stats()["truncated_bytes"] == len(stream) - 5 - first
    book.feed(stream[-5:])  # the tail was kept: the frame still completes
    assert book.queue(1, "B", 990) == [(2, 10)]


def test_end_of_session_marker_and_trailing_bytes_through_the_book():
    book = ItchBook()
    book.feed(itch.frame([add(1)]) + b"abc")
    book.feed(b"de")
    book.finish()
    stats = book.stats()
    assert stats["end_of_session"] is True and stats["trailing_bytes"] == 5
    assert stats["truncated_bytes"] == 0 and stats["messages"] == 1


def test_malformed_frames_are_skipped_and_counted():
    book = ItchBook()
    book.feed(b"\x00\x0aD" + bytes(9) + itch.frame([add(1)], end=False))
    stats = book.stats()
    assert stats["malformed"] == 1 and stats["messages"] == 2 and stats["live_orders"] == 1


def test_locates_are_sorted_and_depth_limits_the_snapshot():
    book = book_of(
        add(1, "B", 1, 100, locate=9),
        add(2, "B", 1, 101, locate=3),
        add(3, "B", 1, 102, locate=3),
        add(4, "B", 1, 103, locate=3),
    )
    assert book.locates() == [3, 9]
    assert book.snapshot(3, 2)["bids"] == [(103, 1, 1), (102, 1, 1)]
    assert book.snapshot(3, 0) == {"bids": [], "asks": []}


def test_argument_validation():
    book = ItchBook()
    with pytest.raises(ValueError):
        book.queue(1, "X", 0)
    with pytest.raises(ValueError):
        book.snapshot(70000, 1)
    with pytest.raises(ValueError):
        book.snapshot(1, -1)
    with pytest.raises(ValueError):
        book.stock(-1)
