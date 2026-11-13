# Nasdaq ITCH 5.0 feed handler and limit order book (design)

Date: 2026-11-13. Status: approved (option 4 of the 2026-11-12 menu, career track).

## Why

Free-data strategy research has failed the same way many times (see
[project history](../project-history.md)). This track shows exchange-grade market-data
engineering in C++ at $0: a parser for Nasdaq TotalView-ITCH 5.0, a full-depth limit
order book rebuilt from every order message, a Python reference it must match exactly,
and honest throughput and latency numbers on a real trading day. It does not touch the
paper bots and is not used by any strategy.

## Data

- Nasdaq publishes free sample days at `https://emi.nasdaq.com/ITCH/Nasdaq%20ITCH/` in
  BinaryFILE format: a gzip stream with no MoldUDP framing; every message is prefixed
  by a 2-byte big-endian length; a zero-length message marks the end of the session.
- Benchmark day: `01302019.NASDAQ_ITCH50.gz` (4,764,426,091 bytes compressed, about
  11.2 GB raw, 368,366,634 messages published for this day). The 2025-2026 files are
  5-18 GB and include the 2023 `O` message.
- License: Nasdaq publishes no terms for the samples. They are used locally only and
  kept in the git-ignored `data/itch/`. Raw or reconstructable data (messages, order
  books, per-order output) is never committed or published; only aggregates (message
  counts, counters, timing, a few closing prices).

## Message format

Every integer is big-endian and unsigned. Every message starts with the same 11-byte
header: type (1), stock locate (2), tracking number (2), timestamp (6, nanoseconds
since midnight). Prices are Price(4) with 4 implied decimals, except `V`, which uses
Price(8) with 8 decimals. Stock symbols are 8 ASCII bytes, right-padded with spaces.

| Type | Length | Fields after the header (bytes) | Book effect |
| --- | --- | --- | --- |
| `S` system event | 12 | event code (1) | none |
| `R` stock directory | 39 | stock (8), market category (1), financial status (1), round lot (4), round lots only (1), issue classification (1), issue sub-type (2), authenticity (1), short sale threshold (1), IPO flag (1), LULD tier (1), ETP flag (1), ETP leverage (4), inverse (1) | directory |
| `H` trading action | 25 | stock (8), trading state (1), reserved (1), reason (4) | trading state |
| `Y` Reg SHO | 20 | stock (8), action (1) | none |
| `L` participant position | 26 | MPID (4), stock (8), primary MM (1), MM mode (1), participant state (1) | none |
| `V` MWCB decline level | 35 | level 1 (8), level 2 (8), level 3 (8), Price(8) | none |
| `W` MWCB status | 12 | breached level (1) | none |
| `K` IPO quoting period | 28 | stock (8), release time (4), release qualifier (1), IPO price (4) | none |
| `J` LULD auction collar | 35 | stock (8), reference price (4), upper (4), lower (4), extension (4) | none |
| `h` operational halt | 21 | stock (8), market code (1), halt action (1) | none |
| `A` add order | 36 | order ref (8), side (1), shares (4), stock (8), price (4) | add |
| `F` add order with MPID | 40 | as `A`, then attribution (4) | add |
| `E` order executed | 31 | order ref (8), executed shares (4), match number (8) | execute at the order price |
| `C` executed with price | 36 | order ref (8), executed shares (4), match number (8), printable (1), price (4) | execute at the message price |
| `X` order cancel | 23 | order ref (8), cancelled shares (4) | reduce |
| `D` order delete | 19 | order ref (8) | remove |
| `U` order replace | 35 | original ref (8), new ref (8), shares (4), price (4) | remove, then add |
| `P` trade (non-cross) | 44 | order ref (8), side (1), shares (4), stock (8), price (4), match number (8) | none (print) |
| `Q` cross trade | 40 | shares (8), stock (8), cross price (4), match number (8), cross type (1) | none (print) |
| `B` broken trade | 19 | match number (8) | none |
| `I` NOII | 50 | paired (8), imbalance (8), direction (1), stock (8), far (4), near (4), reference (4), cross type (1), price variation (1) | none |
| `N` RPII | 20 | stock (8), interest flag (1) | none |
| `O` DLCR price discovery | 48 | stock (8), open eligibility (1), min price (4), max price (4), near execution price (4), near execution time (8), lower collar (4), upper collar (4) | none |

Rules that the implementation must follow:

- Order refs are 64-bit, unique within a day and sparse (up to about 329M on
  2019-01-30); stock locate can exceed 12,000 (12,809 in 2026), so per-stock state is
  a 65,536-entry array indexed by locate.
- An order executed down to zero shares is removed; no `D` follows.
- `U` removes the original order and adds the new ref with the original side and
  stock, the new shares and price, at the back of its level's queue.
- `E`, `C`, `X`, `D` and `U` take the stock from the stored order, not the header.
- `P`, `Q` and `B` never change the book.
- An unknown type is skipped by its length prefix and counted. A known type whose
  length prefix differs from the table is counted as malformed and skipped.
- The 6-byte timestamp is read byte by byte (never an 8-byte load past the field).

## Design

All C++ lives in namespace `hftcore` in the static library `hftcore_core`, with the
existing warning flags (`-Wall -Wextra -Wpedantic -Werror`).

### Parser (`cpp/include/hftcore/itch.hpp`, `cpp/src/itch.cpp`)

- `load_be16/32/64` via `std::memcpy` plus `std::byteswap`; `load_be48` assembles six
  bytes.
- One plain struct per message type with decoded fields (prices as `uint32_t` in
  1/10,000 dollars, `V` levels as `uint64_t` in 1/10^8 dollars, stocks as 8-byte
  arrays), a `message_length(char type)` table (0 for unknown types) and
  `decode_*` functions that take a pointer to the message body.
- `BinaryFileReader` walks a byte span: it reads each length prefix, stops at the
  zero-length end marker (`end_of_session`), skips and counts unknown types and
  length mismatches (`unknown_types`, `malformed`), and reports a truncated tail
  (`truncated_bytes`) instead of reading past the buffer. For streaming input it
  reports how many bytes it consumed so a caller can carry an incomplete tail into
  the next chunk.
- Dispatch is a template `for_each_message(span, Visitor&)` with one overload per
  message struct, so parse-only code pays no virtual calls.

### Order book (`cpp/include/hftcore/order_book.hpp`, `cpp/src/order_book.cpp`)

- Order pool: `std::vector<Order>` with a free list. `Order` holds ref, shares,
  price, locate, side and FIFO prev/next pool indices.
- Order map `OrderMap`: ref to pool index, open addressing with linear probing on
  `detail::mix(ref)` (from `id_set.hpp`), backward-shift deletion (no tombstones:
  about 190M erases a day), grows to keep the load at or below 0.5.
- Levels: each side of each stock is a `std::vector<Level>` sorted so the best price
  is at the back (bids ascending, asks descending), found by binary search. `Level`
  holds price, total shares, order count and the FIFO head and tail.
- Operations: add, execute (returns the fill price; an order reaching zero is
  removed), cancel, delete, replace. An unknown ref is counted and ignored; executing
  or cancelling more shares than remain is counted as an overfill and removes the
  order.
- Queries: best bid and ask, `depth(locate, side, n)`, the FIFO queue of a level,
  live order count, peak live orders, level count, and `validate()`, which checks
  that each level's total equals the sum of its orders' shares and its count equals
  its order count, that levels are strictly sorted, that no level is empty, and that
  the map and pool agree.

### Feed handler (`cpp/include/hftcore/itch_book.hpp`, `cpp/src/itch_book.cpp`)

`ItchBookBuilder` consumes decoded messages and keeps:

- the stock directory (locate to symbol) from `R` and the trading state from `H`;
- one book per locate, allocated on the first add;
- an optional symbol filter: only stocks named in it get books; messages whose header
  locate is filtered out are skipped before any lookup, so they never count as
  unknown refs;
- per-stock trade prints: `E`, printable `C` (`Y`), `P`, and `Q` with nonzero shares;
  count, share volume, last price, and the opening (`O`) and closing (`C`) cross
  prices from `Q`;
- counters: messages by type, unknown types, malformed, unknown refs, duplicate
  refs, invalid (zero-share) orders, overfills, filtered messages, crossed books
  (successful `A`, `F` or `U` messages after which the touched stock's best bid is at
  or above its best ask), live and peak live orders and levels.

### Python reference (`hft/itch.py`) and bindings

- A message encoder for every type in the table plus BinaryFILE framing, used by
  tests to build streams.
- A plain reference book (dicts and sorted lists, no performance tricks) and a
  reference feed handler with the same counters and the same snapshot format.
- The pybind class `hftcore.ItchBook(symbols=None)` offers `feed(bytes)` (an
  incomplete tail is carried to the next call), `finish()`, `snapshot(locate,
  depth)`, `queue(locate, side, price)`, `stock(locate)` and `stats()`. The Python
  reference exposes the same methods with the same return values. A difference is a
  bug in one of them.

### Benchmark and replay CLI (`cpp/bench/`)

- `hftcore_bench_itch FILE`: mmaps a decompressed BinaryFILE and runs (a) parse-only,
  (b) parse plus full book, reporting wall ns per message and messages per second,
  and (c) per-message latency of parse plus book, timed with `rdtscp` after
  `lfence`, calibrated against `steady_clock`, with the timer overhead measured and
  reported. Samples go into a log-linear histogram (storing 368M samples would take
  3 GB); it reports p50, p99, p99.9 and max overall and for `A`, `D`, `U`, `E` and `X`,
  plus peak RSS, peak live orders and peak levels.
- `hftcore_itch_replay [--symbols A,B] [--validate-every N] FILE|-`: reads a file or
  stdin (`gzip -dc f.gz | hftcore_itch_replay -`) and prints an end-of-day summary:
  message counts by type, the unknown, malformed, unknown-ref and overfill counters,
  crossed books, live orders at the end, and, for chosen symbols, the closing cross
  price against the last book mid. Aggregates only.

## Performance goals (set before measuring)

On one core of the development machine (WSL2 on a shared laptop): parse-only mean at
or below 25 ns per message, parse plus full book mean at or below 120 ns per message,
and peak RSS below 1 GiB for the 2019-01-30 day. Published calibration points
(self-reported by their authors): parse-only 5-25 ns, a tuned full book 40-120 ns, a
naive `std::map` plus `unordered_map` book about 250-300 ns. Results are reported as
measured, whether the goals are met or not. Runs are pinned with `taskset` to cores
other than 4-5 (used by the paper bots).

## Verification

- Catch2 tests for every message length and field, endianness, the 6-byte timestamp,
  unknown and future types, truncated tails, the end marker, length mismatches; FIFO
  order, executions to zero, replace semantics, map deletion under collisions,
  unknown refs and a randomized test calling `validate()` after every operation; the
  feed handler's directory, filter, print and counter rules.
- Python parity: seeded random streams through both implementations with identical
  snapshots, queues, stock summaries and counters (`tests/test_itch_parity.py`,
  skipped without `hftcore` unless `HFT_REQUIRE_HFTCORE=1`).
- Real day: the message count matches 368,366,634, unknown refs and overfills are
  zero, `validate()` passes at sampled checkpoints, and the first 5M messages give
  identical C++ and Python snapshots. CI never uses real data.

## Non-goals

No live ITCH feed (paid), no MoldUDP64 or SoupBinTCP session handling (the framing
layer is separate so it could be added), no use in trading strategies, no CI runs on
real data.
