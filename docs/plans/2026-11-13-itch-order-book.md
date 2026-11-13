# Nasdaq ITCH 5.0 Feed Handler and Order Book Implementation Plan

**Goal:** Parse Nasdaq TotalView-ITCH 5.0 BinaryFILE data, rebuild a full-depth limit
order book for every stock in C++ (`hftcore`), prove it against a Python reference, and
measure throughput and latency on the 2019-01-30 sample day.

**Spec:** `docs/specs/2026-11-13-itch-order-book-design.md` (binding; message table,
rules and counter definitions live there).

**Tech stack:** C++20 (g++), CMake + Ninja, Catch2 v3.7.1, pybind11 2.13.6 via
scikit-build-core; Python 3.12, pytest, ruff.

Tasks run in order on branch `feat/itch-order-book` (worktree
`.worktrees/itch-order-book`). One commit per task at least; Conventional Commits.

## Global Constraints

- All C++ in namespace `hftcore` (parser types in `hftcore::itch`), compiled into
  `hftcore_core` with `-Wall -Wextra -Wpedantic -Werror`; header/source pairs like
  the existing units; every file starts with a comment saying what it is for.
- Big-endian loads use `std::memcpy` plus `std::byteswap` (C++23 is not available
  everywhere: if `std::byteswap` is missing in the toolchain, use
  `__builtin_bswap16/32/64`). The 6-byte timestamp is assembled from bytes, never an
  8-byte load. No reads past the end of the input span (the ASan/UBSan job must pass).
- Use at most 4 cores and never cores 4-5 (paper bots): prefix builds and runs with
  `taskset -c 0-3` and build with `-j 4`. Never touch the running paper bots.
- Build and test commands (from the worktree root; the shared venv lives in the main
  checkout at `../../.venv`):
  ```bash
  export PATH="$PWD/../../.venv/bin:$PATH"
  taskset -c 0-3 cmake -S cpp -B cpp/build -G Ninja -DHFTCORE_BUILD_TESTS=ON
  taskset -c 0-3 cmake --build cpp/build -j 4
  ctest --test-dir cpp/build --output-on-failure
  # sanitizers (separate build dir)
  taskset -c 0-3 cmake -S cpp -B cpp/build-asan -G Ninja -DCMAKE_BUILD_TYPE=Debug \
    -DHFTCORE_SANITIZE=ON -DHFTCORE_BUILD_BENCH=OFF
  taskset -c 0-3 cmake --build cpp/build-asan -j 4 && ctest --test-dir cpp/build-asan
  ```
  Python module: never install into the shared venv (the paper bots use it). Install
  into a git-ignored target directory and put it on `PYTHONPATH`:
  ```bash
  ../../.venv/bin/python -m pip install -q --no-deps --target build/hftcore-py ./cpp
  PYTHONPATH=build/hftcore-py HFT_REQUIRE_HFTCORE=1 ../../.venv/bin/python -m pytest -q tests/test_itch.py tests/test_itch_parity.py
  ../../.venv/bin/ruff check hft tests benchmarks scripts
  ../../.venv/bin/ruff format --check hft tests benchmarks scripts
  ```
  Check once that `python -c "import hft; print(hft.__file__)"` run from the worktree
  resolves to the worktree copy.
- Add each new source to `hftcore_core` and each new test file to `hftcore_tests` in
  `cpp/CMakeLists.txt`. Benchmarks are built by default (CI compiles them) but never
  run in CI.
- Raw ITCH data lives only in the main checkout's git-ignored `data/itch/`; never
  commit, print in bulk, or publish raw messages or per-order output. Tests use
  synthetic messages only.
- No absolute home paths in tracked files. No `Co-Authored-By` trailers. Never print
  `.env` values.
- Delete temporary files you create; leave `cpp/build*` and `build/hftcore-py` (git
  ignored) for later tasks.

## Shared interfaces

Later tasks build on these exact names. A task may add helpers but must not rename
or change these signatures.

### `cpp/include/hftcore/itch.hpp` (Task 1)

```cpp
namespace hftcore::itch {
std::uint16_t load_be16(const std::uint8_t* p);
std::uint32_t load_be32(const std::uint8_t* p);
std::uint64_t load_be48(const std::uint8_t* p);   // 6-byte timestamp
std::uint64_t load_be64(const std::uint8_t* p);
using Stock = std::array<char, 8>;                 // space padded, as on the wire
std::string_view trim(const Stock&);               // without trailing spaces

struct Header { char type; std::uint16_t locate; std::uint16_t tracking; std::uint64_t timestamp; };
// One struct per message, fields in wire order, named after the spec table:
struct SystemEvent { Header h; char event_code; };                                     // S
struct StockDirectory { Header h; Stock stock; char market_category; char financial_status;
  std::uint32_t round_lot_size; char round_lots_only; char issue_classification;
  std::array<char, 2> issue_subtype; char authenticity; char short_sale_threshold;
  char ipo_flag; char luld_tier; char etp_flag; std::uint32_t etp_leverage; char inverse; }; // R
struct TradingAction { Header h; Stock stock; char state; char reserved; std::array<char, 4> reason; }; // H
struct RegSho { Header h; Stock stock; char action; };                                 // Y
struct ParticipantPosition { Header h; std::array<char, 4> mpid; Stock stock; char primary;
  char mode; char state; };                                                            // L
struct MwcbDecline { Header h; std::uint64_t level1, level2, level3; };                // V, Price(8)
struct MwcbStatus { Header h; char level; };                                           // W
struct IpoQuoting { Header h; Stock stock; std::uint32_t release_time; char release_qualifier;
  std::uint32_t price; };                                                              // K
struct LuldCollar { Header h; Stock stock; std::uint32_t reference, upper, lower, extension; }; // J
struct OperationalHalt { Header h; Stock stock; char market_code; char action; };      // h
struct AddOrder { Header h; std::uint64_t ref; char side; std::uint32_t shares; Stock stock;
  std::uint32_t price; std::array<char, 4> attribution; };  // A and F; h.type tells which,
                                                            // attribution is spaces for A
struct OrderExecuted { Header h; std::uint64_t ref; std::uint32_t shares; std::uint64_t match; }; // E
struct OrderExecutedWithPrice { Header h; std::uint64_t ref; std::uint32_t shares;
  std::uint64_t match; char printable; std::uint32_t price; };                         // C
struct OrderCancel { Header h; std::uint64_t ref; std::uint32_t shares; };             // X
struct OrderDelete { Header h; std::uint64_t ref; };                                   // D
struct OrderReplace { Header h; std::uint64_t ref; std::uint64_t new_ref; std::uint32_t shares;
  std::uint32_t price; };                                                              // U
struct Trade { Header h; std::uint64_t ref; char side; std::uint32_t shares; Stock stock;
  std::uint32_t price; std::uint64_t match; };                                         // P
struct CrossTrade { Header h; std::uint64_t shares; Stock stock; std::uint32_t price;
  std::uint64_t match; char cross_type; };                                             // Q
struct BrokenTrade { Header h; std::uint64_t match; };                                 // B
struct Noii { Header h; std::uint64_t paired, imbalance; char direction; Stock stock;
  std::uint32_t far_price, near_price, reference; char cross_type; char price_variation; }; // I
struct Rpii { Header h; Stock stock; char interest; };                                 // N
struct PriceDiscovery { Header h; Stock stock; char open_eligibility; std::uint32_t min_price,
  max_price, near_price; std::uint64_t near_time; std::uint32_t lower_collar, upper_collar; }; // O

constexpr std::size_t message_length(char type) noexcept;   // spec table; 0 = unknown type

struct ReadStats {
  std::uint64_t messages = 0;          // every frame with length > 0 (known or not)
  std::array<std::uint64_t, 256> by_type{};  // frames by type byte (all frames > 0)
  std::uint64_t unknown_types = 0;     // type not in the table: skipped by length
  std::uint64_t malformed = 0;         // known type, length differs from the table: skipped
  std::uint64_t trailing_bytes = 0;    // bytes seen after the end-of-session marker
  bool end_of_session = false;         // zero-length frame seen
};
enum class FrameResult { Ok, Unknown, Malformed };

// Decodes one message body (type byte first, `length` bytes) and calls
// visitor(const Msg&) when that call is well-formed (`if constexpr (requires ...)`),
// so a visitor implements only the messages it needs.
template <class Visitor>
FrameResult dispatch(const std::uint8_t* body, std::size_t length, Visitor& visitor);

// Walks BinaryFILE bytes (2-byte big-endian length prefix per frame). Stops before an
// incomplete frame and returns the bytes consumed so the caller can carry the tail.
// At a zero-length frame it sets end_of_session, consumes it, and from then on counts
// all remaining bytes (in this and later calls) as trailing_bytes and consumes them.
template <class Visitor>
std::size_t parse_binary_file(std::span<const std::uint8_t> data, Visitor& visitor,
                              ReadStats& stats);
}  // namespace hftcore::itch
```

### `cpp/include/hftcore/order_book.hpp` (Task 2)

```cpp
namespace hftcore {
enum class Side : char { Buy = 'B', Sell = 'S' };
struct LevelView { std::uint32_t price; std::uint64_t shares; std::uint32_t orders;
  bool operator==(const LevelView&) const = default; };

class OrderMap {   // ref -> pool index; open addressing, linear probing, backward shift
 public:
  static constexpr std::uint32_t kNone = 0xffffffffu;
  explicit OrderMap(std::size_t initial_slots = 1 << 16);  // rounded up to a power of 2
  bool insert(std::uint64_t ref, std::uint32_t index);      // false if ref present
  std::uint32_t find(std::uint64_t ref) const;               // kNone if absent
  bool erase(std::uint64_t ref);                             // false if absent
  std::size_t size() const; std::size_t slots() const;
};

class OrderBook {   // all stocks; per-locate sides indexed by a 65,536-entry array
 public:
  enum class Result { Ok, UnknownRef, DuplicateRef, Overfill, Invalid };
  struct Fill { Result result; std::uint16_t locate; Side side; std::uint32_t price;
                std::uint32_t shares; };   // shares actually removed from the book
  Result add(std::uint64_t ref, std::uint16_t locate, Side side, std::uint32_t shares,
             std::uint32_t price);                 // shares == 0 -> Invalid, no change
  Fill execute(std::uint64_t ref, std::uint32_t shares);  // price = order price
  Fill cancel(std::uint64_t ref, std::uint32_t shares);
  Fill remove(std::uint64_t ref);                  // D: shares = remaining shares
  Fill replace(std::uint64_t ref, std::uint64_t new_ref, std::uint32_t shares,
               std::uint32_t price);               // Fill describes the new order
  std::optional<LevelView> best(std::uint16_t locate, Side side) const;
  std::vector<LevelView> depth(std::uint16_t locate, Side side, std::size_t n) const;  // best first
  std::vector<std::pair<std::uint64_t, std::uint32_t>> queue(std::uint16_t locate, Side side,
                                                             std::uint32_t price) const;  // FIFO
  bool crossed(std::uint16_t locate) const;        // both sides present and bid >= ask
  std::vector<std::uint16_t> locates() const;      // sorted locates with live orders
  std::size_t live_orders() const; std::size_t peak_live_orders() const;
  std::size_t levels() const; std::size_t peak_levels() const;
  std::string validate() const;                    // "" when every invariant holds
};
}
```

Book rules (the Python reference implements the same):

- `add`: `shares == 0` gives `Invalid`; a live ref gives `DuplicateRef`; neither
  changes the book. Otherwise the order joins the back of its price level's queue.
- `execute`/`cancel`: unknown ref gives `UnknownRef`. `shares == 0` is `Ok` with no
  change. `shares` below remaining reduces the order in place (keeps its queue
  position). `shares == remaining` removes it (`Ok`). `shares > remaining` removes it
  and returns `Overfill` with `Fill.shares = remaining`.
- `remove`: unknown ref gives `UnknownRef`; otherwise removes the order.
- `replace`: checks before changing anything: unknown original gives `UnknownRef`;
  `shares == 0` gives `Invalid`; `new_ref` live and different from `ref` gives
  `DuplicateRef`. Otherwise removes the original and adds `new_ref` with the original
  locate and side, the new shares and price, at the back of its level.
- Every level that loses its last order is erased.

### `cpp/include/hftcore/itch_book.hpp` (Task 3)

```cpp
namespace hftcore {
struct StockInfo { std::string symbol; char trading_state = ' '; std::uint64_t trades = 0;
  std::uint64_t volume = 0; std::uint32_t last_price = 0;
  std::optional<std::uint32_t> open_cross, close_cross; };
struct ItchStats {
  itch::ReadStats read; std::uint64_t truncated_bytes = 0;
  std::uint64_t unknown_refs = 0, duplicate_refs = 0, overfills = 0, invalid = 0,
                crossed = 0, filtered = 0;
  std::size_t live_orders = 0, peak_live_orders = 0, levels = 0, peak_levels = 0; };

class ItchBookBuilder {
 public:
  explicit ItchBookBuilder(std::vector<std::string> symbols = {});  // empty = every stock
  void feed(std::span<const std::uint8_t> data);  // carries an incomplete tail; copies at most one frame
  void finish();                                   // truncated_bytes = carried tail size
  // Visitor entry points (public so benchmarks can drive parse_binary_file directly):
  void operator()(const itch::StockDirectory&); void operator()(const itch::TradingAction&);
  void operator()(const itch::AddOrder&); void operator()(const itch::OrderExecuted&);
  void operator()(const itch::OrderExecutedWithPrice&); void operator()(const itch::OrderCancel&);
  void operator()(const itch::OrderDelete&); void operator()(const itch::OrderReplace&);
  void operator()(const itch::Trade&); void operator()(const itch::CrossTrade&);
  const OrderBook& book() const; const StockInfo& stock(std::uint16_t locate) const;
  ItchStats stats() const;
  itch::ReadStats& read_stats();                    // for callers driving parse_binary_file
};
}
```

Feed-handler rules (spec section "Feed handler"):

- `R` stores the trimmed symbol for its locate; with a filter, the locate is allowed
  only if the symbol is in the filter. `H` stores the trading state (stocks are
  stored whether or not they are filtered).
- `A`/`F`/`E`/`C`/`X`/`D`/`U`/`P`/`Q` whose header locate is not allowed increment
  `filtered` and are ignored. Book operations otherwise use the stored order's locate.
- Book results map to counters: `UnknownRef` -> `unknown_refs`, `DuplicateRef` ->
  `duplicate_refs`, `Overfill` -> `overfills`, `Invalid` -> `invalid`.
- Prints update the stock's `trades += 1`, `volume += shares`, `last_price = price`:
  `E` (Fill shares at the order price, only when Fill shares > 0), `C` with
  `printable == 'Y'` (Fill shares at the message price, only when > 0), `P` (message
  shares and price), `Q` with `shares > 0` (message shares and price; cross type `O`
  sets `open_cross`, `C` sets `close_cross`). Unknown refs never print.
- After an `A`, `F` or `U` that returned `Ok`, `crossed += 1` when
  `book.crossed(locate)`.

### Python reference `hft/itch.py` and `hftcore.ItchBook` (Task 4)

Both expose the same class shape with identical return values:

```python
class ItchBook:
    def __init__(self, symbols: Iterable[str] | None = None) -> None: ...
    def feed(self, data: bytes) -> None: ...          # carries an incomplete tail
    def finish(self) -> None: ...
    def snapshot(self, locate: int, depth: int) -> dict[str, list[tuple[int, int, int]]]:
        """{"bids": [(price, shares, orders), ...], "asks": [...]}, best first."""
    def queue(self, locate: int, side: str, price: int) -> list[tuple[int, int]]:
        """[(ref, shares), ...] in FIFO order; side is "B" or "S"."""
    def stock(self, locate: int) -> dict:
        """symbol, trading_state, trades, volume, last_price, open_cross, close_cross
        (None when unset)."""
    def locates(self) -> list[int]:
        """Sorted locates that have at least one live order."""
    def stats(self) -> dict:
        """messages, by_type ({"A": n, ...}, nonzero only), unknown_types, malformed,
        trailing_bytes, end_of_session, truncated_bytes, unknown_refs, duplicate_refs,
        overfills, invalid, crossed, filtered, live_orders, peak_live_orders, levels,
        peak_levels."""
    def validate(self) -> str: ...                    # "" when consistent
```

---

## Task 1: ITCH 5.0 parser

**Files:** create `cpp/include/hftcore/itch.hpp`, `cpp/src/itch.cpp` (non-template
helpers such as `trim`, if any), `cpp/tests/test_itch.cpp`; modify
`cpp/CMakeLists.txt`.

Implement the `itch.hpp` interface above exactly, with the lengths from the spec
table (`S`12 `R`39 `H`25 `Y`20 `L`26 `V`35 `W`12 `K`28 `J`35 `h`21 `A`36 `F`40 `E`31
`C`36 `X`23 `D`19 `U`35 `P`44 `Q`40 `B`19 `I`50 `N`20 `O`48). `message_length` is a
`constexpr` 256-entry table lookup. Decoders read each field at its wire offset; add a
`static_assert` per type that the sum of field sizes equals the table length.

Tests first (`test_itch.cpp`, a small byte-builder helper in the test file):

- Every one of the 23 types: build a message with distinctive values in every field,
  dispatch it, and check every decoded field and the header (locate, tracking,
  timestamp).
- Endianness: `load_be16/32/48/64` on known byte patterns (for example
  `01 02 03 04 05 06` -> `0x010203040506`), including the maximum 48-bit timestamp;
  the 6-byte load at the very end of a buffer allocated with exactly that many bytes
  (ASan catches an over-read).
- `parse_binary_file`: a stream of several messages followed by the zero-length
  marker -> `end_of_session`, counts by type; bytes after the marker go to
  `trailing_bytes`; an unknown type (`'Z'`, and a fake future type `'@'` with
  length 7) is skipped and counted, and the next message still decodes; a known type
  with a wrong length (`A` with 35 bytes) counts as malformed and is skipped; a
  truncated tail (a length prefix without its body, and a lone first byte of a length
  prefix) is not consumed (the return value stops before it); feeding the stream one
  byte at a time with carried tails yields the same visits and stats as one call.
- A visitor that handles only `AddOrder` compiles and receives only adds.

Commit: `feat(cpp): parse Nasdaq ITCH 5.0 BinaryFILE messages`.

## Task 2: Order map and full-depth order book

**Files:** create `cpp/include/hftcore/order_book.hpp`, `cpp/src/order_book.cpp`,
`cpp/tests/test_order_book.cpp`; modify `cpp/CMakeLists.txt`.

Implement the `order_book.hpp` interface and book rules above.

- Order pool: `std::vector<Order>` plus a free list of indices; `Order` holds ref,
  shares, price, locate, side, prev and next (pool indices, `kNone` at the ends).
- `OrderMap`: slots of `{ref, index}` with `index == kNone` meaning empty; home slot
  `detail::mix(ref) & mask` (`detail::mix` from `id_set.hpp`); linear probing;
  backward-shift deletion (no tombstones); doubles when `size + 1 > slots / 2`.
- Sides: a 65,536-entry array of per-locate `{bids, asks}` vectors of `Level
  {price, shares (uint64), orders, head, tail}`; bids sorted ascending and asks
  descending so the best price is at `back()`; find levels with `std::lower_bound`
  using the side's comparator.
- `peak_live_orders` and `peak_levels` are high-water marks of `live_orders()` and
  `levels()` (total levels over all stocks and sides).
- `validate()` checks: every level's shares equal the sum of its queue's shares and its
  orders equal the queue length; prev/next links are consistent; levels strictly
  sorted with the best at the back; no empty level; every live order is in the map
  and in exactly one level of its locate and side; map size equals live orders.

Tests first (`test_order_book.cpp`):

- FIFO inside a level: three adds at one price, `queue` returns them in add order;
  a partial execute of the first keeps its position; removing the middle one keeps
  the others in order.
- Execute down to zero in two partial steps removes the order and, when last, the
  level; `Fill.price` is the order price; overfill returns `Overfill` with the
  remaining shares and removes the order.
- Replace keeps side and locate, moves to the back of the new level, frees the old
  ref (a later `execute` of it is `UnknownRef`), and the rejection cases (unknown
  original, zero shares, live new ref) leave the book unchanged.
- `best`/`depth` ordering for both sides; `crossed` for locked and crossed books.
- `OrderMap`: many refs forced into colliding probe chains (pick refs whose
  `mix(ref) & mask` land on the same slot for a small map), erase from the start,
  middle and end of a chain, and check every remaining ref is still found; growth
  keeps every ref; erase of a missing ref returns false.
- Unknown refs on execute, cancel, remove and replace return `UnknownRef` without
  crashing; duplicate add returns `DuplicateRef`; zero-share add returns `Invalid`.
- Randomized (fixed seed, about 200k operations over a few locates and a narrow
  price band so levels collide): mixed add/execute/cancel/remove/replace including
  unknown refs; `validate()` returns `""` after every operation, and at the end the
  book equals a simple `std::map`-based model kept in the test (levels, queues).

Commit: `feat(cpp): add a full-depth order book with an open-addressing order map`.

## Task 3: ITCH feed handler

**Files:** create `cpp/include/hftcore/itch_book.hpp`, `cpp/src/itch_book.cpp`,
`cpp/tests/test_itch_book.cpp`; modify `cpp/CMakeLists.txt`.

Implement `ItchBookBuilder` and the feed-handler rules above. `feed` keeps a small
tail buffer: if a previous call left an incomplete frame, complete that one frame
from the new data first, then parse the rest of the new data in place, and keep the
new incomplete tail (if any). Per-stock state is a 65,536-entry array indexed by
locate (`StockInfo` for unseen locates has an empty symbol).

Tests first (`test_itch_book.cpp`, building byte streams with a helper):

- Directory: `R` sets the symbol (trimmed); `H` sets the trading state.
- Symbol filter: with `{"AAPL"}`, adds/executes/prints for another locate count as
  `filtered` and never reach the book or unknown refs; AAPL is booked.
- `P`, `Q` and `B` leave the book unchanged (compare `depth` before and after) and
  `P`/`Q` update trades, volume and last price; `Q` with types `O` and `C` sets the
  cross prices; `Q` with zero shares prints nothing.
- `E` prints at the order price, `C` with `Y` prints at its own price, `C` with `N`
  does not print; both reduce the book.
- `U` keeps side and stock even when the header locate differs from the stored one
  (use the stored locate), and a `U` for an unknown ref counts once in `unknown_refs`.
- Counters: unknown refs, duplicate refs, overfills, invalid adds, crossed after a
  crossing add, by-type counts, live and peak live orders.
- Chunking: the same stream fed whole and fed in random chunk sizes (fixed seed,
  1-50 bytes) gives identical books and stats; `finish()` with a partial frame sets
  `truncated_bytes`.

Commit: `feat(cpp): rebuild per-stock books from ITCH messages`.

## Task 4: Python reference, bindings and parity

**Files:** create `hft/itch.py`, `tests/test_itch.py`, `tests/test_itch_parity.py`;
modify `cpp/src/bindings.cpp` (add `ItchBook`).

`hft/itch.py` (module docstring; type annotations; no performance tricks):

- `MESSAGE_FIELDS: dict[str, tuple[tuple[str, int, str], ...]]` describing the body
  after the 11-byte header for all 23 types (name, size, kind: `"u"` unsigned
  big-endian int, `"a"` ASCII text padded with spaces, `"c"` single char), with field
  names matching the C++ struct members; `MESSAGE_LENGTHS` derived from it and
  asserted equal to the spec table.
- `encode(type: str, *, locate=0, tracking=0, timestamp=0, **fields) -> bytes` (missing
  fields default to 0 or spaces) and `frame(messages: Iterable[bytes], *, end: bool =
  True) -> bytes` (BinaryFILE length prefixes, optional zero-length marker).
- `decode(body: bytes) -> dict` and a frame reader with the same `ReadStats` semantics
  as the C++ `parse_binary_file` (messages, by_type, unknown_types, malformed,
  trailing_bytes, end_of_session, carried tail).
- `ItchBook`: the reference implementation of the shared Python interface above. Keep
  orders in a dict `ref -> [locate, side, price, shares]`, levels per `(locate, side)`
  as a dict `price -> list of refs` (FIFO) plus a `bisect`-maintained sorted price list
  per side so best-price lookups are cheap; same book rules and feed-handler rules.

`cpp/src/bindings.cpp`: `py::class_<ItchBookBuilder>(m, "ItchBook")` with the same
method names and return types (`feed` takes `py::bytes` without copying more than the
buffer view; `stats()["by_type"]` contains only nonzero types; `snapshot` returns
lists of tuples; `stock` returns `None` for unset crosses).

Tests:

- `tests/test_itch.py` (reference only, always runs): encode/decode round trip for
  all 23 types; lengths equal the table; reference book rules (FIFO, execute to zero,
  overfill, replace, unknown and duplicate refs, invalid adds, crossed count, filter,
  prints and cross prices, trailing bytes, truncated tail, chunked feeding).
- `tests/test_itch_parity.py`: import `hftcore` with the existing skip pattern
  (`HFT_REQUIRE_HFTCORE=1` turns the skip into a failure; copy the pattern from
  `tests/test_engine_parity.py`). Seeded random streams (at least 20 seeds, about
  20k messages each) over a few locates with directory messages, adds (A and F),
  partial and full executions (E and C, printable and not), cancels, deletes,
  replaces, overfills, unknown refs, duplicate refs, zero-share adds, P/Q/B, unknown
  types, a malformed frame, with and without a symbol filter, fed in random chunk
  sizes. After the whole stream (and at a few checkpoints) `stats()`, `locates()`,
  `snapshot(locate, 1000)` for every locate, `queue` for every level and `stock` for
  every locate must be equal between `hft.itch.ItchBook` and `hftcore.ItchBook`, and
  `validate()` returns `""` for both.

Commit: `feat(itch): add the Python reference book and hftcore bindings` (tests can
be a separate `test:` commit).

## Task 5: Benchmark, replay CLI and real-data parity script

**Files:** create `cpp/bench/bench_itch.cpp`, `cpp/bench/itch_replay.cpp`,
`scripts/itch_parity.py`; modify `cpp/CMakeLists.txt` (targets
`hftcore_bench_itch`, `hftcore_itch_replay` inside `HFTCORE_BUILD_BENCH`).

`hftcore_bench_itch FILE [--repeat N]` (decompressed BinaryFILE, mmapped read-only):

- Pass (a) parse-only: `parse_binary_file` with a visitor that folds a few fields of
  every message into a checksum (printed, so nothing is optimized away).
- Pass (b) parse plus full book: a fresh `ItchBookBuilder` driven by
  `parse_binary_file` over the mapping.
- For (a) and (b): messages, wall seconds (`steady_clock`), mean ns per message,
  messages per second; repeat N times (default 3) and print each run and the median.
- Pass (c) per-message latency: a fresh builder; for each frame, time
  `dispatch(body, length, builder)` with `rdtscp` (preceded by `_mm_lfence`) on x86-64
  (fall back to `steady_clock` elsewhere); calibrate ticks to ns against `steady_clock`
  over at least 200 ms; measure and print the timer overhead (median of 1M empty
  timed regions) and subtract it (clamped at 0). Record into a log-linear histogram
  (for example 2 significant digits, 64 exponent buckets; no per-sample storage).
  Report p50, p99, p99.9 and max overall and for `A`, `D`, `U`, `E` and `X`.
- Print peak RSS (`getrusage` `ru_maxrss`), peak live orders, peak levels, the
  `ItchStats` counters and the end-of-session flag.

`hftcore_itch_replay [--symbols A,B] [--validate-every N] FILE|-`:

- Reads the file or stdin in 1 MiB chunks into `ItchBookBuilder::feed`, then
  `finish()`. With `--validate-every N` it calls `book().validate()` every N messages
  (count via `stats().read.messages` between chunks is fine; validate at chunk
  boundaries when the count crossed a multiple of N) and exits non-zero with the
  message on the first failure.
- Prints aggregates only: messages, counts by type, unknown types, malformed,
  trailing and truncated bytes, end-of-session flag, unknown refs, duplicate refs,
  overfills, invalid, crossed, filtered, live orders at the end, peak live orders and
  levels, validations run; and, for each `--symbols` stock, trades, volume, open and
  close cross prices and the last book mid (best bid and ask at the end, or `-`), as
  dollars with 4 decimals.

`scripts/itch_parity.py FILE --messages N [--depth D]`: reads the first N messages
(frames) of a decompressed BinaryFILE without loading the whole file, feeds the same
bytes to `hft.itch.ItchBook` and `hftcore.ItchBook`, and compares `stats()` (minus
fields that depend on input end, if any), `locates()`, `snapshot(locate, D)` and every
`queue` and `stock` for every locate seen. Prints only counts: messages compared,
locates compared, levels compared, mismatches (and the first mismatching locate and
field, without prices or refs). Exit code 0 only when identical. Includes a module
docstring; ruff clean.

Smoke-test both binaries and the script on a small synthetic BinaryFILE written by
`hft.itch.frame` into a temporary file (delete it afterwards). Do not touch real data.

Commit: `feat(cpp): benchmark and replay CLI for ITCH order books` and
`feat(scripts): compare C++ and Python ITCH books on real data`.

## Task 6: Real-data verification and measurements

No code changes unless a bug is found (then: failing regression test first, fix, and
report it). Uses the main checkout's `data/itch/01302019.NASDAQ_ITCH50.gz`
(4,764,426,091 bytes). All runs with `taskset -c 0-3` (single-threaded runs on one core,
for example `taskset -c 2`), and the RSS limit of 8 GiB.

1. Verify the download size; decompress once with `taskset -c 0-3 pigz -dc` into
   `data/itch/01302019.NASDAQ_ITCH50` (about 11.2 GB; keep it for reruns).
2. `hftcore_itch_replay` on the raw file (Release build): the message count must be
   368,366,634; unknown refs and overfills 0; record every printed counter, and the
   close cross versus last mid for AAPL, MSFT, SPY, QQQ (aggregates only).
3. A validation run: `hftcore_itch_replay --validate-every 10000000` on the Release
   build (validate is O(book) so keep the interval large); record the number of
   validations and that all passed.
4. `hftcore_bench_itch` with `--repeat 3` pinned to one core (not 4-5); record all
   output, the CPU model (`lscpu`), compiler version and kernel.
5. `scripts/itch_parity.py` on the first 5,000,000 messages; record the output.
6. Optional if time allows: none (the 2025-2026 files are 5-18 GB; skip unless the
   owner asks).

Write every result (verbatim program output plus the environment) to the report file
given in the dispatch; do not edit docs in this task.

## Task 7: Documentation

**Files:** modify `docs/performance-results.md`, `cpp/README.md`, `README.md`,
`docs/architecture.md`, `AGENTS.md`, `CONTRIBUTING.md`, `docs/project-history.md`
(and `docs/agent-workflow.md` only if the workflow changed).

- `docs/performance-results.md`: a new dated section "Nasdaq ITCH 5.0 order book
  (November 13, 2026)" in the style of the existing C++ sections: data (day, sizes,
  message count, license stance), method (passes, timer, histogram, pinning), the
  measured table, the goals from the spec marked met or not met, verification
  (counters, validations, 5M-message parity), and the machine caveat (WSL2 on a
  shared laptop; self-reported calibration figures are not comparable measurements).
- `cpp/README.md`: component rows for `itch`, `order_book`, `itch_book`; build and run
  commands for the two new binaries (with `gzip -dc ... | hftcore_itch_replay -`).
- `README.md`: an engineering-highlights line with the measured numbers and a link;
  keep the limitations line accurate (no live ITCH feed; samples used locally only).
- `docs/architecture.md`: an `hftcore` C++ subsection covering the ITCH units and how
  they relate to (and are separate from) the trading engine.
- `AGENTS.md`: commands (bench, replay, parity script) and ownership (`hft/itch.py`,
  the new C++ units); `CONTRIBUTING.md` likewise.
- `docs/project-history.md`: a dated row in the goals table (career track, option 4)
  and an entry recording what was built and measured.

Every number must come from the Task 6 report. Commit: `docs: record the ITCH order
book design, results and commands`.
