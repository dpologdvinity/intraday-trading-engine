// Unit tests for the ITCH feed handler: stock directory and trading state, the symbol
// filter, prints and cross prices, the book effect of every message, the data-quality
// counters, and chunked feeding (random chunk sizes must give the same books and
// stats as one whole feed). Messages are synthetic; no real exchange data is used.
#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <random>
#include <string>
#include <string_view>
#include <vector>

#include "hftcore/itch_book.hpp"

using namespace hftcore;

namespace {

constexpr std::uint16_t kAapl = 1;
constexpr std::uint16_t kMsft = 2;

// Big-endian byte builder for synthetic messages.
struct Bytes {
  std::vector<std::uint8_t> v;
  Bytes& ch(char c) {
    v.push_back(static_cast<std::uint8_t>(c));
    return *this;
  }
  Bytes& be(std::uint64_t x, int n) {
    for (int i = n - 1; i >= 0; --i) v.push_back(static_cast<std::uint8_t>(x >> (8 * i)));
    return *this;
  }
  Bytes& u16(std::uint16_t x) { return be(x, 2); }
  Bytes& u32(std::uint32_t x) { return be(x, 4); }
  Bytes& u48(std::uint64_t x) { return be(x, 6); }
  Bytes& u64(std::uint64_t x) { return be(x, 8); }
  Bytes& text(std::string_view s, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) ch(i < s.size() ? s[i] : ' ');
    return *this;
  }
  Bytes& stock(std::string_view s) { return text(s, 8); }
  Bytes& append(const Bytes& o) {
    v.insert(v.end(), o.v.begin(), o.v.end());
    return *this;
  }
};

Bytes header(char type, std::uint16_t locate) {
  Bytes b;
  b.ch(type).u16(locate).u16(7).u48(123456789);
  return b;
}

// A BinaryFILE frame: 2-byte length prefix followed by the message.
Bytes frame(const Bytes& msg) {
  Bytes b;
  b.u16(static_cast<std::uint16_t>(msg.v.size())).append(msg);
  return b;
}

Bytes directory(std::uint16_t locate, std::string_view symbol) {
  Bytes b = header('R', locate);
  b.stock(symbol).ch('Q').ch('N').u32(100).ch('N').ch('C').text("  ", 2);
  b.ch('P').ch('N').ch('N').ch('1').ch('N').u32(0).ch('N');
  return frame(b);
}
Bytes action(std::uint16_t locate, std::string_view symbol, char state) {
  Bytes b = header('H', locate);
  b.stock(symbol).ch(state).ch(' ').text("", 4);
  return frame(b);
}
Bytes add(std::uint16_t locate, std::uint64_t ref, char side, std::uint32_t shares,
          std::uint32_t price, std::string_view symbol = "AAPL") {
  Bytes b = header('A', locate);
  b.u64(ref).ch(side).u32(shares).stock(symbol).u32(price);
  return frame(b);
}
Bytes add_attributed(std::uint16_t locate, std::uint64_t ref, char side, std::uint32_t shares,
                     std::uint32_t price) {
  Bytes b = header('F', locate);
  b.u64(ref).ch(side).u32(shares).stock("AAPL").u32(price).text("MPID", 4);
  return frame(b);
}
Bytes executed(std::uint16_t locate, std::uint64_t ref, std::uint32_t shares) {
  Bytes b = header('E', locate);
  b.u64(ref).u32(shares).u64(99);
  return frame(b);
}
Bytes executed_price(std::uint16_t locate, std::uint64_t ref, std::uint32_t shares, char printable,
                     std::uint32_t price) {
  Bytes b = header('C', locate);
  b.u64(ref).u32(shares).u64(99).ch(printable).u32(price);
  return frame(b);
}
Bytes cancel(std::uint16_t locate, std::uint64_t ref, std::uint32_t shares) {
  Bytes b = header('X', locate);
  b.u64(ref).u32(shares);
  return frame(b);
}
Bytes remove(std::uint16_t locate, std::uint64_t ref) {
  Bytes b = header('D', locate);
  b.u64(ref);
  return frame(b);
}
Bytes replace(std::uint16_t locate, std::uint64_t ref, std::uint64_t new_ref,
              std::uint32_t shares, std::uint32_t price) {
  Bytes b = header('U', locate);
  b.u64(ref).u64(new_ref).u32(shares).u32(price);
  return frame(b);
}
Bytes trade(std::uint16_t locate, std::uint32_t shares, std::uint32_t price) {
  Bytes b = header('P', locate);
  b.u64(0).ch('B').u32(shares).stock("AAPL").u32(price).u64(5);
  return frame(b);
}
Bytes cross(std::uint16_t locate, std::uint64_t shares, std::uint32_t price, char type) {
  Bytes b = header('Q', locate);
  b.u64(shares).stock("AAPL").u32(price).u64(6).ch(type);
  return frame(b);
}
Bytes broken(std::uint16_t locate) {
  Bytes b = header('B', locate);
  b.u64(5);
  return frame(b);
}
Bytes system_event() {
  Bytes b = header('S', 0);
  b.ch('O');
  return frame(b);
}

std::vector<std::uint8_t> concat(std::initializer_list<Bytes> parts) {
  Bytes all;
  for (const Bytes& p : parts) all.append(p);
  return all.v;
}

void feed_all(ItchBookBuilder& b, const std::vector<std::uint8_t>& bytes) {
  b.feed(std::span<const std::uint8_t>(bytes));
}

// Every level of every live locate, both sides, best first.
std::vector<std::vector<LevelView>> all_depth(const OrderBook& book) {
  std::vector<std::vector<LevelView>> out;
  for (std::uint16_t locate : book.locates()) {
    out.push_back(book.depth(locate, Side::Buy, 100000));
    out.push_back(book.depth(locate, Side::Sell, 100000));
  }
  return out;
}

bool same_stock(const StockInfo& a, const StockInfo& b) {
  return a.symbol == b.symbol && a.trading_state == b.trading_state && a.trades == b.trades &&
         a.volume == b.volume && a.last_price == b.last_price && a.open_cross == b.open_cross &&
         a.close_cross == b.close_cross;
}

bool same_stats(const ItchStats& a, const ItchStats& b) {
  return a.read.messages == b.read.messages && a.read.by_type == b.read.by_type &&
         a.read.unknown_types == b.read.unknown_types && a.read.malformed == b.read.malformed &&
         a.read.trailing_bytes == b.read.trailing_bytes &&
         a.read.end_of_session == b.read.end_of_session &&
         a.truncated_bytes == b.truncated_bytes && a.unknown_refs == b.unknown_refs &&
         a.duplicate_refs == b.duplicate_refs && a.overfills == b.overfills &&
         a.invalid == b.invalid && a.crossed == b.crossed && a.filtered == b.filtered &&
         a.live_orders == b.live_orders && a.peak_live_orders == b.peak_live_orders &&
         a.levels == b.levels && a.peak_levels == b.peak_levels;
}

}  // namespace

TEST_CASE("directory sets the trimmed symbol and H sets the trading state", "[itch_book]") {
  ItchBookBuilder b;
  CHECK(b.stock(kAapl).symbol.empty());  // never seen: empty symbol
  CHECK(b.stock(kAapl).trading_state == ' ');
  feed_all(b, concat({directory(kAapl, "AAPL"), directory(kMsft, "MSFT"),
                      action(kAapl, "AAPL", 'H'), system_event()}));
  CHECK(b.stock(kAapl).symbol == "AAPL");
  CHECK(b.stock(kMsft).symbol == "MSFT");
  CHECK(b.stock(kAapl).trading_state == 'H');
  CHECK(b.stock(kMsft).trading_state == ' ');
  feed_all(b, concat({action(kAapl, "AAPL", 'T')}));
  CHECK(b.stock(kAapl).trading_state == 'T');
  CHECK(b.stats().read.messages == 5);
}

TEST_CASE("an R message with a longer symbol keeps only the trimmed text", "[itch_book]") {
  ItchBookBuilder b;
  feed_all(b, concat({directory(kAapl, "BRK.A")}));
  CHECK(b.stock(kAapl).symbol == "BRK.A");
  feed_all(b, concat({directory(kAapl, "A")}));  // a later R replaces it
  CHECK(b.stock(kAapl).symbol == "A");
}

TEST_CASE("the symbol filter books only listed stocks and counts the rest as filtered",
          "[itch_book]") {
  ItchBookBuilder b({"AAPL"});
  feed_all(b, concat({directory(kAapl, "AAPL"), directory(kMsft, "MSFT"),
                      action(kMsft, "MSFT", 'T')}));
  CHECK(b.stock(kMsft).symbol == "MSFT");  // stored although filtered
  CHECK(b.stock(kMsft).trading_state == 'T');
  feed_all(b, concat({add(kAapl, 1, 'B', 100, 1000),
                      add(kMsft, 2, 'B', 100, 2000),         // filtered
                      executed(kMsft, 2, 50),                // filtered, not an unknown ref
                      executed_price(kMsft, 2, 50, 'Y', 1),  // filtered
                      cancel(kMsft, 2, 10), remove(kMsft, 2),
                      replace(kMsft, 2, 3, 10, 5),
                      trade(kMsft, 10, 5), cross(kMsft, 10, 5, 'O')}));
  const ItchStats s = b.stats();
  CHECK(s.filtered == 8);
  CHECK(s.unknown_refs == 0);
  CHECK(s.live_orders == 1);
  CHECK(b.book().locates() == std::vector<std::uint16_t>{kAapl});
  CHECK(b.stock(kMsft).trades == 0);
  CHECK(b.stock(kMsft).volume == 0);
  CHECK_FALSE(b.stock(kMsft).open_cross.has_value());
  // The booked stock still works.
  feed_all(b, concat({executed(kAapl, 1, 40)}));
  CHECK(b.stock(kAapl).trades == 1);
  CHECK(b.book().best(kAapl, Side::Buy)->shares == 60);
}

TEST_CASE("a filtered locate with no directory entry is not allowed", "[itch_book]") {
  ItchBookBuilder b({"AAPL"});
  feed_all(b, concat({add(kAapl, 1, 'B', 100, 1000)}));  // no R yet
  CHECK(b.stats().filtered == 1);
  CHECK(b.stats().live_orders == 0);
  // Without a filter the same stream is booked.
  ItchBookBuilder all;
  feed_all(all, concat({add(kAapl, 1, 'B', 100, 1000)}));
  CHECK(all.stats().filtered == 0);
  CHECK(all.stats().live_orders == 1);
}

TEST_CASE("an R message re-listing a locate changes whether it is allowed", "[itch_book]") {
  ItchBookBuilder b({"AAPL"});
  feed_all(b, concat({directory(kAapl, "AAPL"), add(kAapl, 1, 'B', 100, 1000),
                      directory(kAapl, "XXXX"), add(kAapl, 2, 'B', 100, 900)}));
  CHECK(b.stats().filtered == 1);
  CHECK(b.stats().live_orders == 1);
}

TEST_CASE("P, Q and B leave the book unchanged; P and Q print", "[itch_book]") {
  ItchBookBuilder b;
  feed_all(b, concat({directory(kAapl, "AAPL"), add(kAapl, 1, 'B', 100, 1000),
                      add(kAapl, 2, 'S', 100, 1010)}));
  const auto before = all_depth(b.book());
  feed_all(b, concat({trade(kAapl, 30, 1005), broken(kAapl), cross(kAapl, 500, 990, 'O'),
                      cross(kAapl, 700, 1008, 'C')}));
  CHECK(all_depth(b.book()) == before);
  const StockInfo& s = b.stock(kAapl);
  CHECK(s.trades == 3);
  CHECK(s.volume == 30 + 500 + 700);
  CHECK(s.last_price == 1008);
  CHECK(s.open_cross == 990);
  CHECK(s.close_cross == 1008);
  CHECK(b.stats().read.by_type['B'] == 1);
}

TEST_CASE("other cross types print without setting a cross price; zero shares print nothing",
          "[itch_book]") {
  ItchBookBuilder b;
  feed_all(b, concat({directory(kAapl, "AAPL"), cross(kAapl, 100, 1010, 'H'),
                      cross(kAapl, 0, 2000, 'O'), cross(kAapl, 0, 3000, 'C')}));
  const StockInfo& s = b.stock(kAapl);
  CHECK(s.trades == 1);
  CHECK(s.volume == 100);
  CHECK(s.last_price == 1010);
  CHECK_FALSE(s.open_cross.has_value());
  CHECK_FALSE(s.close_cross.has_value());
}

TEST_CASE("cross volume uses 64-bit shares", "[itch_book]") {
  ItchBookBuilder b;
  feed_all(b, concat({cross(kAapl, 5'000'000'000ULL, 1000, 'C')}));
  CHECK(b.stock(kAapl).volume == 5'000'000'000ULL);
}

TEST_CASE("E prints at the order price; C prints at its own price only when printable",
          "[itch_book]") {
  ItchBookBuilder b;
  feed_all(b, concat({directory(kAapl, "AAPL"), add(kAapl, 1, 'B', 100, 1000),
                      add(kAapl, 2, 'S', 100, 1010), add_attributed(kAapl, 3, 'B', 100, 990)}));
  feed_all(b, concat({executed(kAapl, 1, 40)}));
  CHECK(b.stock(kAapl).trades == 1);
  CHECK(b.stock(kAapl).volume == 40);
  CHECK(b.stock(kAapl).last_price == 1000);
  CHECK(b.book().best(kAapl, Side::Buy)->shares == 60);

  feed_all(b, concat({executed_price(kAapl, 2, 30, 'Y', 1004)}));
  CHECK(b.stock(kAapl).trades == 2);
  CHECK(b.stock(kAapl).volume == 70);
  CHECK(b.stock(kAapl).last_price == 1004);  // the message price, not the order price 1010
  CHECK(b.book().best(kAapl, Side::Sell)->shares == 70);

  feed_all(b, concat({executed_price(kAapl, 3, 25, 'N', 995)}));
  CHECK(b.stock(kAapl).trades == 2);  // not printable: no print
  CHECK(b.stock(kAapl).volume == 70);
  CHECK(b.stock(kAapl).last_price == 1004);
  CHECK(b.book().depth(kAapl, Side::Buy, 5).back() == LevelView{990, 75, 1});  // but reduced
}

TEST_CASE("X and D change the book without printing", "[itch_book]") {
  ItchBookBuilder b;
  feed_all(b, concat({directory(kAapl, "AAPL"), add(kAapl, 1, 'B', 100, 1000),
                      add(kAapl, 2, 'B', 50, 1000), cancel(kAapl, 1, 30)}));
  CHECK(b.book().best(kAapl, Side::Buy) == LevelView{1000, 120, 2});
  feed_all(b, concat({remove(kAapl, 2)}));
  CHECK(b.book().best(kAapl, Side::Buy) == LevelView{1000, 70, 1});
  CHECK(b.stock(kAapl).trades == 0);
  CHECK(b.stats().unknown_refs == 0);
}

TEST_CASE("U keeps side and the stored stock even when the header locate differs",
          "[itch_book]") {
  ItchBookBuilder b;
  feed_all(b, concat({directory(kAapl, "AAPL"), directory(kMsft, "MSFT"),
                      add(kAapl, 1, 'S', 100, 1010)}));
  feed_all(b, concat({replace(kMsft, 1, 11, 80, 1020)}));  // header locate says MSFT
  CHECK(b.book().locates() == std::vector<std::uint16_t>{kAapl});
  CHECK(b.book().best(kAapl, Side::Sell) == LevelView{1020, 80, 1});
  CHECK_FALSE(b.book().best(kMsft, Side::Sell).has_value());
  CHECK(b.book().queue(kAapl, Side::Sell, 1020).front().first == 11);
  CHECK(b.stats().unknown_refs == 0);
}

TEST_CASE("a U for an unknown ref counts once in unknown_refs", "[itch_book]") {
  ItchBookBuilder b;
  feed_all(b, concat({directory(kAapl, "AAPL"), replace(kAapl, 9, 10, 100, 1000)}));
  const ItchStats s = b.stats();
  CHECK(s.unknown_refs == 1);
  CHECK(s.live_orders == 0);
  CHECK(s.duplicate_refs == 0);
  CHECK(s.invalid == 0);
}

TEST_CASE("data-quality counters", "[itch_book]") {
  ItchBookBuilder b;
  feed_all(b, concat({directory(kAapl, "AAPL"), add(kAapl, 1, 'B', 100, 1000),
                      add(kAapl, 2, 'B', 100, 990)}));
  // Unknown refs: E, C, X, D never print.
  feed_all(b, concat({executed(kAapl, 77, 5), executed_price(kAapl, 77, 5, 'Y', 1000),
                      cancel(kAapl, 77, 5), remove(kAapl, 77)}));
  CHECK(b.stats().unknown_refs == 4);
  CHECK(b.stock(kAapl).trades == 0);
  // Duplicate refs: A and U (the book is unchanged).
  feed_all(b, concat({add(kAapl, 1, 'S', 10, 1100), replace(kAapl, 2, 1, 10, 1000)}));
  CHECK(b.stats().duplicate_refs == 2);
  CHECK(b.stats().live_orders == 2);
  // Invalid: zero-share A and U, and an A with an unknown side byte.
  feed_all(b, concat({add(kAapl, 5, 'B', 0, 1000), replace(kAapl, 2, 6, 0, 1000),
                      add(kAapl, 7, 'X', 10, 1000)}));
  CHECK(b.stats().invalid == 3);
  CHECK(b.stats().live_orders == 2);
  // Overfills: E and X for more than remains; E prints the remaining shares.
  feed_all(b, concat({executed(kAapl, 2, 500)}));
  CHECK(b.stats().overfills == 1);
  CHECK(b.stock(kAapl).trades == 1);
  CHECK(b.stock(kAapl).volume == 100);
  CHECK(b.stock(kAapl).last_price == 990);
  feed_all(b, concat({cancel(kAapl, 1, 101)}));
  CHECK(b.stats().overfills == 2);
  CHECK(b.stats().live_orders == 0);
  CHECK(b.stats().peak_live_orders == 2);
  CHECK(b.stats().peak_levels == 2);
  CHECK(b.stats().levels == 0);
}

TEST_CASE("an E for zero shares prints nothing", "[itch_book]") {
  ItchBookBuilder b;
  feed_all(b, concat({directory(kAapl, "AAPL"), add(kAapl, 1, 'B', 100, 1000),
                      executed(kAapl, 1, 0), executed_price(kAapl, 1, 0, 'Y', 1000)}));
  CHECK(b.stock(kAapl).trades == 0);
  CHECK(b.book().best(kAapl, Side::Buy)->shares == 100);
}

TEST_CASE("crossed counts accepted A, F and U that leave the book crossed", "[itch_book]") {
  ItchBookBuilder b;
  feed_all(b, concat({directory(kAapl, "AAPL"), add(kAapl, 1, 'B', 100, 1000),
                      add(kAapl, 2, 'S', 100, 1010)}));
  CHECK(b.stats().crossed == 0);
  feed_all(b, concat({add(kAapl, 3, 'S', 100, 1000)}));  // ask 1000 >= bid 1000
  CHECK(b.stats().crossed == 1);
  feed_all(b, concat({add_attributed(kAapl, 4, 'B', 10, 995)}));  // still crossed
  CHECK(b.stats().crossed == 2);
  feed_all(b, concat({replace(kAapl, 4, 5, 10, 996)}));
  CHECK(b.stats().crossed == 3);
  feed_all(b, concat({executed(kAapl, 3, 100)}));  // E does not count
  CHECK(b.stats().crossed == 3);
  CHECK(b.book().crossed(kAapl) == false);
  // A rejected add on a crossed book is not counted again.
  feed_all(b, concat({add(kAapl, 6, 'S', 10, 990), add(kAapl, 6, 'S', 10, 990)}));
  CHECK(b.stats().crossed == 4);
  CHECK(b.stats().duplicate_refs == 1);
}

TEST_CASE("by-type counts, live orders and an F add", "[itch_book]") {
  ItchBookBuilder b;
  feed_all(b, concat({system_event(), directory(kAapl, "AAPL"), add(kAapl, 1, 'B', 100, 1000),
                      add_attributed(kAapl, 2, 'S', 100, 1010)}));
  const ItchStats s = b.stats();
  CHECK(s.read.messages == 4);
  CHECK(s.read.by_type['S'] == 1);
  CHECK(s.read.by_type['R'] == 1);
  CHECK(s.read.by_type['A'] == 1);
  CHECK(s.read.by_type['F'] == 1);
  CHECK(s.live_orders == 2);
  CHECK(s.levels == 2);
  CHECK(b.book().validate().empty());
}

TEST_CASE("an incomplete frame is carried and completed by the next feed", "[itch_book]") {
  const auto bytes = concat({directory(kAapl, "AAPL"), add(kAapl, 1, 'B', 100, 1000)});
  const std::size_t frame_len = 2 + 36;
  // Split inside the length prefix, inside the body, and one byte short.
  for (std::size_t cut : {bytes.size() - frame_len + 1, bytes.size() - frame_len + 2,
                          bytes.size() - frame_len + 20, bytes.size() - 1}) {
    ItchBookBuilder b;
    b.feed(std::span<const std::uint8_t>(bytes.data(), cut));
    CHECK(b.stats().live_orders == 0);
    b.feed(std::span<const std::uint8_t>(bytes.data() + cut, bytes.size() - cut));
    b.finish();
    CHECK(b.stats().live_orders == 1);
    CHECK(b.stats().truncated_bytes == 0);
    CHECK(b.stats().read.messages == 2);
  }
}

TEST_CASE("feeding one byte at a time equals feeding the whole stream", "[itch_book]") {
  const auto bytes = concat({directory(kAapl, "AAPL"), add(kAapl, 1, 'B', 100, 1000),
                             add(kAapl, 2, 'S', 100, 1010), executed(kAapl, 1, 10)});
  ItchBookBuilder whole, slow;
  feed_all(whole, bytes);
  for (std::uint8_t byte : bytes) slow.feed(std::span<const std::uint8_t>(&byte, 1));
  whole.finish();
  slow.finish();
  CHECK(all_depth(whole.book()) == all_depth(slow.book()));
  CHECK(same_stats(whole.stats(), slow.stats()));
}

TEST_CASE("random chunk sizes give the same books and stats as one whole feed", "[itch_book]") {
  // A busy synthetic stream over three stocks, with every order message type and some
  // bad refs, ending with the end-of-session marker and trailing bytes.
  std::mt19937 rng(12345);
  Bytes stream;
  stream.append(system_event());
  for (std::uint16_t loc = 1; loc <= 3; ++loc) {
    stream.append(directory(loc, std::string("S") + std::to_string(loc)));
    stream.append(action(loc, std::string("S") + std::to_string(loc), 'T'));
  }
  std::vector<std::uint64_t> live;
  std::uint64_t next_ref = 1;
  for (int i = 0; i < 3000; ++i) {
    const auto loc = static_cast<std::uint16_t>(1 + rng() % 3);
    const std::uint32_t price = 1000 + rng() % 20;
    switch (rng() % 8) {
      case 0:
      case 1:
      case 2:
        live.push_back(next_ref);
        stream.append(add(loc, next_ref++, rng() % 2 ? 'B' : 'S', 1 + rng() % 200, price));
        break;
      case 3:
        if (!live.empty()) stream.append(executed(loc, live[rng() % live.size()], 1 + rng() % 150));
        break;
      case 4:
        if (!live.empty()) stream.append(cancel(loc, live[rng() % live.size()], 1 + rng() % 150));
        break;
      case 5:
        if (!live.empty()) stream.append(remove(loc, live[rng() % live.size()]));
        break;
      case 6:
        if (!live.empty()) {
          live.push_back(next_ref);
          stream.append(replace(loc, live[rng() % live.size()], next_ref++, 1 + rng() % 200,
                                price));
        }
        break;
      default:
        stream.append(trade(loc, 1 + rng() % 50, price));
        stream.append(cross(loc, 1 + rng() % 500, price, rng() % 2 ? 'O' : 'C'));
        stream.append(broken(loc));
        stream.append(executed(loc, 999999, 1));  // unknown ref
        break;
    }
  }
  stream.u16(0).append(Bytes{}.ch('x').ch('y').ch('z'));  // end of session + trailing bytes

  ItchBookBuilder whole;
  feed_all(whole, stream.v);
  whole.finish();
  REQUIRE(whole.stats().live_orders > 0);
  REQUIRE(whole.stats().unknown_refs > 0);
  REQUIRE(whole.stats().read.end_of_session);
  REQUIRE(whole.stats().read.trailing_bytes == 3);
  REQUIRE(whole.book().validate().empty());

  std::mt19937 chunker(777);
  for (int round = 0; round < 5; ++round) {
    ItchBookBuilder chunked;
    std::size_t pos = 0;
    while (pos < stream.v.size()) {
      const std::size_t n = std::min<std::size_t>(1 + chunker() % 50, stream.v.size() - pos);
      chunked.feed(std::span<const std::uint8_t>(stream.v.data() + pos, n));
      pos += n;
    }
    chunked.finish();
    CHECK(all_depth(whole.book()) == all_depth(chunked.book()));
    CHECK(same_stats(whole.stats(), chunked.stats()));
    for (std::uint16_t loc = 0; loc <= 4; ++loc) {
      CHECK(same_stock(whole.stock(loc), chunked.stock(loc)));
    }
    CHECK(chunked.book().validate().empty());
  }
}

TEST_CASE("finish with a partial frame sets truncated_bytes", "[itch_book]") {
  const auto bytes = concat({directory(kAapl, "AAPL"), add(kAapl, 1, 'B', 100, 1000)});
  ItchBookBuilder b;
  b.feed(std::span<const std::uint8_t>(bytes.data(), bytes.size() - 5));
  b.finish();
  const ItchStats s = b.stats();
  CHECK(s.read.messages == 1);
  CHECK(s.live_orders == 0);
  CHECK(s.truncated_bytes == 38 - 5);

  ItchBookBuilder clean;
  feed_all(clean, bytes);
  clean.finish();
  CHECK(clean.stats().truncated_bytes == 0);
}

TEST_CASE("a lone length-prefix byte is carried; finish counts it", "[itch_book]") {
  ItchBookBuilder b;
  const std::uint8_t one = 0x00;
  b.feed(std::span<const std::uint8_t>(&one, 1));
  b.finish();
  CHECK(b.stats().truncated_bytes == 1);
}

TEST_CASE("the end-of-session marker split across feeds counts trailing bytes", "[itch_book]") {
  Bytes stream;
  stream.append(directory(kAapl, "AAPL")).u16(0).ch('t').ch('t');
  ItchBookBuilder b;
  const auto& v = stream.v;
  const std::size_t marker = v.size() - 4;
  b.feed(std::span<const std::uint8_t>(v.data(), marker + 1));  // first marker byte only
  CHECK_FALSE(b.stats().read.end_of_session);
  b.feed(std::span<const std::uint8_t>(v.data() + marker + 1, v.size() - marker - 1));
  b.feed(std::span<const std::uint8_t>(v.data(), 3));  // more trailing bytes later
  b.finish();
  const ItchStats s = b.stats();
  CHECK(s.read.end_of_session);
  CHECK(s.read.trailing_bytes == 2 + 3);
  CHECK(s.truncated_bytes == 0);
  CHECK(s.read.messages == 1);
}

TEST_CASE("unknown and malformed frames are skipped and counted", "[itch_book]") {
  Bytes bad_type;
  bad_type.u16(3).ch('?').ch('a').ch('b');
  Bytes short_add;
  short_add.u16(5).ch('A').ch('a').ch('b').ch('c').ch('d');
  ItchBookBuilder b;
  feed_all(b, concat({bad_type, short_add, directory(kAapl, "AAPL")}));
  const ItchStats s = b.stats();
  CHECK(s.read.unknown_types == 1);
  CHECK(s.read.malformed == 1);
  CHECK(b.stock(kAapl).symbol == "AAPL");
}
