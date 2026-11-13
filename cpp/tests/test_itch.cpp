// Unit tests for the ITCH 5.0 parser: big-endian loads, every message type decoded
// field by field from synthetic bytes, the BinaryFILE frame walker (unknown types,
// malformed lengths, truncated tails, the end-of-session marker, byte-at-a-time
// streaming) and selective visitors. No real exchange data is used.
#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

#include "hftcore/itch.hpp"

using namespace hftcore::itch;

namespace {

constexpr std::uint16_t kLocate = 0x1234;
constexpr std::uint16_t kTracking = 0x5678;
constexpr std::uint64_t kTimestamp = 0x0A0B0C0D0E0FULL;

// Big-endian byte builder for synthetic messages.
struct Bytes {
  std::vector<std::uint8_t> v;
  Bytes& ch(char c) { return u8(static_cast<std::uint8_t>(c)); }
  Bytes& u8(std::uint8_t x) {
    v.push_back(x);
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
  // ASCII text right-padded with spaces to n bytes.
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

Bytes header(char type, std::uint16_t locate = kLocate) {
  Bytes b;
  b.ch(type).u16(locate).u16(kTracking).u48(kTimestamp);
  return b;
}

// A BinaryFILE frame: 2-byte length prefix followed by the message.
Bytes frame(const Bytes& msg) {
  Bytes b;
  b.u16(static_cast<std::uint16_t>(msg.v.size())).append(msg);
  return b;
}

// Captures the last decoded message of each type.
struct Capture {
  std::tuple<std::optional<SystemEvent>, std::optional<StockDirectory>,
             std::optional<TradingAction>, std::optional<RegSho>,
             std::optional<ParticipantPosition>, std::optional<MwcbDecline>,
             std::optional<MwcbStatus>, std::optional<IpoQuoting>, std::optional<LuldCollar>,
             std::optional<OperationalHalt>, std::optional<AddOrder>, std::optional<OrderExecuted>,
             std::optional<OrderExecutedWithPrice>, std::optional<OrderCancel>,
             std::optional<OrderDelete>, std::optional<OrderReplace>, std::optional<Trade>,
             std::optional<CrossTrade>, std::optional<BrokenTrade>, std::optional<Noii>,
             std::optional<Rpii>, std::optional<PriceDiscovery>>
      store;
  int calls = 0;
  template <class M>
  void operator()(const M& m) {
    std::get<std::optional<M>>(store) = m;
    ++calls;
  }
};

// Decodes `msg` from a buffer of exactly its size (ASan flags any over-read).
template <class M>
M decode(const Bytes& msg) {
  auto exact = std::make_unique<std::uint8_t[]>(msg.v.size());
  std::copy(msg.v.begin(), msg.v.end(), exact.get());
  Capture cap;
  REQUIRE(dispatch(exact.get(), msg.v.size(), cap) == FrameResult::Ok);
  REQUIRE(cap.calls == 1);
  REQUIRE(std::get<std::optional<M>>(cap.store).has_value());
  return *std::get<std::optional<M>>(cap.store);
}

void check_header(const Header& h, char type) {
  CHECK(h.type == type);
  CHECK(h.locate == kLocate);
  CHECK(h.tracking == kTracking);
  CHECK(h.timestamp == kTimestamp);
}

// An `A` message cut to 35 bytes (the table says 36): a known type with a wrong length.
Bytes short_add() {
  Bytes b = header('A');
  b.u64(7).ch('B').u32(1).stock("AAPL").u32(1500000);
  b.v.pop_back();
  return b;
}

std::string sv(const Stock& s) { return std::string(trim(s)); }
std::string str(const char* data, std::size_t n) { return std::string(data, n); }

}  // namespace

TEST_CASE("big-endian loads decode known byte patterns") {
  const std::uint8_t b[] = {0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08};
  CHECK(load_be16(b) == 0x0102);
  CHECK(load_be32(b) == 0x01020304u);
  CHECK(load_be48(b) == 0x010203040506ULL);
  CHECK(load_be64(b) == 0x0102030405060708ULL);
  const std::uint8_t max[] = {0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff};
  CHECK(load_be48(max) == 0xffffffffffffULL);
  CHECK(load_be64(max) == 0xffffffffffffffffULL);
}

TEST_CASE("loads read exactly their width at the end of a buffer") {
  auto b2 = std::make_unique<std::uint8_t[]>(2);
  b2[0] = 0xab, b2[1] = 0xcd;
  CHECK(load_be16(b2.get()) == 0xabcd);
  auto b4 = std::make_unique<std::uint8_t[]>(4);
  for (int i = 0; i < 4; ++i) b4[i] = static_cast<std::uint8_t>(0x10 + i);
  CHECK(load_be32(b4.get()) == 0x10111213u);
  auto b6 = std::make_unique<std::uint8_t[]>(6);
  for (int i = 0; i < 6; ++i) b6[i] = static_cast<std::uint8_t>(0xa0 + i);
  CHECK(load_be48(b6.get()) == 0xa0a1a2a3a4a5ULL);
  auto b8 = std::make_unique<std::uint8_t[]>(8);
  for (int i = 0; i < 8; ++i) b8[i] = static_cast<std::uint8_t>(0x80 + i);
  CHECK(load_be64(b8.get()) == 0x8081828384858687ULL);
}

TEST_CASE("trim drops trailing spaces only") {
  CHECK(sv({'A', 'A', 'P', 'L', ' ', ' ', ' ', ' '}) == "AAPL");
  CHECK(sv({'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'}) == "ABCDEFGH");
  CHECK(sv({'B', 'R', 'K', ' ', 'B', ' ', ' ', ' '}) == "BRK B");
  CHECK(sv({' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '}).empty());
}

TEST_CASE("message_length matches the specification table") {
  const std::tuple<char, std::size_t> expected[] = {
      {'S', 12}, {'R', 39}, {'H', 25}, {'Y', 20}, {'L', 26}, {'V', 35}, {'W', 12}, {'K', 28},
      {'J', 35}, {'h', 21}, {'A', 36}, {'F', 40}, {'E', 31}, {'C', 36}, {'X', 23}, {'D', 19},
      {'U', 35}, {'P', 44}, {'Q', 40}, {'B', 19}, {'I', 50}, {'N', 20}, {'O', 48}};
  for (const auto& [type, length] : expected) CHECK(message_length(type) == length);
  std::size_t known = 0;
  for (int t = 0; t < 256; ++t) known += message_length(static_cast<char>(t)) != 0;
  CHECK(known == 23);
  CHECK(message_length('Z') == 0);
  CHECK(message_length('a') == 0);
  CHECK(message_length('\0') == 0);
  static_assert(message_length('A') == 36);  // usable in constant expressions
}

TEST_CASE("S system event") {
  auto m = decode<SystemEvent>(header('S').ch('O'));
  check_header(m.h, 'S');
  CHECK(m.event_code == 'O');
}

TEST_CASE("R stock directory") {
  Bytes b = header('R');
  b.stock("AAPL").ch('Q').ch('N').u32(100).ch('Y').ch('C').text("EQ", 2).ch('P').ch('1').ch('2');
  b.ch('3').ch('4').u32(0xDEADBEEF).ch('5');
  auto m = decode<StockDirectory>(b);
  check_header(m.h, 'R');
  CHECK(sv(m.stock) == "AAPL");
  CHECK(m.market_category == 'Q');
  CHECK(m.financial_status == 'N');
  CHECK(m.round_lot_size == 100);
  CHECK(m.round_lots_only == 'Y');
  CHECK(m.issue_classification == 'C');
  CHECK(str(m.issue_subtype.data(), 2) == "EQ");
  CHECK(m.authenticity == 'P');
  CHECK(m.short_sale_threshold == '1');
  CHECK(m.ipo_flag == '2');
  CHECK(m.luld_tier == '3');
  CHECK(m.etp_flag == '4');
  CHECK(m.etp_leverage == 0xDEADBEEFu);
  CHECK(m.inverse == '5');
}

TEST_CASE("H trading action") {
  auto m = decode<TradingAction>(header('H').stock("MSFT").ch('T').ch('x').text("ABCD", 4));
  check_header(m.h, 'H');
  CHECK(sv(m.stock) == "MSFT");
  CHECK(m.state == 'T');
  CHECK(m.reserved == 'x');
  CHECK(str(m.reason.data(), 4) == "ABCD");
}

TEST_CASE("Y Reg SHO") {
  auto m = decode<RegSho>(header('Y').stock("GME").ch('2'));
  check_header(m.h, 'Y');
  CHECK(sv(m.stock) == "GME");
  CHECK(m.action == '2');
}

TEST_CASE("L participant position") {
  auto m =
      decode<ParticipantPosition>(header('L').text("MPID", 4).stock("IBM").ch('Y').ch('N').ch('A'));
  check_header(m.h, 'L');
  CHECK(str(m.mpid.data(), 4) == "MPID");
  CHECK(sv(m.stock) == "IBM");
  CHECK(m.primary == 'Y');
  CHECK(m.mode == 'N');
  CHECK(m.state == 'A');
}

TEST_CASE("V MWCB decline level") {
  auto m = decode<MwcbDecline>(
      header('V').u64(0x1111111111111111ULL).u64(0x2222222222222222ULL).u64(0x3333333333333333ULL));
  check_header(m.h, 'V');
  CHECK(m.level1 == 0x1111111111111111ULL);
  CHECK(m.level2 == 0x2222222222222222ULL);
  CHECK(m.level3 == 0x3333333333333333ULL);
}

TEST_CASE("W MWCB status") {
  auto m = decode<MwcbStatus>(header('W').ch('2'));
  check_header(m.h, 'W');
  CHECK(m.level == '2');
}

TEST_CASE("K IPO quoting period") {
  auto m = decode<IpoQuoting>(header('K').stock("NEWCO").u32(0x01020304).ch('A').u32(0x05060708));
  check_header(m.h, 'K');
  CHECK(sv(m.stock) == "NEWCO");
  CHECK(m.release_time == 0x01020304u);
  CHECK(m.release_qualifier == 'A');
  CHECK(m.price == 0x05060708u);
}

TEST_CASE("J LULD auction collar") {
  auto m = decode<LuldCollar>(header('J').stock("TSLA").u32(11).u32(22).u32(33).u32(44));
  check_header(m.h, 'J');
  CHECK(sv(m.stock) == "TSLA");
  CHECK(m.reference == 11);
  CHECK(m.upper == 22);
  CHECK(m.lower == 33);
  CHECK(m.extension == 44);
}

TEST_CASE("h operational halt") {
  auto m = decode<OperationalHalt>(header('h').stock("AMD").ch('Q').ch('H'));
  check_header(m.h, 'h');
  CHECK(sv(m.stock) == "AMD");
  CHECK(m.market_code == 'Q');
  CHECK(m.action == 'H');
}

TEST_CASE("A add order") {
  auto m = decode<AddOrder>(
      header('A').u64(0x0102030405060708ULL).ch('B').u32(1234).stock("AAPL").u32(1500000));
  check_header(m.h, 'A');
  CHECK(m.ref == 0x0102030405060708ULL);
  CHECK(m.side == 'B');
  CHECK(m.shares == 1234);
  CHECK(sv(m.stock) == "AAPL");
  CHECK(m.price == 1500000);
  CHECK(str(m.attribution.data(), 4) == "    ");
}

TEST_CASE("F add order with MPID") {
  auto m = decode<AddOrder>(
      header('F').u64(99).ch('S').u32(77).stock("NVDA").u32(4200000).text("GSCO", 4));
  check_header(m.h, 'F');
  CHECK(m.ref == 99);
  CHECK(m.side == 'S');
  CHECK(m.shares == 77);
  CHECK(sv(m.stock) == "NVDA");
  CHECK(m.price == 4200000);
  CHECK(str(m.attribution.data(), 4) == "GSCO");
}

TEST_CASE("E order executed") {
  auto m = decode<OrderExecuted>(
      header('E').u64(0xAABBCCDDEEFF0011ULL).u32(500).u64(0x1122334455667788ULL));
  check_header(m.h, 'E');
  CHECK(m.ref == 0xAABBCCDDEEFF0011ULL);
  CHECK(m.shares == 500);
  CHECK(m.match == 0x1122334455667788ULL);
}

TEST_CASE("C order executed with price") {
  auto m = decode<OrderExecutedWithPrice>(
      header('C').u64(42).u32(300).u64(0x99AABBCCDDEEFF00ULL).ch('Y').u32(2500000));
  check_header(m.h, 'C');
  CHECK(m.ref == 42);
  CHECK(m.shares == 300);
  CHECK(m.match == 0x99AABBCCDDEEFF00ULL);
  CHECK(m.printable == 'Y');
  CHECK(m.price == 2500000);
}

TEST_CASE("X order cancel") {
  auto m = decode<OrderCancel>(header('X').u64(7).u32(25));
  check_header(m.h, 'X');
  CHECK(m.ref == 7);
  CHECK(m.shares == 25);
}

TEST_CASE("D order delete") {
  auto m = decode<OrderDelete>(header('D').u64(0xFEDCBA9876543210ULL));
  check_header(m.h, 'D');
  CHECK(m.ref == 0xFEDCBA9876543210ULL);
}

TEST_CASE("U order replace") {
  auto m = decode<OrderReplace>(header('U').u64(1001).u64(1002).u32(800).u32(3300000));
  check_header(m.h, 'U');
  CHECK(m.ref == 1001);
  CHECK(m.new_ref == 1002);
  CHECK(m.shares == 800);
  CHECK(m.price == 3300000);
}

TEST_CASE("P trade") {
  auto m = decode<Trade>(
      header('P').u64(55).ch('B').u32(10).stock("META").u32(5100000).u64(0x0F0E0D0C0B0A0908ULL));
  check_header(m.h, 'P');
  CHECK(m.ref == 55);
  CHECK(m.side == 'B');
  CHECK(m.shares == 10);
  CHECK(sv(m.stock) == "META");
  CHECK(m.price == 5100000);
  CHECK(m.match == 0x0F0E0D0C0B0A0908ULL);
}

TEST_CASE("Q cross trade") {
  auto m = decode<CrossTrade>(
      header('Q').u64(0x0000000100000000ULL).stock("ORCL").u32(1250000).u64(31337).ch('C'));
  check_header(m.h, 'Q');
  CHECK(m.shares == 0x0000000100000000ULL);
  CHECK(sv(m.stock) == "ORCL");
  CHECK(m.price == 1250000);
  CHECK(m.match == 31337);
  CHECK(m.cross_type == 'C');
}

TEST_CASE("B broken trade") {
  auto m = decode<BrokenTrade>(header('B').u64(0x7766554433221100ULL));
  check_header(m.h, 'B');
  CHECK(m.match == 0x7766554433221100ULL);
}

TEST_CASE("I NOII") {
  Bytes b = header('I');
  b.u64(111).u64(222).ch('B').stock("INTC").u32(1).u32(2).u32(3).ch('O').ch('L');
  auto m = decode<Noii>(b);
  check_header(m.h, 'I');
  CHECK(m.paired == 111);
  CHECK(m.imbalance == 222);
  CHECK(m.direction == 'B');
  CHECK(sv(m.stock) == "INTC");
  CHECK(m.far_price == 1);
  CHECK(m.near_price == 2);
  CHECK(m.reference == 3);
  CHECK(m.cross_type == 'O');
  CHECK(m.price_variation == 'L');
}

TEST_CASE("N RPII") {
  auto m = decode<Rpii>(header('N').stock("COIN").ch('S'));
  check_header(m.h, 'N');
  CHECK(sv(m.stock) == "COIN");
  CHECK(m.interest == 'S');
}

TEST_CASE("O price discovery") {
  Bytes b = header('O');
  b.stock("PLTR").ch('Y').u32(10).u32(20).u32(30).u64(0x0102030405060708ULL).u32(40).u32(50);
  auto m = decode<PriceDiscovery>(b);
  check_header(m.h, 'O');
  CHECK(sv(m.stock) == "PLTR");
  CHECK(m.open_eligibility == 'Y');
  CHECK(m.min_price == 10);
  CHECK(m.max_price == 20);
  CHECK(m.near_price == 30);
  CHECK(m.near_time == 0x0102030405060708ULL);
  CHECK(m.lower_collar == 40);
  CHECK(m.upper_collar == 50);
}

TEST_CASE("header decodes the maximum 48-bit timestamp and a high locate") {
  Bytes b;
  b.ch('S').u16(0xFFFF).u16(0xFFFF).u48(0xFFFFFFFFFFFFULL).ch('C');
  auto m = decode<SystemEvent>(b);
  CHECK(m.h.locate == 0xFFFF);
  CHECK(m.h.tracking == 0xFFFF);
  CHECK(m.h.timestamp == 0xFFFFFFFFFFFFULL);
}

TEST_CASE("dispatch rejects unknown types and wrong lengths without calling the visitor") {
  Capture cap;
  Bytes z = header('Z').ch('x');
  CHECK(dispatch(z.v.data(), z.v.size(), cap) == FrameResult::Unknown);
  Bytes shortA = short_add();
  REQUIRE(shortA.v.size() == 35);
  CHECK(dispatch(shortA.v.data(), shortA.v.size(), cap) == FrameResult::Malformed);
  Bytes longS = header('S').ch('O').ch('O');
  CHECK(dispatch(longS.v.data(), longS.v.size(), cap) == FrameResult::Malformed);
  CHECK(cap.calls == 0);
}

namespace {

// Records the visit order (type, ref) so streamed and one-shot parses can be compared.
struct Trace {
  std::vector<std::pair<char, std::uint64_t>> visits;
  void operator()(const AddOrder& m) { visits.emplace_back(m.h.type, m.ref); }
  void operator()(const OrderExecuted& m) { visits.emplace_back(m.h.type, m.ref); }
  void operator()(const OrderCancel& m) { visits.emplace_back(m.h.type, m.ref); }
  void operator()(const OrderDelete& m) { visits.emplace_back(m.h.type, m.ref); }
  void operator()(const SystemEvent& m) { visits.emplace_back(m.h.type, m.event_code); }
};

Bytes add(std::uint64_t ref) {
  return frame(header('A').u64(ref).ch('B').u32(100).stock("AAPL").u32(1500000));
}

void check_same_stats(const ReadStats& a, const ReadStats& b) {
  CHECK(a.messages == b.messages);
  CHECK(a.by_type == b.by_type);
  CHECK(a.unknown_types == b.unknown_types);
  CHECK(a.malformed == b.malformed);
  CHECK(a.trailing_bytes == b.trailing_bytes);
  CHECK(a.end_of_session == b.end_of_session);
}

std::size_t parse(const Bytes& b, Trace& t, ReadStats& s) {
  return parse_binary_file(std::span<const std::uint8_t>(b.v), t, s);
}

}  // namespace

TEST_CASE("parse_binary_file walks frames and stops at the end-of-session marker") {
  Bytes stream;
  stream.append(frame(header('S').ch('O')));
  stream.append(add(1)).append(add(2));
  stream.append(frame(header('E').u64(1).u32(40).u64(9)));
  stream.append(frame(header('X').u64(2).u32(10)));
  stream.append(frame(header('D').u64(2)));
  stream.u16(0);
  const std::size_t marker_end = stream.v.size();
  stream.u8(0xAA).u8(0xBB).u8(0xCC);
  Trace t;
  ReadStats s;
  CHECK(parse(stream, t, s) == stream.v.size());
  CHECK(s.end_of_session);
  CHECK(s.messages == 6);
  CHECK(s.by_type['S'] == 1);
  CHECK(s.by_type['A'] == 2);
  CHECK(s.by_type['E'] == 1);
  CHECK(s.by_type['X'] == 1);
  CHECK(s.by_type['D'] == 1);
  CHECK(s.unknown_types == 0);
  CHECK(s.malformed == 0);
  CHECK(s.trailing_bytes == stream.v.size() - marker_end);
  REQUIRE(t.visits.size() == 6);
  CHECK(t.visits[1] == std::pair<char, std::uint64_t>{'A', 1});
  CHECK(t.visits[5] == std::pair<char, std::uint64_t>{'D', 2});

  // Later calls count everything as trailing bytes and consume it.
  Bytes more;
  more.u16(36).u8(1).u8(2);
  CHECK(parse(more, t, s) == more.v.size());
  CHECK(s.trailing_bytes == stream.v.size() - marker_end + more.v.size());
  CHECK(s.messages == 6);
  CHECK(t.visits.size() == 6);
}

TEST_CASE("parse_binary_file skips unknown types and malformed lengths") {
  Bytes stream;
  stream.append(add(1));
  stream.append(frame(header('Z').ch('x')));  // unknown 'Z'
  Bytes future;
  future.ch('@').u8(1).u8(2).u8(3).u8(4).u8(5).u8(6);  // fake future type, length 7
  stream.append(frame(future));
  Bytes shortA = short_add();
  stream.append(frame(shortA));
  stream.append(add(2));
  Trace t;
  ReadStats s;
  CHECK(parse(stream, t, s) == stream.v.size());
  CHECK(s.messages == 5);
  CHECK(s.unknown_types == 2);
  CHECK(s.malformed == 1);
  CHECK(s.by_type['Z'] == 1);
  CHECK(s.by_type['@'] == 1);
  CHECK(s.by_type['A'] == 3);
  CHECK(!s.end_of_session);
  REQUIRE(t.visits.size() == 2);  // the malformed A never reaches the visitor
  CHECK(t.visits[0].second == 1);
  CHECK(t.visits[1].second == 2);
}

TEST_CASE("parse_binary_file does not consume a truncated tail") {
  const Bytes first = add(1);
  Bytes prefix_only = first;
  prefix_only.u16(36);  // a length prefix without its body
  Trace t;
  ReadStats s;
  CHECK(parse(prefix_only, t, s) == first.v.size());
  CHECK(s.messages == 1);

  Bytes partial_body = first;
  partial_body.u16(36).u8('A').u8(0);  // body cut short
  CHECK(parse(partial_body, t, s) == first.v.size());

  Bytes lone_byte = first;
  lone_byte.u8(0x00);  // first byte of a length prefix
  CHECK(parse(lone_byte, t, s) == first.v.size());

  Bytes lone_nonzero = first;
  lone_nonzero.u8(0x01);
  CHECK(parse(lone_nonzero, t, s) == first.v.size());
  CHECK(!s.end_of_session);
  CHECK(s.trailing_bytes == 0);

  ReadStats s2;
  Trace t2;
  Bytes empty;
  CHECK(parse(empty, t2, s2) == 0);
  CHECK(s2.messages == 0);
}

TEST_CASE("streaming one byte at a time matches a single call") {
  Bytes stream;
  stream.append(frame(header('S').ch('O')));
  stream.append(add(1)).append(add(2));
  stream.append(frame(header('Z').ch('x')));
  Bytes shortA = short_add();
  stream.append(frame(shortA));
  stream.append(frame(header('E').u64(1).u32(40).u64(9)));
  stream.append(frame(header('D').u64(2)));
  stream.u16(0).u8(1).u8(2).u8(3);

  Trace whole;
  ReadStats whole_stats;
  CHECK(parse(stream, whole, whole_stats) == stream.v.size());

  Trace streamed;
  ReadStats streamed_stats;
  std::vector<std::uint8_t> tail;
  for (std::uint8_t byte : stream.v) {
    tail.push_back(byte);
    const std::size_t used = parse_binary_file(tail, streamed, streamed_stats);
    REQUIRE(used <= tail.size());
    tail.erase(tail.begin(), tail.begin() + static_cast<std::ptrdiff_t>(used));
  }
  CHECK(tail.empty());
  CHECK(streamed.visits == whole.visits);
  check_same_stats(streamed_stats, whole_stats);
}

TEST_CASE("a visitor handling only AddOrder receives only adds") {
  struct AddsOnly {
    std::vector<std::uint64_t> refs;
    void operator()(const AddOrder& m) { refs.push_back(m.ref); }
  };
  Bytes stream;
  stream.append(frame(header('S').ch('O')));
  stream.append(add(11));
  stream.append(frame(header('D').u64(11)));
  stream.append(frame(header('F').u64(12).ch('S').u32(5).stock("MSFT").u32(10).text("GSCO", 4)));
  stream.append(frame(header('E').u64(12).u32(5).u64(1)));
  AddsOnly v;
  ReadStats s;
  CHECK(parse_binary_file(std::span<const std::uint8_t>(stream.v), v, s) == stream.v.size());
  CHECK(v.refs == std::vector<std::uint64_t>{11, 12});
  CHECK(s.messages == 5);
}
