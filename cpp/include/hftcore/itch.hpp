// Nasdaq TotalView-ITCH 5.0 parser: big-endian loads, one struct per message type,
// the message length table, a visitor-based dispatcher and a BinaryFILE frame walker.
// Parse-only code pays no virtual calls: the visitor is a template parameter and
// implements only the messages it needs.
#pragma once

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <string_view>

namespace hftcore::itch {

namespace detail {
constexpr std::uint16_t bswap(std::uint16_t v) noexcept { return __builtin_bswap16(v); }
constexpr std::uint32_t bswap(std::uint32_t v) noexcept { return __builtin_bswap32(v); }
constexpr std::uint64_t bswap(std::uint64_t v) noexcept { return __builtin_bswap64(v); }

template <class T>
inline T load_be(const std::uint8_t* p) noexcept {
  T v;
  std::memcpy(&v, p, sizeof(T));
  if constexpr (std::endian::native == std::endian::little) v = bswap(v);
  return v;
}
}  // namespace detail

// Reads exactly 2, 4, 6 and 8 bytes at p; never touches a byte past them.
inline std::uint16_t load_be16(const std::uint8_t* p) { return detail::load_be<std::uint16_t>(p); }
inline std::uint32_t load_be32(const std::uint8_t* p) { return detail::load_be<std::uint32_t>(p); }
inline std::uint64_t load_be64(const std::uint8_t* p) { return detail::load_be<std::uint64_t>(p); }
// The 6-byte timestamp is assembled from bytes, never an 8-byte load.
inline std::uint64_t load_be48(const std::uint8_t* p) {
  std::uint64_t v = 0;
  for (int i = 0; i < 6; ++i) v = (v << 8) | p[i];
  return v;
}

using Stock = std::array<char, 8>;  // space padded, as on the wire
// The symbol without its trailing spaces (a view into the argument).
std::string_view trim(const Stock& stock);

struct Header {
  char type;
  std::uint16_t locate;
  std::uint16_t tracking;
  std::uint64_t timestamp;  // nanoseconds since midnight
};

// One struct per message, fields in wire order. Prices are Price(4) in 1/10,000
// dollars except the V levels (Price(8), 1/10^8 dollars).
struct SystemEvent {  // S
  Header h;
  char event_code;
};
struct StockDirectory {  // R
  Header h;
  Stock stock;
  char market_category;
  char financial_status;
  std::uint32_t round_lot_size;
  char round_lots_only;
  char issue_classification;
  std::array<char, 2> issue_subtype;
  char authenticity;
  char short_sale_threshold;
  char ipo_flag;
  char luld_tier;
  char etp_flag;
  std::uint32_t etp_leverage;
  char inverse;
};
struct TradingAction {  // H
  Header h;
  Stock stock;
  char state;
  char reserved;
  std::array<char, 4> reason;
};
struct RegSho {  // Y
  Header h;
  Stock stock;
  char action;
};
struct ParticipantPosition {  // L
  Header h;
  std::array<char, 4> mpid;
  Stock stock;
  char primary;
  char mode;
  char state;
};
struct MwcbDecline {  // V
  Header h;
  std::uint64_t level1, level2, level3;
};
struct MwcbStatus {  // W
  Header h;
  char level;
};
struct IpoQuoting {  // K
  Header h;
  Stock stock;
  std::uint32_t release_time;
  char release_qualifier;
  std::uint32_t price;
};
struct LuldCollar {  // J
  Header h;
  Stock stock;
  std::uint32_t reference, upper, lower, extension;
};
struct OperationalHalt {  // h
  Header h;
  Stock stock;
  char market_code;
  char action;
};
struct AddOrder {  // A and F: h.type tells which; attribution is spaces for A
  Header h;
  std::uint64_t ref;
  char side;
  std::uint32_t shares;
  Stock stock;
  std::uint32_t price;
  std::array<char, 4> attribution;
};
struct OrderExecuted {  // E
  Header h;
  std::uint64_t ref;
  std::uint32_t shares;
  std::uint64_t match;
};
struct OrderExecutedWithPrice {  // C
  Header h;
  std::uint64_t ref;
  std::uint32_t shares;
  std::uint64_t match;
  char printable;
  std::uint32_t price;
};
struct OrderCancel {  // X
  Header h;
  std::uint64_t ref;
  std::uint32_t shares;
};
struct OrderDelete {  // D
  Header h;
  std::uint64_t ref;
};
struct OrderReplace {  // U
  Header h;
  std::uint64_t ref;
  std::uint64_t new_ref;
  std::uint32_t shares;
  std::uint32_t price;
};
struct Trade {  // P
  Header h;
  std::uint64_t ref;
  char side;
  std::uint32_t shares;
  Stock stock;
  std::uint32_t price;
  std::uint64_t match;
};
struct CrossTrade {  // Q
  Header h;
  std::uint64_t shares;
  Stock stock;
  std::uint32_t price;
  std::uint64_t match;
  char cross_type;
};
struct BrokenTrade {  // B
  Header h;
  std::uint64_t match;
};
struct Noii {  // I
  Header h;
  std::uint64_t paired, imbalance;
  char direction;
  Stock stock;
  std::uint32_t far_price, near_price, reference;
  char cross_type;
  char price_variation;
};
struct Rpii {  // N
  Header h;
  Stock stock;
  char interest;
};
struct PriceDiscovery {  // O
  Header h;
  Stock stock;
  char open_eligibility;
  std::uint32_t min_price, max_price, near_price;
  std::uint64_t near_time;
  std::uint32_t lower_collar, upper_collar;
};

namespace detail {
constexpr std::array<std::uint8_t, 256> make_length_table() {
  std::array<std::uint8_t, 256> t{};
  const struct {
    char type;
    std::uint8_t length;
  } entries[] = {{'S', 12}, {'R', 39}, {'H', 25}, {'Y', 20}, {'L', 26}, {'V', 35},
                 {'W', 12}, {'K', 28}, {'J', 35}, {'h', 21}, {'A', 36}, {'F', 40},
                 {'E', 31}, {'C', 36}, {'X', 23}, {'D', 19}, {'U', 35}, {'P', 44},
                 {'Q', 40}, {'B', 19}, {'I', 50}, {'N', 20}, {'O', 48}};
  for (const auto& e : entries) t[static_cast<std::uint8_t>(e.type)] = e.length;
  return t;
}
inline constexpr std::array<std::uint8_t, 256> kLengthTable = make_length_table();
}  // namespace detail

// Wire length of a message (type byte included); 0 = unknown type.
constexpr std::size_t message_length(char type) noexcept {
  return detail::kLengthTable[static_cast<std::uint8_t>(type)];
}

struct ReadStats {
  std::uint64_t messages = 0;                // every frame with length > 0 (known or not)
  std::array<std::uint64_t, 256> by_type{};  // frames by type byte (all frames > 0)
  std::uint64_t unknown_types = 0;           // type not in the table: skipped by length
  std::uint64_t malformed = 0;               // known type, length differs from the table: skipped
  std::uint64_t trailing_bytes = 0;          // bytes seen after the end-of-session marker
  bool end_of_session = false;               // zero-length frame seen
};
enum class FrameResult { Ok, Unknown, Malformed };

namespace detail {
// Sequential big-endian reader over one message body; the decoders read fields in
// wire order, so the field list in each decoder is the layout.
class Cursor {
 public:
  explicit Cursor(const std::uint8_t* p) : p_(p) {}
  char ch() { return static_cast<char>(*p_++); }
  std::uint16_t u16() { return advance(load_be16(p_), 2); }
  std::uint32_t u32() { return advance(load_be32(p_), 4); }
  std::uint64_t u48() { return advance(load_be48(p_), 6); }
  std::uint64_t u64() { return advance(load_be64(p_), 8); }
  template <std::size_t N>
  std::array<char, N> chars() {
    std::array<char, N> out;
    std::memcpy(out.data(), p_, N);
    p_ += N;
    return out;
  }
  Stock stock() { return chars<8>(); }
  Header header() {
    Header h;
    h.type = ch();
    h.locate = u16();
    h.tracking = u16();
    h.timestamp = u48();
    return h;
  }

 private:
  template <class T>
  T advance(T v, std::size_t n) {
    p_ += n;
    return v;
  }
  const std::uint8_t* p_;
};

// Wire size of Header; the static_asserts below add each message's field sizes to it.
inline constexpr std::size_t kHdr = 11;
}  // namespace detail

// Each assert states that the header plus the field sizes equals the table length.
static_assert(detail::kHdr + 1 == message_length('S'));
static_assert(detail::kHdr + 8 + 1 + 1 + 4 + 1 + 1 + 2 + 1 + 1 + 1 + 1 + 1 + 4 + 1 ==
              message_length('R'));
static_assert(detail::kHdr + 8 + 1 + 1 + 4 == message_length('H'));
static_assert(detail::kHdr + 8 + 1 == message_length('Y'));
static_assert(detail::kHdr + 4 + 8 + 1 + 1 + 1 == message_length('L'));
static_assert(detail::kHdr + 8 + 8 + 8 == message_length('V'));
static_assert(detail::kHdr + 1 == message_length('W'));
static_assert(detail::kHdr + 8 + 4 + 1 + 4 == message_length('K'));
static_assert(detail::kHdr + 8 + 4 + 4 + 4 + 4 == message_length('J'));
static_assert(detail::kHdr + 8 + 1 + 1 == message_length('h'));
static_assert(detail::kHdr + 8 + 1 + 4 + 8 + 4 == message_length('A'));
static_assert(detail::kHdr + 8 + 1 + 4 + 8 + 4 + 4 == message_length('F'));
static_assert(detail::kHdr + 8 + 4 + 8 == message_length('E'));
static_assert(detail::kHdr + 8 + 4 + 8 + 1 + 4 == message_length('C'));
static_assert(detail::kHdr + 8 + 4 == message_length('X'));
static_assert(detail::kHdr + 8 == message_length('D'));
static_assert(detail::kHdr + 8 + 8 + 4 + 4 == message_length('U'));
static_assert(detail::kHdr + 8 + 1 + 4 + 8 + 4 + 8 == message_length('P'));
static_assert(detail::kHdr + 8 + 8 + 4 + 8 + 1 == message_length('Q'));
static_assert(detail::kHdr + 8 == message_length('B'));
static_assert(detail::kHdr + 8 + 8 + 1 + 8 + 4 + 4 + 4 + 1 + 1 == message_length('I'));
static_assert(detail::kHdr + 8 + 1 == message_length('N'));
static_assert(detail::kHdr + 8 + 1 + 4 + 4 + 4 + 8 + 4 + 4 == message_length('O'));

namespace detail {
// Calls visitor(msg) only when the visitor accepts that message type.
template <class Visitor, class Msg>
inline void emit(Visitor& visitor, const Msg& msg) {
  if constexpr (requires { visitor(msg); }) visitor(msg);
}
}  // namespace detail

// Decodes one message body (type byte first, `length` bytes) and calls
// visitor(const Msg&) when that call is well-formed, so a visitor implements only
// the messages it needs. Reads nothing past `length` bytes.
template <class Visitor>
FrameResult dispatch(const std::uint8_t* body, std::size_t length, Visitor& visitor) {
  if (length == 0) return FrameResult::Malformed;
  const char type = static_cast<char>(body[0]);
  const std::size_t expected = message_length(type);
  if (expected == 0) return FrameResult::Unknown;
  if (length != expected) return FrameResult::Malformed;
  detail::Cursor c(body);
  const Header h = c.header();
  switch (type) {
    case 'S':
      detail::emit(visitor, SystemEvent{h, c.ch()});
      break;
    case 'R': {
      StockDirectory m{};
      m.h = h;
      m.stock = c.stock();
      m.market_category = c.ch();
      m.financial_status = c.ch();
      m.round_lot_size = c.u32();
      m.round_lots_only = c.ch();
      m.issue_classification = c.ch();
      m.issue_subtype = c.chars<2>();
      m.authenticity = c.ch();
      m.short_sale_threshold = c.ch();
      m.ipo_flag = c.ch();
      m.luld_tier = c.ch();
      m.etp_flag = c.ch();
      m.etp_leverage = c.u32();
      m.inverse = c.ch();
      detail::emit(visitor, m);
      break;
    }
    case 'H': {
      TradingAction m{};
      m.h = h;
      m.stock = c.stock();
      m.state = c.ch();
      m.reserved = c.ch();
      m.reason = c.chars<4>();
      detail::emit(visitor, m);
      break;
    }
    case 'Y': {
      RegSho m{};
      m.h = h;
      m.stock = c.stock();
      m.action = c.ch();
      detail::emit(visitor, m);
      break;
    }
    case 'L': {
      ParticipantPosition m{};
      m.h = h;
      m.mpid = c.chars<4>();
      m.stock = c.stock();
      m.primary = c.ch();
      m.mode = c.ch();
      m.state = c.ch();
      detail::emit(visitor, m);
      break;
    }
    case 'V': {
      MwcbDecline m{};
      m.h = h;
      m.level1 = c.u64();
      m.level2 = c.u64();
      m.level3 = c.u64();
      detail::emit(visitor, m);
      break;
    }
    case 'W':
      detail::emit(visitor, MwcbStatus{h, c.ch()});
      break;
    case 'K': {
      IpoQuoting m{};
      m.h = h;
      m.stock = c.stock();
      m.release_time = c.u32();
      m.release_qualifier = c.ch();
      m.price = c.u32();
      detail::emit(visitor, m);
      break;
    }
    case 'J': {
      LuldCollar m{};
      m.h = h;
      m.stock = c.stock();
      m.reference = c.u32();
      m.upper = c.u32();
      m.lower = c.u32();
      m.extension = c.u32();
      detail::emit(visitor, m);
      break;
    }
    case 'h': {
      OperationalHalt m{};
      m.h = h;
      m.stock = c.stock();
      m.market_code = c.ch();
      m.action = c.ch();
      detail::emit(visitor, m);
      break;
    }
    case 'A':
    case 'F': {
      AddOrder m{};
      m.h = h;
      m.ref = c.u64();
      m.side = c.ch();
      m.shares = c.u32();
      m.stock = c.stock();
      m.price = c.u32();
      if (type == 'F') {
        m.attribution = c.chars<4>();
      } else {
        m.attribution.fill(' ');
      }
      detail::emit(visitor, m);
      break;
    }
    case 'E': {
      OrderExecuted m{};
      m.h = h;
      m.ref = c.u64();
      m.shares = c.u32();
      m.match = c.u64();
      detail::emit(visitor, m);
      break;
    }
    case 'C': {
      OrderExecutedWithPrice m{};
      m.h = h;
      m.ref = c.u64();
      m.shares = c.u32();
      m.match = c.u64();
      m.printable = c.ch();
      m.price = c.u32();
      detail::emit(visitor, m);
      break;
    }
    case 'X': {
      OrderCancel m{};
      m.h = h;
      m.ref = c.u64();
      m.shares = c.u32();
      detail::emit(visitor, m);
      break;
    }
    case 'D':
      detail::emit(visitor, OrderDelete{h, c.u64()});
      break;
    case 'U': {
      OrderReplace m{};
      m.h = h;
      m.ref = c.u64();
      m.new_ref = c.u64();
      m.shares = c.u32();
      m.price = c.u32();
      detail::emit(visitor, m);
      break;
    }
    case 'P': {
      Trade m{};
      m.h = h;
      m.ref = c.u64();
      m.side = c.ch();
      m.shares = c.u32();
      m.stock = c.stock();
      m.price = c.u32();
      m.match = c.u64();
      detail::emit(visitor, m);
      break;
    }
    case 'Q': {
      CrossTrade m{};
      m.h = h;
      m.shares = c.u64();
      m.stock = c.stock();
      m.price = c.u32();
      m.match = c.u64();
      m.cross_type = c.ch();
      detail::emit(visitor, m);
      break;
    }
    case 'B':
      detail::emit(visitor, BrokenTrade{h, c.u64()});
      break;
    case 'I': {
      Noii m{};
      m.h = h;
      m.paired = c.u64();
      m.imbalance = c.u64();
      m.direction = c.ch();
      m.stock = c.stock();
      m.far_price = c.u32();
      m.near_price = c.u32();
      m.reference = c.u32();
      m.cross_type = c.ch();
      m.price_variation = c.ch();
      detail::emit(visitor, m);
      break;
    }
    case 'N': {
      Rpii m{};
      m.h = h;
      m.stock = c.stock();
      m.interest = c.ch();
      detail::emit(visitor, m);
      break;
    }
    case 'O': {
      PriceDiscovery m{};
      m.h = h;
      m.stock = c.stock();
      m.open_eligibility = c.ch();
      m.min_price = c.u32();
      m.max_price = c.u32();
      m.near_price = c.u32();
      m.near_time = c.u64();
      m.lower_collar = c.u32();
      m.upper_collar = c.u32();
      detail::emit(visitor, m);
      break;
    }
    default:  // unreachable: the length table already rejected every other type
      return FrameResult::Unknown;
  }
  return FrameResult::Ok;
}

// Walks BinaryFILE bytes (2-byte big-endian length prefix per frame). Stops before an
// incomplete frame and returns the bytes consumed so the caller can carry the tail.
// At a zero-length frame it sets end_of_session, consumes it, and from then on counts
// all remaining bytes (in this and later calls) as trailing_bytes and consumes them.
template <class Visitor>
std::size_t parse_binary_file(std::span<const std::uint8_t> data, Visitor& visitor,
                              ReadStats& stats) {
  if (stats.end_of_session) {
    stats.trailing_bytes += data.size();
    return data.size();
  }
  std::size_t pos = 0;
  while (data.size() - pos >= 2) {
    const std::size_t length = load_be16(data.data() + pos);
    if (length == 0) {
      stats.end_of_session = true;
      pos += 2;
      stats.trailing_bytes += data.size() - pos;
      return data.size();
    }
    if (data.size() - pos - 2 < length) break;  // incomplete frame: leave it for the caller
    const std::uint8_t* body = data.data() + pos + 2;
    ++stats.messages;
    ++stats.by_type[body[0]];
    switch (dispatch(body, length, visitor)) {
      case FrameResult::Ok:
        break;
      case FrameResult::Unknown:
        ++stats.unknown_types;
        break;
      case FrameResult::Malformed:
        ++stats.malformed;
        break;
    }
    pos += 2 + length;
  }
  return pos;
}

}  // namespace hftcore::itch
