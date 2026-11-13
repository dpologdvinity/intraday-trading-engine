// ITCH feed handler: turns a stream of Nasdaq TotalView-ITCH 5.0 BinaryFILE bytes into
// per-stock order books. It is the parser's visitor (R/H/A/F/E/C/X/D/U/P/Q), keeps the
// stock directory per locate, an optional symbol filter, the trade/volume/cross
// counters and the data-quality counters, and carries an incomplete frame between
// feed() calls. The rules are mirrored exactly by the Python reference in hft/itch.py.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <set>
#include <span>
#include <string>
#include <vector>

#include "hftcore/itch.hpp"
#include "hftcore/order_book.hpp"

namespace hftcore {

// Directory and print state of one stock (one locate). A locate never seen in an R
// message has an empty symbol.
struct StockInfo {
  std::string symbol;        // trimmed, from the R message
  char trading_state = ' ';  // from the latest H message
  std::uint64_t trades = 0;  // prints counted (E, C printable, P, Q with shares)
  std::uint64_t volume = 0;
  std::uint32_t last_price = 0;  // price of the latest print, 1/10,000 dollars
  std::optional<std::uint32_t> open_cross, close_cross;  // Q with cross type O / C
};

struct ItchStats {
  itch::ReadStats read;
  std::uint64_t truncated_bytes = 0;  // incomplete tail left when finish() was called
  std::uint64_t unknown_refs = 0;     // E/C/X/D/U for a ref that is not live
  std::uint64_t duplicate_refs = 0;   // A/F/U whose new ref is already live
  std::uint64_t overfills = 0;        // E/C/X for more shares than remain
  std::uint64_t invalid = 0;          // A/F/U with zero shares, or A/F with an unknown side
  std::uint64_t crossed = 0;          // A/F/U accepted while the stock's book is crossed
  std::uint64_t filtered = 0;         // book/print messages for a locate outside the filter
  std::size_t live_orders = 0, peak_live_orders = 0, levels = 0, peak_levels = 0;
};

class ItchBookBuilder {
 public:
  // Only stocks whose R-message symbol is listed are booked; empty = every stock.
  explicit ItchBookBuilder(std::vector<std::string> symbols = {});

  // Consumes BinaryFILE bytes. An incomplete frame left by a previous call is completed
  // from the new data first (copying at most one frame), then the rest is parsed in
  // place and its own incomplete tail, if any, is carried to the next call.
  void feed(std::span<const std::uint8_t> data);
  // Marks the end of input: truncated_bytes = the size of the carried incomplete tail.
  void finish();

  // Visitor entry points (public so benchmarks can drive parse_binary_file directly).
  void operator()(const itch::StockDirectory& m);
  void operator()(const itch::TradingAction& m);
  void operator()(const itch::AddOrder& m);
  void operator()(const itch::OrderExecuted& m);
  void operator()(const itch::OrderExecutedWithPrice& m);
  void operator()(const itch::OrderCancel& m);
  void operator()(const itch::OrderDelete& m);
  void operator()(const itch::OrderReplace& m);
  void operator()(const itch::Trade& m);
  void operator()(const itch::CrossTrade& m);

  const OrderBook& book() const { return book_; }
  const StockInfo& stock(std::uint16_t locate) const { return stocks_[locate]; }
  ItchStats stats() const;
  itch::ReadStats& read_stats() { return read_; }  // for callers driving parse_binary_file

 private:
  bool allowed(std::uint16_t locate) const { return allowed_[locate] != 0; }
  // Counts a rejected book result; returns true when the result left the book changed
  // (Ok or Overfill).
  bool tally(OrderBook::Result r);
  void print(std::uint16_t locate, std::uint64_t shares, std::uint32_t price);
  void check_crossed(std::uint16_t locate);

  std::set<std::string, std::less<>> filter_;
  bool filtering_;
  std::vector<std::uint8_t> allowed_;  // 65,536 flags indexed by locate (O(1) check)
  std::vector<StockInfo> stocks_;      // 65,536 entries indexed by locate
  OrderBook book_;
  itch::ReadStats read_;
  // Appends bytes to the carried tail (never reallocates; see kMaxFrame).
  void append_tail(std::span<const std::uint8_t> bytes);

  // The largest frame BinaryFILE can carry: the 2-byte length prefix plus a body of at
  // most 65,535 bytes (the prefix is a big-endian uint16). A carried tail is always an
  // incomplete frame, so it is shorter than this and a fixed buffer holds it with no
  // heap allocation in feed().
  static constexpr std::size_t kMaxFrame = 2 + 65535;
  std::array<std::uint8_t, kMaxFrame> tail_{};  // an incomplete frame carried between feed() calls
  std::size_t tail_size_ = 0;                   // bytes of tail_ in use
  std::uint64_t truncated_bytes_ = 0;
  std::uint64_t unknown_refs_ = 0, duplicate_refs_ = 0, overfills_ = 0, invalid_ = 0,
                crossed_ = 0, filtered_ = 0;
};

}  // namespace hftcore
