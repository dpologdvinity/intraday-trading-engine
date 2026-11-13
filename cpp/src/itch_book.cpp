// ITCH feed handler (see itch_book.hpp): applies parsed messages to the order book and
// the per-stock directory, and carries an incomplete frame between feed() calls.
#include "hftcore/itch_book.hpp"

#include <algorithm>
#include <cassert>
#include <cstring>

namespace hftcore {

namespace {
constexpr std::size_t kLocates = 65536;
constexpr std::size_t kLengthPrefix = 2;
}  // namespace

ItchBookBuilder::ItchBookBuilder(std::vector<std::string> symbols)
    : filter_(symbols.begin(), symbols.end()),
      filtering_(!symbols.empty()),
      allowed_(kLocates, filtering_ ? 0 : 1),
      stocks_(kLocates) {}

void ItchBookBuilder::feed(std::span<const std::uint8_t> data) {
  // Complete the carried frame first, taking only the bytes it still needs.
  if (tail_size_ != 0) {
    if (tail_size_ < kLengthPrefix) {
      const std::size_t take = std::min(kLengthPrefix - tail_size_, data.size());
      append_tail(data.first(take));
      data = data.subspan(take);
      if (tail_size_ < kLengthPrefix) return;
    }
    const std::size_t need = kLengthPrefix + itch::load_be16(tail_.data());
    const std::size_t take = std::min(need - tail_size_, data.size());
    append_tail(data.first(take));
    data = data.subspan(take);
    if (tail_size_ < need) return;
    // Exactly one whole frame (or the end-of-session marker): parse it from the buffer.
    itch::parse_binary_file(std::span<const std::uint8_t>(tail_.data(), tail_size_), *this,
                            read_);
    tail_size_ = 0;
  }
  const std::size_t used = itch::parse_binary_file(data, *this, read_);
  append_tail(data.subspan(used));
}

void ItchBookBuilder::append_tail(std::span<const std::uint8_t> bytes) {
  // parse_binary_file stops only before a frame whose length prefix promises more bytes
  // than remain, and the completion step above never takes more than the frame needs,
  // so the carried tail stays shorter than one maximal frame and always fits.
  assert(bytes.size() <= tail_.size() - tail_size_);
  if (bytes.empty()) return;
  std::memcpy(tail_.data() + tail_size_, bytes.data(), bytes.size());
  tail_size_ += bytes.size();
}

void ItchBookBuilder::finish() { truncated_bytes_ = tail_size_; }

ItchStats ItchBookBuilder::stats() const {
  ItchStats s;
  s.read = read_;
  s.truncated_bytes = truncated_bytes_;
  s.unknown_refs = unknown_refs_;
  s.duplicate_refs = duplicate_refs_;
  s.overfills = overfills_;
  s.invalid = invalid_;
  s.crossed = crossed_;
  s.filtered = filtered_;
  s.live_orders = book_.live_orders();
  s.peak_live_orders = book_.peak_live_orders();
  s.levels = book_.levels();
  s.peak_levels = book_.peak_levels();
  return s;
}

bool ItchBookBuilder::tally(OrderBook::Result r) {
  switch (r) {
    case OrderBook::Result::Ok:
      return true;
    case OrderBook::Result::Overfill:
      ++overfills_;
      return true;
    case OrderBook::Result::UnknownRef:
      ++unknown_refs_;
      return false;
    case OrderBook::Result::DuplicateRef:
      ++duplicate_refs_;
      return false;
    case OrderBook::Result::Invalid:
      ++invalid_;
      return false;
  }
  return false;
}

void ItchBookBuilder::print(std::uint16_t locate, std::uint64_t shares, std::uint32_t price) {
  StockInfo& s = stocks_[locate];
  ++s.trades;
  s.volume += shares;
  s.last_price = price;
}

void ItchBookBuilder::check_crossed(std::uint16_t locate) {
  if (book_.crossed(locate)) ++crossed_;
}

void ItchBookBuilder::operator()(const itch::StockDirectory& m) {
  StockInfo& s = stocks_[m.h.locate];
  s.symbol = std::string(itch::trim(m.stock));
  const bool listed = !filtering_ || filter_.find(std::string_view(s.symbol)) != filter_.end();
  allowed_[m.h.locate] = listed ? 1 : 0;
}

void ItchBookBuilder::operator()(const itch::TradingAction& m) {
  stocks_[m.h.locate].trading_state = m.state;
}

void ItchBookBuilder::operator()(const itch::AddOrder& m) {
  const std::uint16_t locate = m.h.locate;
  if (!allowed(locate)) {
    ++filtered_;
    return;
  }
  if (m.side != 'B' && m.side != 'S') {  // the book only knows buy and sell
    ++invalid_;
    return;
  }
  const Side side = m.side == 'B' ? Side::Buy : Side::Sell;
  if (tally(book_.add(m.ref, locate, side, m.shares, m.price))) check_crossed(locate);
}

void ItchBookBuilder::operator()(const itch::OrderExecuted& m) {
  if (!allowed(m.h.locate)) {
    ++filtered_;
    return;
  }
  const OrderBook::Fill f = book_.execute(m.ref, m.shares);
  if (tally(f.result) && f.shares > 0) print(f.locate, f.shares, f.price);
}

void ItchBookBuilder::operator()(const itch::OrderExecutedWithPrice& m) {
  if (!allowed(m.h.locate)) {
    ++filtered_;
    return;
  }
  const OrderBook::Fill f = book_.execute(m.ref, m.shares);
  if (tally(f.result) && m.printable == 'Y' && f.shares > 0) print(f.locate, f.shares, m.price);
}

void ItchBookBuilder::operator()(const itch::OrderCancel& m) {
  if (!allowed(m.h.locate)) {
    ++filtered_;
    return;
  }
  tally(book_.cancel(m.ref, m.shares).result);
}

void ItchBookBuilder::operator()(const itch::OrderDelete& m) {
  if (!allowed(m.h.locate)) {
    ++filtered_;
    return;
  }
  tally(book_.remove(m.ref).result);
}

void ItchBookBuilder::operator()(const itch::OrderReplace& m) {
  if (!allowed(m.h.locate)) {
    ++filtered_;
    return;
  }
  const OrderBook::Fill f = book_.replace(m.ref, m.new_ref, m.shares, m.price);
  if (tally(f.result)) check_crossed(f.locate);  // the stored locate, not the header's
}

void ItchBookBuilder::operator()(const itch::Trade& m) {
  if (!allowed(m.h.locate)) {
    ++filtered_;
    return;
  }
  print(m.h.locate, m.shares, m.price);
}

void ItchBookBuilder::operator()(const itch::CrossTrade& m) {
  if (!allowed(m.h.locate)) {
    ++filtered_;
    return;
  }
  if (m.shares == 0) return;
  print(m.h.locate, m.shares, m.price);
  if (m.cross_type == 'O') stocks_[m.h.locate].open_cross = m.price;
  if (m.cross_type == 'C') stocks_[m.h.locate].close_cross = m.price;
}

}  // namespace hftcore
