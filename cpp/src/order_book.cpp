// Order map and order book implementation (see order_book.hpp for the rules).
#include "hftcore/order_book.hpp"

#include <algorithm>
#include <bit>

#include "hftcore/id_set.hpp"

namespace hftcore {

// ---------------------------------------------------------------- OrderMap

OrderMap::OrderMap(std::size_t initial_slots)
    : slots_(std::bit_ceil(std::max<std::size_t>(initial_slots, 4))) {}

std::size_t OrderMap::home(std::uint64_t ref) const {
  return detail::mix(ref) & (slots_.size() - 1);
}

bool OrderMap::insert(std::uint64_t ref, std::uint32_t index) {
  if (size_ + 1 > slots_.size() / 2) grow();
  const std::size_t mask = slots_.size() - 1;
  std::size_t i = home(ref);
  while (slots_[i].index != kNone) {
    if (slots_[i].ref == ref) return false;
    i = (i + 1) & mask;
  }
  slots_[i] = {ref, index};
  ++size_;
  return true;
}

std::uint32_t OrderMap::find(std::uint64_t ref) const {
  const std::size_t mask = slots_.size() - 1;
  std::size_t i = home(ref);
  while (slots_[i].index != kNone) {
    if (slots_[i].ref == ref) return slots_[i].index;
    i = (i + 1) & mask;
  }
  return kNone;
}

bool OrderMap::erase(std::uint64_t ref) {
  const std::size_t mask = slots_.size() - 1;
  std::size_t i = home(ref);
  while (true) {
    if (slots_[i].index == kNone) return false;
    if (slots_[i].ref == ref) break;
    i = (i + 1) & mask;
  }
  // Backward shift: pull later members of the probe chain into the hole unless their
  // home slot lies cyclically in (hole, candidate], which would break their lookup.
  std::size_t j = i;
  while (true) {
    j = (j + 1) & mask;
    if (slots_[j].index == kNone) break;
    const std::size_t k = home(slots_[j].ref);
    const bool stays = i <= j ? (i < k && k <= j) : (i < k || k <= j);
    if (stays) continue;
    slots_[i] = slots_[j];
    i = j;
  }
  slots_[i] = Slot{};
  --size_;
  return true;
}

void OrderMap::grow() {
  std::vector<Slot> old;
  old.swap(slots_);
  slots_.assign(old.size() * 2, Slot{});
  const std::size_t mask = slots_.size() - 1;
  for (const Slot& s : old) {
    if (s.index == kNone) continue;
    std::size_t i = home(s.ref);
    while (slots_[i].index != kNone) i = (i + 1) & mask;
    slots_[i] = s;
  }
}

// --------------------------------------------------------------- OrderBook

OrderBook::OrderBook() : sides_(65536) {}

std::vector<OrderBook::Level>::iterator OrderBook::lower(std::vector<Level>& levels, Side side,
                                                         std::uint32_t price) {
  if (side == Side::Buy) {
    return std::lower_bound(levels.begin(), levels.end(), price,
                            [](const Level& l, std::uint32_t p) { return l.price < p; });
  }
  return std::lower_bound(levels.begin(), levels.end(), price,
                          [](const Level& l, std::uint32_t p) { return l.price > p; });
}

std::vector<OrderBook::Level>::const_iterator OrderBook::lower(const std::vector<Level>& levels,
                                                               Side side, std::uint32_t price) {
  if (side == Side::Buy) {
    return std::lower_bound(levels.begin(), levels.end(), price,
                            [](const Level& l, std::uint32_t p) { return l.price < p; });
  }
  return std::lower_bound(levels.begin(), levels.end(), price,
                          [](const Level& l, std::uint32_t p) { return l.price > p; });
}

OrderBook::Level& OrderBook::level_of(const Order& o) {
  auto& levels = side_levels(sides_[o.locate], o.side);
  return *lower(levels, o.side, o.price);  // exists whenever the order is live
}

void OrderBook::insert_order(std::uint64_t ref, std::uint16_t locate, Side side,
                             std::uint32_t shares, std::uint32_t price) {
  std::uint32_t index;
  if (free_head_ != kNone) {
    index = free_head_;
    free_head_ = pool_[index].next;
  } else {
    index = static_cast<std::uint32_t>(pool_.size());
    pool_.emplace_back();
  }
  map_.insert(ref, index);

  Sides& s = sides_[locate];
  if (!s.touched) {
    s.touched = true;
    touched_.push_back(locate);
  }
  auto& levels = side_levels(s, side);
  auto it = lower(levels, side, price);
  if (it == levels.end() || it->price != price) {
    it = levels.insert(it, Level{price, 0, 0, kNone, kNone});
    if (++levels_ > peak_levels_) peak_levels_ = levels_;
  }
  Order& o = pool_[index];
  o = {ref, shares, price, locate, side, it->tail, kNone};
  if (it->tail != kNone) {
    pool_[it->tail].next = index;
  } else {
    it->head = index;
  }
  it->tail = index;
  it->shares += shares;
  ++it->orders;
  if (++live_ > peak_live_) peak_live_ = live_;
}

void OrderBook::erase_order(std::uint32_t index) {
  Order& o = pool_[index];
  auto& levels = side_levels(sides_[o.locate], o.side);
  auto it = lower(levels, o.side, o.price);
  if (o.prev != kNone) {
    pool_[o.prev].next = o.next;
  } else {
    it->head = o.next;
  }
  if (o.next != kNone) {
    pool_[o.next].prev = o.prev;
  } else {
    it->tail = o.prev;
  }
  it->shares -= o.shares;
  if (--it->orders == 0) {
    levels.erase(it);
    --levels_;
  }
  map_.erase(o.ref);
  o.next = free_head_;
  free_head_ = index;
  --live_;
}

OrderBook::Result OrderBook::add(std::uint64_t ref, std::uint16_t locate, Side side,
                                 std::uint32_t shares, std::uint32_t price) {
  if (shares == 0) return Result::Invalid;
  if (map_.find(ref) != kNone) return Result::DuplicateRef;
  insert_order(ref, locate, side, shares, price);
  return Result::Ok;
}

OrderBook::Fill OrderBook::reduce(std::uint64_t ref, std::uint32_t shares) {
  const std::uint32_t index = map_.find(ref);
  if (index == kNone) return rejected(Result::UnknownRef);
  Order& o = pool_[index];
  Fill f{Result::Ok, o.locate, o.side, o.price, shares};
  if (shares == 0) return f;
  if (shares < o.shares) {
    level_of(o).shares -= shares;
    o.shares -= shares;
    return f;
  }
  if (shares > o.shares) f.result = Result::Overfill;
  f.shares = o.shares;
  erase_order(index);
  return f;
}

OrderBook::Fill OrderBook::execute(std::uint64_t ref, std::uint32_t shares) {
  return reduce(ref, shares);
}

OrderBook::Fill OrderBook::cancel(std::uint64_t ref, std::uint32_t shares) {
  return reduce(ref, shares);
}

OrderBook::Fill OrderBook::remove(std::uint64_t ref) {
  const std::uint32_t index = map_.find(ref);
  if (index == kNone) return rejected(Result::UnknownRef);
  const Order& o = pool_[index];
  Fill f{Result::Ok, o.locate, o.side, o.price, o.shares};
  erase_order(index);
  return f;
}

OrderBook::Fill OrderBook::replace(std::uint64_t ref, std::uint64_t new_ref, std::uint32_t shares,
                                   std::uint32_t price) {
  const std::uint32_t index = map_.find(ref);
  if (index == kNone) return rejected(Result::UnknownRef);
  if (shares == 0) return rejected(Result::Invalid);
  if (new_ref != ref && map_.find(new_ref) != kNone) return rejected(Result::DuplicateRef);
  const std::uint16_t locate = pool_[index].locate;
  const Side side = pool_[index].side;
  erase_order(index);
  insert_order(new_ref, locate, side, shares, price);
  return {Result::Ok, locate, side, price, shares};
}

std::optional<LevelView> OrderBook::best(std::uint16_t locate, Side side) const {
  const auto& levels = side_levels(sides_[locate], side);
  if (levels.empty()) return std::nullopt;
  const Level& l = levels.back();
  return LevelView{l.price, l.shares, l.orders};
}

std::vector<LevelView> OrderBook::depth(std::uint16_t locate, Side side, std::size_t n) const {
  const auto& levels = side_levels(sides_[locate], side);
  std::vector<LevelView> out;
  out.reserve(std::min(n, levels.size()));
  for (auto it = levels.rbegin(); it != levels.rend() && out.size() < n; ++it) {
    out.push_back({it->price, it->shares, it->orders});
  }
  return out;
}

std::vector<std::pair<std::uint64_t, std::uint32_t>> OrderBook::queue(std::uint16_t locate,
                                                                      Side side,
                                                                      std::uint32_t price) const {
  std::vector<std::pair<std::uint64_t, std::uint32_t>> out;
  const auto& levels = side_levels(sides_[locate], side);
  auto it = lower(levels, side, price);
  if (it == levels.end() || it->price != price) return out;
  for (std::uint32_t i = it->head; i != kNone; i = pool_[i].next) {
    out.emplace_back(pool_[i].ref, pool_[i].shares);
  }
  return out;
}

bool OrderBook::crossed(std::uint16_t locate) const {
  const Sides& s = sides_[locate];
  return !s.bids.empty() && !s.asks.empty() && s.bids.back().price >= s.asks.back().price;
}

std::vector<std::uint16_t> OrderBook::locates() const {
  std::vector<std::uint16_t> out;
  for (std::uint16_t loc : touched_) {
    if (!sides_[loc].bids.empty() || !sides_[loc].asks.empty()) out.push_back(loc);
  }
  std::sort(out.begin(), out.end());
  return out;
}

std::string OrderBook::validate() const {
  std::vector<char> seen(pool_.size(), 0);  // 1 = in a level queue, 2 = on the free list
  std::size_t queued = 0, level_count = 0;

  for (std::uint16_t loc : touched_) {
    const Sides& s = sides_[loc];
    for (Side side : {Side::Buy, Side::Sell}) {
      const auto& levels = side_levels(s, side);
      for (std::size_t li = 0; li < levels.size(); ++li) {
        const Level& l = levels[li];
        ++level_count;
        if (li > 0) {
          const std::uint32_t prev = levels[li - 1].price;
          const bool sorted = side == Side::Buy ? prev < l.price : prev > l.price;
          if (!sorted) return "levels not strictly sorted at locate " + std::to_string(loc);
        }
        if (l.orders == 0) return "empty level at locate " + std::to_string(loc);
        std::uint64_t shares = 0;
        std::uint32_t count = 0, prev = kNone;
        for (std::uint32_t i = l.head; i != kNone; i = pool_[i].next) {
          if (i >= pool_.size()) return "queue index out of range";
          if (seen[i]) return "order in more than one place";
          seen[i] = 1;
          const Order& o = pool_[i];
          if (o.prev != prev) return "inconsistent prev link";
          if (o.locate != loc || o.side != side || o.price != l.price) {
            return "order filed under the wrong level";
          }
          if (map_.find(o.ref) != i) return "live order missing from the map";
          shares += o.shares;
          ++count;
          prev = i;
          if (count > pool_.size()) return "queue cycle";
        }
        if (l.tail != prev) return "tail does not match the queue end";
        if (shares != l.shares) return "level shares differ from the queue sum";
        if (count != l.orders) return "level order count differs from the queue length";
        queued += count;
      }
    }
  }
  if (level_count != levels_) return "levels() differs from the level count";
  if (queued != live_) return "live_orders() differs from the queued orders";
  if (map_.size() != live_) return "map size differs from live orders";

  std::size_t free_count = 0;
  for (std::uint32_t i = free_head_; i != kNone; i = pool_[i].next) {
    if (i >= pool_.size() || seen[i]) return "free list corrupt";
    seen[i] = 2;
    ++free_count;
  }
  if (free_count + live_ != pool_.size()) return "pool slot neither live nor free";
  return "";
}

}  // namespace hftcore
