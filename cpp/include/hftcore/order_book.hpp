// Full-depth limit order book for every stock in an ITCH feed: an open-addressing map
// from order reference to a slot in an order pool, and per-stock (locate) bid and ask
// price levels holding FIFO queues of orders. The rules (partial executes keep queue
// position, replace goes to the back, empty levels are erased) are mirrored exactly by
// the Python reference in hft/itch.py.
#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace hftcore {

enum class Side : char { Buy = 'B', Sell = 'S' };

struct LevelView {
  std::uint32_t price;
  std::uint64_t shares;
  std::uint32_t orders;
  bool operator==(const LevelView&) const = default;
};

// ref -> pool index. Linear probing, home slot detail::mix(ref) & mask, backward-shift
// deletion (no tombstones), doubles when size + 1 > slots / 2 so probes stay short.
class OrderMap {
 public:
  static constexpr std::uint32_t kNone = 0xffffffffu;  // empty slot / absent ref

  // The slot count is rounded up to a power of two (at least 4).
  explicit OrderMap(std::size_t initial_slots = 1 << 16);

  // Inserts ref -> index (index must not be kNone); false if ref is already present.
  bool insert(std::uint64_t ref, std::uint32_t index);
  std::uint32_t find(std::uint64_t ref) const;  // kNone if absent
  bool erase(std::uint64_t ref);                // false if absent
  std::size_t size() const { return size_; }
  std::size_t slots() const { return slots_.size(); }

 private:
  struct Slot {
    std::uint64_t ref = 0;
    std::uint32_t index = kNone;
  };
  std::size_t home(std::uint64_t ref) const;
  void grow();
  std::vector<Slot> slots_;
  std::size_t size_ = 0;
};

// All stocks; per-locate sides are reached through a 65,536-entry array.
class OrderBook {
 public:
  enum class Result { Ok, UnknownRef, DuplicateRef, Overfill, Invalid };
  // `shares` is what was actually removed from the book (for replace: the new order's
  // shares). locate/side/price are zero/Buy/0 when the ref was unknown or rejected.
  struct Fill {
    Result result;
    std::uint16_t locate;
    Side side;
    std::uint32_t price;
    std::uint32_t shares;
  };

  OrderBook();

  Result add(std::uint64_t ref, std::uint16_t locate, Side side, std::uint32_t shares,
             std::uint32_t price);
  Fill execute(std::uint64_t ref, std::uint32_t shares);  // price = the order's price
  Fill cancel(std::uint64_t ref, std::uint32_t shares);
  Fill remove(std::uint64_t ref);  // D: shares = remaining shares
  Fill replace(std::uint64_t ref, std::uint64_t new_ref, std::uint32_t shares,
               std::uint32_t price);  // Fill describes the new order

  std::optional<LevelView> best(std::uint16_t locate, Side side) const;
  std::vector<LevelView> depth(std::uint16_t locate, Side side, std::size_t n) const;  // best first
  // (ref, shares) in FIFO order for one level; empty if the level does not exist.
  std::vector<std::pair<std::uint64_t, std::uint32_t>> queue(std::uint16_t locate, Side side,
                                                             std::uint32_t price) const;
  bool crossed(std::uint16_t locate) const;    // both sides present and bid >= ask
  std::vector<std::uint16_t> locates() const;  // sorted locates with live orders

  std::size_t live_orders() const { return live_; }
  std::size_t peak_live_orders() const { return peak_live_; }
  std::size_t levels() const { return levels_; }  // over all stocks and both sides
  std::size_t peak_levels() const { return peak_levels_; }

  // Checks every structural invariant; returns "" when they hold, else a description.
  std::string validate() const;

 private:
  static constexpr std::uint32_t kNone = OrderMap::kNone;

  struct Order {
    std::uint64_t ref;
    std::uint32_t shares;
    std::uint32_t price;
    std::uint16_t locate;
    Side side;
    std::uint32_t prev;  // pool indices within the level queue, kNone at the ends
    std::uint32_t next;  // (a free pool entry reuses `next` as the free-list link)
  };
  struct Level {
    std::uint32_t price;
    std::uint64_t shares;
    std::uint32_t orders;
    std::uint32_t head;
    std::uint32_t tail;
  };
  // Bids sorted ascending and asks descending, so the best price is always back().
  struct Sides {
    std::vector<Level> bids, asks;
    bool touched = false;  // listed in touched_ (so scans skip the 65,536-entry array)
  };

  static Fill rejected(Result r) { return {r, 0, Side::Buy, 0, 0}; }
  std::vector<Level>& side_levels(Sides& s, Side side) {
    return side == Side::Buy ? s.bids : s.asks;
  }
  const std::vector<Level>& side_levels(const Sides& s, Side side) const {
    return side == Side::Buy ? s.bids : s.asks;
  }
  // Index of the first level not before `price` in the side's order (may be end()).
  static std::vector<Level>::iterator lower(std::vector<Level>& levels, Side side,
                                            std::uint32_t price);
  static std::vector<Level>::const_iterator lower(const std::vector<Level>& levels, Side side,
                                                  std::uint32_t price);
  Level& level_of(const Order& o);
  void insert_order(std::uint64_t ref, std::uint16_t locate, Side side, std::uint32_t shares,
                    std::uint32_t price);
  void erase_order(std::uint32_t index);  // unlinks, drops an empty level, frees the slot
  Fill reduce(std::uint64_t ref, std::uint32_t shares);  // shared by execute and cancel

  OrderMap map_;
  std::vector<Order> pool_;
  std::uint32_t free_head_ = kNone;
  std::vector<Sides> sides_;  // 65,536 entries
  std::vector<std::uint16_t> touched_;
  std::size_t live_ = 0, peak_live_ = 0, levels_ = 0, peak_levels_ = 0;
};

}  // namespace hftcore
