// Unit tests for the ITCH order map and full-depth order book: FIFO queues, partial and
// full executes, replace semantics, rejections that leave the book unchanged, probe
// chain handling in the open-addressing map, and a seeded randomized run checked after
// every operation by validate() and at the end against an independent std::map model.
#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <map>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "hftcore/id_set.hpp"
#include "hftcore/order_book.hpp"

using namespace hftcore;
using Result = OrderBook::Result;
using Queue = std::vector<std::pair<std::uint64_t, std::uint32_t>>;

namespace {
constexpr std::uint16_t kLoc = 7;
}

TEST_CASE("OrderMap rounds slots up to a power of two and starts empty", "[order_map]") {
  OrderMap m(100);
  CHECK(m.slots() == 128);
  CHECK(m.size() == 0);
  CHECK(m.find(5) == OrderMap::kNone);
  CHECK(OrderMap(1).slots() == 4);
}

TEST_CASE("OrderMap insert, find, duplicate insert and erase of a missing ref", "[order_map]") {
  OrderMap m(64);
  CHECK(m.insert(10, 1));
  CHECK(m.insert(0, 2));  // ref 0 is an ordinary key
  CHECK_FALSE(m.insert(10, 3));
  CHECK(m.find(10) == 1);
  CHECK(m.find(0) == 2);
  CHECK(m.size() == 2);
  CHECK_FALSE(m.erase(11));
  CHECK(m.erase(10));
  CHECK_FALSE(m.erase(10));
  CHECK(m.find(10) == OrderMap::kNone);
  CHECK(m.find(0) == 2);
  CHECK(m.size() == 1);
}

namespace {
// Refs whose home slot in a map of `slots` slots is `slot`.
std::vector<std::uint64_t> colliding_refs(std::size_t slots, std::size_t slot, std::size_t n) {
  std::vector<std::uint64_t> refs;
  for (std::uint64_t r = 1; refs.size() < n; ++r) {
    if ((detail::mix(r) & (slots - 1)) == slot) refs.push_back(r);
  }
  return refs;
}
}  // namespace

TEST_CASE("OrderMap keeps every ref findable when erasing from a collision chain", "[order_map]") {
  // 8 refs on one home slot of a 64-slot map (capacity 32, so no growth) plus 2 refs on
  // the next slot, so the chains overlap. Erase the start, middle and end of the chain.
  for (std::size_t victim : {0u, 3u, 7u, 8u}) {
    OrderMap m(64);
    std::vector<std::uint64_t> chain = colliding_refs(64, 20, 8);
    std::vector<std::uint64_t> next = colliding_refs(64, 21, 2);
    std::vector<std::uint64_t> all = chain;
    all.insert(all.end(), next.begin(), next.end());
    // The victim index 8 is the first ref of the neighbouring chain.
    for (std::size_t i = 0; i < all.size(); ++i) {
      REQUIRE(m.insert(all[i], static_cast<std::uint32_t>(i)));
    }
    REQUIRE(m.slots() == 64);
    REQUIRE(m.erase(all[victim]));
    CHECK(m.size() == all.size() - 1);
    for (std::size_t i = 0; i < all.size(); ++i) {
      CHECK(m.find(all[i]) == (i == victim ? OrderMap::kNone : static_cast<std::uint32_t>(i)));
    }
    // Erase everything else in a different order; the map must end up empty.
    for (std::size_t i = all.size(); i-- > 0;) {
      if (i != victim) REQUIRE(m.erase(all[i]));
    }
    CHECK(m.size() == 0);
    for (auto r : all) CHECK(m.find(r) == OrderMap::kNone);
  }
}

TEST_CASE("OrderMap backward shift handles a chain that wraps past the last slot", "[order_map]") {
  OrderMap m(16);
  std::vector<std::uint64_t> refs = colliding_refs(16, 15, 4);  // slots 15, 0, 1, 2
  for (std::size_t i = 0; i < refs.size(); ++i) {
    REQUIRE(m.insert(refs[i], static_cast<std::uint32_t>(i)));
  }
  REQUIRE(m.erase(refs[0]));
  for (std::size_t i = 1; i < refs.size(); ++i) {
    CHECK(m.find(refs[i]) == static_cast<std::uint32_t>(i));
  }
  REQUIRE(m.erase(refs[2]));
  CHECK(m.find(refs[1]) == 1);
  CHECK(m.find(refs[3]) == 3);
}

TEST_CASE("OrderMap doubles at size + 1 > slots / 2 and keeps every ref", "[order_map]") {
  OrderMap m(16);  // capacity 8
  for (std::uint64_t r = 0; r < 8; ++r)
    REQUIRE(m.insert(r * 1000 + 1, static_cast<std::uint32_t>(r)));
  CHECK(m.slots() == 16);
  REQUIRE(m.insert(99999, 8));  // the ninth entry needs 9 > 8
  CHECK(m.slots() == 32);
  for (std::uint64_t r = 9; r < 5000; ++r)
    REQUIRE(m.insert(r * 7919 + 3, static_cast<std::uint32_t>(r)));
  CHECK(m.size() == 5000);
  CHECK(m.slots() >= 2 * m.size());
  for (std::uint64_t r = 0; r < 8; ++r) CHECK(m.find(r * 1000 + 1) == r);
  CHECK(m.find(99999) == 8);
  for (std::uint64_t r = 9; r < 5000; ++r) CHECK(m.find(r * 7919 + 3) == r);
}

TEST_CASE("OrderBook keeps FIFO order inside a level", "[order_book]") {
  OrderBook b;
  REQUIRE(b.add(1, kLoc, Side::Buy, 100, 5000) == Result::Ok);
  REQUIRE(b.add(2, kLoc, Side::Buy, 200, 5000) == Result::Ok);
  REQUIRE(b.add(3, kLoc, Side::Buy, 300, 5000) == Result::Ok);
  CHECK(b.queue(kLoc, Side::Buy, 5000) == Queue{{1, 100}, {2, 200}, {3, 300}});

  auto f = b.execute(1, 40);  // partial execute keeps the first order in front
  CHECK(f.result == Result::Ok);
  CHECK(b.queue(kLoc, Side::Buy, 5000) == Queue{{1, 60}, {2, 200}, {3, 300}});
  CHECK(*b.best(kLoc, Side::Buy) == LevelView{5000, 560, 3});

  CHECK(b.cancel(2, 50).result == Result::Ok);  // a partial cancel also keeps position
  CHECK(b.queue(kLoc, Side::Buy, 5000) == Queue{{1, 60}, {2, 150}, {3, 300}});

  CHECK(b.remove(2).shares == 150);  // removing the middle order keeps the others in order
  CHECK(b.queue(kLoc, Side::Buy, 5000) == Queue{{1, 60}, {3, 300}});
  CHECK(*b.best(kLoc, Side::Buy) == LevelView{5000, 360, 2});
  CHECK(b.validate() == "");
}

TEST_CASE("OrderBook executes down to zero and erases the emptied level", "[order_book]") {
  OrderBook b;
  REQUIRE(b.add(1, kLoc, Side::Sell, 100, 5100) == Result::Ok);
  REQUIRE(b.add(2, kLoc, Side::Sell, 50, 5200) == Result::Ok);
  auto f = b.execute(1, 30);
  CHECK(f.result == Result::Ok);
  CHECK(f.locate == kLoc);
  CHECK(f.side == Side::Sell);
  CHECK(f.price == 5100);
  CHECK(f.shares == 30);
  f = b.execute(1, 70);
  CHECK(f.result == Result::Ok);
  CHECK(f.shares == 70);
  CHECK(f.price == 5100);
  CHECK(b.queue(kLoc, Side::Sell, 5100).empty());  // the last order took the level with it
  CHECK(*b.best(kLoc, Side::Sell) == LevelView{5200, 50, 1});
  CHECK(b.live_orders() == 1);
  CHECK(b.levels() == 1);
  CHECK(b.execute(1, 1).result == Result::UnknownRef);
  CHECK(b.validate() == "");
}

TEST_CASE("OrderBook overfill removes the order and reports the remaining shares", "[order_book]") {
  OrderBook b;
  REQUIRE(b.add(1, kLoc, Side::Buy, 100, 5000) == Result::Ok);
  REQUIRE(b.add(2, kLoc, Side::Buy, 100, 5000) == Result::Ok);
  auto f = b.execute(1, 130);
  CHECK(f.result == Result::Overfill);
  CHECK(f.shares == 100);
  CHECK(f.price == 5000);
  CHECK(b.queue(kLoc, Side::Buy, 5000) == Queue{{2, 100}});
  f = b.cancel(2, 101);
  CHECK(f.result == Result::Overfill);
  CHECK(f.shares == 100);
  CHECK(b.live_orders() == 0);
  CHECK(b.levels() == 0);
  CHECK(b.validate() == "");
}

TEST_CASE("OrderBook execute or cancel of zero shares changes nothing", "[order_book]") {
  OrderBook b;
  REQUIRE(b.add(1, kLoc, Side::Buy, 100, 5000) == Result::Ok);
  auto f = b.execute(1, 0);
  CHECK(f.result == Result::Ok);
  CHECK(f.shares == 0);
  CHECK(b.cancel(1, 0).result == Result::Ok);
  CHECK(b.queue(kLoc, Side::Buy, 5000) == Queue{{1, 100}});
  CHECK(b.validate() == "");
}

TEST_CASE("OrderBook replace keeps side and locate and moves to the back of the new level",
          "[order_book]") {
  OrderBook b;
  REQUIRE(b.add(1, kLoc, Side::Buy, 100, 5000) == Result::Ok);
  REQUIRE(b.add(2, kLoc, Side::Buy, 100, 5000) == Result::Ok);
  REQUIRE(b.add(3, kLoc, Side::Buy, 100, 4900) == Result::Ok);

  // Same price: loses time priority and takes the new size.
  auto f = b.replace(1, 10, 80, 5000);
  CHECK(f.result == Result::Ok);
  CHECK(f.locate == kLoc);
  CHECK(f.side == Side::Buy);
  CHECK(f.price == 5000);
  CHECK(f.shares == 80);
  CHECK(b.queue(kLoc, Side::Buy, 5000) == Queue{{2, 100}, {10, 80}});
  CHECK(b.execute(1, 10).result == Result::UnknownRef);  // the old ref is gone

  // New price: the old level is erased when it empties, the new one is created.
  f = b.replace(3, 30, 120, 5100);
  CHECK(f.result == Result::Ok);
  CHECK(f.price == 5100);
  CHECK(b.queue(kLoc, Side::Buy, 4900).empty());
  CHECK(b.queue(kLoc, Side::Buy, 5100) == Queue{{30, 120}});
  CHECK(*b.best(kLoc, Side::Buy) == LevelView{5100, 120, 1});

  // Replacing an order with its own ref is a valid replace.
  f = b.replace(2, 2, 55, 5000);
  CHECK(f.result == Result::Ok);
  CHECK(b.queue(kLoc, Side::Buy, 5000) == Queue{{10, 80}, {2, 55}});
  CHECK(b.live_orders() == 3);
  CHECK(b.validate() == "");
}

TEST_CASE("OrderBook replace rejections leave the book unchanged", "[order_book]") {
  OrderBook b;
  REQUIRE(b.add(1, kLoc, Side::Sell, 100, 5000) == Result::Ok);
  REQUIRE(b.add(2, kLoc, Side::Sell, 100, 5000) == Result::Ok);
  const auto before = b.queue(kLoc, Side::Sell, 5000);

  CHECK(b.replace(99, 5, 10, 5000).result == Result::UnknownRef);
  CHECK(b.replace(1, 5, 0, 5000).result == Result::Invalid);
  CHECK(b.replace(1, 2, 10, 5000).result == Result::DuplicateRef);  // new ref is live
  CHECK(b.replace(99, 2, 0, 5000).result == Result::UnknownRef);    // unknown wins over invalid
  CHECK(b.queue(kLoc, Side::Sell, 5000) == before);
  CHECK(b.live_orders() == 2);
  CHECK(b.execute(5, 1).result == Result::UnknownRef);
  CHECK(b.validate() == "");
}

TEST_CASE("OrderBook best and depth order each side best first", "[order_book]") {
  OrderBook b;
  CHECK_FALSE(b.best(kLoc, Side::Buy).has_value());
  CHECK(b.depth(kLoc, Side::Sell, 5).empty());
  for (std::uint32_t p : {5000u, 5200u, 4900u, 5100u}) {
    REQUIRE(b.add(p, kLoc, Side::Buy, 10, p) == Result::Ok);
  }
  REQUIRE(b.add(5100, kLoc, Side::Buy, 5, 5100) == Result::DuplicateRef);
  REQUIRE(b.add(1000001, kLoc, Side::Buy, 7, 5100) == Result::Ok);
  for (std::uint32_t p : {6000u, 5500u, 5800u}) {
    REQUIRE(b.add(p + 100000, kLoc, Side::Sell, 20, p) == Result::Ok);
  }
  CHECK(b.depth(kLoc, Side::Buy, 3) ==
        std::vector<LevelView>{{5200, 10, 1}, {5100, 17, 2}, {5000, 10, 1}});
  CHECK(b.depth(kLoc, Side::Buy, 10).size() == 4);
  CHECK(b.depth(kLoc, Side::Sell, 10) ==
        std::vector<LevelView>{{5500, 20, 1}, {5800, 20, 1}, {6000, 20, 1}});
  CHECK(b.depth(kLoc, Side::Sell, 0).empty());
  CHECK(*b.best(kLoc, Side::Buy) == LevelView{5200, 10, 1});
  CHECK(*b.best(kLoc, Side::Sell) == LevelView{5500, 20, 1});
  CHECK(b.levels() == 7);
  CHECK(b.validate() == "");
}

TEST_CASE("OrderBook crossed is true for locked and crossed books only", "[order_book]") {
  OrderBook b;
  CHECK_FALSE(b.crossed(kLoc));
  REQUIRE(b.add(1, kLoc, Side::Buy, 10, 5000) == Result::Ok);
  CHECK_FALSE(b.crossed(kLoc));  // one side only
  REQUIRE(b.add(2, kLoc, Side::Sell, 10, 5001) == Result::Ok);
  CHECK_FALSE(b.crossed(kLoc));
  REQUIRE(b.add(3, kLoc, Side::Sell, 10, 5000) == Result::Ok);
  CHECK(b.crossed(kLoc));  // locked
  REQUIRE(b.add(4, kLoc, Side::Buy, 10, 5002) == Result::Ok);
  CHECK(b.crossed(kLoc));  // crossed
  CHECK_FALSE(b.crossed(kLoc + 1));
  b.remove(4);
  b.remove(1);
  b.remove(3);
  CHECK_FALSE(b.crossed(kLoc));
}

TEST_CASE("OrderBook stocks are independent and locates lists live ones sorted", "[order_book]") {
  OrderBook b;
  CHECK(b.locates().empty());
  REQUIRE(b.add(1, 300, Side::Buy, 10, 5000) == Result::Ok);
  REQUIRE(b.add(2, 5, Side::Sell, 10, 5000) == Result::Ok);
  REQUIRE(b.add(3, 65535, Side::Buy, 10, 5000) == Result::Ok);
  REQUIRE(b.add(4, 5, Side::Buy, 10, 4000) == Result::Ok);
  CHECK(b.locates() == std::vector<std::uint16_t>{5, 300, 65535});
  CHECK_FALSE(b.best(300, Side::Sell).has_value());
  CHECK(*b.best(5, Side::Sell) == LevelView{5000, 10, 1});
  b.remove(1);  // the last order of locate 300 disappears from the list
  CHECK(b.locates() == std::vector<std::uint16_t>{5, 65535});
  b.remove(2);
  CHECK(b.locates() == std::vector<std::uint16_t>{5, 65535});  // locate 5 still has order 4
  b.remove(4);
  CHECK(b.locates() == std::vector<std::uint16_t>{65535});
  REQUIRE(b.add(5, 300, Side::Sell, 1, 1) == Result::Ok);  // and comes back when used again
  CHECK(b.locates() == std::vector<std::uint16_t>{300, 65535});
  CHECK(b.validate() == "");
}

TEST_CASE("OrderBook rejects unknown refs, duplicates and zero shares", "[order_book]") {
  OrderBook b;
  CHECK(b.execute(1, 10).result == Result::UnknownRef);
  CHECK(b.cancel(1, 10).result == Result::UnknownRef);
  CHECK(b.remove(1).result == Result::UnknownRef);
  CHECK(b.replace(1, 2, 10, 5000).result == Result::UnknownRef);
  CHECK(b.add(1, kLoc, Side::Buy, 0, 5000) == Result::Invalid);
  CHECK(b.live_orders() == 0);
  CHECK(b.levels() == 0);
  REQUIRE(b.add(1, kLoc, Side::Buy, 10, 5000) == Result::Ok);
  CHECK(b.add(1, kLoc + 1, Side::Sell, 99, 6000) == Result::DuplicateRef);
  CHECK(b.live_orders() == 1);
  CHECK(b.queue(kLoc, Side::Buy, 5000) == Queue{{1, 10}});
  CHECK(b.best(kLoc + 1, Side::Sell) == std::nullopt);
  CHECK(b.validate() == "");
}

TEST_CASE("OrderBook peaks are high-water marks and pool slots are reused", "[order_book]") {
  OrderBook b;
  for (std::uint64_t r = 1; r <= 10; ++r) {
    REQUIRE(b.add(r, kLoc, Side::Buy, 10, static_cast<std::uint32_t>(5000 + r)) == Result::Ok);
  }
  CHECK(b.live_orders() == 10);
  CHECK(b.levels() == 10);
  for (std::uint64_t r = 1; r <= 8; ++r) b.remove(r);
  CHECK(b.live_orders() == 2);
  CHECK(b.levels() == 2);
  CHECK(b.peak_live_orders() == 10);
  CHECK(b.peak_levels() == 10);
  for (std::uint64_t r = 20; r < 25; ++r) {
    REQUIRE(b.add(r, kLoc, Side::Sell, 10, static_cast<std::uint32_t>(r)) == Result::Ok);
  }
  CHECK(b.live_orders() == 7);
  CHECK(b.peak_live_orders() == 10);
  CHECK(b.validate() == "");
}

namespace {

// Independent model: orders by ref with an arrival sequence; levels are derived by
// grouping, so nothing is shared with the book's linked-list implementation.
struct Model {
  struct Ord {
    std::uint16_t locate;
    Side side;
    std::uint32_t price;
    std::uint32_t shares;
    std::uint64_t seq;
  };
  std::map<std::uint64_t, Ord> orders;
  std::uint64_t seq = 0;

  Result add(std::uint64_t ref, std::uint16_t loc, Side side, std::uint32_t shares,
             std::uint32_t price) {
    if (shares == 0) return Result::Invalid;
    if (orders.count(ref)) return Result::DuplicateRef;
    orders[ref] = {loc, side, price, shares, seq++};
    return Result::Ok;
  }
  // Execute and cancel share the same book effect; returns the removed shares.
  std::pair<Result, std::uint32_t> reduce(std::uint64_t ref, std::uint32_t shares) {
    auto it = orders.find(ref);
    if (it == orders.end()) return {Result::UnknownRef, 0};
    if (shares == 0) return {Result::Ok, 0};
    if (shares < it->second.shares) {
      it->second.shares -= shares;
      return {Result::Ok, shares};
    }
    std::uint32_t remaining = it->second.shares;
    orders.erase(it);
    return {shares == remaining ? Result::Ok : Result::Overfill, remaining};
  }
  // Levels of one side, best first, each with its FIFO queue.
  std::vector<std::pair<std::uint32_t, Queue>> levels(std::uint16_t loc, Side side) const {
    std::map<std::uint32_t,
             std::vector<std::pair<std::uint64_t, std::pair<std::uint64_t, std::uint32_t>>>>
        by_price;
    for (const auto& [ref, o] : orders) {
      if (o.locate == loc && o.side == side) by_price[o.price].push_back({o.seq, {ref, o.shares}});
    }
    std::vector<std::pair<std::uint32_t, Queue>> out;
    for (auto& [price, v] : by_price) {
      std::sort(v.begin(), v.end());
      Queue q;
      for (auto& e : v) q.push_back(e.second);
      out.push_back({price, std::move(q)});
    }
    if (side == Side::Buy) std::reverse(out.begin(), out.end());
    return out;
  }
};

void compare_to_model(const OrderBook& b, const Model& m, std::uint16_t max_locate) {
  std::size_t total_levels = 0;
  std::vector<std::uint16_t> live;
  for (std::uint16_t loc = 1; loc <= max_locate; ++loc) {
    bool any = false;
    for (Side side : {Side::Buy, Side::Sell}) {
      auto expected = m.levels(loc, side);
      auto depth = b.depth(loc, side, 1000000);
      REQUIRE(depth.size() == expected.size());
      total_levels += expected.size();
      any = any || !expected.empty();
      for (std::size_t i = 0; i < expected.size(); ++i) {
        std::uint64_t shares = 0;
        for (auto& e : expected[i].second) shares += e.second;
        CHECK(depth[i] == LevelView{expected[i].first, shares,
                                    static_cast<std::uint32_t>(expected[i].second.size())});
        CHECK(b.queue(loc, side, expected[i].first) == expected[i].second);
      }
    }
    if (any) live.push_back(loc);
  }
  CHECK(b.locates() == live);
  CHECK(b.live_orders() == m.orders.size());
  CHECK(b.levels() == total_levels);
}

}  // namespace

TEST_CASE("OrderBook matches a simple model over a randomized operation stream",
          "[order_book][randomized]") {
  std::mt19937_64 rng(20261113);
  OrderBook b;
  Model m;
  constexpr std::uint16_t kLocates = 3;
  constexpr int kOps = 100000;
  constexpr std::uint64_t kRefSpace = 400;  // ~half live at any time; hits unknown refs too
  std::size_t peak_live = 0, peak_levels = 0;

  auto pick_ref = [&] { return 1 + rng() % kRefSpace; };
  auto pick_price = [&] { return static_cast<std::uint32_t>(10000 + rng() % 12); };

  for (int op = 0; op < kOps; ++op) {
    std::uint64_t ref = pick_ref();
    switch (rng() % 10) {
      case 0:
      case 1:
      case 2: {  // add
        auto loc = static_cast<std::uint16_t>(1 + rng() % kLocates);
        Side side = (rng() & 1) ? Side::Buy : Side::Sell;
        auto shares = static_cast<std::uint32_t>(rng() % 5 == 0 ? 0 : 1 + rng() % 500);
        auto price = pick_price();
        CHECK(b.add(ref, loc, side, shares, price) == m.add(ref, loc, side, shares, price));
        break;
      }
      case 3:
      case 4:
      case 5: {  // execute or cancel
        auto shares = static_cast<std::uint32_t>(rng() % 6 == 0 ? 0 : 1 + rng() % 300);
        OrderBook::Fill f;
        std::pair<Result, std::uint32_t> e;
        if (rng() & 1) {
          f = b.execute(ref, shares);
        } else {
          f = b.cancel(ref, shares);
        }
        e = m.reduce(ref, shares);
        CHECK(f.result == e.first);
        CHECK(f.shares == e.second);
        break;
      }
      case 6:
      case 7: {  // remove
        auto it = m.orders.find(ref);
        auto f = b.remove(ref);
        if (it == m.orders.end()) {
          CHECK(f.result == Result::UnknownRef);
        } else {
          CHECK(f.result == Result::Ok);
          CHECK(f.shares == it->second.shares);
          CHECK(f.locate == it->second.locate);
          CHECK(f.side == it->second.side);
          CHECK(f.price == it->second.price);
          m.orders.erase(it);
        }
        break;
      }
      default: {  // replace; the new ref comes from the same space, so it is often live
        std::uint64_t new_ref = (rng() % 8 == 0) ? ref : pick_ref();
        auto shares = static_cast<std::uint32_t>(rng() % 6 == 0 ? 0 : 1 + rng() % 500);
        auto price = pick_price();
        auto it = m.orders.find(ref);
        auto f = b.replace(ref, new_ref, shares, price);
        if (it == m.orders.end()) {
          CHECK(f.result == Result::UnknownRef);
        } else if (shares == 0) {
          CHECK(f.result == Result::Invalid);
        } else if (new_ref != ref && m.orders.count(new_ref)) {
          CHECK(f.result == Result::DuplicateRef);
        } else {
          CHECK(f.result == Result::Ok);
          CHECK(f.locate == it->second.locate);
          CHECK(f.side == it->second.side);
          CHECK(f.price == price);
          CHECK(f.shares == shares);
          Model::Ord o = it->second;
          m.orders.erase(it);
          m.add(new_ref, o.locate, o.side, shares, price);
        }
        break;
      }
    }
    std::string problem = b.validate();
    REQUIRE(problem == "");
    CHECK(b.live_orders() == m.orders.size());
    peak_live = std::max(peak_live, m.orders.size());
    peak_levels = std::max(peak_levels, b.levels());
    if (op % 5000 == 0) compare_to_model(b, m, kLocates);
  }
  compare_to_model(b, m, kLocates);
  CHECK(b.peak_live_orders() == peak_live);
  CHECK(b.peak_levels() == peak_levels);
  CHECK(peak_live > 100);  // the run actually exercised a populated book
}
