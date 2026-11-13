#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <optional>
#include <variant>
#include <string>

#include "hftcore/bar_aggregator.hpp"
#include "hftcore/decimal.hpp"
#include "hftcore/frame_router.hpp"
#include "hftcore/itch_book.hpp"
#include "hftcore/market_engine.hpp"
#include "hftcore/replay.hpp"
#include "hftcore/timestamp.hpp"
#include "hftcore/version.hpp"

namespace py = pybind11;
using namespace hftcore;

namespace {

constexpr const char* kCloseExcluded = "479CGHIMNPQRTUVZ";

py::tuple bar_tuple(const Bar& b) {
  return py::make_tuple(b.start_ns, b.end_ns, b.quote_ns, b.session_id, b.tradable, b.open,
                        b.high, b.low, b.close, b.volume, b.bid, b.ask, b.bid_size, b.ask_size,
                        b.vwap);
}

py::list bar_list(const std::vector<Bar>& bars) {
  py::list out;
  for (const auto& b : bars) out.append(bar_tuple(b));
  return out;
}

bool is_int(const py::handle& value) { return py::isinstance<py::int_>(value); }

// hft.feed.event_timestamp
std::int64_t event_timestamp(const py::dict& event) {
  py::object stamp = event.attr("get")("event_ns");
  if (stamp.is_none()) {
    py::object text = event.attr("get")("t");
    if (!py::isinstance<py::str>(text)) {
      throw py::value_error("timestamp must be RFC3339 text");
    }
    try {
      return parse_timestamp(text.cast<std::string>());
    } catch (const std::invalid_argument& error) {
      throw py::value_error(error.what());
    }
  }
  if (!is_int(stamp) || py::isinstance<py::bool_>(stamp) || stamp.cast<long long>() <= 0) {
    throw py::value_error("event_ns must be positive integer nanoseconds");
  }
  return stamp.cast<std::int64_t>();
}

std::optional<std::string> identity(const py::dict& event) {
  if (event.contains("i")) return py::str(event["i"]).cast<std::string>();
  if (event.contains("quote_id")) return py::str(event["quote_id"]).cast<std::string>();
  return std::nullopt;
}

double decimal_float(const py::handle& value, int power) {
  std::string text = py::str(value).cast<std::string>();
  if (power == 0) return py::float_(py::str(text)).cast<double>();
  try {
    return scale_decimal(text, power);
  } catch (const std::invalid_argument&) {
    // Defer to Python for forms the fast path does not handle (e.g. "NaN").
    py::object decimal = py::module_::import("decimal").attr("Decimal");
    return py::float_(decimal(text) * py::int_(100)).cast<double>();
  }
}

bool excluded_trade(const py::dict& event) {
  py::object conditions = event.attr("get")("c", py::list());
  py::object tape = event.attr("get")("z", "C");
  std::string excluded = kCloseExcluded;
  excluded += tape.equal(py::str("C")) ? 'W' : 'B';
  for (const auto& c : conditions) {
    if (!py::isinstance<py::str>(c)) continue;
    auto code = c.cast<std::string>();
    if (code.size() == 1 && excluded.find(code[0]) != std::string::npos) return true;
  }
  return false;
}

struct Parsed {
  enum class Kind { Ignored, Quote, Trade, Correction, Cancellation } kind = Kind::Ignored;
  std::int64_t stamp = 0, arrival = 0;
  Quote quote{};
  double price = 0, size = 0;
  bool excluded = false;
  std::optional<std::string> id;
  std::optional<std::string_view> id_view() const {
    return id ? std::optional<std::string_view>(*id) : std::nullopt;
  }
};

// Python-compatible parsing and validation of hft.feed.BarAggregator.add input.
Parsed parse_event(const std::string& symbol, const py::dict& event, const py::object& arrival_ns) {
  Parsed p;
  if (!event.attr("get")("S").equal(py::str(symbol))) return p;
  py::object kind_obj = event.attr("get")("T");
  if (!py::isinstance<py::str>(kind_obj)) return p;
  auto kind = kind_obj.cast<std::string>();
  if (kind != "q" && kind != "t" && kind != "c" && kind != "x") return p;
  p.stamp = event_timestamp(event);
  py::object arrival_obj =
      arrival_ns.is_none() ? event.attr("get")("arrival_ns", p.stamp) : arrival_ns;
  if (!is_int(arrival_obj) || arrival_obj.cast<long long>() < 0) {
    throw py::value_error("invalid arrival_ns");
  }
  p.arrival = arrival_obj.cast<std::int64_t>();
  p.id = identity(event);
  if (kind == "c" || kind == "x") {
    p.kind = kind == "x" ? Parsed::Kind::Cancellation : Parsed::Kind::Correction;
  } else if (kind == "q") {
    bool shares = py::bool_(event.attr("get")("sizes_in_shares"));
    p.kind = Parsed::Kind::Quote;
    p.quote = Quote{p.stamp,
                    p.arrival,
                    decimal_float(event["bp"], 0),
                    decimal_float(event["ap"], 0),
                    decimal_float(event["bs"], shares ? 0 : 2),
                    decimal_float(event["as"], shares ? 0 : 2)};
  } else {
    p.kind = Parsed::Kind::Trade;
    p.price = py::float_(event["p"]).cast<double>();
    p.size = py::float_(event["s"]).cast<double>();
    p.excluded = excluded_trade(event);
  }
  return p;
}

template <class Fn>
auto rethrow_value_errors(Fn&& fn) -> decltype(fn()) {
  try {
    return fn();
  } catch (const std::invalid_argument& error) {
    throw py::value_error(error.what());
  } catch (const std::logic_error& error) {
    throw py::value_error(error.what());
  }
}

py::list add_event(BarAggregator& agg, const py::dict& event, const py::object& arrival_ns) {
  Parsed p = parse_event(agg.symbol(), event, arrival_ns);
  return rethrow_value_errors([&] {
    switch (p.kind) {
      case Parsed::Kind::Ignored: return py::list();
      case Parsed::Kind::Correction:
      case Parsed::Kind::Cancellation:
        return bar_list(agg.add_correction(p.stamp, p.arrival,
                                           p.kind == Parsed::Kind::Cancellation));
      case Parsed::Kind::Quote: return bar_list(agg.add_quote(p.quote, p.id_view()));
      case Parsed::Kind::Trade:
        return bar_list(
            agg.add_trade(p.stamp, p.arrival, p.price, p.size, p.excluded, p.id_view()));
    }
    return py::list();
  });
}

py::object update_object(const Update& update) {
  if (const auto* gap = std::get_if<GapEvent>(&update)) {
    return py::make_tuple("gap", gap->now_ns, gap->reason);
  }
  const auto& u = std::get<BarUpdate>(update);
  py::object market = py::none();
  if (u.market) {
    py::list values;
    for (float v : *u.market) values.append(static_cast<double>(v));
    market = values;
  }
  return py::make_tuple("bar", bar_tuple(u.bar), u.now_ns, u.accepted,
                        u.reset ? py::object(py::str(*u.reset)) : py::object(py::none()), u.ready,
                        market, u.action ? py::object(py::int_(*u.action)) : py::object(py::none()));
}

py::list update_list(const std::vector<Update>& updates) {
  py::list out;
  for (const auto& u : updates) out.append(update_object(u));
  return out;
}

// hft.market_engine.PyMarketEngine.on_event: the arrival-gap check runs for every
// event, including ones the aggregator ignores.
py::list engine_event(MarketEngine& engine, const py::dict& event, std::int64_t arrival_ns) {
  if (!engine.aggregator()) throw py::value_error("start a session before market events");
  Parsed p = parse_event(engine.symbol(), event, py::int_(arrival_ns));
  return rethrow_value_errors([&] {
    switch (p.kind) {
      case Parsed::Kind::Ignored: return update_list(engine.on_ignored(arrival_ns));
      case Parsed::Kind::Correction:
      case Parsed::Kind::Cancellation:
        return update_list(engine.on_correction(p.stamp, p.arrival,
                                                p.kind == Parsed::Kind::Cancellation));
      case Parsed::Kind::Quote: return update_list(engine.on_quote(p.quote, p.id_view()));
      case Parsed::Kind::Trade:
        return update_list(
            engine.on_trade(p.stamp, p.arrival, p.price, p.size, p.excluded, p.id_view()));
    }
    return py::list();
  });
}

Strategy parse_strategy_name(const std::string& name) {
  if (name == "model") return Strategy::Model;
  if (name == "ema-crossover") return Strategy::EmaCrossover;
  if (name == "hold-day") return Strategy::HoldDay;
  throw py::value_error("unknown strategy '" + name + "'");
}

template <class T>
using Array = py::array_t<T, py::array::c_style | py::array::forcecast>;

template <class T>
std::span<const T> view(const Array<T>& a) {
  return {a.data(), static_cast<std::size_t>(a.size())};
}

// (offsets, data) numpy views of an Arrow string column, or None.
IdColumn id_column(const py::object& ids, std::size_t n, bool& present,
                   std::vector<py::object>& keep) {
  present = !ids.is_none();
  IdColumn col;
  if (!present) return col;
  auto pair = ids.cast<py::tuple>();
  py::array offsets = pair[0].cast<py::array>();
  auto data = pair[1].cast<Array<std::uint8_t>>();
  keep.push_back(offsets);
  keep.push_back(data);
  if (static_cast<std::size_t>(offsets.size()) != n + 1) throw py::value_error("id offsets size");
  if (offsets.dtype().is(py::dtype::of<std::int32_t>())) {
    col.offsets32 = static_cast<const std::int32_t*>(offsets.data());
  } else if (offsets.dtype().is(py::dtype::of<std::int64_t>())) {
    col.offsets64 = static_cast<const std::int64_t*>(offsets.data());
  } else {
    throw py::value_error("id offsets must be int32 or int64");
  }
  col.data = reinterpret_cast<const char*>(data.data());
  col.size = n;
  return col;
}

py::tuple replay_session(std::string symbol, std::string session_id, std::int64_t open_ns,
                         std::int64_t close_ns, int bar_seconds, const Array<std::int64_t>& quote_ns,
                         const py::object& quote_arrival, const Array<double>& bid,
                         const Array<double>& ask, const Array<double>& bid_size,
                         const Array<double>& ask_size, const py::object& quote_ids,
                         const Array<std::int64_t>& trade_ns, const py::object& trade_arrival,
                         const Array<double>& price, const Array<double>& size,
                         const Array<std::uint8_t>& trade_excluded, const py::object& trade_ids) {
  ReplayInput in;
  std::vector<py::object> keep;
  in.symbol = std::move(symbol);
  in.session = Session{std::move(session_id), open_ns, close_ns};
  in.bar_seconds = bar_seconds;
  in.quote_ns = view(quote_ns);
  Array<std::int64_t> qa, ta;
  if (!quote_arrival.is_none()) {
    qa = quote_arrival.cast<Array<std::int64_t>>();
    in.quote_arrival = view(qa);
  }
  in.bid = view(bid);
  in.ask = view(ask);
  in.bid_size = view(bid_size);
  in.ask_size = view(ask_size);
  in.quote_id_column = id_column(quote_ids, in.quote_ns.size(), in.quote_ids, keep);
  in.trade_ns = view(trade_ns);
  if (!trade_arrival.is_none()) {
    ta = trade_arrival.cast<Array<std::int64_t>>();
    in.trade_arrival = view(ta);
  }
  in.price = view(price);
  in.size = view(size);
  in.trade_excluded = view(trade_excluded);
  in.trade_id_column = id_column(trade_ids, in.trade_ns.size(), in.trade_ids, keep);
  const std::size_t nq = in.quote_ns.size(), nt = in.trade_ns.size();
  if (in.bid.size() != nq || in.ask.size() != nq || in.bid_size.size() != nq ||
      in.ask_size.size() != nq || (!in.quote_arrival.empty() && in.quote_arrival.size() != nq) ||
      in.price.size() != nt || in.size.size() != nt || in.trade_excluded.size() != nt ||
      (!in.trade_arrival.empty() && in.trade_arrival.size() != nt)) {
    throw py::value_error("replay array length mismatch");
  }
  ReplayResult result;
  std::string error;
  {
    py::gil_scoped_release release;  // pure C++; Python exceptions only after reacquiring
    try {
      result = replay(in);
    } catch (const std::exception& e) {
      error = e.what();
    }
  }
  if (!error.empty()) throw py::value_error(error);
  py::list bars;
  for (const auto& b : result.bars) bars.append(bar_tuple(b));
  return py::make_tuple(bars, py::array_t<std::int64_t>(result.ready_ns.size(), result.ready_ns.data()),
                        py::array_t<std::int64_t>(result.gaps_ns.size(), result.gaps_ns.data()),
                        result.quality);
}

// --- ITCH feed handler (hft/itch.py is the reference with the same interface) ----------

// chr(code) for one byte, as the reference's latin-1 text.
py::str byte_str(unsigned char code) {
  return py::reinterpret_steal<py::str>(PyUnicode_FromOrdinal(code));
}

py::str latin1_str(const std::string& text) {
  return py::reinterpret_steal<py::str>(
      PyUnicode_DecodeLatin1(text.data(), static_cast<Py_ssize_t>(text.size()), nullptr));
}

struct IntArg {
  long long value;
  bool overflow;  // does not fit in 64 bits; `value` is then only a sign (+1 / -1)
};

IntArg int_arg(const py::object& obj, const char* what) {
  if (!py::isinstance<py::int_>(obj)) throw py::type_error(std::string(what) + " must be an integer");
  int overflow = 0;
  const long long value = PyLong_AsLongLongAndOverflow(obj.ptr(), &overflow);
  if (value == -1 && PyErr_Occurred()) throw py::error_already_set();
  return {overflow != 0 ? overflow : value, overflow != 0};
}

std::uint16_t locate_arg(const py::object& obj) {
  const IntArg a = int_arg(obj, "locate");
  if (a.overflow || a.value < 0 || a.value > 65535) {
    throw py::value_error("locate must be in 0..65535");
  }
  return static_cast<std::uint16_t>(a.value);
}

py::list level_list(const std::vector<LevelView>& levels) {
  py::list out;
  for (const auto& l : levels) out.append(py::make_tuple(l.price, l.shares, l.orders));
  return out;
}

Side side_arg(const std::string& side) {
  if (side == "B") return Side::Buy;
  if (side == "S") return Side::Sell;
  throw py::value_error("side must be 'B' or 'S'");
}

std::vector<std::string> filter_symbols(const py::object& symbols) {
  std::vector<std::string> out;
  if (symbols.is_none()) return out;
  if (py::isinstance<py::str>(symbols) || py::isinstance<py::bytes>(symbols)) {
    throw py::type_error("symbols must be a collection of symbols, not a string");
  }
  for (const py::handle item : py::iter(symbols)) {
    if (!py::isinstance<py::str>(item)) throw py::type_error("symbols must be text");
    // Symbols are compared as the wire's single-byte text (UnicodeEncodeError is a ValueError).
    out.push_back(py::bytes(item.attr("encode")("latin-1")).cast<std::string>());
  }
  return out;
}

void bind_itch(py::module_& m) {
  py::class_<ItchBookBuilder>(m, "ItchBook")
      .def(py::init([](const py::object& symbols) {
             return std::make_unique<ItchBookBuilder>(filter_symbols(symbols));
           }),
           py::arg("symbols") = py::none())
      .def(
          "feed",
          [](ItchBookBuilder& book, const py::object& data) {
            // Any contiguous bytes-like object, read in place (the builder copies at most
            // the one frame it carries between calls).
            Py_buffer view;
            if (PyObject_GetBuffer(data.ptr(), &view, PyBUF_SIMPLE) != 0) {
              throw py::error_already_set();
            }
            try {
              book.feed(std::span<const std::uint8_t>(static_cast<const std::uint8_t*>(view.buf),
                                                      static_cast<std::size_t>(view.len)));
            } catch (...) {
              PyBuffer_Release(&view);
              throw;
            }
            PyBuffer_Release(&view);
          },
          py::arg("data"))
      .def("finish", &ItchBookBuilder::finish)
      .def(
          "snapshot",
          [](const ItchBookBuilder& book, const py::object& locate, const py::object& depth) {
            const std::uint16_t loc = locate_arg(locate);
            const IntArg n = int_arg(depth, "depth");
            if (n.value < 0) throw py::value_error("depth must not be negative");
            const std::size_t limit = n.overflow ? SIZE_MAX : static_cast<std::size_t>(n.value);
            py::dict out;
            out["bids"] = level_list(book.book().depth(loc, Side::Buy, limit));
            out["asks"] = level_list(book.book().depth(loc, Side::Sell, limit));
            return out;
          },
          py::arg("locate"), py::arg("depth"))
      .def(
          "queue",
          [](const ItchBookBuilder& book, const py::object& locate, const std::string& side,
             const py::object& price) {
            const std::uint16_t loc = locate_arg(locate);
            const Side s = side_arg(side);
            const IntArg p = int_arg(price, "price");
            py::list out;
            if (p.overflow || p.value < 0 || p.value > 0xffffffffLL) return out;  // no such level
            for (const auto& [ref, shares] :
                 book.book().queue(loc, s, static_cast<std::uint32_t>(p.value))) {
              out.append(py::make_tuple(ref, shares));
            }
            return out;
          },
          py::arg("locate"), py::arg("side"), py::arg("price"))
      .def(
          "stock",
          [](const ItchBookBuilder& book, const py::object& locate) {
            const StockInfo& s = book.stock(locate_arg(locate));
            py::dict out;
            out["symbol"] = latin1_str(s.symbol);
            out["trading_state"] = byte_str(static_cast<unsigned char>(s.trading_state));
            out["trades"] = s.trades;
            out["volume"] = s.volume;
            out["last_price"] = s.last_price;
            out["open_cross"] = s.open_cross ? py::object(py::int_(*s.open_cross)) : py::none();
            out["close_cross"] = s.close_cross ? py::object(py::int_(*s.close_cross)) : py::none();
            return out;
          },
          py::arg("locate"))
      .def("locates", [](const ItchBookBuilder& book) { return book.book().locates(); })
      .def("stats",
           [](const ItchBookBuilder& book) {
             const ItchStats s = book.stats();
             py::dict by_type;
             for (std::size_t code = 0; code < s.read.by_type.size(); ++code) {
               if (s.read.by_type[code] != 0) {
                 by_type[byte_str(static_cast<unsigned char>(code))] = s.read.by_type[code];
               }
             }
             py::dict out;
             out["messages"] = s.read.messages;
             out["by_type"] = by_type;
             out["unknown_types"] = s.read.unknown_types;
             out["malformed"] = s.read.malformed;
             out["trailing_bytes"] = s.read.trailing_bytes;
             out["end_of_session"] = s.read.end_of_session;
             out["truncated_bytes"] = s.truncated_bytes;
             out["unknown_refs"] = s.unknown_refs;
             out["duplicate_refs"] = s.duplicate_refs;
             out["overfills"] = s.overfills;
             out["invalid"] = s.invalid;
             out["crossed"] = s.crossed;
             out["filtered"] = s.filtered;
             out["live_orders"] = s.live_orders;
             out["peak_live_orders"] = s.peak_live_orders;
             out["levels"] = s.levels;
             out["peak_levels"] = s.peak_levels;
             return out;
           })
      .def("validate", [](const ItchBookBuilder& book) { return book.book().validate(); });
}

}  // namespace

PYBIND11_MODULE(hftcore, m) {
  m.doc() = "Low-latency market-data replay and decision engine";
  m.def("version", [] { return std::string(hftcore::version()); },
        "Engine version recorded in run identities");

  py::class_<BarAggregator>(m, "BarAggregator")
      .def(py::init([](std::string symbol, int bar_seconds, std::optional<std::string> session_id,
                       std::optional<std::int64_t> open_ns, std::optional<std::int64_t> close_ns,
                       std::int64_t late_tolerance_ns, std::size_t max_events) {
             std::optional<Session> session;
             if (session_id) session = Session{*session_id, open_ns.value(), close_ns.value()};
             try {
               return BarAggregator(std::move(symbol), bar_seconds, session, late_tolerance_ns,
                                    max_events);
             } catch (const std::invalid_argument& error) {
               throw py::value_error(error.what());
             }
           }),
           py::arg("symbol"), py::arg("bar_seconds") = 5, py::arg("session_id") = py::none(),
           py::arg("open_ns") = py::none(), py::arg("close_ns") = py::none(),
           py::arg("late_tolerance_ns") = 250'000'000, py::arg("max_events") = 100'000)
      .def("add", &add_event, py::arg("event"), py::arg("arrival_ns") = py::none())
      .def(
          "advance_to",
          [](BarAggregator& agg, const py::object& now_ns) {
            if (!is_int(now_ns)) throw py::value_error("clock must be integer nanoseconds");
            return bar_list(agg.advance_to(now_ns.cast<std::int64_t>()));
          },
          py::arg("now_ns"))
      .def("quality", [](const BarAggregator& agg) { return agg.quality(); })
      .def_property(
          "start", [](const BarAggregator& agg) { return agg.start(); },
          [](BarAggregator& agg, std::int64_t start) { agg.set_start(start); });

  py::class_<MarketEngine>(m, "MarketEngine")
      .def(py::init([](std::string symbol, const std::string& strategy, int bar_seconds) {
             return MarketEngine(std::move(symbol), parse_strategy_name(strategy), bar_seconds);
           }),
           py::arg("symbol"), py::arg("strategy") = "model", py::arg("bar_seconds") = 5)
      .def(
          "start_session",
          [](MarketEngine& e, std::string session_id, std::int64_t open_ns, std::int64_t close_ns,
             std::optional<std::int64_t> now_ns) {
            e.start_session(Session{std::move(session_id), open_ns, close_ns}, now_ns);
          },
          py::arg("session_id"), py::arg("open_ns"), py::arg("close_ns"),
          py::arg("now_ns") = py::none())
      .def("end_session", &MarketEngine::end_session)
      .def("reset_history", &MarketEngine::reset_history)
      .def("mark_gap", &MarketEngine::mark_gap, py::arg("now_ns"))
      .def("on_event", &engine_event, py::arg("event"), py::arg("arrival_ns"))
      .def(
          "advance_to",
          [](MarketEngine& e, std::int64_t now_ns) {
            return rethrow_value_errors([&] { return update_list(e.advance_to(now_ns)); });
          },
          py::arg("now_ns"))
      .def("history",
           [](const MarketEngine& e) {
             py::list out;
             for (const auto& b : e.history()) out.append(bar_tuple(b));
             return out;
           })
      .def("quality",
           [](MarketEngine& e) {
             return e.aggregator() ? e.aggregator()->quality()
                                   : std::map<std::string, std::int64_t>{};
           })
      .def("aggregator_start",
           [](MarketEngine& e) {
             return e.aggregator() ? e.aggregator()->start() : std::nullopt;
           })
      .def_property_readonly("last_event_ns", &MarketEngine::last_event_ns)
      .def_property_readonly("last_tick_ns", &MarketEngine::last_tick_ns)
      .def_property_readonly("warmup_after_ns", &MarketEngine::warmup_after_ns);

  m.def("replay", &replay_session, "Bars, ready times, gap times and quality for one session",
        py::arg("symbol"), py::arg("session_id"), py::arg("open_ns"), py::arg("close_ns"),
        py::arg("bar_seconds"), py::arg("quote_ns"), py::arg("quote_arrival"), py::arg("bid"),
        py::arg("ask"), py::arg("bid_size"), py::arg("ask_size"), py::arg("quote_ids"),
        py::arg("trade_ns"), py::arg("trade_arrival"), py::arg("price"), py::arg("size"),
        py::arg("trade_excluded"), py::arg("trade_ids"));

  py::class_<FrameRouter>(m, "FrameRouter")
      .def(py::init<>())
      .def("add", &FrameRouter::add, py::arg("engine"), py::keep_alive<1, 2>(),
           "Route a symbol's messages to this engine (kept alive by the router)")
      .def(
          "on_frame",
          [](FrameRouter& router, py::bytes raw, std::int64_t arrival_ns) {
            std::string_view text = raw;
            FrameResult result;
            try {
              result = router.on_frame(text, arrival_ns);
            } catch (const std::invalid_argument& error) {
              throw py::value_error(error.what());
            }
            py::list updates, quotes;
            for (const auto& u : result.updates) {
              updates.append(py::make_tuple(u.symbol, update_object(u.update), u.message));
            }
            for (const auto& q : result.quotes) {
              quotes.append(py::make_tuple(q.symbol, q.quote.event_ns, q.quote.bid, q.quote.ask,
                                           q.quote.bid_size, q.quote.ask_size, q.message));
            }
            return py::make_tuple(updates, quotes, result.malformed);
          },
          py::arg("raw"), py::arg("arrival_ns"));

  bind_itch(m);
}
