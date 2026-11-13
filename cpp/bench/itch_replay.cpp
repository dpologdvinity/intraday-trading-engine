// Replays a decompressed Nasdaq ITCH 5.0 BinaryFILE (a path, or "-" for stdin so that
// `gzip -dc day.gz | hftcore_itch_replay -` works) through the order book builder in
// 1 MiB chunks and prints aggregates only: message counts, data-quality counters, book
// sizes and, for the requested stocks, trades, volume, cross prices and the final
// best bid and ask. No per-order data is ever printed.
//
//   hftcore_itch_replay [--symbols A,B] [--validate-every N] FILE|-
//
// --symbols limits the book to those stocks (the counters then describe only them).
// --validate-every N runs the book's invariant check at the first chunk boundary after
// the message count crosses each multiple of N, and once more at the end; the first
// failure is printed and the exit status is 1.
#include <unistd.h>

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <string>
#include <vector>

#include "hftcore/itch_book.hpp"

using namespace hftcore;

namespace {

constexpr std::size_t kChunk = std::size_t{1} << 20;

using ull = unsigned long long;

[[noreturn]] void usage(const char* argv0) {
  std::fprintf(stderr, "usage: %s [--symbols A,B] [--validate-every N] FILE|-\n", argv0);
  std::exit(2);
}

std::vector<std::string> split_symbols(const std::string& list) {
  std::vector<std::string> out;
  std::size_t start = 0;
  while (start <= list.size()) {
    const std::size_t end = list.find(',', start);
    const std::string name = list.substr(start, end == std::string::npos ? end : end - start);
    if (!name.empty()) out.push_back(name);
    if (end == std::string::npos) break;
    start = end + 1;
  }
  return out;
}

// Prices are 1/10,000 dollars.
std::string dollars(std::uint64_t price) {
  char buf[40];
  std::snprintf(buf, sizeof buf, "$%llu.%04llu", static_cast<ull>(price / 10000),
                static_cast<ull>(price % 10000));
  return buf;
}

std::string optional_dollars(const std::optional<std::uint32_t>& price) {
  return price ? dollars(*price) : "-";
}

// The midpoint rounded half up to a whole 1/10,000 dollar.
std::string mid_dollars(std::uint32_t bid, std::uint32_t ask) {
  return dollars((std::uint64_t{bid} + ask + 1) / 2);
}

std::string type_label(unsigned char code) {
  char buf[8];
  if (code >= 0x21 && code <= 0x7e) {
    std::snprintf(buf, sizeof buf, "%c", code);
  } else {
    std::snprintf(buf, sizeof buf, "0x%02x", code);
  }
  return buf;
}

}  // namespace

int main(int argc, char** argv) {
  std::vector<std::string> symbols;
  std::uint64_t validate_every = 0;
  const char* path = nullptr;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--symbols" && i + 1 < argc) {
      symbols = split_symbols(argv[++i]);
    } else if (arg == "--validate-every" && i + 1 < argc) {
      char* end = nullptr;
      errno = 0;
      const long long n = std::strtoll(argv[++i], &end, 10);
      if (errno != 0 || *end != '\0' || n <= 0) usage(argv[0]);
      validate_every = static_cast<std::uint64_t>(n);
    } else if ((arg == "-" || arg[0] != '-') && !path) {
      path = argv[i];
    } else {
      usage(argv[0]);
    }
  }
  if (!path) usage(argv[0]);

  const bool from_stdin = std::strcmp(path, "-") == 0;
  FILE* file = from_stdin ? stdin : std::fopen(path, "rb");
  if (!file) {
    std::fprintf(stderr, "hftcore_itch_replay: cannot open %s: %s\n", path, std::strerror(errno));
    return 2;
  }

  ItchBookBuilder builder(symbols);
  std::vector<std::uint8_t> buffer(kChunk);
  std::uint64_t validations = 0, next_validation = validate_every;

  auto validate = [&](const char* when) {
    ++validations;
    const std::string problem = builder.book().validate();
    if (!problem.empty()) {
      std::fprintf(stderr, "validation failed %s (after %llu messages): %s\n", when,
                   static_cast<ull>(builder.stats().read.messages), problem.c_str());
      std::exit(1);
    }
  };

  for (;;) {
    const std::size_t got = std::fread(buffer.data(), 1, buffer.size(), file);
    if (got > 0) {
      builder.feed({buffer.data(), got});
      if (validate_every > 0) {
        const std::uint64_t messages = builder.stats().read.messages;
        if (messages >= next_validation) {
          validate("at a chunk boundary");
          next_validation = (messages / validate_every + 1) * validate_every;
        }
      }
    }
    if (got < buffer.size()) {
      if (std::ferror(file)) {
        std::fprintf(stderr, "hftcore_itch_replay: read error: %s\n", std::strerror(errno));
        return 2;
      }
      break;  // end of input
    }
  }
  if (!from_stdin) std::fclose(file);
  builder.finish();
  if (validate_every > 0) validate("at the end");

  const ItchStats s = builder.stats();
  std::printf("filter: %s\n", symbols.empty() ? "all stocks" : "listed symbols only");
  std::printf("messages: %llu\n", static_cast<ull>(s.read.messages));
  std::printf("by type:");
  for (std::size_t code = 0; code < s.read.by_type.size(); ++code) {
    if (s.read.by_type[code] != 0) {
      std::printf(" %s=%llu", type_label(static_cast<unsigned char>(code)).c_str(),
                  static_cast<ull>(s.read.by_type[code]));
    }
  }
  std::printf("\n");
  std::printf("unknown types: %llu\nmalformed: %llu\ntrailing bytes: %llu\ntruncated bytes: %llu\n",
              static_cast<ull>(s.read.unknown_types), static_cast<ull>(s.read.malformed),
              static_cast<ull>(s.read.trailing_bytes), static_cast<ull>(s.truncated_bytes));
  std::printf("end of session marker: %s\n", s.read.end_of_session ? "yes" : "no");
  std::printf("unknown refs: %llu\nduplicate refs: %llu\noverfills: %llu\ninvalid: %llu\n",
              static_cast<ull>(s.unknown_refs), static_cast<ull>(s.duplicate_refs),
              static_cast<ull>(s.overfills), static_cast<ull>(s.invalid));
  std::printf("crossed: %llu\nfiltered: %llu\n", static_cast<ull>(s.crossed),
              static_cast<ull>(s.filtered));
  std::printf("live orders at the end: %zu\npeak live orders: %zu\nlevels at the end: %zu\n"
              "peak levels: %zu\n",
              s.live_orders, s.peak_live_orders, s.levels, s.peak_levels);
  std::printf("validations run: %llu\n", static_cast<ull>(validations));

  for (const std::string& symbol : symbols) {
    std::optional<std::uint16_t> locate;
    for (std::uint32_t l = 0; l <= 0xffff; ++l) {
      if (builder.stock(static_cast<std::uint16_t>(l)).symbol == symbol) {
        locate = static_cast<std::uint16_t>(l);
        break;
      }
    }
    if (!locate) {
      std::printf("%s: not in the stock directory\n", symbol.c_str());
      continue;
    }
    const StockInfo& info = builder.stock(*locate);
    const auto bid = builder.book().best(*locate, Side::Buy);
    const auto ask = builder.book().best(*locate, Side::Sell);
    std::printf("%s: trades=%llu volume=%llu open cross=%s close cross=%s", symbol.c_str(),
                static_cast<ull>(info.trades), static_cast<ull>(info.volume),
                optional_dollars(info.open_cross).c_str(),
                optional_dollars(info.close_cross).c_str());
    if (bid && ask) {
      std::printf(" best bid=%s best ask=%s mid=%s\n", dollars(bid->price).c_str(),
                  dollars(ask->price).c_str(), mid_dollars(bid->price, ask->price).c_str());
    } else {
      std::printf(" last book mid=-\n");
    }
  }
  return 0;
}
