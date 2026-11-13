// Throughput and per-message latency of the ITCH 5.0 parser and order book on a
// decompressed BinaryFILE trading day (memory-mapped read-only):
//   (a) parse only: every decoded message folds fields into a checksum;
//   (b) parse plus the full order book (a fresh ItchBookBuilder per run);
//   (c) per-message latency of one dispatch() call into the builder, timed with rdtscp
//       and recorded in a log-linear histogram (no per-message samples are kept).
// Passes (a) and (b) repeat (--repeat N, default 3) and report each run and the median;
// pass (c) runs once. Output is aggregates only: no per-order or per-stock data.
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <span>
#include <string>
#include <vector>

#if defined(__x86_64__)
#include <x86intrin.h>
#endif

#include "hftcore/itch.hpp"
#include "hftcore/itch_book.hpp"

using namespace hftcore;
using Clock = std::chrono::steady_clock;

namespace {

// The mapping is consumed in windows so peak anonymous memory can be sampled between
// them; the parser carries no state between windows beyond the returned byte count.
constexpr std::size_t kWindow = std::size_t{256} << 20;

[[noreturn]] void fail(const char* what, const char* detail = "") {
  std::fprintf(stderr, "bench_itch: %s%s%s\n", what, *detail ? ": " : "", detail);
  std::exit(2);
}

// --- timer ------------------------------------------------------------------

#if defined(__x86_64__)
constexpr bool kTsc = true;
inline std::uint64_t tick_begin() {
  unsigned aux;
  _mm_lfence();  // earlier loads complete before the clock is read
  const std::uint64_t t = __rdtscp(&aux);
  _mm_lfence();  // the timed region does not start before the clock is read
  return t;
}
inline std::uint64_t tick_end() {
  unsigned aux;
  const std::uint64_t t = __rdtscp(&aux);  // waits for the timed region to finish
  _mm_lfence();
  return t;
}
#else
constexpr bool kTsc = false;
inline std::uint64_t tick_now() {
  return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch())
          .count());
}
inline std::uint64_t tick_begin() { return tick_now(); }
inline std::uint64_t tick_end() { return tick_now(); }
#endif

// Ticks per nanosecond, calibrated against steady_clock over at least 200 ms.
double calibrate_ticks_per_ns() {
  if (!kTsc) return 1.0;
  const auto c0 = Clock::now();
  const std::uint64_t t0 = tick_begin();
  while (Clock::now() - c0 < std::chrono::milliseconds(250)) {
  }
  const std::uint64_t t1 = tick_end();
  const auto c1 = Clock::now();
  const double ns = static_cast<double>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(c1 - c0).count());
  return static_cast<double>(t1 - t0) / ns;
}

// Median cost, in ticks, of an empty begin/end pair (the timer's own overhead).
std::uint64_t calibrate_overhead_ticks() {
  constexpr std::size_t kRuns = 1'000'000;
  std::vector<std::uint32_t> v(kRuns);
  for (auto& x : v) {
    const std::uint64_t a = tick_begin();
    const std::uint64_t b = tick_end();
    x = static_cast<std::uint32_t>(std::min<std::uint64_t>(b - a, 0xffffffffu));
  }
  std::nth_element(v.begin(), v.begin() + kRuns / 2, v.end());
  return v[kRuns / 2];
}

// --- log-linear histogram -----------------------------------------------------

// Values below 2^kSubBits are exact; above, each power of two is split into 2^kSubBits
// equal buckets, so a bucket is at most 1/64 (1.6%) of its lower bound wide and a
// reported percentile (the bucket midpoint) is within 0.8% of the recorded value.
class Histogram {
 public:
  static constexpr int kSubBits = 6;
  static constexpr std::size_t kSub = std::size_t{1} << kSubBits;

  void record(std::uint64_t v) {
    ++counts_[index(v)];
    ++n_;
    max_ = std::max(max_, v);
  }
  std::uint64_t count() const { return n_; }
  std::uint64_t max() const { return max_; }

  // Value (ticks) at percentile p in [0, 100]: nearest rank, bucket midpoint.
  double percentile(double p) const {
    if (n_ == 0) return 0;
    const auto rank = std::max<std::uint64_t>(
        1, static_cast<std::uint64_t>(std::ceil(p / 100.0 * static_cast<double>(n_))));
    std::uint64_t seen = 0;
    for (std::size_t i = 0; i < counts_.size(); ++i) {
      seen += counts_[i];
      if (seen >= rank) {
        const auto [lo, width] = bucket(i);
        return static_cast<double>(lo) + static_cast<double>(width - 1) / 2.0;
      }
    }
    return static_cast<double>(max_);
  }

 private:
  static std::size_t index(std::uint64_t v) {
    if (v < kSub) return static_cast<std::size_t>(v);
    const int shift = std::bit_width(v) - 1 - kSubBits;
    return (static_cast<std::size_t>(shift) + 1) * kSub + ((v >> shift) - kSub);
  }
  // Lower bound and width of bucket i.
  static std::pair<std::uint64_t, std::uint64_t> bucket(std::size_t i) {
    if (i < kSub) return {i, 1};
    const std::size_t shift = i / kSub - 1;
    return {(std::uint64_t{kSub} + i % kSub) << shift, std::uint64_t{1} << shift};
  }

  std::array<std::uint64_t, (64 - kSubBits + 1) * kSub> counts_{};
  std::uint64_t n_ = 0, max_ = 0;
};

// --- parse-only visitor ---------------------------------------------------------

// Folds the header and the interesting fields of every message into a checksum so the
// decoder cannot be optimized away. One generic call operator covers all messages.
struct ChecksumVisitor {
  std::uint64_t sum = 0;

  void fold(std::uint64_t x) { sum = std::rotl(sum, 7) ^ (x + 0x9e3779b97f4a7c15ULL); }

  template <class M>
  void operator()(const M& m) {
    fold(m.h.timestamp ^ (std::uint64_t{m.h.locate} << 48) ^ static_cast<unsigned char>(m.h.type));
    if constexpr (requires { m.ref; }) fold(m.ref);
    if constexpr (requires { m.new_ref; }) fold(m.new_ref);
    if constexpr (requires { m.shares; }) fold(m.shares);
    if constexpr (requires { m.price; }) fold(m.price);
    if constexpr (requires { m.stock; }) fold(static_cast<unsigned char>(m.stock[0]));
    if constexpr (requires { m.side; }) fold(static_cast<unsigned char>(m.side));
  }
};

// --- file and process helpers -------------------------------------------------------

struct Mapping {
  const std::uint8_t* data = nullptr;
  std::size_t size = 0;
};

Mapping map_file(const char* path) {
  const int fd = ::open(path, O_RDONLY);
  if (fd < 0) fail("cannot open", path);
  struct stat st {};
  if (::fstat(fd, &st) != 0 || st.st_size <= 0) fail("empty or unreadable file", path);
  void* p = ::mmap(nullptr, static_cast<std::size_t>(st.st_size), PROT_READ, MAP_PRIVATE, fd, 0);
  ::close(fd);
  if (p == MAP_FAILED) fail("mmap failed", path);
  ::madvise(p, static_cast<std::size_t>(st.st_size), MADV_SEQUENTIAL);
  return {static_cast<const std::uint8_t*>(p), static_cast<std::size_t>(st.st_size)};
}

// Resident anonymous memory (the books, not the mapped file) in MiB from /proc.
double rss_anon_mib() {
  std::FILE* f = std::fopen("/proc/self/status", "r");
  if (!f) return 0;
  char line[256];
  double kib = 0;
  while (std::fgets(line, sizeof line, f)) {
    if (std::strncmp(line, "RssAnon:", 8) == 0) {
      kib = std::strtod(line + 8, nullptr);
      break;
    }
  }
  std::fclose(f);
  return kib / 1024.0;
}

double peak_rss_mib() {
  rusage ru{};
  ::getrusage(RUSAGE_SELF, &ru);
  return static_cast<double>(ru.ru_maxrss) / 1024.0;  // Linux reports KiB
}

double seconds_since(Clock::time_point t0) {
  return std::chrono::duration<double>(Clock::now() - t0).count();
}

// Runs parse_binary_file over the mapping window by window; returns the bytes left
// unparsed (an incomplete frame at the end of the file). `peak_anon` tracks the largest
// anonymous RSS seen between windows.
template <class Visitor>
std::size_t parse_all(const Mapping& m, Visitor& visitor, itch::ReadStats& stats,
                      double& peak_anon) {
  std::size_t pos = 0;
  while (pos < m.size) {
    const std::size_t len = std::min(kWindow, m.size - pos);
    const std::size_t used = itch::parse_binary_file({m.data + pos, len}, visitor, stats);
    if (used == 0) break;  // incomplete frame at the end of the file: nothing more to parse
    pos += used;
    peak_anon = std::max(peak_anon, rss_anon_mib());
  }
  return m.size - pos;
}

struct Run {
  std::uint64_t messages;
  double seconds;
};

double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  const std::size_t n = v.size();
  return n % 2 ? v[n / 2] : (v[n / 2 - 1] + v[n / 2]) / 2.0;
}

void print_run(const char* label, std::size_t k, std::uint64_t messages, double seconds) {
  std::printf("  %-6s %2zu  messages=%llu  wall=%8.3f s  mean=%7.1f ns/msg  rate=%7.2f Mmsg/s\n",
              label, k, static_cast<unsigned long long>(messages), seconds,
              seconds * 1e9 / static_cast<double>(messages),
              static_cast<double>(messages) / seconds / 1e6);
}

void print_median(const std::vector<Run>& runs) {
  std::vector<double> secs;
  for (const Run& r : runs) secs.push_back(r.seconds);
  const double s = median(secs);
  const std::uint64_t n = runs.front().messages;
  std::printf("  median     messages=%llu  wall=%8.3f s  mean=%7.1f ns/msg  rate=%7.2f Mmsg/s\n",
              static_cast<unsigned long long>(n), s, s * 1e9 / static_cast<double>(n),
              static_cast<double>(n) / s / 1e6);
}

// --- pass (c): per-message latency ------------------------------------------------------

struct Latency {
  Histogram all;
  std::array<Histogram, 5> by_type;  // A D U E X
  static constexpr std::array<char, 5> kTypes = {'A', 'D', 'U', 'E', 'X'};
};

void print_latency(const char* name, const Histogram& h, double ns_per_tick) {
  if (h.count() == 0) {
    std::printf("  %-8s n=0\n", name);
    return;
  }
  std::printf("  %-8s n=%-11llu p50=%8.1f ns  p99=%8.1f ns  p99.9=%9.1f ns  max=%11.1f ns\n", name,
              static_cast<unsigned long long>(h.count()), h.percentile(50) * ns_per_tick,
              h.percentile(99) * ns_per_tick, h.percentile(99.9) * ns_per_tick,
              static_cast<double>(h.max()) * ns_per_tick);
}

}  // namespace

int main(int argc, char** argv) {
  const char* path = nullptr;
  long repeat = 3;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--repeat") == 0 && i + 1 < argc) {
      repeat = std::strtol(argv[++i], nullptr, 10);
    } else if (argv[i][0] != '-' && !path) {
      path = argv[i];
    } else {
      path = nullptr;
      break;
    }
  }
  if (!path || repeat < 1) {
    std::fprintf(stderr, "usage: %s FILE [--repeat N]\n", argv[0]);
    return 2;
  }

  const Mapping map = map_file(path);
  std::printf("file: %.1f MiB mapped read-only (MAP_PRIVATE, MADV_SEQUENTIAL)\n",
              static_cast<double>(map.size) / (1024.0 * 1024.0));
  double peak_anon = rss_anon_mib();

  // (a) parse only
  std::printf("\n(a) parse only (checksum of decoded fields)\n");
  std::vector<Run> parse_runs;
  std::uint64_t checksum = 0;
  itch::ReadStats parse_stats;
  for (long k = 0; k < repeat; ++k) {
    ChecksumVisitor visitor;
    itch::ReadStats stats;
    const auto t0 = Clock::now();
    parse_all(map, visitor, stats, peak_anon);
    const double secs = seconds_since(t0);
    parse_runs.push_back({stats.messages, secs});
    print_run("run", static_cast<std::size_t>(k + 1), stats.messages, secs);
    checksum = visitor.sum;
    parse_stats = stats;
  }
  print_median(parse_runs);
  std::printf("  checksum=%016llx\n", static_cast<unsigned long long>(checksum));

  // (b) parse plus the full order book
  std::printf("\n(b) parse plus full order book\n");
  std::vector<Run> book_runs;
  ItchStats book_stats;
  for (long k = 0; k < repeat; ++k) {
    auto builder = std::make_unique<ItchBookBuilder>();
    const auto t0 = Clock::now();
    const std::size_t left = parse_all(map, *builder, builder->read_stats(), peak_anon);
    const double secs = seconds_since(t0);
    book_stats = builder->stats();
    book_stats.truncated_bytes = left;
    book_runs.push_back({book_stats.read.messages, secs});
    print_run("run", static_cast<std::size_t>(k + 1), book_stats.read.messages, secs);
  }
  print_median(book_runs);

  // (c) per-message latency of dispatch() into the builder
  const double ticks_per_ns = calibrate_ticks_per_ns();
  const std::uint64_t overhead = calibrate_overhead_ticks();
  const double ns_per_tick = 1.0 / ticks_per_ns;
  std::printf("\n(c) per-message latency: one dispatch() call into the builder\n");
  std::printf("  timer: %s, %.4f ticks/ns (%.3f GHz), overhead %llu ticks = %.1f ns "
              "(median of 1,000,000 empty regions, subtracted, clamped at 0)\n",
              kTsc ? "rdtscp after lfence" : "steady_clock", ticks_per_ns, ticks_per_ns,
              static_cast<unsigned long long>(overhead), static_cast<double>(overhead) * ns_per_tick);
  std::printf("  histogram: log-linear, %zu buckets per power of two; percentiles are bucket "
              "midpoints, within 0.8%% of the recorded value (max is exact)\n",
              Histogram::kSub);

  auto latency = std::make_unique<Latency>();
  std::array<Histogram*, 256> route{};
  for (std::size_t i = 0; i < Latency::kTypes.size(); ++i) {
    route[static_cast<unsigned char>(Latency::kTypes[i])] = &latency->by_type[i];
  }
  auto builder = std::make_unique<ItchBookBuilder>();
  itch::ReadStats& rs = builder->read_stats();
  std::size_t pos = 0;
  while (pos + 2 <= map.size && !rs.end_of_session) {
    const std::size_t length = itch::load_be16(map.data + pos);
    if (length == 0) {
      rs.end_of_session = true;
      pos += 2;
      rs.trailing_bytes += map.size - pos;
      pos = map.size;
      break;
    }
    if (map.size - pos - 2 < length) break;  // incomplete frame at the end of the file
    const std::uint8_t* body = map.data + pos + 2;
    ++rs.messages;
    ++rs.by_type[body[0]];
    const std::uint64_t t0 = tick_begin();
    const itch::FrameResult result = itch::dispatch(body, length, *builder);
    const std::uint64_t t1 = tick_end();
    switch (result) {
      case itch::FrameResult::Ok:
        break;
      case itch::FrameResult::Unknown:
        ++rs.unknown_types;
        break;
      case itch::FrameResult::Malformed:
        ++rs.malformed;
        break;
    }
    const std::uint64_t raw = t1 - t0;
    const std::uint64_t ticks = raw > overhead ? raw - overhead : 0;
    latency->all.record(ticks);
    if (Histogram* h = route[body[0]]) h->record(ticks);
    pos += 2 + length;
    if ((rs.messages & 0xfffff) == 0) peak_anon = std::max(peak_anon, rss_anon_mib());
  }
  peak_anon = std::max(peak_anon, rss_anon_mib());
  ItchStats latency_stats = builder->stats();
  latency_stats.truncated_bytes = pos < map.size ? map.size - pos : 0;
  print_latency("all", latency->all, ns_per_tick);
  for (std::size_t i = 0; i < Latency::kTypes.size(); ++i) {
    const char name[2] = {Latency::kTypes[i], 0};
    print_latency(name, latency->by_type[i], ns_per_tick);
  }
  const std::string verdict = builder->book().validate();
  std::printf("  book validate after pass (c): %s\n", verdict.empty() ? "ok" : verdict.c_str());

  // Cross-check: the hand-driven walk of pass (c) must match pass (b) exactly.
  const bool consistent = latency_stats.read.messages == book_stats.read.messages &&
                          latency_stats.live_orders == book_stats.live_orders &&
                          latency_stats.peak_live_orders == book_stats.peak_live_orders &&
                          latency_stats.peak_levels == book_stats.peak_levels &&
                          latency_stats.unknown_refs == book_stats.unknown_refs &&
                          latency_stats.read.trailing_bytes == book_stats.read.trailing_bytes &&
                          latency_stats.read.end_of_session == book_stats.read.end_of_session;
  std::printf("  pass (c) counters match pass (b): %s\n", consistent ? "yes" : "NO");
  const bool parse_consistent = parse_stats.messages == book_stats.read.messages;
  std::printf("  pass (a) message count matches pass (b): %s\n", parse_consistent ? "yes" : "NO");

  // Totals from the full-book run (pass (b)).
  const ItchStats& s = book_stats;
  std::printf("\nbook after the file (pass (b))\n");
  std::printf("  messages=%llu  unknown types=%llu  malformed=%llu  trailing bytes=%llu  "
              "truncated bytes=%llu  end-of-session marker=%s\n",
              static_cast<unsigned long long>(s.read.messages),
              static_cast<unsigned long long>(s.read.unknown_types),
              static_cast<unsigned long long>(s.read.malformed),
              static_cast<unsigned long long>(s.read.trailing_bytes),
              static_cast<unsigned long long>(s.truncated_bytes),
              s.read.end_of_session ? "yes" : "no");
  std::printf("  unknown refs=%llu  duplicate refs=%llu  overfills=%llu  invalid=%llu  "
              "crossed=%llu  filtered=%llu\n",
              static_cast<unsigned long long>(s.unknown_refs),
              static_cast<unsigned long long>(s.duplicate_refs),
              static_cast<unsigned long long>(s.overfills),
              static_cast<unsigned long long>(s.invalid),
              static_cast<unsigned long long>(s.crossed),
              static_cast<unsigned long long>(s.filtered));
  std::printf("  live orders at the end=%zu  peak live orders=%zu  levels at the end=%zu  "
              "peak levels=%zu\n",
              s.live_orders, s.peak_live_orders, s.levels, s.peak_levels);
  std::printf("  peak RSS=%.0f MiB (ru_maxrss; includes touched mapped-file pages)  "
              "peak anonymous RSS=%.0f MiB (sampled between windows; the books and buffers)\n",
              peak_rss_mib(), peak_anon);
  std::printf("  parse-only checksum=%016llx\n", static_cast<unsigned long long>(checksum));
  return consistent && parse_consistent && verdict.empty() ? 0 : 1;
}
