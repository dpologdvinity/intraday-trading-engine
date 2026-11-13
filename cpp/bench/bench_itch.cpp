// Throughput and per-message latency of the ITCH 5.0 parser and order book on a
// decompressed BinaryFILE trading day (memory-mapped read-only):
//   (a) parse only: every decoded message folds every decoded field into a checksum;
//   (b) parse plus the full order book (a fresh ItchBookBuilder per run);
//   (c) per-message latency of one dispatch() call into the builder, timed with rdtscp
//       and recorded in a log-linear histogram (no per-message samples are kept).
// Passes (a) and (b) repeat (--repeat N, default 3) and report each run and the median;
// pass (c) runs once. Output is aggregates only: no per-order or per-stock data.
//
// Run it pinned to ONE core (`taskset -c N hftcore_bench_itch FILE`): rdtscp readings from
// different cores are not comparable. Pass (c) counts CPU-id changes and prints the count.
//
// The file may be larger than the page cache, so input I/O is kept out of the timings:
// the mapping is consumed in 256 MiB windows; each window is pre-faulted (one byte per
// page touched) before the clock starts and released (MADV_DONTNEED) after it. The
// headline rate is CPU-bound (the sum of the timed spans); end-to-end wall time including
// page-in is printed next to it, with page-fault counts for the timed spans and for the
// untimed page-in. A non-zero major-fault count inside a timed span means the input was
// evicted between page-in and use: do not publish that run as parser cost.
//
// The checksum folds every decoded field (header, numbers, characters and the symbol,
// subtype, reason, MPID and attribution texts); no decoded field is left unread.
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
#include <type_traits>
#include <vector>

#if defined(__x86_64__)
#include <x86intrin.h>
#endif

#include "hftcore/itch.hpp"
#include "hftcore/itch_book.hpp"

using namespace hftcore;
using Clock = std::chrono::steady_clock;
using ull = unsigned long long;

namespace {

constexpr std::size_t kWindow = std::size_t{256} << 20;
constexpr std::size_t kPage = 4096;
constexpr std::size_t kLead = std::size_t{1} << 20;  // pass (c) refills when this close to the end

[[noreturn]] void fail(const char* what, const char* detail = "") {
  std::fprintf(stderr, "bench_itch: %s%s%s\n", what, *detail ? ": " : "", detail);
  std::exit(2);
}

// --- timer ------------------------------------------------------------------

#if defined(__x86_64__)
constexpr bool kTsc = true;
inline std::uint64_t tick_begin(unsigned& cpu) {
  _mm_lfence();  // earlier loads complete before the clock is read
  const std::uint64_t t = __rdtscp(&cpu);
  _mm_lfence();  // the timed region does not start before the clock is read
  return t;
}
inline std::uint64_t tick_end(unsigned& cpu) {
  const std::uint64_t t = __rdtscp(&cpu);  // waits for the timed region to finish
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
inline std::uint64_t tick_begin(unsigned& cpu) {
  cpu = 0;
  return tick_now();
}
inline std::uint64_t tick_end(unsigned& cpu) {
  cpu = 0;
  return tick_now();
}
#endif

// Ticks per nanosecond, calibrated against steady_clock over at least 200 ms.
double calibrate_ticks_per_ns() {
  if (!kTsc) return 1.0;
  unsigned cpu;
  const auto c0 = Clock::now();
  const std::uint64_t t0 = tick_begin(cpu);
  while (Clock::now() - c0 < std::chrono::milliseconds(250)) {
  }
  const std::uint64_t t1 = tick_end(cpu);
  const auto c1 = Clock::now();
  const double ns =
      static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(c1 - c0).count());
  return static_cast<double>(t1 - t0) / ns;
}

// Median cost, in ticks, of an empty begin/end pair (the timer's own overhead).
std::uint64_t calibrate_overhead_ticks() {
  constexpr std::size_t kRuns = 1'000'000;
  std::vector<std::uint32_t> v(kRuns);
  unsigned cpu;
  for (auto& x : v) {
    const std::uint64_t a = tick_begin(cpu);
    const std::uint64_t b = tick_end(cpu);
    x = static_cast<std::uint32_t>(std::min<std::uint64_t>(b - a, 0xffffffffu));
  }
  std::nth_element(v.begin(), v.begin() + kRuns / 2, v.end());
  return v[kRuns / 2];
}

// --- log-linear histogram -----------------------------------------------------

// Values below 2^kSubBits are exact; above, each power of two is split into 2^kSubBits
// equal buckets, so a bucket is at most 1/64 (1.6%) of its lower bound wide and a
// reported percentile (the bucket midpoint, never above the recorded maximum) is within
// 0.8% of the recorded value.
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

  // Value (ticks) at percentile p in [0, 100]: nearest rank, bucket midpoint, <= max.
  double percentile(double p) const {
    if (n_ == 0) return 0;
    const auto rank = std::max<std::uint64_t>(
        1, static_cast<std::uint64_t>(std::ceil(p / 100.0 * static_cast<double>(n_))));
    std::uint64_t seen = 0;
    for (std::size_t i = 0; i < counts_.size(); ++i) {
      seen += counts_[i];
      if (seen >= rank) {
        const auto [lo, width] = bucket(i);
        const double mid = static_cast<double>(lo) + static_cast<double>(width - 1) / 2.0;
        return std::min(mid, static_cast<double>(max_));
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

// Every message field that any struct in itch.hpp declares. The visitor folds whichever
// of them the message has, so a message's whole decoded payload feeds the checksum and
// the compiler cannot drop any of the decoding.
#define ITCH_FIELDS(X)                                                                          \
  X(ref) X(new_ref) X(match) X(shares) X(price) X(side) X(stock) X(attribution) X(printable)    \
  X(event_code) X(market_category) X(financial_status) X(round_lot_size) X(round_lots_only)     \
  X(issue_classification) X(issue_subtype) X(authenticity) X(short_sale_threshold) X(ipo_flag)  \
  X(luld_tier) X(etp_flag) X(etp_leverage) X(inverse) X(state) X(reserved) X(reason) X(action)  \
  X(mpid) X(primary) X(mode) X(level1) X(level2) X(level3) X(level) X(release_time)             \
  X(release_qualifier) X(reference) X(upper) X(lower) X(extension) X(market_code)               \
  X(cross_type) X(paired) X(imbalance) X(direction) X(far_price) X(near_price)                  \
  X(price_variation) X(interest) X(open_eligibility) X(min_price) X(max_price) X(near_time)     \
  X(lower_collar) X(upper_collar)

struct ChecksumVisitor {
  std::uint64_t sum = 0;

  void fold(std::uint64_t x) { sum = std::rotl(sum, 7) ^ (x + 0x9e3779b97f4a7c15ULL); }

  template <class T>
  void field(const T& v) {
    if constexpr (std::is_integral_v<T>) {
      fold(static_cast<std::uint64_t>(static_cast<std::make_unsigned_t<T>>(v)));
    } else {  // std::array<char, N>, N <= 8
      std::uint64_t packed = 0;
      std::memcpy(&packed, v.data(), v.size());
      fold(packed);
    }
  }

  template <class M>
  void operator()(const M& m) {
    fold(m.h.timestamp ^ (std::uint64_t{m.h.locate} << 48) ^ (std::uint64_t{m.h.tracking} << 32) ^
         static_cast<unsigned char>(m.h.type));
#define ITCH_FOLD(name) \
  if constexpr (requires { m.name; }) field(m.name);
    ITCH_FIELDS(ITCH_FOLD)
#undef ITCH_FOLD
  }
};

// --- file, paging and process helpers --------------------------------------------------

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

struct Faults {
  long major = 0, minor = 0;
  Faults operator-(const Faults& o) const { return {major - o.major, minor - o.minor}; }
  Faults& operator+=(const Faults& o) {
    major += o.major;
    minor += o.minor;
    return *this;
  }
};

Faults faults_now() {
  rusage ru{};
  ::getrusage(RUSAGE_SELF, &ru);
  return {ru.ru_majflt, ru.ru_minflt};
}

// Pre-faults the mapping ahead of the parser (one byte per page) and releases what the
// parser has finished with, so the working set stays at about one window and no page-in
// happens inside a timed span. Page-in faults are tallied separately.
class Pager {
 public:
  explicit Pager(const Mapping& m) : m_(m) {}

  // Makes [pos, pos + bytes) resident (clamped to the file).
  void prepare(std::size_t pos, std::size_t bytes) {
    const std::size_t hi = std::min(m_.size, pos + bytes);
    const std::size_t lo = std::max(touched_, pos) & ~(kPage - 1);
    if (hi <= lo) return;
    const Faults before = faults_now();
    volatile std::uint8_t sink = 0;
    std::uint8_t acc = 0;
    for (std::size_t p = lo; p < hi; p += kPage) acc = static_cast<std::uint8_t>(acc + m_.data[p]);
    acc = static_cast<std::uint8_t>(acc + m_.data[hi - 1]);
    sink = acc;
    (void)sink;
    touched_ = hi;
    paged_ += faults_now() - before;
  }
  // Drops the page mappings below pos (the page cache keeps the data).
  void release_before(std::size_t pos) {
    const std::size_t hi = pos & ~(kPage - 1);
    if (hi > released_) {
      ::madvise(const_cast<std::uint8_t*>(m_.data) + released_, hi - released_, MADV_DONTNEED);
      released_ = hi;
    }
  }
  std::size_t touched() const { return touched_; }
  const Faults& paged() const { return paged_; }

 private:
  const Mapping& m_;
  std::size_t touched_ = 0, released_ = 0;
  Faults paged_;
};

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

// --- passes (a) and (b) -------------------------------------------------------------------

struct PassResult {
  double cpu_seconds = 0;   // sum of the timed spans (parse/book work only)
  double wall_seconds = 0;  // end to end, including page-in, release and RSS sampling
  std::size_t left = 0;     // bytes of an incomplete frame at the end of the file
  Faults timed;             // faults inside timed spans
  Faults paged;             // faults while paging the input in (untimed)
};

// Parses the mapping window by window. Only parse_binary_file is inside a timed span.
template <class Visitor>
PassResult run_pass(const Mapping& m, Visitor& visitor, itch::ReadStats& stats,
                    double& peak_anon) {
  PassResult r;
  Pager pager(m);
  const auto wall0 = Clock::now();
  std::size_t pos = 0;
  while (pos < m.size) {
    pager.prepare(pos, kWindow);  // untimed page-in
    const std::size_t len = std::min(kWindow, m.size - pos);
    const Faults f0 = faults_now();
    const auto t0 = Clock::now();
    const std::size_t used = itch::parse_binary_file({m.data + pos, len}, visitor, stats);
    r.cpu_seconds += seconds_since(t0);
    r.timed += faults_now() - f0;
    if (used == 0) break;  // incomplete frame at the end of the file: nothing more to parse
    pos += used;
    pager.release_before(pos);
    peak_anon = std::max(peak_anon, rss_anon_mib());  // outside the timed span
  }
  pager.release_before(m.size);
  r.left = m.size - pos;
  r.paged = pager.paged();
  r.wall_seconds = seconds_since(wall0);
  return r;
}

double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  const std::size_t n = v.size();
  return n % 2 ? v[n / 2] : (v[n / 2 - 1] + v[n / 2]) / 2.0;
}

void print_run(std::size_t k, std::uint64_t messages, const PassResult& r) {
  const double n = static_cast<double>(messages);
  std::printf("  run %2zu  cpu-bound %8.3f s  %7.1f ns/msg  %7.2f Mmsg/s | end-to-end %8.3f s "
              "(incl. page-in)\n",
              k, r.cpu_seconds, r.cpu_seconds * 1e9 / n, n / r.cpu_seconds / 1e6, r.wall_seconds);
  std::printf("          faults in timed spans: major=%ld minor=%ld (minor includes the book's own "
              "memory growth) | page-in (untimed): major=%ld minor=%ld\n",
              r.timed.major, r.timed.minor, r.paged.major, r.paged.minor);
}

void print_median(std::uint64_t messages, const std::vector<PassResult>& runs) {
  std::vector<double> cpu, wall;
  for (const PassResult& r : runs) {
    cpu.push_back(r.cpu_seconds);
    wall.push_back(r.wall_seconds);
  }
  const double c = median(cpu), n = static_cast<double>(messages);
  std::printf("  median  cpu-bound %8.3f s  %7.1f ns/msg  %7.2f Mmsg/s | end-to-end %8.3f s\n", c,
              c * 1e9 / n, n / c / 1e6, median(wall));
  long major = 0;
  for (const PassResult& r : runs) major += r.timed.major;
  if (major > 0) {
    std::printf("  WARNING: %ld major page fault(s) inside timed spans: input was evicted after "
                "page-in; these timings include disk reads\n",
                major);
  }
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
              static_cast<ull>(h.count()), h.percentile(50) * ns_per_tick,
              h.percentile(99) * ns_per_tick, h.percentile(99.9) * ns_per_tick,
              static_cast<double>(h.max()) * ns_per_tick);
}

bool same_read(const itch::ReadStats& a, const itch::ReadStats& b) {
  return a.messages == b.messages && a.by_type == b.by_type && a.unknown_types == b.unknown_types &&
         a.malformed == b.malformed && a.trailing_bytes == b.trailing_bytes &&
         a.end_of_session == b.end_of_session;
}

bool same_stats(const ItchStats& a, const ItchStats& b) {
  return same_read(a.read, b.read) && a.truncated_bytes == b.truncated_bytes &&
         a.unknown_refs == b.unknown_refs && a.duplicate_refs == b.duplicate_refs &&
         a.overfills == b.overfills && a.invalid == b.invalid && a.crossed == b.crossed &&
         a.filtered == b.filtered && a.live_orders == b.live_orders &&
         a.peak_live_orders == b.peak_live_orders && a.levels == b.levels &&
         a.peak_levels == b.peak_levels;
}

}  // namespace

int main(int argc, char** argv) {
  const char* path = nullptr;
  long repeat = 3;
  bool ok_args = true;
  for (int i = 1; i < argc && ok_args; ++i) {
    if (std::strcmp(argv[i], "--repeat") == 0 && i + 1 < argc) {
      char* end = nullptr;
      repeat = std::strtol(argv[++i], &end, 10);
      ok_args = *argv[i] != '\0' && *end == '\0' && repeat >= 1;
    } else if (argv[i][0] != '-' && !path) {
      path = argv[i];
    } else {
      ok_args = false;
    }
  }
  if (!ok_args || !path) {
    std::fprintf(stderr,
                 "usage: taskset -c N %s FILE [--repeat N]\n"
                 "  FILE: decompressed ITCH BinaryFILE; N >= 1 repeats of passes (a) and (b).\n"
                 "  Pin to ONE core (taskset -c N): cycle-counter timings need a single CPU.\n",
                 argv[0]);
    return 2;
  }

  const Mapping map = map_file(path);
  std::printf("file: %.1f MiB mapped read-only (MAP_PRIVATE, MADV_SEQUENTIAL), consumed in %zu MiB "
              "windows, each pre-faulted before and released after its timed span\n",
              static_cast<double>(map.size) / (1024.0 * 1024.0), kWindow >> 20);
  double peak_anon = rss_anon_mib();

  // (a) parse only
  std::printf("\n(a) parse only (checksum of every decoded field)\n");
  std::vector<PassResult> parse_runs;
  std::uint64_t checksum = 0;
  itch::ReadStats parse_stats;
  for (long k = 0; k < repeat; ++k) {
    ChecksumVisitor visitor;
    itch::ReadStats stats;
    const PassResult r = run_pass(map, visitor, stats, peak_anon);
    if (stats.messages == 0) {
      std::fprintf(stderr, "bench_itch: no complete ITCH frame in %s\n", path);
      return 1;
    }
    parse_runs.push_back(r);
    print_run(static_cast<std::size_t>(k + 1), stats.messages, r);
    checksum = visitor.sum;
    parse_stats = stats;
  }
  print_median(parse_stats.messages, parse_runs);
  std::printf("  checksum=%016llx\n", static_cast<ull>(checksum));

  // (b) parse plus the full order book
  std::printf("\n(b) parse plus full order book\n");
  std::vector<PassResult> book_runs;
  ItchStats book_stats;
  for (long k = 0; k < repeat; ++k) {
    auto builder = std::make_unique<ItchBookBuilder>();
    const PassResult r = run_pass(map, *builder, builder->read_stats(), peak_anon);
    book_stats = builder->stats();
    book_stats.truncated_bytes = r.left;
    book_runs.push_back(r);
    print_run(static_cast<std::size_t>(k + 1), book_stats.read.messages, r);
  }
  print_median(book_stats.read.messages, book_runs);

  // (c) per-message latency of dispatch() into the builder
  const double ticks_per_ns = calibrate_ticks_per_ns();
  const std::uint64_t overhead = calibrate_overhead_ticks();
  const double ns_per_tick = 1.0 / ticks_per_ns;
  std::printf("\n(c) per-message latency: one dispatch() call into the builder\n");
  std::printf("  timer: %s, %.4f ticks/ns (%.3f GHz), overhead %llu ticks = %.1f ns "
              "(median of 1,000,000 empty regions, subtracted, clamped at 0)\n",
              kTsc ? "rdtscp after lfence" : "steady_clock", ticks_per_ns, ticks_per_ns,
              static_cast<ull>(overhead), static_cast<double>(overhead) * ns_per_tick);
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
  Pager pager(map);
  const Faults pass_faults0 = faults_now();
  volatile std::uint8_t body_sink = 0;
  unsigned cpu_begin = 0, cpu_end = 0, last_cpu = ~0u;
  std::uint64_t cpu_changes = 0;
  std::size_t pos = 0;
  while (pos + 2 <= map.size && !rs.end_of_session) {
    if (pos + kLead > pager.touched() && pager.touched() < map.size) {
      pager.prepare(pos, kWindow);  // page-in between messages, never inside a timed region
      pager.release_before(pos);
      peak_anon = std::max(peak_anon, rss_anon_mib());
    }
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
    body_sink = body[length - 1];  // the whole body is resident before the clock starts
    ++rs.messages;
    ++rs.by_type[body[0]];
    const std::uint64_t t0 = tick_begin(cpu_begin);
    const itch::FrameResult result = itch::dispatch(body, length, *builder);
    const std::uint64_t t1 = tick_end(cpu_end);
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
    if (cpu_begin != last_cpu || cpu_end != cpu_begin) ++cpu_changes;
    last_cpu = cpu_end;
    const std::uint64_t raw = t1 - t0;
    const std::uint64_t ticks = raw > overhead ? raw - overhead : 0;
    latency->all.record(ticks);
    if (Histogram* h = route[body[0]]) h->record(ticks);
    pos += 2 + length;
    if ((rs.messages & 0xfffff) == 0) peak_anon = std::max(peak_anon, rss_anon_mib());
  }
  (void)body_sink;
  pager.release_before(map.size);
  peak_anon = std::max(peak_anon, rss_anon_mib());
  const Faults pass_faults = faults_now() - pass_faults0;
  ItchStats latency_stats = builder->stats();
  latency_stats.truncated_bytes = pos < map.size ? map.size - pos : 0;
  print_latency("all", latency->all, ns_per_tick);
  for (std::size_t i = 0; i < Latency::kTypes.size(); ++i) {
    const char name[2] = {Latency::kTypes[i], 0};
    print_latency(name, latency->by_type[i], ns_per_tick);
  }
  // The first change is the first reading; more means the process moved between CPUs.
  std::printf("  CPU-id changes seen by rdtscp: %llu (1 = pinned to one core; more means the "
              "timings mix cores)\n",
              static_cast<ull>(cpu_changes));
  std::printf("  page faults, whole pass: major=%ld minor=%ld; of which input page-in (outside "
              "timed regions): major=%ld minor=%ld. Remaining faults are the book's and "
              "histograms' own memory growth; remaining MAJOR faults would mean input pages were "
              "evicted after page-in\n",
              pass_faults.major, pass_faults.minor, pager.paged().major, pager.paged().minor);
  const std::string verdict = builder->book().validate();
  std::printf("  book validate after pass (c): %s\n", verdict.empty() ? "ok" : verdict.c_str());

  // Cross-checks: every counter of the hand-driven walk of pass (c) must equal pass (b),
  // and pass (a)'s frame counters must equal pass (b)'s.
  const bool consistent = same_stats(latency_stats, book_stats);
  std::printf("  pass (c) counters match pass (b): %s\n", consistent ? "yes" : "NO");
  const bool parse_consistent = same_read(parse_stats, book_stats.read);
  std::printf("  pass (a) frame counters match pass (b): %s\n", parse_consistent ? "yes" : "NO");

  // Totals from the full-book run (pass (b)).
  const ItchStats& s = book_stats;
  std::printf("\nbook after the file (pass (b))\n");
  std::printf("  messages=%llu  unknown types=%llu  malformed=%llu  trailing bytes=%llu  "
              "truncated bytes=%llu  end-of-session marker=%s\n",
              static_cast<ull>(s.read.messages), static_cast<ull>(s.read.unknown_types),
              static_cast<ull>(s.read.malformed), static_cast<ull>(s.read.trailing_bytes),
              static_cast<ull>(s.truncated_bytes), s.read.end_of_session ? "yes" : "no");
  std::printf("  unknown refs=%llu  duplicate refs=%llu  overfills=%llu  invalid=%llu  "
              "crossed=%llu  filtered=%llu\n",
              static_cast<ull>(s.unknown_refs), static_cast<ull>(s.duplicate_refs),
              static_cast<ull>(s.overfills), static_cast<ull>(s.invalid),
              static_cast<ull>(s.crossed), static_cast<ull>(s.filtered));
  std::printf("  live orders at the end=%zu  peak live orders=%zu  levels at the end=%zu  "
              "peak levels=%zu\n",
              s.live_orders, s.peak_live_orders, s.levels, s.peak_levels);
  std::printf("  peak RSS=%.0f MiB (ru_maxrss; input windows are released after use, so this is "
              "about one window plus the books)  peak anonymous RSS=%.0f MiB (sampled outside "
              "timed spans; the books and buffers)\n",
              peak_rss_mib(), peak_anon);
  std::printf("  parse-only checksum=%016llx\n", static_cast<ull>(checksum));
  return consistent && parse_consistent && verdict.empty() ? 0 : 1;
}
