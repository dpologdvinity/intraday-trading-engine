// End-to-end check of the hftcore_itch_replay CLI (registered with ctest only when both
// tests and benchmarks are built). Writes a tiny synthetic BinaryFILE, runs the replay
// binary (given as argv[1]) on the file and on stdin, and checks the aggregate lines:
// message count, end-of-session flag, one stock's trades, volume and prices, validation
// runs, and that stdin gives the same output as the file. No real exchange data is used.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <string_view>
#include <vector>

namespace {

struct Bytes {
  std::vector<std::uint8_t> v;
  Bytes& ch(char c) {
    v.push_back(static_cast<std::uint8_t>(c));
    return *this;
  }
  Bytes& be(std::uint64_t x, int n) {
    for (int i = n - 1; i >= 0; --i) v.push_back(static_cast<std::uint8_t>(x >> (8 * i)));
    return *this;
  }
  Bytes& text(std::string_view s, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) ch(i < s.size() ? s[i] : ' ');
    return *this;
  }
  // The frame: 2-byte length prefix, then the message body.
  void frame_into(std::vector<std::uint8_t>& out) const {
    out.push_back(static_cast<std::uint8_t>(v.size() >> 8));
    out.push_back(static_cast<std::uint8_t>(v.size()));
    out.insert(out.end(), v.begin(), v.end());
  }
};

Bytes header(char type, std::uint16_t locate) {
  Bytes b;
  b.ch(type).be(locate, 2).be(0, 2).be(1, 6);
  return b;
}

Bytes directory(std::uint16_t locate, std::string_view symbol) {
  Bytes b = header('R', locate);
  b.text(symbol, 8).ch('Q').ch('N').be(100, 4).ch('N').ch('C').text("", 2).ch('P').ch('N').ch('N');
  b.ch('1').ch('N').be(0, 4).ch('N');
  return b;
}

Bytes add(std::uint16_t locate, std::uint64_t ref, char side, std::uint32_t shares,
          std::string_view symbol, std::uint32_t price) {
  Bytes b = header('A', locate);
  b.be(ref, 8).ch(side).be(shares, 4).text(symbol, 8).be(price, 4);
  return b;
}

std::string run(const std::string& command) {
  std::string out;
  std::FILE* pipe = popen(command.c_str(), "r");
  if (!pipe) return out;
  char buf[4096];
  std::size_t got;
  while ((got = std::fread(buf, 1, sizeof buf, pipe)) > 0) out.append(buf, got);
  pclose(pipe);
  return out;
}

int failures = 0;

void expect(const std::string& output, const std::string& line) {
  if (output.find(line) == std::string::npos) {
    std::fprintf(stderr, "missing in output: %s\n", line.c_str());
    ++failures;
  }
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 3) {
    std::fprintf(stderr, "usage: %s REPLAY_BINARY TEMP_FILE\n", argv[0]);
    return 2;
  }
  const std::string replay = argv[1], file = argv[2];

  std::vector<std::uint8_t> bytes;
  directory(1, "AAPL").frame_into(bytes);
  directory(2, "MSFT").frame_into(bytes);
  add(1, 1, 'B', 100, "AAPL", 1000000).frame_into(bytes);  // $100.0000
  add(1, 2, 'S', 50, "AAPL", 1000100).frame_into(bytes);
  add(2, 3, 'B', 10, "MSFT", 2000000).frame_into(bytes);
  {
    Bytes e = header('E', 1);  // 40 shares of order 1 trade at its price
    e.be(1, 8).be(40, 4).be(77, 8);
    e.frame_into(bytes);
    Bytes p = header('P', 1);  // 25 shares off-book at $100.0050
    p.be(0, 8).ch('B').be(25, 4).text("AAPL", 8).be(1000050, 4).be(78, 8);
    p.frame_into(bytes);
    Bytes q = header('Q', 1);  // closing cross, 1000 shares at $100.0040
    q.be(1000, 8).text("AAPL", 8).be(1000040, 4).be(79, 8).ch('C');
    q.frame_into(bytes);
    Bytes d = header('D', 2);  // MSFT's only order disappears
    d.be(3, 8);
    d.frame_into(bytes);
  }
  bytes.push_back(0);  // end-of-session marker
  bytes.push_back(0);

  std::FILE* f = std::fopen(file.c_str(), "wb");
  if (!f || std::fwrite(bytes.data(), 1, bytes.size(), f) != bytes.size()) {
    std::fprintf(stderr, "cannot write %s\n", file.c_str());
    return 2;
  }
  std::fclose(f);

  const std::string args = " --symbols AAPL,MSFT --validate-every 1 ";
  const std::string from_file = run(replay + args + file);
  const std::string from_stdin = run("cat " + file + " | " + replay + args + "-");

  expect(from_file, "messages: 9\n");
  expect(from_file, "by type: A=3 D=1 E=1 P=1 Q=1 R=2\n");
  expect(from_file, "end of session marker: yes\n");
  expect(from_file, "malformed: 0\n");
  expect(from_file, "live orders at the end: 2\n");
  expect(from_file, "validations run: 2\n");  // after the (single) chunk and at the end
  expect(from_file, "AAPL: trades=3 volume=1065 open cross=- close cross=$100.0040 | ");
  expect(from_file, "best bid=$100.0000 best ask=$100.0100 mid=$100.00500 (exact)");
  expect(from_file, "MSFT: trades=0 volume=0 open cross=- close cross=- | book mid at end of input");
  expect(from_file, "note: --symbols is set");
  if (from_file.empty() || from_file != from_stdin) {
    std::fprintf(stderr, "stdin output differs from file output\n");
    ++failures;
  }
  std::remove(file.c_str());
  if (failures) {
    std::fprintf(stderr, "--- output ---\n%s", from_file.c_str());
    return 1;
  }
  return 0;
}
