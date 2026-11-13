// Non-template helpers for the ITCH 5.0 parser (see itch.hpp).
#include "hftcore/itch.hpp"

namespace hftcore::itch {

std::string_view trim(const Stock& stock) {
  std::size_t n = stock.size();
  while (n > 0 && stock[n - 1] == ' ') --n;
  return std::string_view(stock.data(), n);
}

}  // namespace hftcore::itch
