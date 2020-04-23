#include "../aligned.h"
#include "../stop_watch.h"
#include "../test/test_matrices.h"
#include "../tile/multiply.h"
#include "../tile/dot.h"

#include <chrono>
#include <iomanip>
#include <random>
#include <vector>

namespace intgemm {
namespace {

typedef TestMatrices8::AccessT Accessor;

template <Index A_rows, Index B_cols> static inline void BenchmarkOne(Accessor access, Tile shape) {
  const std::size_t kTries = 4;
  auto start = std::chrono::steady_clock::now();
  // Burn in.
  AVX512VNNI::Multiply<Accessor, AVX512VNNI::Shifted8, A_rows, B_cols>(access, shape);
  for (std::size_t t = 0; t < kTries; ++t) {
    // TODO: try various multipliers, guard against old compilers, etc.
    AVX512VNNI::Multiply<Accessor, AVX512VNNI::Shifted8, A_rows, B_cols>(access, shape);
  }
  auto end = std::chrono::steady_clock::now();
  double took = std::chrono::duration<double>(end - start).count() / kTries;
  std::cout << std::setw(8) << std::setprecision(4) << took << ' ' << std::setw(2) << A_rows << 'x' << std::setw(2) << B_cols << std::endl;
}

template <std::size_t... Iterator> static inline void BenchmarkKernels(Tile shape, index_sequence<Iterator...>) {
  constexpr Index ColsMax = 16;
  TestMatrices8 matrices(shape);
  using unfurl = int[];
  (void)unfurl{0, (
    BenchmarkOne<(Iterator / ColsMax) + 1, (Iterator % ColsMax) + 1>(matrices.Accessor(), shape)
  , 0)...};
}

} // namespace
} // namespace intgemm

int main() {
  intgemm::BenchmarkKernels({1024, 1024, 1024}, intgemm::make_index_sequence<16*16>());
}
