#include "../tile/access.h"
#include "../tile/dot.h"

#include "test.h"
#include "../aligned.h"

namespace intgemm {
namespace AVX2 {
namespace {

INTGEMM_AVX2 void OneIteration() {
  AlignedVector<int8_t> A(1 * sizeof(Register));
  AlignedVector<int8_t> B(sizeof(Register) * 1);
  AlignedVector<int32_t> C(sizeof(Register) / sizeof(int32_t)/* Raw sums */);

  memset(C.begin(), 0, sizeof(Register));

  std::iota(A.begin(), A.end(), 7 /* made up */);
  std::iota(B.begin(), B.end(), 1 /* made up */);

  typedef RowMajorAccess<int8_t> InputA;
  typedef ColMajorAccess<int8_t> InputB;
  typedef RegisterRowMajorAccess Output;
  Access<InputA, InputB, Output> access(
      InputA(A.begin(), sizeof(Register)),
      InputB(B.begin(), sizeof(Register)),
      Output(reinterpret_cast<Register*>(C.begin()), 1));
  MatrixTile<1, 1, AVX2::Shifted8>::Run(access);

  const std::size_t kStride = sizeof(int32_t) / sizeof(int8_t);
  for (std::size_t i = 0; i < sizeof(Register) / sizeof(int32_t); ++i) {
    int32_t sum = 0;
    for (std::size_t j = i * kStride; j < (i+1) * kStride; ++j) {
      sum += static_cast<int32_t>(A[j]) * static_cast<int32_t>(B[j]);
    }
    CHECK(C[i] == sum);
  }
}

TEST_CASE("Basic Tile", "[tile]") {
  if (kCPU >= CPUType::AVX2) {
    OneIteration();
  }
}

} // namespace
} // namespace AVX2
} // namespace intgemm
