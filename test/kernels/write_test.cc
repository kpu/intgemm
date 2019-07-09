#include "test/test.h"
#include "aligned.h"
#include "kernels.h"

#include <numeric>

namespace intgemm {

template <CPUType CPUType_, typename ElemType_>
void kernel_write_test() {
  if (kCPU < CPUType_)
    return;

  using vec_t = vector_t<CPUType_, ElemType_>;
  constexpr static auto VECTOR_LENGTH = sizeof(vec_t) / sizeof(ElemType_);

  AlignedVector<ElemType_> input(VECTOR_LENGTH);
  AlignedVector<ElemType_> output(VECTOR_LENGTH);

  std::iota(input.begin(), input.end(), 0);

  kernels::write(*input.template as<vec_t>(), output.begin(), 0);
  for (auto i = 0; i < VECTOR_LENGTH; ++i)
    CHECK(output[i] == i);
}

template INTGEMM_SSE2 void kernel_write_test<CPUType::SSE2, int>();
template INTGEMM_SSE2 void kernel_write_test<CPUType::SSE2, float>();
template INTGEMM_SSE2 void kernel_write_test<CPUType::SSE2, double>();
TEST_CASE("Kernel: write/int SSE2",) { return kernel_write_test<CPUType::SSE2, int>(); }
TEST_CASE("Kernel: write/float SSE2",) { return kernel_write_test<CPUType::SSE2, float>(); }
TEST_CASE("Kernel: write/double SSE2",) { return kernel_write_test<CPUType::SSE2, double>(); }

template INTGEMM_AVX2 void kernel_write_test<CPUType::AVX2, int>();
template INTGEMM_AVX2 void kernel_write_test<CPUType::AVX2, float>();
template INTGEMM_AVX2 void kernel_write_test<CPUType::AVX2, double>();
TEST_CASE("Kernel: write/int AVX2",) { return kernel_write_test<CPUType::AVX2, int>(); }
TEST_CASE("Kernel: write/float AVX2",) { return kernel_write_test<CPUType::AVX2, float>(); }
TEST_CASE("Kernel: write/double AVX2",) { return kernel_write_test<CPUType::AVX2, double>(); }

#ifndef INTGEMM_NO_AVX512
template INTGEMM_AVX512BW void kernel_write_test<CPUType::AVX512BW, int>();
template INTGEMM_AVX512BW void kernel_write_test<CPUType::AVX512BW, float>();
template INTGEMM_AVX512BW void kernel_write_test<CPUType::AVX512BW, double>();
TEST_CASE("Kernel: write/int AVX512BW",) { return kernel_write_test<CPUType::AVX512BW, int>(); }
TEST_CASE("Kernel: write/float AVX512BW",) { return kernel_write_test<CPUType::AVX512BW, float>(); }
TEST_CASE("Kernel: write/double AVX512BW",) { return kernel_write_test<CPUType::AVX512BW, double>(); }
#endif

}
