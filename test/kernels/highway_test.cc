#include "test/test.h"
#include "aligned.h"
#include "kernels.h"

#include <numeric>

namespace intgemm {

template <CPUType CPUType_, typename ElemType_>
void kernel_highway_test() {
  if (kCPU < CPUType_)
    return;

  using vec_t = vector_t<CPUType_, ElemType_>;
  constexpr static auto VECTOR_LENGTH = sizeof(vec_t) / sizeof(ElemType_);

  AlignedVector<ElemType_> input1(VECTOR_LENGTH);
  AlignedVector<ElemType_> input2(VECTOR_LENGTH);
  AlignedVector<ElemType_> weight(VECTOR_LENGTH);
  AlignedVector<ElemType_> output(VECTOR_LENGTH);

  std::iota(input1.begin(), input1.end(), 0);
  std::iota(input2.begin(), input2.end(), 100);
  std::fill(weight.begin(), weight.end(), 0.1);

  *output.template as<vec_t>() = kernels::highway(*input1.template as<vec_t>(), *input2.template as<vec_t>(), *weight.template as<vec_t>());
  for (auto i = 0; i < output.size(); ++i)
    CHECK_EPS(output[i], input1[i] * weight[0] + input2[i] * (1 - weight[0]), 0.00001);
}

template INTGEMM_SSE2 void kernel_highway_test<CPUType::SSE2, float>();
template INTGEMM_SSE2 void kernel_highway_test<CPUType::SSE2, double>();
KERNEL_TEST_CASE("highway/float SSE2") { return kernel_highway_test<CPUType::SSE2, float>(); }
KERNEL_TEST_CASE("highway/double SSE2") { return kernel_highway_test<CPUType::SSE2, double>(); }

template INTGEMM_AVX2 void kernel_highway_test<CPUType::AVX2, float>();
template INTGEMM_AVX2 void kernel_highway_test<CPUType::AVX2, double>();
KERNEL_TEST_CASE("highway/float AVX2") { return kernel_highway_test<CPUType::AVX2, float>(); }
KERNEL_TEST_CASE("highway/double AVX2") { return kernel_highway_test<CPUType::AVX2, double>(); }

#ifndef INTGEMM_NO_AVX512
template INTGEMM_AVX512BW void kernel_highway_test<CPUType::AVX512BW, float>();
template INTGEMM_AVX512BW void kernel_highway_test<CPUType::AVX512BW, double>();
KERNEL_TEST_CASE("highway/float AVX512BW") { return kernel_highway_test<CPUType::AVX512BW, float>(); }
KERNEL_TEST_CASE("highway/double AVX512BW") { return kernel_highway_test<CPUType::AVX512BW, double>(); }
#endif

}
