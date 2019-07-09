#include "test/test.h"
#include "aligned.h"
#include "kernels.h"

#include <numeric>

namespace intgemm {

template <CPUType CPUType_, typename ElemType_>
void kernel_add_bias_test() {
  if (kCPU < CPUType_)
    return;

  using vec_t = vector_t<CPUType_, ElemType_>;
  constexpr static auto VECTOR_LENGTH = sizeof(vec_t) / sizeof(ElemType_);

  AlignedVector<ElemType_> input(VECTOR_LENGTH);
  AlignedVector<ElemType_> bias(VECTOR_LENGTH);
  AlignedVector<ElemType_> output(VECTOR_LENGTH);

  std::iota(input.begin(), input.end(), 0);
  std::fill(bias.begin(), bias.end(), 100);

  *output.template as<vec_t>() = kernels::add_bias(*input.template as<vec_t>(), bias.begin(), 0);
  for (auto i = 0; i < output.size(); ++i)
    CHECK(output[i] == 100 + i);
}

template INTGEMM_SSE2 void kernel_add_bias_test<CPUType::SSE2, float>();
KERNEL_TEST_CASE("add_bias/float SSE2") { return kernel_add_bias_test<CPUType::SSE2, float>(); }

template INTGEMM_AVX2 void kernel_add_bias_test<CPUType::AVX2, float>();
KERNEL_TEST_CASE("add_bias/float AVX2") { return kernel_add_bias_test<CPUType::AVX2, float>(); }

#ifndef INTGEMM_NO_AVX512
template INTGEMM_AVX512BW void kernel_add_bias_test<CPUType::AVX512BW, float>();
KERNEL_TEST_CASE("add_bias/float AVX512BW") { return kernel_add_bias_test<CPUType::AVX512BW, float>(); }
#endif

}
