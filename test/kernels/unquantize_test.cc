#include "test/test.h"
#include "aligned.h"
#include "kernels.h"

#include <numeric>

namespace intgemm {

template <CPUType CPUType_>
void kernel_unquantize_test() {
  if (kCPU < CPUType_)
    return;

  using input_vec_t = vector_t<CPUType_, int>;
  using output_vec_t = vector_t<CPUType_, float>;

  AlignedVector<int> input(sizeof(input_vec_t) / sizeof(int));
  AlignedVector<float> output(sizeof(output_vec_t) / sizeof(float));

  std::iota(input.begin(), input.end(), 0);
  auto unquant_mult = set1_ps<output_vec_t>(0.5f);

  *output.template as<output_vec_t>() = kernels::unquantize(*input.template as<input_vec_t>(), unquant_mult);
  for (auto i = 0; i < output.size(); ++i)
    CHECK(output[i] == i * 0.5f);
}

template INTGEMM_SSE2 void kernel_unquantize_test<CPUType::SSE2>();
TEST_CASE("Kernel: unquantize SSE2",) { return kernel_unquantize_test<CPUType::SSE2>(); }

template INTGEMM_AVX2 void kernel_unquantize_test<CPUType::AVX2>();
TEST_CASE("Kernel: unquantize AVX2",) { return kernel_unquantize_test<CPUType::AVX2>(); }

#ifndef INTGEMM_NO_AVX512
template INTGEMM_AVX512BW void kernel_unquantize_test<CPUType::AVX512BW>();
TEST_CASE("Kernel: unquantize AVX512BW",) { return kernel_unquantize_test<CPUType::AVX512BW>(); }
#endif

}