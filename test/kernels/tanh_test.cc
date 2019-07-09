#include "test/test.h"
#include "aligned.h"
#include "kernels.h"

#include <numeric>

namespace intgemm {

template <CPUType CPUType_>
void kernel_tanh_test() {
  if (kCPU < CPUType_)
    return;

  using vec_t = vector_t<CPUType_, float>;
  constexpr static int VECTOR_LENGTH = sizeof(vec_t) / sizeof(float);

  AlignedVector<float> input(VECTOR_LENGTH);
  AlignedVector<float> output(VECTOR_LENGTH);

  std::generate(input.begin(), input.end(), [] () { static int n = -4; return n++ / 4.f; });

  *output.template as<vec_t>() = kernels::tanh(*input.template as<vec_t>());
  for (auto i = 0; i < output.size(); ++i)
    CHECK_EPS(output[i], tanh(input[i]), 0.001f);
}

template INTGEMM_AVX2 void kernel_tanh_test<CPUType::AVX2>();
KERNEL_TEST_CASE("tanh AVX2") { return kernel_tanh_test<CPUType::AVX2>(); }

}
