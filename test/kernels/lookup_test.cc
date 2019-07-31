#include "test/test.h"
#include "aligned.h"
#include "kernels.h"

#include <numeric>

namespace intgemm {

template <CPUType CPUType_>
void kernel_lookup_uint8_test() {
  if (kCPU < CPUType_)
    return;

  using vec_t = vector_t<CPUType_, int>;
  constexpr static auto VECTOR_LENGTH = sizeof(vec_t) / sizeof(int8_t);

  AlignedVector<uint8_t> input(VECTOR_LENGTH);
  int8_t lut[256];
  AlignedVector<int8_t> output(VECTOR_LENGTH);

  std::iota(input.begin(), input.end(), 0);
  std::iota(lut, lut + 256, -5);

  *output.template as<vec_t>() = kernels::lookup_8b(*input.template as<vec_t>(), lut);
  for (auto i = 0; i < output.size(); ++i)
    CHECK(output[i] == lut[input[i]]);
}

template INTGEMM_SSE2 void kernel_lookup_uint8_test<CPUType::SSE2>();
KERNEL_TEST_CASE("lookup/int8 SSE2") { return kernel_lookup_uint8_test<CPUType::SSE2>(); }

template INTGEMM_AVX2 void kernel_lookup_uint8_test<CPUType::AVX2>();
KERNEL_TEST_CASE("lookup/int8 AVX2") { return kernel_lookup_uint8_test<CPUType::AVX2>(); }

#ifdef INTGEMM_COMPILER_SUPPORTS_AVX512
template INTGEMM_AVX512BW void kernel_lookup_uint8_test<CPUType::AVX512BW>();
KERNEL_TEST_CASE("lookup/int8 AVX512BW") { return kernel_lookup_uint8_test<CPUType::AVX512BW>(); }
#endif

}
