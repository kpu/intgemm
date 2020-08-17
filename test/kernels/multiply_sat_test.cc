#include "../test.h"
#include "../../aligned.h"
#include "../../kernels.h"

#include <cstdint>
#include <cstddef>
#include <numeric>

namespace intgemm {

template <CPUType CPUType_, typename Type_>
void kernel_multiply_sat_test() {
  if (kCPU < CPUType_)
    return;

  using vec_t = vector_t<CPUType_, Type_>;
  constexpr int VECTOR_LENGTH = sizeof(vec_t) / sizeof(Type_);

  AlignedVector<Type_> input1(VECTOR_LENGTH);
  AlignedVector<Type_> input2(VECTOR_LENGTH);
  AlignedVector<Type_> output(VECTOR_LENGTH);

  std::iota(input1.begin(), input1.end(), static_cast<Type_>(-VECTOR_LENGTH / 2));
  std::iota(input2.begin(), input2.end(), static_cast<Type_>(-VECTOR_LENGTH / 3));

  // TODO: try all shifts.  The shift must be an immediate.
  int8_t shift = 1;
  *output.template as<vec_t>() = kernels::multiply_sat<Type_>(*input1.template as<vec_t>(), *input2.template as<vec_t>(), shift);
  for (std::size_t i = 0; i < output.size(); ++i) {
    auto ref = (int64_t(input1[i]) * input2[i]) >> shift;
    auto ref_sat = Type_(std::min<int64_t>(std::numeric_limits<Type_>::max(), std::max<int64_t>(std::numeric_limits<Type_>::min(), ref)));
    CHECK(output[i] == ref_sat);
  }
}

template INTGEMM_SSE2 void kernel_multiply_sat_test<CPUType::SSE2, int8_t>();
template INTGEMM_SSE2 void kernel_multiply_sat_test<CPUType::SSE2, int16_t>();
KERNEL_TEST_CASE("multiply_sat/int8 SSE2") { return kernel_multiply_sat_test<CPUType::SSE2, int8_t>(); }
KERNEL_TEST_CASE("multiply_sat/int16 SSE2") { return kernel_multiply_sat_test<CPUType::SSE2, int16_t>(); }

template INTGEMM_AVX2 void kernel_multiply_sat_test<CPUType::AVX2, int8_t>();
template INTGEMM_AVX2 void kernel_multiply_sat_test<CPUType::AVX2, int16_t>();
KERNEL_TEST_CASE("multiply_sat/int8 AVX2") { return kernel_multiply_sat_test<CPUType::AVX2, int8_t>(); }
KERNEL_TEST_CASE("multiply_sat/int16 AVX2") { return kernel_multiply_sat_test<CPUType::AVX2, int16_t>(); }

#ifdef INTGEMM_COMPILER_SUPPORTS_AVX512BW
template INTGEMM_AVX512BW void kernel_multiply_sat_test<CPUType::AVX512BW, int8_t>();
template INTGEMM_AVX512BW void kernel_multiply_sat_test<CPUType::AVX512BW, int16_t>();
KERNEL_TEST_CASE("multiply_sat/int8 AVX512BW") { return kernel_multiply_sat_test<CPUType::AVX512BW, int8_t>(); }
KERNEL_TEST_CASE("multiply_sat/int16 AVX512BW") { return kernel_multiply_sat_test<CPUType::AVX512BW, int16_t>(); }
#endif

}
