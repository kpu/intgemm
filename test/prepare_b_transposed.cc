#include "test.h"
#include "../aligned.h"
#include "../avx2_gemm.h"
#include "../avx512_gemm.h"
#include "../sse2_gemm.h"
#include "../ssse3_gemm.h"

#include <cstring>
#include <iostream>
#include <math.h>

namespace intgemm {
namespace {

template <typename Backend>
void PrepareBTransposedRef(const float* input, typename Backend::Integer* output, float quant_mult, Index rows, Index cols) {
  using vec_t = intgemm::vector_t<Backend::kUses, typename Backend::Integer>;
  constexpr Index vec_len = sizeof(vec_t) / sizeof(typename Backend::Integer);

  for (Index i = 0; i < rows * cols / 8; i += vec_len)
    for (Index j = 0; j < 8; ++j)
      for (Index k = 0; k < vec_len; ++k) {
        Index col = (i + k) % cols;
        Index row = 8 * ((i + k) / cols) + j;
        *output++ = input[row * cols + col] * quant_mult;
      }
}

template <typename Backend>
bool Test(const AlignedVector<float>& input, Index rows, Index cols, float quant_mult) {
  bool success = true;

  AlignedVector<typename Backend::Integer> output(input.size());
  Backend::PrepareBTransposed(input.begin(), output.begin(), quant_mult, cols, rows);

  AlignedVector<typename Backend::Integer> reference(input.size());
  PrepareBTransposedRef<Backend>(input.begin(), reference.begin(), quant_mult, rows, cols);

  for (std::size_t i = 0; i < output.size(); ++i) {
    if (output[i] != reference[i]) {
      UNSCOPED_INFO("Error at " << i << ", output = " << int(output[i]) << ", reference = " << int(reference[i]));
      success = false;
      break;
    }
  }
  return success;
}

template <typename Backend>
bool TestMany(Index rows, Index cols) {
  AlignedVector<float> input(rows * cols);
  const float quant_mult = 2.f;

  std::generate(input.begin(), input.end(), []() {
    static constexpr int divider = sizeof(intgemm::vector_t<Backend::kUses, typename Backend::Integer>) / sizeof(typename Backend::Integer);
    static int value = 0;
    return (value++) % divider;
  });

  return Test<Backend>(input, rows, cols, quant_mult);
}

TEST_CASE("PrepareBTransposed SSE2", "") {
  if (kCPU < CPUType::SSE2)
    return;

  CHECK(TestMany<SSE2_16bit>(128, 4));
}

TEST_CASE("PrepareBTransposed SSSE3", "") {
  if (kCPU < CPUType::SSSE3)
    return;

  CHECK(TestMany<SSSE3_8bit>(128, 4));
}

TEST_CASE("PrepareBTransposed AVX2", "") {
  if (kCPU < CPUType::AVX2)
    return;

  CHECK(TestMany<AVX2_8bit>(128, 8));
  CHECK(TestMany<AVX2_16bit>(128, 8));
}

#ifdef INTGEMM_COMPILER_SUPPORTS_AVX512
  TEST_CASE("PrepareBTransposed AVX512", "") {
    if (kCPU < CPUType::AVX512BW)
      return;

    CHECK(TestMany<AVX512_8bit>(128, 16));
    CHECK(TestMany<AVX512_16bit>(128, 16));
  }
#endif

}
}
