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

template <typename Backend, Index TileColumns>
void PrepareBRef(const float* input, typename Backend::Integer* output, float quant_mult, Index B_rows, Index B_cols) {
  using vec_t = intgemm::vector_t<Backend::kUses, typename Backend::Integer>;
  constexpr Index vec_len = sizeof(vec_t) / sizeof(typename Backend::Integer);

  for (Index c = 0; c < B_cols; c += TileColumns)
    for (Index r = 0; r < B_rows; r += vec_len)
      for (Index ci = 0; ci < TileColumns; ++ci)
        for (Index ri = 0; ri < vec_len; ++ri) {
          *output++ = input[(r + ri) * B_cols + (c + ci)] * quant_mult;
        }
}

template <typename Backend, Index TileColumns>
bool TestInner(const AlignedVector<float>& input, Index B_rows, Index B_cols, float quant_mult) {
  bool success = true;

  AlignedVector<typename Backend::Integer> output(input.size());
  Backend::template PrepareB<TileColumns>(input.begin(), output.begin(), quant_mult, B_rows, B_cols);

  AlignedVector<typename Backend::Integer> reference(input.size());
  PrepareBRef<Backend, TileColumns>(input.begin(), reference.begin(), quant_mult, B_rows, B_cols);

  if (TileColumns != 7 && TileColumns != 8) {
    std::cout << "Input:" << std::endl;
    for (int i = 0; i < B_rows; ++i) {
      for (int j = 0; j < B_cols; ++j)
        std::cout << input[i * B_cols + j] << ", ";
      std::cout << std::endl;
    }
    
    std::cout << "Output:" << std::endl;
    for (int i = 0; i < B_rows; ++i) {
      for (int j = 0; j < B_cols; ++j)
        std::cout << output[i * B_cols + j] << ", ";
      std::cout << std::endl;
    }

    std::cout << "Reference:" << std::endl;
    for (int i = 0; i < B_rows; ++i) {
      for (int j = 0; j < B_cols; ++j)
        std::cout << reference[i * B_cols + j] << ", ";
      std::cout << std::endl;
    }
  }

  for (std::size_t i = 0; i < output.size(); ++i) {
    if (output[i] != reference[i]) {
      UNSCOPED_INFO("Error at " << i << ", output = " << int(output[i]) << ", reference = " << int(reference[i]));
      success = false;
      break;
    }
  }
  return success;
}

template <typename Backend, Index TileColumns>
bool Test(Index B_rows, Index B_cols, float quant_mult) {
  AlignedVector<float> input(B_rows * B_cols);

  std::generate(input.begin(), input.end(), []() {
    static constexpr int divider = sizeof(intgemm::vector_t<Backend::kUses, typename Backend::Integer>) / sizeof(typename Backend::Integer);
    static int value = 0;
    return (value++) % divider;
  });

  return TestInner<Backend, TileColumns>(input, B_rows, B_cols, quant_mult);
}

TEST_CASE("PrepareB SSE2", "") {
  if (kCPU < CPUType::SSE2)
    return;

  // CHECK(Test<SSE2_16bit, 1>(32, 32, 2.0f));
  // CHECK(Test<SSE2_16bit, 2>(32, 2*16, 2.0f));
  // CHECK(Test<SSE2_16bit, 3>(32, 3*16, 2.0f));
  // CHECK(Test<SSE2_16bit, 4>(32, 4*8, 2.0f));
  // CHECK(Test<SSE2_16bit, 5>(32, 5*8, 2.0f));
  // CHECK(Test<SSE2_16bit, 6>(32, 6*8, 2.0f));
  CHECK(Test<SSE2_16bit, 7>(32, 7*8, 2.0f));
  CHECK(Test<SSE2_16bit, 8>(32, 32, 2.0f));
  // CHECK(Test<SSE2_16bit, 9>(32, 9*16, 2.0f));
}

TEST_CASE("PrepareB SSSE3", "") {
  if (kCPU < CPUType::SSSE3)
    return;

  // CHECK(Test<SSSE3_8bit, 1>(32, 1*128, 2.0f));
  // CHECK(Test<SSSE3_8bit, 2>(32, 2*128, 2.0f));
  // CHECK(Test<SSSE3_8bit, 3>(32, 3*128, 2.0f));
  // CHECK(Test<SSSE3_8bit, 4>(32, 4*128, 2.0f));
  // CHECK(Test<SSSE3_8bit, 5>(32, 5*128, 2.0f));
  // CHECK(Test<SSSE3_8bit, 6>(32, 6*128, 2.0f));
  // CHECK(Test<SSSE3_8bit, 7>(32, 7*128, 2.0f));
  // CHECK(Test<SSSE3_8bit, 8>(32, 8*128, 2.0f));
  // CHECK(Test<SSSE3_8bit, 9>(32, 9*128, 2.0f));
}

TEST_CASE("PrepareB AVX2", "") {
  if (kCPU < CPUType::AVX2)
    return;

  // CHECK(Test<AVX2_8bit, 1>(32, 1*128, 2.0f));
  // CHECK(Test<AVX2_8bit, 2>(32, 2*128, 2.0f));
  // CHECK(Test<AVX2_8bit, 3>(32, 3*128, 2.0f));
  // CHECK(Test<AVX2_8bit, 4>(32, 4*128, 2.0f));
  // CHECK(Test<AVX2_8bit, 5>(32, 5*128, 2.0f));
  // CHECK(Test<AVX2_8bit, 6>(32, 6*128, 2.0f));
  // CHECK(Test<AVX2_8bit, 7>(32, 7*128, 2.0f));
  // CHECK(Test<AVX2_8bit, 8>(32, 8*128, 2.0f));
  // CHECK(Test<AVX2_8bit, 9>(32, 9*128, 2.0f));

  // CHECK(Test<AVX2_16bit, 1>(32, 1*128, 2.0f));
  // CHECK(Test<AVX2_16bit, 2>(32, 2*128, 2.0f));
  // CHECK(Test<AVX2_16bit, 3>(32, 3*128, 2.0f));
  // CHECK(Test<AVX2_16bit, 4>(32, 4*128, 2.0f));
  // CHECK(Test<AVX2_16bit, 5>(32, 5*128, 2.0f));
  // CHECK(Test<AVX2_16bit, 6>(32, 6*128, 2.0f));
  // CHECK(Test<AVX2_16bit, 7>(32, 7*128, 2.0f));
  // CHECK(Test<AVX2_16bit, 8>(32, 8*128, 2.0f));
  // CHECK(Test<AVX2_16bit, 9>(32, 9*128, 2.0f));
}

#ifdef INTGEMM_COMPILER_SUPPORTS_AVX512
  TEST_CASE("PrepareB AVX512", "") {
    if (kCPU < CPUType::AVX512BW)
      return;

    // CHECK(Test<AVX512_8bit, 1>(32, 1*128, 2.0f));
    // CHECK(Test<AVX512_8bit, 2>(32, 2*128, 2.0f));
    // CHECK(Test<AVX512_8bit, 3>(32, 3*128, 2.0f));
    // CHECK(Test<AVX512_8bit, 4>(32, 4*128, 2.0f));
    // CHECK(Test<AVX512_8bit, 5>(32, 5*128, 2.0f));
    // CHECK(Test<AVX512_8bit, 6>(32, 6*128, 2.0f));
    // CHECK(Test<AVX512_8bit, 7>(32, 7*128, 2.0f));
    // CHECK(Test<AVX512_8bit, 8>(32, 8*128, 2.0f));
    // CHECK(Test<AVX512_8bit, 9>(32, 9*128, 2.0f));

    // CHECK(Test<AVX512_16bit, 1>(32, 1*128, 2.0f));
    // CHECK(Test<AVX512_16bit, 2>(32, 2*128, 2.0f));
    // CHECK(Test<AVX512_16bit, 3>(32, 3*128, 2.0f));
    // CHECK(Test<AVX512_16bit, 4>(32, 4*128, 2.0f));
    // CHECK(Test<AVX512_16bit, 5>(32, 5*128, 2.0f));
    // CHECK(Test<AVX512_16bit, 6>(32, 6*128, 2.0f));
    // CHECK(Test<AVX512_16bit, 7>(32, 7*128, 2.0f));
    // CHECK(Test<AVX512_16bit, 8>(32, 8*128, 2.0f));
    // CHECK(Test<AVX512_16bit, 9>(32, 9*128, 2.0f));
  }
#endif

}
}
