#include "test.h"
#include "../aligned.h"
#include "../avx2_gemm.h"
#include "../avx512_gemm.h"
#include "../sse2_gemm.h"
#include "../ssse3_gemm.h"

#include <cstring>
#include <iostream>
#include <math.h>
#include <random>

namespace intgemm {
namespace {

template <typename Backend, Index TileRows, Index TileColumnsMultiplier>
bool Test(const AlignedVector<float>& A, const AlignedVector<float>& B, Index A_rows, Index width, Index B_cols, float quant_mult) {
  AlignedVector<typename Backend::Integer> A_quantized(A.size());
  AlignedVector<typename Backend::Integer> B_prepared(B.size());
  AlignedVector<int32_t> output(A_rows * B_cols);

  AlignedVector<typename Backend::Integer> B_quantized(B.size());
  AlignedVector<int32_t> reference(output.size());

  Backend::PrepareA(A.begin(), A_quantized.begin(), quant_mult, A_rows, width);
  Backend::template PrepareB<TileColumnsMultiplier>(B.begin(), B_prepared.begin(), quant_mult, width, B_cols);
  Backend::template Multiply<TileRows, TileColumnsMultiplier>(A_quantized.begin(), B_prepared.begin(), A_rows, width, B_cols, callbacks::Write<int32_t>(output.begin()));

  references::Quantize(B.begin(), B_quantized.begin(), quant_mult, B_quantized.size());
  references::Multiply(A_quantized.begin(), B_quantized.begin(), reference.begin(), A_rows, width, B_cols, [&](int32_t sum, const callbacks::OutputBufferInfo& info) {
    return sum;
  });

  bool success = true;
  for (std::size_t i = 0; i < output.size(); ++i) {
    if (output[i] != reference[i]) {
      UNSCOPED_INFO("Error at " << i << ", output = " << int(output[i]) << ", reference = " << int(reference[i]));
      success = false;
      break;
    }
  }
  return success;
}

template <typename Backend, Index TileRows, Index TileColumnsMultiplier>
bool TestMany(Index A_rows, Index width, Index B_cols, float quant_mult) {
  AlignedVector<float> A(A_rows * width);
  AlignedVector<float> B(width * B_cols);

  std::mt19937 gen;
  std::uniform_int_distribution<typename Backend::Integer> dist(0, 32);

  gen.seed(0);

  std::generate(A.begin(), A.end(), [&]() {
    return dist(gen);
  });

  std::generate(B.begin(), B.end(), [&]() {
    return dist(gen);
  });

  return Test<Backend, TileRows, TileColumnsMultiplier>(A, B, A_rows, width, B_cols, quant_mult);
}

TEST_CASE("Multiply SSE2 16bit - custom tiling", "") {
  if (kCPU < CPUType::SSE2)
    return;

  CHECK(TestMany<SSE2_16bit, 1, 1>(256, 256, 256, 1.0f));
  CHECK(TestMany<SSE2_16bit, 2, 1>(256, 256, 256, 1.0f));
  CHECK(TestMany<SSE2_16bit, 1, 2>(256, 256, 256, 1.0f));
  CHECK(TestMany<SSE2_16bit, 2, 2>(256, 256, 256, 1.0f));
  CHECK(TestMany<SSE2_16bit, 4, 1>(256, 256, 256, 1.0f));
  CHECK(TestMany<SSE2_16bit, 1, 4>(256, 256, 256, 1.0f));
  CHECK(TestMany<SSE2_16bit, 4, 4>(256, 256, 256, 1.0f));
}

TEST_CASE("Multiply SSSE3 8bit - custom tiling", "") {
  if (kCPU < CPUType::SSSE3)
    return;

  CHECK(TestMany<SSSE3_8bit, 1, 1>(256, 256, 256, 1.0f));
  CHECK(TestMany<SSSE3_8bit, 2, 1>(256, 256, 256, 1.0f));
  CHECK(TestMany<SSSE3_8bit, 1, 2>(256, 256, 256, 1.0f));
  CHECK(TestMany<SSSE3_8bit, 2, 2>(256, 256, 256, 1.0f));
  CHECK(TestMany<SSSE3_8bit, 4, 1>(256, 256, 256, 1.0f));
  CHECK(TestMany<SSSE3_8bit, 1, 4>(256, 256, 256, 1.0f));
  CHECK(TestMany<SSSE3_8bit, 4, 4>(256, 256, 256, 1.0f));
}

TEST_CASE("Multiply AVX2 8bit - custom tiling", "") {
  if (kCPU < CPUType::AVX2)
    return;

  CHECK(TestMany<AVX2_8bit, 1, 1>(256, 256, 256, 1.0f));
  CHECK(TestMany<AVX2_8bit, 2, 1>(256, 256, 256, 1.0f));
  CHECK(TestMany<AVX2_8bit, 1, 2>(256, 256, 256, 1.0f));
  CHECK(TestMany<AVX2_8bit, 2, 2>(256, 256, 256, 1.0f));
  CHECK(TestMany<AVX2_8bit, 4, 1>(256, 256, 256, 1.0f));
  CHECK(TestMany<AVX2_8bit, 1, 4>(256, 256, 256, 1.0f));
  CHECK(TestMany<AVX2_8bit, 4, 4>(256, 256, 256, 1.0f));
}

TEST_CASE("Multiply AVX2 16bit - custom tiling", "") {
  if (kCPU < CPUType::AVX2)
    return;

  CHECK(TestMany<AVX2_16bit, 1, 1>(256, 256, 256, 1.0f));
  CHECK(TestMany<AVX2_16bit, 2, 1>(256, 256, 256, 1.0f));
  CHECK(TestMany<AVX2_16bit, 1, 2>(256, 256, 256, 1.0f));
  CHECK(TestMany<AVX2_16bit, 2, 2>(256, 256, 256, 1.0f));
  CHECK(TestMany<AVX2_16bit, 4, 1>(256, 256, 256, 1.0f));
  CHECK(TestMany<AVX2_16bit, 1, 4>(256, 256, 256, 1.0f));
  CHECK(TestMany<AVX2_16bit, 4, 4>(256, 256, 256, 1.0f));
}

// #ifdef INTGEMM_COMPILER_SUPPORTS_AVX512
//   TEST_CASE("Multiply AVX512BW 8bit - custom tiling", "") {
//     if (kCPU < CPUType::AVX512BW)
//       return;

//     CHECK(TestMany<AVX512_8bit, 1, 1>(256, 256, 256, 1.0f));
//     CHECK(TestMany<AVX512_8bit, 2, 1>(256, 256, 256, 1.0f));
//     CHECK(TestMany<AVX512_8bit, 1, 2>(256, 256, 256, 1.0f));
//     CHECK(TestMany<AVX512_8bit, 2, 2>(256, 256, 256, 1.0f));
//     CHECK(TestMany<AVX512_8bit, 4, 1>(256, 256, 256, 1.0f));
//     CHECK(TestMany<AVX512_8bit, 1, 4>(256, 256, 256, 1.0f));
//     CHECK(TestMany<AVX512_8bit, 4, 4>(256, 256, 256, 1.0f));
//   }

//   TEST_CASE("Multiply AVX512BW 16bit - custom tiling", "") {
//     if (kCPU < CPUType::AVX512BW)
//       return;

//     CHECK(TestMany<AVX512_16bit, 1, 1>(256, 256, 256, 1.0f));
//     CHECK(TestMany<AVX512_16bit, 2, 1>(256, 256, 256, 1.0f));
//     CHECK(TestMany<AVX512_16bit, 1, 2>(256, 256, 256, 1.0f));
//     CHECK(TestMany<AVX512_16bit, 2, 2>(256, 256, 256, 1.0f));
//     CHECK(TestMany<AVX512_16bit, 4, 1>(256, 256, 256, 1.0f));
//     CHECK(TestMany<AVX512_16bit, 1, 4>(256, 256, 256, 1.0f));
//     CHECK(TestMany<AVX512_16bit, 4, 4>(256, 256, 256, 1.0f));
//   }
// #endif

#ifdef INTGEMM_COMPILER_SUPPORTS_AVX512VNNI
  TEST_CASE("Multiply AVX512VNNI 8bit - custom tiling", "") {
    if (kCPU < CPUType::AVX512VNNI)
      return;

    CHECK(TestMany<AVX512VNNI_8bit, 1, 1>(256, 256, 256, 1.0f));
    CHECK(TestMany<AVX512VNNI_8bit, 2, 1>(256, 256, 256, 1.0f));
    CHECK(TestMany<AVX512VNNI_8bit, 1, 2>(256, 256, 256, 1.0f));
    CHECK(TestMany<AVX512VNNI_8bit, 2, 2>(256, 256, 256, 1.0f));
    CHECK(TestMany<AVX512VNNI_8bit, 4, 1>(256, 256, 256, 1.0f));
    CHECK(TestMany<AVX512VNNI_8bit, 1, 4>(256, 256, 256, 1.0f));
    CHECK(TestMany<AVX512VNNI_8bit, 4, 4>(256, 256, 256, 1.0f));
  }
#endif

}
}
