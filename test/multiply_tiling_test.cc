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

template <typename Type>
void fillRandom(AlignedVector<Type>& matrix) {
  static std::mt19937 gen(0);
  static std::uniform_int_distribution<uint8_t> dist(0, 32);

  std::generate(matrix.begin(), matrix.end(), [&]() {
    return dist(gen);
  });
}

template <typename Type>
bool compare(const AlignedVector<Type>& reference, const AlignedVector<Type>& output) {
  for (std::size_t i = 0; i < output.size(); ++i) {
    if (output[i] != reference[i]) {
      UNSCOPED_INFO("Error at " << i << ", output = " << int(output[i]) << ", reference = " << int(reference[i]));
      return false;
    }
  }
  return true;
}

template <typename Backend, Index TileRows, Index TileColumnsMultiplier>
bool Test(Index A_rows, Index width, Index B_cols, float quant_mult) {
  AlignedVector<float> A(A_rows * width);
  AlignedVector<float> B(width * B_cols);

  fillRandom(A);
  fillRandom(B);

  AlignedVector<typename Backend::Integer> A_quantized(A.size());
  AlignedVector<typename Backend::Integer> B_prepared(B.size());
  AlignedVector<int32_t> output(A_rows * B_cols);

  AlignedVector<typename Backend::Integer> B_quantized(B.size());
  AlignedVector<int32_t> reference(output.size());

  Backend::PrepareA(A.begin(), A_quantized.begin(), quant_mult, A_rows, width);
  Backend::template PrepareB<TileColumnsMultiplier>(B.begin(), B_prepared.begin(), quant_mult, width, B_cols);
  Backend::template Multiply<TileRows, TileColumnsMultiplier>(A_quantized.begin(), B_prepared.begin(), A_rows, width, B_cols, callbacks::Write<int32_t>(output.begin()));

  references::Quantize(B.begin(), B_quantized.begin(), quant_mult, B_quantized.size());
  references::Multiply(A_quantized.begin(), B_quantized.begin(), reference.begin(), A_rows, width, B_cols, [&](int32_t sum, const callbacks::OutputBufferInfo&) {
    return sum;
  });

  return compare(reference, output);
}

template <typename Backend, Index TileRows, Index TileColumnsMultiplier>
bool TestShift(Index A_rows, Index width, Index B_cols, float quant_mult) {
  AlignedVector<float> A(A_rows * width);
  AlignedVector<float> B(width * B_cols);

  fillRandom(A);
  fillRandom(B);

  AlignedVector<uint8_t> A_quantized(A.size());   // make it more generic, e.g.: make_unsigned<typename Backend::Integer>::type or sth
  AlignedVector<int8_t> B_prepared(B.size());
  AlignedVector<int32_t> output(A_rows * B_cols);

  AlignedVector<int8_t> B_quantized(B.size());
  AlignedVector<int32_t> reference(output.size());

  Backend::PrepareA(A.begin(), A_quantized.begin(), quant_mult, A_rows, width);
  Backend::template PrepareB<TileColumnsMultiplier>(B.begin(), B_prepared.begin(), quant_mult, width, B_cols);
  Backend::template Multiply8Shift<TileRows, TileColumnsMultiplier>(A_quantized.begin(), B_prepared.begin(), A_rows, width, B_cols, callbacks::Write<int32_t>(output.begin()));

  references::Quantize(B.begin(), B_quantized.begin(), quant_mult, B_quantized.size());
  references::Multiply(A_quantized.begin(), B_quantized.begin(), reference.begin(), A_rows, width, B_cols, [&](int32_t sum, const callbacks::OutputBufferInfo&) {
    return sum;
  });

  return compare(reference, output);
}

TEST_CASE("Multiply SSE2 16bit - custom tiling", "") {
  if (kCPU < CPUType::SSE2)
    return;

  CHECK(Test<SSE2_16bit, 1, 1>(256, 256, 256, 1.0f));
  CHECK(Test<SSE2_16bit, 2, 1>(256, 256, 256, 1.0f));
  CHECK(Test<SSE2_16bit, 1, 2>(256, 256, 256, 1.0f));
  CHECK(Test<SSE2_16bit, 2, 2>(256, 256, 256, 1.0f));
  CHECK(Test<SSE2_16bit, 4, 1>(256, 256, 256, 1.0f));
  CHECK(Test<SSE2_16bit, 1, 4>(256, 256, 256, 1.0f));
  CHECK(Test<SSE2_16bit, 4, 4>(256, 256, 256, 1.0f));
}

TEST_CASE("Multiply SSSE3 8bit - custom tiling", "") {
  if (kCPU < CPUType::SSSE3)
    return;

  CHECK(Test<SSSE3_8bit, 1, 1>(256, 256, 256, 1.0f));
  CHECK(Test<SSSE3_8bit, 2, 1>(256, 256, 256, 1.0f));
  CHECK(Test<SSSE3_8bit, 1, 2>(256, 256, 256, 1.0f));
  CHECK(Test<SSSE3_8bit, 2, 2>(256, 256, 256, 1.0f));
  CHECK(Test<SSSE3_8bit, 4, 1>(256, 256, 256, 1.0f));
  CHECK(Test<SSSE3_8bit, 1, 4>(256, 256, 256, 1.0f));
  CHECK(Test<SSSE3_8bit, 4, 4>(256, 256, 256, 1.0f));
}

TEST_CASE("Multiply AVX2 8bit - custom tiling", "") {
  if (kCPU < CPUType::AVX2)
    return;

  CHECK(Test<AVX2_8bit, 1, 1>(256, 256, 256, 1.0f));
  CHECK(Test<AVX2_8bit, 2, 1>(256, 256, 256, 1.0f));
  CHECK(Test<AVX2_8bit, 1, 2>(256, 256, 256, 1.0f));
  CHECK(Test<AVX2_8bit, 2, 2>(256, 256, 256, 1.0f));
  CHECK(Test<AVX2_8bit, 4, 1>(256, 256, 256, 1.0f));
  CHECK(Test<AVX2_8bit, 1, 4>(256, 256, 256, 1.0f));
  CHECK(Test<AVX2_8bit, 4, 4>(256, 256, 256, 1.0f));
}

TEST_CASE("Multiply AVX2 16bit - custom tiling", "") {
  if (kCPU < CPUType::AVX2)
    return;

  CHECK(Test<AVX2_16bit, 1, 1>(256, 256, 256, 1.0f));
  CHECK(Test<AVX2_16bit, 2, 1>(256, 256, 256, 1.0f));
  CHECK(Test<AVX2_16bit, 1, 2>(256, 256, 256, 1.0f));
  CHECK(Test<AVX2_16bit, 2, 2>(256, 256, 256, 1.0f));
  CHECK(Test<AVX2_16bit, 4, 1>(256, 256, 256, 1.0f));
  CHECK(Test<AVX2_16bit, 1, 4>(256, 256, 256, 1.0f));
  CHECK(Test<AVX2_16bit, 4, 4>(256, 256, 256, 1.0f));
}

// #ifdef INTGEMM_COMPILER_SUPPORTS_AVX512
//   TEST_CASE("Multiply AVX512BW 8bit - custom tiling", "") {
//     if (kCPU < CPUType::AVX512BW)
//       return;

//     CHECK(Test<AVX512_8bit, 1, 1>(256, 256, 256, 1.0f));
//     CHECK(Test<AVX512_8bit, 2, 1>(256, 256, 256, 1.0f));
//     CHECK(Test<AVX512_8bit, 1, 2>(256, 256, 256, 1.0f));
//     CHECK(Test<AVX512_8bit, 2, 2>(256, 256, 256, 1.0f));
//     CHECK(Test<AVX512_8bit, 4, 1>(256, 256, 256, 1.0f));
//     CHECK(Test<AVX512_8bit, 1, 4>(256, 256, 256, 1.0f));
//     CHECK(Test<AVX512_8bit, 4, 4>(256, 256, 256, 1.0f));
//   }

//   TEST_CASE("Multiply AVX512BW 16bit - custom tiling", "") {
//     if (kCPU < CPUType::AVX512BW)
//       return;

//     CHECK(Test<AVX512_16bit, 1, 1>(256, 256, 256, 1.0f));
//     CHECK(Test<AVX512_16bit, 2, 1>(256, 256, 256, 1.0f));
//     CHECK(Test<AVX512_16bit, 1, 2>(256, 256, 256, 1.0f));
//     CHECK(Test<AVX512_16bit, 2, 2>(256, 256, 256, 1.0f));
//     CHECK(Test<AVX512_16bit, 4, 1>(256, 256, 256, 1.0f));
//     CHECK(Test<AVX512_16bit, 1, 4>(256, 256, 256, 1.0f));
//     CHECK(Test<AVX512_16bit, 4, 4>(256, 256, 256, 1.0f));
//   }
// #endif

#ifdef INTGEMM_COMPILER_SUPPORTS_AVX512VNNI
  TEST_CASE("Multiply AVX512VNNI 8bit - custom tiling", "") {
    if (kCPU < CPUType::AVX512VNNI)
      return;

    CHECK(Test<AVX512VNNI_8bit, 1, 1>(256, 256, 256, 1.0f));
    CHECK(Test<AVX512VNNI_8bit, 2, 1>(256, 256, 256, 1.0f));
    CHECK(Test<AVX512VNNI_8bit, 1, 2>(256, 256, 256, 1.0f));
    CHECK(Test<AVX512VNNI_8bit, 2, 2>(256, 256, 256, 1.0f));
    CHECK(Test<AVX512VNNI_8bit, 4, 1>(256, 256, 256, 1.0f));
    CHECK(Test<AVX512VNNI_8bit, 1, 4>(256, 256, 256, 1.0f));
    CHECK(Test<AVX512VNNI_8bit, 4, 4>(256, 256, 256, 1.0f));
  }

  TEST_CASE("Multiply AVX512VNNI Shift 8bit - custom tiling", "") {
    if (kCPU < CPUType::AVX512VNNI)
      return;

    CHECK(TestShift<AVX512VNNI_8bit, 1, 1>(256, 256, 256, 1.0f));
    CHECK(TestShift<AVX512VNNI_8bit, 2, 1>(256, 256, 256, 1.0f));
    CHECK(TestShift<AVX512VNNI_8bit, 1, 2>(256, 256, 256, 1.0f));
    CHECK(TestShift<AVX512VNNI_8bit, 2, 2>(256, 256, 256, 1.0f));
    CHECK(TestShift<AVX512VNNI_8bit, 4, 1>(256, 256, 256, 1.0f));
    CHECK(TestShift<AVX512VNNI_8bit, 1, 4>(256, 256, 256, 1.0f));
    CHECK(TestShift<AVX512VNNI_8bit, 4, 4>(256, 256, 256, 1.0f));
  }
#endif

}
}
