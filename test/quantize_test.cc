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

void QuantizeRef(const float *input, int16_t *output, float quant_mult, std::size_t size) {
  for (std::size_t i = 0; i < size; ++i) {
    float value = roundf(input[i] * quant_mult);
    value = std::max(-32768.0f, value);
    value = std::min(32767.0f, value);
    output[i] = value;
  }
}

void QuantizeRef(const float *input, int8_t *output, float quant_mult, std::size_t size) {
  for (std::size_t i = 0; i < size; ++i) {
    float value = roundf(input[i] * quant_mult);
    value = std::max(-127.0f, value);
    value = std::min(127.0f, value);
    output[i] = value;
  }
}

float QuantizerStddRef(AlignedVector<float>& vals, int num_items, int stdnum) {
  float normal_sums = 0;
  float squares_sum = 0;
  //std::for_each(vals.begin(), vals.end(), [&] (int n) {normal_sums+=n;});
  //std::for_each(vals.begin(), vals.end(), [&] (int n) {squares_sum+=n*n;});
  for (auto&& it : vals) {
    normal_sums += it;
  }

  for (auto&& it : vals) {
    squares_sum += it*it;
  }

  float mean = normal_sums/num_items;
  return mean + (std::sqrt((squares_sum/num_items) - (mean*mean)) * stdnum);
}

template <class Backend>
void testQuantizerStd(int num_items, int stdnum) {
  std::mt19937 gen;
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  AlignedVector<float> inputVec(num_items);

  for (auto&& it : inputVec) {
    it = dist(gen);
  }

  float reference = QuantizerStddRef(inputVec, num_items, stdnum);
  float fast = Backend::GetQuantizerStd(inputVec.begin(), inputVec.end(), stdnum);

  float difference = fabs(reference - fast);
  float eps = 0.00002; //Accumulating horizontal sums can lead to errors.

  CHECK_MESSAGE(difference <= eps, "Reference: " << reference << " actual: " << fast);// /*Backend::kName << */" Mismatch:\n" << "Reference: " << reference << " Fast: " << fast << std::endl);

}

template <class I> bool IsOff(float from, I ref, I test) {
  if (ref == test) return false;
  if (ref - test > 1 && test - ref > 1) return true;
  float off_test = fabs((float)test - from);
  float off_ref = fabs((float)ref - from);
  // Allow 0.5 to round either way.
  if (off_test > 0.49 && off_test < 0.51 && off_ref > 0.49 && off_ref < 0.51) return false;
  return true;
}

template <class Backend> bool Test(const float *input_unaligned, float quant_mult, std::size_t size) {
  typedef typename Backend::Integer Integer;
  bool success = true;
  AlignedVector<float> input(size);
  std::memcpy(input.begin(), input_unaligned, sizeof(float) * size);

  AlignedVector<Integer> ref(size);
  AlignedVector<Integer> test(size);
  QuantizeRef(input.begin(), ref.begin(), quant_mult, size);
  Backend::Quantize(input.begin(), test.begin(), quant_mult, size);
  for (std::size_t i = 0; i < size; ++i) {
    if (IsOff(input[i] * quant_mult, ref[i], test[i])) {
      UNSCOPED_INFO("Error at " << i << " from " << input[i] << '*' << quant_mult << '=' << (input[i]*quant_mult) << " ref = " <<  ref[i] << " test = " << test[i]);
      success = false;
    }
  }
  return success;
}

template <class Backend> bool TestMany() {
  bool success = true;
  float input[32] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31};
  success &= Test<Backend>(input, 1.0, 32);
  success &= Test<Backend>(input, 32.0, 32);
  float corners[32] = {-32769, -32768, -32767, -129, -128, -127, -1, 0, 1, 126, 127, 128, 129, 32766, 32768, 32769, -1.9, -1.5, -1.1, -1, -0.9, -0.5, -0.1, 0.0, 0.1, 0.5, 0.9, 1.0, 1.1, 1.5, 1.9, 16056.8};
  success &= Test<Backend>(corners, 1.0, sizeof(corners) / sizeof(float));
  success &= Test<Backend>(corners, -1.0, sizeof(corners) / sizeof(float));
  success &= Test<Backend>(corners, -0.49, sizeof(corners) / sizeof(float));
  return success;
}

TEST_CASE ("Quantize SSE2", "[quantize]") {
  if (kCPU < CPUType::SSE2) return;
  CHECK(TestMany<SSE2_16bit>());
}

TEST_CASE ("Quantize SSE3", "[quantize]") {
  if (kCPU < CPUType::SSSE3) return;
  CHECK(TestMany<SSSE3_8bit>());
}

TEST_CASE ("Quantize AVX2", "[quantize]") {
  if (kCPU < CPUType::AVX2) return;
  CHECK(TestMany<AVX2_8bit>());
  CHECK(TestMany<AVX2_16bit>());
}
#ifdef INTGEMM_COMPILER_SUPPORTS_AVX512
  TEST_CASE ("Quantize AVX512", "[quantize]") {
    if (kCPU < CPUType::AVX512BW) return;
    CHECK(TestMany<AVX512_8bit>());
    CHECK(TestMany<AVX512_16bit>());
  }
#endif

TEST_CASE("QuantizeStd SSSE3", "[quantizerSTD]") {
  if (kCPU < CPUType::SSSE3) return;
  testQuantizerStd<SSSE3_8bit>(64, 1);
  testQuantizerStd<SSSE3_8bit>(64, 2);
  testQuantizerStd<SSSE3_8bit>(256, 1);
  testQuantizerStd<SSSE3_8bit>(256, 2);
  testQuantizerStd<SSSE3_8bit>(2048, 1);
  testQuantizerStd<SSSE3_8bit>(2048, 2);
  testQuantizerStd<SSSE3_8bit>(65536, 1);
  testQuantizerStd<SSSE3_8bit>(65536, 2);
  testQuantizerStd<SSSE3_8bit>(81920, 1);
  testQuantizerStd<SSSE3_8bit>(81920, 2);
  testQuantizerStd<SSSE3_8bit>(120832, 1);
  testQuantizerStd<SSSE3_8bit>(120832, 2);
}

TEST_CASE("QuantizeStd AVX2", "[quantizerSTD]") {
  if (kCPU < CPUType::AVX2) return;
  testQuantizerStd<AVX2_8bit>(64, 1);
  testQuantizerStd<AVX2_8bit>(64, 2);
  testQuantizerStd<AVX2_8bit>(256, 1);
  testQuantizerStd<AVX2_8bit>(256, 2);
  testQuantizerStd<AVX2_8bit>(2048, 1);
  testQuantizerStd<AVX2_8bit>(2048, 2);
  testQuantizerStd<AVX2_8bit>(65536, 1);
  testQuantizerStd<AVX2_8bit>(65536, 2);
  testQuantizerStd<AVX2_8bit>(81920, 1);
  testQuantizerStd<AVX2_8bit>(81920, 2);
  testQuantizerStd<AVX2_8bit>(120832, 1);
  testQuantizerStd<AVX2_8bit>(120832, 2);
}

#ifdef INTGEMM_COMPILER_SUPPORTS_AVX512
TEST_CASE("QuantizeStd AVX512", "[quantizerSTD]") {
  if (kCPU < CPUType::AVX512BW) return;
  testQuantizerStd<AVX512_8bit>(64, 1);
  testQuantizerStd<AVX512_8bit>(64, 2);
  testQuantizerStd<AVX512_8bit>(256, 1);
  testQuantizerStd<AVX512_8bit>(256, 2);
  testQuantizerStd<AVX512_8bit>(2048, 1);
  testQuantizerStd<AVX512_8bit>(2048, 2);
  testQuantizerStd<AVX512_8bit>(65536, 1);
  testQuantizerStd<AVX512_8bit>(65536, 2);
  testQuantizerStd<AVX512_8bit>(81920, 1);
  testQuantizerStd<AVX512_8bit>(81920, 2);
  testQuantizerStd<AVX512_8bit>(120832, 1);
  testQuantizerStd<AVX512_8bit>(120832, 2);
}
#endif

} // namespace
} // namespace intgemm
