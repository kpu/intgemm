#include "Quantize.h"
#include "StopWatch.h"

#include <cstring>
#include <math.h>

#include <iostream>

namespace intgemm {
namespace {

template <class I> bool IsOff(float from, I ref, I test) {
  if (ref == test) return false;
  if (ref - test > 1 && test - ref > 1) return true;
  float off_test = fabs((float)test - from);
  float off_ref = fabs((float)ref - from);
  // Allow 0.5 to round either way.
  if (off_test > 0.49 && off_test < 0.51 && off_ref > 0.49 && off_ref < 0.51) return false;
  return true;
}

bool Test(const float *input_unaligned, float quant_mult, std::size_t size) {
  bool success = true;
  float *input = static_cast<float*>(aligned_alloc(64, sizeof(float) * size));
  std::memcpy(input, input_unaligned, sizeof(float) * size);
  void *mem = aligned_alloc(64, sizeof(int16_t) * size * 2);
  int16_t *ref16 = static_cast<int16_t*>(mem);
  int16_t *test16 = ref16 + size;
  slow::Quantize16(input, ref16, quant_mult, size);
  AVX2::Quantize16(input, test16, quant_mult, size);
  for (std::size_t i = 0; i < size; ++i) {
    if (IsOff(input[i] * quant_mult, ref16[i], test16[i])) {
      std::cerr << "16-bit error at " << i << " from " << input[i] << '*' << quant_mult << '=' << (input[i]*quant_mult) << " ref = " <<  ref16[i] << " test = " << test16[i] << '\n';
      success = false;
    }
  }

  int8_t *ref8 = static_cast<int8_t*>(mem);
  int8_t *test8 = ref8 + size;
  slow::Quantize8(input, ref8, quant_mult, size);
  AVX2::Quantize8(input, test8, quant_mult, size);
  for (std::size_t i = 0; i < size; ++i) {
    if (IsOff(input[i] * quant_mult, ref8[i], test8[i])) {
      std::cerr << "8-bit error at " << i << " from " << input[i] << '*' << quant_mult << "=" << (input[i]*quant_mult) << " ref = " << (int16_t)ref8[i] << " test = " << (int16_t)test8[i] << '\n';
      success = false;
    }
  }

  free(input);
  free(mem);
  return success;
}

void Benchmark(std::size_t size) {
  float *input = (float*)aligned_alloc(64, sizeof(float) * size);
  void *output = aligned_alloc(64, sizeof(int16_t) * size);
  int8_t *out8 = (int8_t*)output;
  int16_t *out16 = (int16_t*)output;
  for (std::size_t i = 0; i < size; ++i) {
    input[i] = i;
  }
#ifdef __AVX512F__
  // Burn in.
  slow::Quantize16(input, out16, 3, size);
  {
    StopWatch w("AVX512 16-bit");
    for (int i = 0; i < 10; ++i)
      AVX512::Quantize16(input, out16, 3, size);
  }
#endif
  slow::Quantize16(input, out16, 3, size);
  {
    StopWatch w("AVX2 16-bit");
    for (int i = 0; i < 10; ++i)
      AVX2::Quantize16(input, out16, 3, size);
  }
  slow::Quantize16(input, out16, 3, size);
  {
    StopWatch w("SSE 16-bit");
    for (int i = 0; i < 10; ++i)
      SSE::Quantize16(input, out16, 3, size);
  }
#ifdef __AVX512F__
  slow::Quantize8(input, out8, 3, size);
  {
    StopWatch w("AVX512 8-bit");
    for (int i = 0; i < 10; ++i)
      AVX512::Quantize8(input, out8, 3, size);
  }
#endif
  slow::Quantize8(input, out8, 3, size);
  {
    StopWatch w("AVX2 8-bit");
    for (int i = 0; i < 10; ++i)
      AVX2::Quantize8(input, out8, 3, size);
  }
  slow::Quantize8(input, out8, 3, size);
  {
    StopWatch w("SSE 8-bit");
    for (int i = 0; i < 10; ++i)
      SSE::Quantize8(input, out8, 3, size);
  }
}

} // namespace
} // namespace intgemm

int main() {
  float input[32] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31};
  bool success = true;
  success &= intgemm::Test(input, 1.0, 32);
  success &= intgemm::Test(input, 32.0, 32);
  float corners[32] = {-32769, -32768, -32767, -129, -128, -127, -1, 0, 1, 126, 127, 128, 129, 32766, 32768, 32769, -1.9, -1.5, -1.1, -1, -0.9, -0.5, -0.1, 0.0, 0.1, 0.5, 0.9, 1.0, 1.1, 1.5, 1.9, 16056.8};
  success &= intgemm::Test(corners, 1.0, sizeof(corners) / sizeof(float));
  success &= intgemm::Test(corners, -1.0, sizeof(corners) / sizeof(float));
  success &= intgemm::Test(corners, -0.49, sizeof(corners) / sizeof(float));
  intgemm::Benchmark(1048576);
  return success ? 0 : 1;
}
