/* Should be a test, just wanted to mess around first and be able to print easily */
#include "log4/log4.h"
#include "aligned.h"

#include <cassert>
#include <cstdint>
#include <stdint.h>

#include <limits>
#include <iostream>

namespace intgemm {
namespace {

// Take the dot product of the bottom 4 bits of a with the bottom 4 bits of b,
// where a and b are in log format.
int16_t BottomDotLog4(uint8_t a, uint8_t b) {
  int16_t shift = 1;
  shift = shift << ((a & 7) + (b & 7));
  if ((a & 8) ^ (b & 8)) {
    shift = -shift;
  }
  return shift;
}

int16_t Saturate(int32_t value) {
  value = std::max<int32_t>(std::numeric_limits<int16_t>::min(), value);
  value = std::min<int32_t>(std::numeric_limits<int16_t>::max(), value);
  return (int16_t)value;
}

int16_t SaturateAdd(int16_t first, int16_t second) {
  return Saturate((int32_t)first + (int32_t)second);
}

// 16-bit columns with saturation.  Closer to implementation.
void SlowSaturateDotLog4(const uint8_t *begin, const uint8_t *end, const uint8_t *with, int16_t *out) {
  assert(!((end - begin) % 2));
  for (; begin != end; ++out) {
    *out = BottomDotLog4(*begin, *with);
    *out = SaturateAdd(*out, BottomDotLog4((*begin) >> 4, (*with) >> 4));
    ++begin, ++with;
    *out = SaturateAdd(*out, BottomDotLog4(*begin, *with));
    *out = SaturateAdd(*out, BottomDotLog4((*begin) >> 4, (*with) >> 4));
    ++begin, ++with;
  }
}

// Reference implementation of log4 dot product without any saturation.
int64_t ReferenceDotLog4(const uint8_t *begin, const uint8_t *end, const uint8_t *with) {
  int64_t result = 0;
  for (; begin != end; ++begin, ++with) {
    result += (int64_t)BottomDotLog4(*begin, *with);
    result += (int64_t)BottomDotLog4((*begin) >> 4, (*with) >> 4);
  }
  return result;
}

#ifndef INTGEMM_NO_AVX512

INTGEMM_AVX512BW void Compare(uint8_t *a, uint8_t *b) {
  __m512i impl = MultiplyLog4(*(__m512i*)a, *(__m512i*)b);
  AlignedVector<int16_t> ref(sizeof(__m512i) / sizeof(int16_t));
  SlowSaturateDotLog4(a, a + sizeof(__m512i), b, ref.begin());
  const int16_t *imp = reinterpret_cast<const int16_t*>(&impl);
  for (unsigned int i = 0; i < sizeof(__m512i) / sizeof(uint16_t); ++i) {
    if (ref[i] != imp[i]) {
      std::cout << ref[i] << ' ' << imp[i] << std::endl;
    }
  }
}

INTGEMM_AVX512BW void Test() {
  if (kCPU < CPUType::CPU_AVX512BW) return;
  // Generate all possible combinations of 4-bit pairs. Note this isn't an
  // exhaustive test case because they should be in each position of the 16-bit
  // blocks.
  AlignedVector<uint8_t> a(128), b(128);
  uint8_t *a_it = a.begin(), *b_it = b.begin();
  for (uint8_t i = 0; i < 16; ++i) {
    for (uint8_t j = 0; j < 16; j += 2, ++a_it, ++b_it) {
      *a_it = (i << 4) | i;
      *b_it = (j << 4) | (j + 1);
    }
  }
  Compare(a.begin(), b.begin());
  Compare(a.begin() + 64, b.begin() + 64);
}


#endif
} // namespace
} // namespace intgemm

int main() {
  intgemm::Test(); 
}
