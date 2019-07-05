#include "aligned.h"
#include "log4/log4.h"

#include "3rd_party/catch.hpp"

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

// Reference implementation of log4 dot product.
int64_t ReferenceDotLog4(const uint8_t *begin, const uint8_t *end, const uint8_t *with) {
  int64_t result = 0;
  for (; begin != end; ++begin, ++with) {
    result += (int64_t)BottomDotLog4(*begin, *with);
    result += (int64_t)BottomDotLog4((*begin) >> 4, (*with) >> 4);
  }
  return result;
}

#ifndef INTGEMM_NO_AVX512

 
TEST_CASE("4-bit dot", "combinations") {
  if (kCPU < CPUType::AVX512BW) return;
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
  std::cout << ReferenceDotLog4(a.begin(), a.end(), b.begin()) << std::endl;
}
#endif

}
} // namespace intgemm
