#pragma once
#include "types.h"
#include <cstdint>
#include <stdint.h>
#include "multiply.h"
// 8 bit is in ssse3_gemm.h

namespace intgemm {

namespace sse2 {
// Same implementation as AVX512, just width.  Grabs 4 32-bit values.
// TODO duplicated function requires the removal of the annonymous namespace
SSE2 inline __m128i QuantizerGrab(const float *input, const __m128 quant_mult_reg) {
  return _mm_cvtps_epi32(_mm_mul_ps(*reinterpret_cast<const __m128*>(input), quant_mult_reg));
}

SELECT_COL_B_DEFINE(SSE2, __m128i)

class QuantizeTile16 {
  public:
    typedef __m128i Integer;

    SSE2 explicit QuantizeTile16(float mult) : mult_reg_(_mm_set1_ps(mult)) {}

    // Quantize 8xfloat into 8xint16_t
    SSE2 inline __m128i Consecutive(const float *input) {
      __m128i g0 = QuantizerGrab(input, mult_reg_);
      __m128i g1 = QuantizerGrab(input + 4, mult_reg_);
      return _mm_packs_epi32(g0, g1);
    }

    SSE2 inline __m128i ForReshape(const float *input, int) {
      return Consecutive(input);
    }

  private:
    const __m128 mult_reg_;
};

// Technically only requires SSE

MAXABSOLUTE_DEFINE(__m128, SSE2)

} //namespace
// This should be pure SSE2 (and below).
struct SSE2_16bit {
  typedef int16_t Integer;

  // Currently A is prepared by quantization but this could theoretically change.
  SSE2 static inline void PrepareA(const float *input, int16_t *output, float quant_mult, Index rows, Index cols) {
    Quantize(input, output, quant_mult, rows * cols);
  }

  SSE2 static void Quantize(const float *input, int16_t *output, float quant_mult, Index size) {
    assert(size % 8 == 0);
    assert(reinterpret_cast<uintptr_t>(input) % 16 == 0);
    assert(reinterpret_cast<uintptr_t>(output) % 16 == 0);
    sse2::QuantizeTile16 q(quant_mult);
    const float *end = input + size;
    for (; input != end; input += 8, output += 8) {
      *reinterpret_cast<__m128i*>(output) = q.Consecutive(input);
    }
  }

  // Tile size for B; B must be a multiple of this block size.
  static const Index kBTileRow = 8;
  static const Index kBTileCol = 8;

  PREPARE_B_16_DEFINE(SSE2, sse2::QuantizeTile16)

  SSE2 static void SelectColumnsB(const int16_t *input, int16_t *output, Index rows, const Index *cols_begin, const Index *cols_end) {
    //TODO #DEFINE
    sse2::SelectColumnsOfB((const __m128i*)input, (__m128i*)output, rows * 2, cols_begin, cols_end);
  }
  MULTIPLY16_DEFINE(__m128i, SSE2, OnSSE2)

  constexpr static const char *const kName = "16-bit SSE2";

  static const CPUType kUses = CPU_SSE2;
};

} // namespace intgemm
