#pragma once

#include "kernels.h"
#include "multiply.h"
#include "types.h"

#include <cstdint>
#include <stdint.h>

// 8 bit is in ssse3_gemm.h

namespace intgemm {

namespace sse2 {

INTGEMM_SSE2 inline __m128i QuantizerGrab(const float *input, const __m128 quant_mult_reg) {
  return kernels::quantize(loadu_ps<__m128>(input), quant_mult_reg);
}

INTGEMM_SELECT_COL_B(INTGEMM_SSE2, __m128i)

class QuantizeTile16 {
  public:
    typedef __m128i Register;

    INTGEMM_SSE2 explicit QuantizeTile16(float mult) : mult_reg_(_mm_set1_ps(mult)) {}

    INTGEMM_SSE2 inline __m128i Consecutive(const float *input) const {
      return Tile(input, input + 4);
    }

    INTGEMM_SSE2 Register ConsecutiveWithWrapping(const float *input, Index cols_left, Index cols, Index row_step) const {
      return Tile(
        input,
        input + 4 + (cols_left <= 4 ? cols * (row_step - 1) : 0));
    }

    INTGEMM_SSE2 inline __m128i ForReshape(const float *input, int) const {
      return Consecutive(input);
    }

  private:
    INTGEMM_SSE2 __m128i Tile(const float *input0, const float *input1) const {
      __m128i g0 = QuantizerGrab(input0, mult_reg_);
      __m128i g1 = QuantizerGrab(input1, mult_reg_);
      return _mm_packs_epi32(g0, g1);
    }

    const __m128 mult_reg_;
};

// Technically only requires SSE

INTGEMM_MAXABSOLUTE(__m128, INTGEMM_SSE2)

INTGEMM_VECTORMEANSTD(__m128, INTGEMM_SSE2)

} //namespace
// This should be pure INTGEMM_SSE2 (and below).
struct SSE2_16bit {
  typedef int16_t Integer;

  // Currently A is prepared by quantization but this could theoretically change.
  INTGEMM_SSE2 static inline void PrepareA(const float *input, int16_t *output, float quant_mult, Index rows, Index cols) {
    Quantize(input, output, quant_mult, rows * cols);
  }

  INTGEMM_SSE2 static void Quantize(const float *input, int16_t *output, float quant_mult, Index size) {
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

  INTGEMM_PREPARE_B_16(INTGEMM_SSE2, sse2::QuantizeTile16)
  INTGEMM_PREPARE_B_QUANTIZED_TRANSPOSED(INTGEMM_SSE2, CPUType::SSE2, int16_t)
  INTGEMM_PREPARE_B_TRANSPOSED(INTGEMM_SSE2, sse2::QuantizeTile16, int16_t)

  INTGEMM_SSE2 static void SelectColumnsB(const int16_t *input, int16_t *output, Index rows, const Index *cols_begin, const Index *cols_end) {
    //TODO #DEFINE
    sse2::SelectColumnsOfB((const __m128i*)input, (__m128i*)output, rows * 2, cols_begin, cols_end);
  }
  INTGEMM_MULTIPLY16(__m128i, INTGEMM_SSE2, CPUType::SSE2)

  constexpr static const char *const kName = "16-bit SSE2";

  static const CPUType kUses = CPUType::SSE2;
};

} // namespace intgemm
