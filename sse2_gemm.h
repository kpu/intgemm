#pragma once
#include "types.h"
#include <cstdint>
#include <stdint.h>
#include "cops.h"
#include "multiply.h"
// 8 bit is in ssse3_gemm.h

namespace intgemm {

namespace sse2 {
// Same implementation as AVX512, just width.  Grabs 4 32-bit values.
// TODO duplicated function requires the removal of the annonymous namespace
SSE2 inline __m128i QuantizerGrab(const float *input, const __m128 quant_mult_reg) {
  return _mm_cvtps_epi32(_mm_mul_ps(*reinterpret_cast<const __m128*>(input), quant_mult_reg));
}

class QuantizeTile16 {
  public:
    typedef __m128i Integer;

    explicit QuantizeTile16(float mult) : mult_reg_(_mm_set1_ps(mult)) {}

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

  SSE2 static void PrepareB(const float *input, int16_t *output, float quant_mult, Index rows, Index cols) {
    //TODO #DEFINE
    PrepareBFor16(input, output, sse2::QuantizeTile16(quant_mult), rows, cols);
  }

  SSE2 static void SelectColumnsB(const int16_t *input, int16_t *output, Index rows, const Index *cols_begin, const Index *cols_end) {
    //TODO #DEFINE
    SelectColumnsOfB((const __m128i*)input, (__m128i*)output, rows * 2, cols_begin, cols_end);
  }

  SSE2 static void Multiply(const int16_t *A, const int16_t *B, float *C, float unquant_mult, Index A_rows, Index width, Index B_cols) {
    //TODO #DEFINE
    Multiply16<__m128i, JustUnquantizeC> (A, B, JustUnquantizeC(C, unquant_mult), A_rows, width, B_cols);
  }

  constexpr static const char *const kName = "16-bit SSE2";

  static const CPUType kUses = CPU_SSE2;
};

// Technically only requires SSE
SSE2 float SSE2_MaxAbsolute(const float *begin, const float *end) {
  return MaxAbsoluteBackend<__m128>(begin, end);
}

} // namespace intgemm
