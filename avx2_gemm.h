#pragma once
#include "types.h"
#include <cstdint>
#include <stdint.h>

#include "interleave.h"
#include "multiply.h"

namespace intgemm {

namespace avx2 {
// Read a vector of floats, multiply them, and cast to 32-bit integer.
// EVIL EVIL CODE DUPLICATION, FIX
INTGEMM_AVX2 inline __m256i QuantizerGrab(const float *input, const __m256 quant_mult_reg) {
  return _mm256_cvtps_epi32(_mm256_mul_ps(*reinterpret_cast<const __m256*>(input), quant_mult_reg));
}

INTGEMM_SELECT_COL_B(INTGEMM_AVX2, __m256i)

class QuantizeTile16 {
  public:
    typedef __m256i Integer;

    INTGEMM_AVX2 explicit QuantizeTile16(float mult) : mult_(_mm256_set1_ps(mult)) {}

    INTGEMM_AVX2 Integer Consecutive(const float *input) {
      return Tile(input, input + 8);
    }

    INTGEMM_AVX2 Integer ForReshape(const float *input, Index cols) {
      // 8 rows in the first 128-bit register, 8 in the second register.
      return Tile(input, input + 8 * cols);
    }

  private:
    INTGEMM_AVX2 __m256i Tile(const float *input0, const float *input1) {
      __m256i g0 = QuantizerGrab(input0, mult_);
      __m256i g1 = QuantizerGrab(input1, mult_);
      __m256i packed = _mm256_packs_epi32(g0, g1);
      // Reorder the packed values because Intel does 0 1 2 3 8 9 10 11 4 5 6 7 12 13 14 15.
      // Technically this could be removed if the PrepareB did the same reordering internally.
      return _mm256_permute4x64_epi64(packed, 0xd8 /* 0, 2, 1, 3 */);
    }

    const __m256 mult_;
};

} // namespace


struct AVX2_16bit {
  typedef int16_t Integer;

  // Currently A is prepared by quantization but this could theoretically change.
  INTGEMM_AVX2 static inline void PrepareA(const float *input, int16_t *output, float quant_mult, Index rows, Index cols) {
    Quantize(input, output, quant_mult, rows * cols);
  }

  // Just quantize everything in order.
  INTGEMM_AVX2 static void Quantize(const float *input, int16_t *output, float quant_mult, Index size) {
    assert(size % 16 == 0);
    assert(reinterpret_cast<uintptr_t>(input) % 32 == 0);
    avx2::QuantizeTile16 q(quant_mult);
    const float *end = input + size;
    for (; input != end; input += 16, output += 16) {
      *reinterpret_cast<__m256i*>(output) = q.Consecutive(input);
    }
  }

  // Tile size for B; B must be a multiple of this block size.
  static const Index kBTileRow = 16;
  static const Index kBTileCol = 8;
/*
  INTGEMM_AVX2 static void PrepareB(const float *input, int16_t *output, float quant_mult, Index rows, Index cols) {
    PrepareBFor16(input, output, avx2::QuantizeTile16(quant_mult), rows, cols);
  }*/
  INTGEMM_PREPARE_B_16(INTGEMM_AVX2, avx2::QuantizeTile16)

  INTGEMM_AVX2 static void SelectColumnsB(const int16_t *input, int16_t *output, Index rows, const Index *cols_begin, const Index *cols_end) {
    avx2::SelectColumnsOfB((const __m256i*)input, (__m256i*)output, rows * 2, cols_begin, cols_end);
  }
  
  INTGEMM_MULTIPLY16(__m256i, INTGEMM_AVX2, OnAVX2)

  constexpr static const char *const kName = "16-bit INTGEMM_AVX2";

  static const CPUType kUses = CPU_AVX2;
};

namespace avx2 {
/* Read 8 floats at a time from input0, input1, input2, and input3.  Quantize
 * them to 8-bit by multiplying with quant_mult_reg then rounding. Concatenate
 * the result into one register and return it.
 */
class QuantizeTile8 {
  public:
    typedef __m256i Integer;

    INTGEMM_AVX2 explicit QuantizeTile8(float quant_mult) : mult_(_mm256_set1_ps(quant_mult)) {}

    INTGEMM_AVX2 inline __m256i Consecutive(const float *input) {
      return Tile(input, input + 8, input + 16, input + 24);
    }

    INTGEMM_AVX2 inline __m256i ForReshape(const float *input, Index cols) {
      // Put higher rows in the second half of the register.  These will jumble
      // around in the same way then conveniently land in the right place.
      return Tile(input, input + 2 * cols, input + 16 * cols, input + 18 * cols);
    }

  private:
    INTGEMM_AVX2 inline __m256i Tile(const float *input0, const float *input1, const float *input2, const float *input3) {
      // Looking at the assembly, gcc has pulled this outside the loops calling this.
      const __m256i neg127 = _mm256_set1_epi8(-127);
      const __m256i shuffle_param = _mm256_set_epi32(7, 3, 6, 2, 5, 1, 4, 0);
      // Grab 4 registers at a time in 32-bit format.
      __m256i g0 = avx2::QuantizerGrab(input0, mult_);
      __m256i g1 = avx2::QuantizerGrab(input1, mult_);
      __m256i g2 = avx2::QuantizerGrab(input2, mult_);
      __m256i g3 = avx2::QuantizerGrab(input3, mult_);
      // Pack 32-bit to 16-bit.
      __m256i packed0 = _mm256_packs_epi32(g0, g1);
      __m256i packed1 = _mm256_packs_epi32(g2, g3);
      // Pack 16-bit to 8-bit.
      __m256i packed = _mm256_packs_epi16(packed0, packed1);
      // Ban -128.
      packed = _mm256_max_epi8(packed, neg127);
      // Currently in 0 1 2 3 8 9 10 11 16 17 18 19 24 25 26 27 4 5 6 7 12 13 14 15 20 21 22 23 28 29 30 31
      // Or as 32-bit integers 0 2 4 6 1 3 5 7
      // Technically this could be removed so long as the rows are bigger than 16
      // and the values are only used for GEMM.
      return _mm256_permutevar8x32_epi32(packed, shuffle_param);
    }
    
    const __m256 mult_;
};

// Technically only requires AVX
INTGEMM_MAXABSOLUTE(__m256, INTGEMM_AVX2)

} // namespace

struct AVX2_8bit {
  typedef int8_t Integer;

  // Currently A is prepared by quantization but this could theoretically change.
  INTGEMM_AVX2 static inline void PrepareA(const float *input, int8_t *output, float quant_mult, Index rows, Index cols) {
    Quantize(input, output, quant_mult, rows * cols);
  }

  // Just quantize everything in order.
  INTGEMM_AVX2 static void Quantize(const float *input, int8_t *output, float quant_mult, Index size) {
    assert(size % 32 == 0);
    assert(reinterpret_cast<uintptr_t>(input) % 32 == 0);
    avx2::QuantizeTile8 q(quant_mult);
    const float *end = input + size;
    for (; input != end; input += 32, output += 32) {
      *reinterpret_cast<__m256i*>(output) = q.Consecutive(input);
    }
  }

  // Tile size for B; B must be a multiple of this block size.
  static const Index kBTileRow = 32;
  static const Index kBTileCol = 8;

/*
  INTGEMM_AVX2 static void PrepareB(const float *input, int8_t *output, float quant_mult, Index rows, Index cols) {
    PrepareBFor8(input, output, avx2::QuantizeTile8(quant_mult), rows, cols);
  }*/

  INTGEMM_PREPARE_B_8(INTGEMM_AVX2, avx2::QuantizeTile8)

  INTGEMM_AVX2 static void SelectColumnsB(const int8_t *input, int8_t *output, Index rows, const Index *cols_begin, const Index *cols_end) {
    avx2::SelectColumnsOfB((const __m256i*)input, (__m256i*)output, rows, cols_begin, cols_end);
  }
/*
  INTGEMM_AVX2 static void Multiply(const int8_t *A, const int8_t *B, float *C, float unquant_mult, Index A_rows, Index width, Index B_cols) {
    //Multiply8_SSE2OrAVX2<Multiply8_AVXAVX2, __m256i, __m256>(A, B, C, unquant_mult, A_rows, width, B_cols);
    Multiply8_SSE2OrAVX2__m256i<JustUnquantizeC>(A, B, JustUnquantizeC(C, unquant_mult), A_rows, width, B_cols);
  }*/
  INTGEMM_MULTIPLY8(__m256i, INTGEMM_AVX2, OnAVX2)
  
  constexpr static const char *const kName = "8-bit INTGEMM_AVX2";

  static const CPUType kUses = CPU_AVX2;
};

} // namespace intgemm
