#include "avx2_gemm.h"
#include "interleave.h"
#include "multiply.h"

#include <cassert>
#include <emmintrin.h>
#include <immintrin.h>
#include <tmmintrin.h>
#include <xmmintrin.h>
#include <cstdint>

namespace intgemm {
#ifdef __AVX2__

// PREPARE A: just quantization in the same memory order.

namespace {
// Read a vector of floats, multiply them, and cast to 32-bit integer.
inline __m256i QuantizerGrab(const float *input, const __m256 quant_mult_reg) {
  return _mm256_cvtps_epi32(_mm256_mul_ps(*reinterpret_cast<const __m256*>(input), quant_mult_reg));
}

inline __m256i QuantizeTile16(const float *input0, const float *input1, __m256 quant_mult_reg) {
  __m256i g0 = QuantizerGrab(input0, quant_mult_reg);
  __m256i g1 = QuantizerGrab(input1, quant_mult_reg);
  __m256i packed = _mm256_packs_epi32(g0, g1);
  // Reorder the packed values because Intel does 0 1 2 3 8 9 10 11 4 5 6 7 12 13 14 15.
  // Technically this could be removed if the PrepareB did the same reordering internally.
  return _mm256_permute4x64_epi64(packed, 0xd8 /* 0, 2, 1, 3 */);
}
} // namespace

// Just quantize everything in order.
void AVX2_16bit::Quantize(const float *input, int16_t *output, float quant_mult, int size) {
  assert(size % 16 == 0);
  assert(reinterpret_cast<uintptr_t>(input) % 32 == 0);
  const __m256 quant_mult_reg = _mm256_set1_ps(quant_mult);
  const float *end = input + size;
  for (; input != end; input += 16, output += 16) {
    *reinterpret_cast<__m256i*>(output) = QuantizeTile16(input, input + 8, quant_mult_reg);
  }
}

namespace {
/* Read 8 floats at a time from input0, input1, input2, and input3.  Quantize
 * them to 8-bit by multiplying with quant_mult_reg then rounding. Concatenate
 * the result into one register and return it.
 */
class QuantizeTile8 {
  public:
    typedef __m256i I;

    explicit QuantizeTile8(float quant_mult) : mult_(_mm256_set1_ps(quant_mult)) {}

    inline __m256i Consecutive(const float *input) {
      return Tile(input, input + 8, input + 16, input + 24);
    }

    inline __m256i ForReshape(const float *input, int cols) {
      // Put higher rows in the second half of the register.  These will jumble
      // around in the same way then conveniently land in the right place.
      return Tile(input, input + 2 * cols, input + 16 * cols, input + 18 * cols);
    }

  private:
    inline __m256i Tile(const float *input0, const float *input1, const float *input2, const float *input3) {
      // Looking at the assembly, gcc has pulled this outside the loops calling this.
      const __m256i neg127 = _mm256_set1_epi8(-127);
      const __m256i shuffle_param = _mm256_set_epi32(7, 3, 6, 2, 5, 1, 4, 0);
      // Grab 4 registers at a time in 32-bit format.
      __m256i g0 = QuantizerGrab(input0, mult_);
      __m256i g1 = QuantizerGrab(input1, mult_);
      __m256i g2 = QuantizerGrab(input2, mult_);
      __m256i g3 = QuantizerGrab(input3, mult_);
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
    
    __m256 mult_;
};
} // namespace

// Just quantize everything in order.
void AVX2_8bit::Quantize(const float *input, int8_t *output, float quant_mult, int size) {
  assert(size % 32 == 0);
  assert(reinterpret_cast<uintptr_t>(input) % 32 == 0);
  QuantizeTile8 q(quant_mult);
  const float *end = input + size;
  for (; input != end; input += 32, output += 32) {
    *reinterpret_cast<__m256i*>(output) = q.Consecutive(input);
  }
}

// PREPARE B: quantize and rearrange.  B is presumed to be constantparameters
// so we can take our time rearranging it in order to save during the multiply.
//
// We presume B starts in row-major order.
//
// A register holds 32 8-bit values or 16 16-bit values and we want that many
// values from the same column in the register.
//
// The multiplier reads 8 rows at a time and we want these reads to be
// contiguous.
//
// Each 8x32 (for 8-bit) or 8x16 (for 16-bit) tile of B is transposed.
// The tiles are stored in column major order.
//
// This matrix shows what index each value of B will be stored at:
//   0  16 ... 240
//   1  17 ... 241
//   2  18 ... 242
//   3  19 ... 243
//   4  20 ... 244
//   5  21 ... 245
//   6  22 ... 246
//   7  23 ... 247
//   8  24 ... 248
//   9  25 ... 249
//  10  26 ... 250
//  11  27 ... 251
//  12  28 ... 252
//  13  29 ... 253
//  14  30 ... 254
//  15  31 ... 255
// 256 272
// 257 273
// ... ...
namespace {

inline void ReshapeToEights16(const float *input, __m256 quant_mult_reg, int cols, __m256i &out0, __m256i &out1, __m256i &out2, __m256i &out3) {
  out0 = QuantizeTile16(input,            input + 8 * cols, quant_mult_reg);
  out2 = QuantizeTile16(input + 1 * cols, input + 9 * cols, quant_mult_reg);
  Interleave16(out0, out2);
  // out0:
  // [0,0,1,1,2,2,3,3] [0,0,1,1,2,2,3,3]
  // out2:
  // [4,4,5,5,6,6,7,7] [4,4,5,5,6,6,7,7]

  // Do it again, just one more time now.
  out1 = QuantizeTile16(input + 2 * cols, input + 10 * cols, quant_mult_reg);
  out3 = QuantizeTile16(input + 3 * cols, input + 11 * cols, quant_mult_reg);
  Interleave16(out1, out3);

  Interleave32(out0, out1);
  Interleave32(out2, out3);
  // out0: 64-bit [0,1] from rows 0-3 [0,1] from rows 8-11
  // out1: 64-bit [2,3] from rows 0-3 [2,3] from rows 8-11
  // out2: 64-bit [5,6] from rows 0-3 [5,6] from rows 8-11
  // out3: 64-bit [7,8] from rows 0-3 [7,8] from rows 8-11
}

} // namespace

void AVX2_16bit::PrepareB(const float *input, int16_t *output_shadow, float quant_mult, int rows, int cols) {
  assert(rows % 16 == 0);
  assert(cols % 8 == 0);
  assert(reinterpret_cast<uintptr_t>(input) % 32 == 0);
  assert(reinterpret_cast<uintptr_t>(output_shadow) % 32 == 0);
  __m256i *output = reinterpret_cast<__m256i*>(output_shadow);
  const __m256 quant_mult_reg = _mm256_set1_ps(quant_mult);
  for (int c = 0; c < cols; c += 8) {
    for (int r = 0; r < rows; r += 16, output += 8) {
      ReshapeToEights16(input + r * cols + c,       quant_mult_reg, cols, output[0], output[2], output[4], output[6]);
      ReshapeToEights16(input + (r + 4) * cols + c, quant_mult_reg, cols, output[1], output[3], output[5], output[7]);
      Interleave64(output[0], output[1]);
      Interleave64(output[2], output[3]);
      Interleave64(output[4], output[5]);
      Interleave64(output[6], output[7]);
    }
  }
}

void AVX2_8bit::PrepareB(const float *input, int8_t *output, float quant_mult, int rows, int cols) {
  PrepareBFor8(input, output, QuantizeTile8(quant_mult), rows, cols);
}

void AVX2_16bit::Multiply(const int16_t *A, const int16_t *B, float *C, float unquant_mult, int A_rows, int width, int B_cols) {
  Multiply16<__m256i, __m256>(A, B, C, unquant_mult, A_rows, width, B_cols);
}

void AVX2_8bit::Multiply(const int8_t *A, const int8_t *B, float *C, float unquant_mult, int A_rows, int width, int B_cols) {
  Multiply8_SSE2OrAVX2<__m256i, __m256>(A, B, C, unquant_mult, A_rows, width, B_cols);
}

#endif // __AVX2__
} // namespace intgemm
