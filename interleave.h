#pragma once

#include <emmintrin.h>
#include <immintrin.h>
#include <tmmintrin.h>
#include <xmmintrin.h>

#include <cassert>

namespace intgemm {

/* This macro defines functions that interleave their arguments like
 * inline void Interleave8(__m256i &first, __m256i &second) {
 *   __m256i temp = _mm256_unpacklo_epi8(first, second);
 *   second = _mm256_unpackhi_epi8(first, second);
 *   first = temp;
 * }
 *
 * Example usage:
 *   INTGEMM_INTERLEAVE(__m128i, )
 *   INTGEMM_INTERLEAVE(__m256i, 256)
 *   INTGEMM_INTERLEAVE(__m512i, 512)
 */
#define INTGEMM_INTERLEAVE(type, prefix) \
 inline void Interleave8(type &first, type &second) { \
  type temp = _mm##prefix##_unpacklo_epi8(first, second); \
  second = _mm##prefix##_unpackhi_epi8(first, second); \
  first = temp; \
} \
inline void Interleave16(type &first, type &second) { \
  type temp = _mm##prefix##_unpacklo_epi16(first, second); \
  second = _mm##prefix##_unpackhi_epi16(first, second); \
  first = temp; \
} \
inline void Interleave32(type &first, type &second) { \
  type temp = _mm##prefix##_unpacklo_epi32(first, second); \
  second = _mm##prefix##_unpackhi_epi32(first, second); \
  first = temp; \
} \
inline void Interleave64(type &first, type &second) { \
  type temp = _mm##prefix##_unpacklo_epi64(first, second); \
  second = _mm##prefix##_unpackhi_epi64(first, second); \
  first = temp; \
}

#ifdef __SSE2__
INTGEMM_INTERLEAVE(__m128i, )
#endif
#ifdef __AVX2__
INTGEMM_INTERLEAVE(__m256i, 256)
#endif
#ifdef __AVX512__
INTGEMM_INTERLEAVE(__m512i, 512)
#endif

template <class Register> inline void Swap(Register &a, Register &b) {
  Register tmp = a;
  a = b;
  b = tmp;
}

/* Transpose registers containing 8 packed 16-bit integers.
 * Each 128-bit lane is handled independently.
 */
template <class Register> inline void Transpose16InLane(Register &r0, Register &r1, Register &r2, Register &r3, Register &r4, Register &r5, Register &r6, Register &r7) {
  // r0: columns 0 1 2 3 4 5 6 7 from row 0
  // r1: columns 0 1 2 3 4 5 6 7 from row 1

  Interleave16(r0, r1);
  Interleave16(r2, r3);
  Interleave16(r4, r5);
  Interleave16(r6, r7);
  // r0: columns 0 0 1 1 2 2 3 3 from rows 0 and 1
  // r1: columns 4 4 5 5 6 6 7 7 from rows 0 and 1
  // r2: columns 0 0 1 1 2 2 3 3 from rows 2 and 3
  // r3: columns 4 4 5 5 6 6 7 7 from rows 2 and 3
  // r4: columns 0 0 1 1 2 2 3 3 from rows 4 and 5
  // r5: columns 4 4 5 5 6 6 7 7 from rows 4 and 5
  // r6: columns 0 0 1 1 2 2 3 3 from rows 6 and 7
  // r7: columns 4 4 5 5 6 6 7 7 from rows 6 and 7

  Interleave32(r0, r2);
  Interleave32(r1, r3);
  Interleave32(r4, r6);
  Interleave32(r5, r7);
  // r0: columns 0 0 0 0 1 1 1 1 from rows 0, 1, 2, and 3
  // r1: columns 4 4 4 4 5 5 5 5 from rows 0, 1, 2, and 3
  // r2: columns 2 2 2 2 3 3 3 3 from rows 0, 1, 2, and 3
  // r3: columns 6 6 6 6 7 7 7 7 from rows 0, 1, 2, and 3
  // r4: columns 0 0 0 0 1 1 1 1 from rows 4, 5, 6, and 7
  // r5: columns 4 4 4 4 5 5 5 5 from rows 4, 5, 6, and 7
  // r6: columns 2 2 2 2 3 3 3 3 from rows 4, 5, 6, and 7
  // r7: columns 6 6 6 6 7 7 7 7 from rows 4, 5, 6, and 7

  Interleave64(r0, r4);
  Interleave64(r1, r5);
  Interleave64(r2, r6);
  Interleave64(r3, r7);
  // r0: columns 0 0 0 0 0 0 0 0 from rows 0 through 7
  // r1: columns 4 4 4 4 4 4 4 4 from rows 0 through 7
  // r2: columns 2 2 2 2 2 2 2 2 from rows 0 through 7
  // r3: columns 6 6 6 6 6 6 6 6 from rows 0 through 7
  // r4: columns 1 1 1 1 1 1 1 1 from rows 0 through 7
  // r5: columns 5 5 5 5 5 5 5 5 from rows 0 through 7
  
  // Empirically gcc is able to remove these movs and just rename the outputs of Interleave64.
  Swap(r1, r4);
  Swap(r3, r6);
}

/* Tranpose registers containing 16 packed 8-bit integers.
 * Each 128-bit lane is handled independently.
 */
template <class Register> inline void Transpose8InLane(
    Register &r0, Register &r1, Register &r2, Register &r3, Register &r4, Register &r5, Register &r6, Register &r7,
    Register &r8, Register &r9, Register &r10, Register &r11, Register &r12, Register &r13, Register &r14, Register &r15) {
  // Get 8-bit values to 16-bit values so they can travel together.
  Interleave8(r0, r1);
  // r0: columns 0 0 1 1 2 2 3 3 4 4 5 5 6 6 7 7 from rows 0 and 1.
  // r1: columns 8 8 9 9 10 10 11 11 12 12 13 13 14 14 15 15 from rows 0 and 1.
  Interleave8(r2, r3);
  // r2: columns 0 0 1 1 2 2 3 3 4 4 5 5 6 6 7 7 from rows 2 and 3.
  Interleave8(r4, r5);
  Interleave8(r6, r7);
  Interleave8(r8, r9);
  Interleave8(r10, r11);
  Interleave8(r12, r13);
  Interleave8(r14, r15);
  Transpose16InLane(r0, r2, r4, r6, r8, r10, r12, r14);
  Transpose16InLane(r1, r3, r5, r7, r9, r11, r13, r15);
  // Permute into correct order.  This is free because the outputs just get pemuted.
  Register tmp;
  tmp = r2;
  r2 = r4;
  r4 = r8;
  r8 = r1;
  r1 = tmp;
  tmp = r3;
  r3 = r6;
  r6 = r12;
  r12 = r9;
  r9 = tmp;
  tmp = r5;
  r5 = r10;
  r10 = tmp;
  tmp = r7;
  r7 = r14;
  r14 = r13;
  r13 = r11;
  r11 = tmp;
}

// This is a helper function for ReshapeBFor8
// It calls the quantizer to retrieve rows 0-7 of input from a row-major
// matrix, then does what it can to transpose.
template <class Quantizer> inline void ReshapeToEights8(const float *input, typename Quantizer::F quant_mult_reg, int cols, typename Quantizer::I &out0, typename Quantizer::I &out1, typename Quantizer::I &out2, typename Quantizer:: I &out3) {
  // Rows 0 and 2
  out0 = Quantizer::ForReshape(input, cols, quant_mult_reg);
  // Rows 1 and 3
  out2 = Quantizer::ForReshape(input + cols, cols, quant_mult_reg);
  // Rows 4 and 6
  out1 = Quantizer::ForReshape(input + 4 * cols, cols, quant_mult_reg);
  // Rows 5 and 7
  out3 = Quantizer::ForReshape(input + 5 * cols, cols, quant_mult_reg);
  Interleave8(out0, out2);
  Interleave16(out0, out2);
  // out0:
  // [0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3] [0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3]
  // Rows 0, 1, 2, and 3 in first 128; rows 16, 17, 18, and 19 in last 128
  // out2:
  // [4,4,4,4,5,5,5,5,6,6,6,6,7,7,7,7] [4,4,4,4,5,5,5,5,6,6,6,6,7,7,7,7]
  // Or as 32-bit blocks: [4,5,6,7]
  Interleave8(out1, out3);
  Interleave16(out1, out3);
  // out1: [0,1,2,3] from rows 4-7 [0,1,2,3] from rows 20-23
  // out3: [5,6,7,8] from rows 4-7 [5,6,7,8] from rows 20-23
  // out1: [0,1,2,3] from rows 4-7 [0,1,2,3] from rows 20-23
  // out3: [5,6,7,8] from rows 4-7 [5,6,7,8] from rows 20-23
  Interleave32(out0, out1);
  Interleave32(out2, out3);
  // out0: 64-bit [0,1] from rows 0-7 [0,1] from rows 16-23
  // out1: 64-bit [2,3] from rows 0-7 [2,3] from rows 16-23
  // out2: 64-bit [5,6] from rows 0-7 [5,6] from rows 16-23
  // out3: 64-bit [7,8] from rows 0-7 [7,8] from rows 16-23
}

// PREPARE B: quantize and rearrange.  B is presumed to be constantparameters
// so we can take our time rearranging it in order to save during the multiply.
//
// We presume B starts in row-major order.
//
// In AVX2, a register holds 32 8-bit values or 16 16-bit values and we want
// that many values from the same column in the register.
//
// The multiplier reads 8 rows at a time and we want these reads to be
// contiguous.
//
// Each 8x32 (for 8-bit) or 8x16 (for 16-bit) tile of B is transposed.
// The tiles are stored in column major order.
//
// For AVX2, this matrix shows what index each value of B will be stored at:
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
template <class Quantizer> inline void PrepareBFor8(const float *input, int8_t *output_shadow, float quant_mult, int rows, int cols) {
  typedef typename Quantizer::I Register;
  // Currently all multipliers have a stride of 8 columns.
  const int kColStride = 8;
  assert(cols % kColStride == 0);
  assert(rows % sizeof(Register) == 0);
  assert(reinterpret_cast<uintptr_t>(input) % sizeof(Register) == 0);
  Register *output = reinterpret_cast<Register*>(output_shadow);
  assert(reinterpret_cast<uintptr_t>(output) % sizeof(Register) == 0);

  typename Quantizer::F quant_mult_reg = Quantizer::Broadcast(quant_mult);
  for (int c = 0; c < cols; c += kColStride) {
    for (int r = 0; r < rows; r += sizeof(Register), output += 8) {
      ReshapeToEights8<Quantizer>(input + r * cols + c,       quant_mult_reg, cols, output[0], output[2], output[4], output[6]);
      // Read everything 8 rows later in B.
      ReshapeToEights8<Quantizer>(input + (r + 8) * cols + c, quant_mult_reg, cols, output[1], output[3], output[5], output[7]);
      // Interleave the results from 8 rows later to finally get:
      // B's column c in output[0]
      // B's column c + 1 in output[1] etc
      Interleave64(output[0], output[1]);
      Interleave64(output[2], output[3]);
      Interleave64(output[4], output[5]);
      Interleave64(output[6], output[7]);
    }
  }
}

} // namespace intgemm
