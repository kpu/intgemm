#include "AVX_Matrix_Mult.h"

#include <cassert>
#include <emmintrin.h>
#include <immintrin.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <tmmintrin.h>
#include <xmmintrin.h>

#include <iostream>
#include <iomanip>

void Print8(__m512i value) {
  int8_t out[64];
  _mm512_storeu_si512(&out, value);
  for (int i = 0; i < 64; ++i) {
    std::cout << std::setw(3) << (int16_t)out[i] << ' ';
  }
  std::cout << '\n';
}

void Print16(__m512i value) {
  int16_t out[32];
  _mm512_storeu_si512(&out, value);
  for (int i = 0; i < 32; ++i) {
    std::cout << std::setw(2) << out[i] << ' ';
  }
  std::cout << '\n';
}

namespace {
// Load from memory, multiply, and convert to int32_t.
inline __m512i QuantizerGrab(const float *input, const __m512 quant_mult_reg) {
  // Load 16 floats
  __m512 val = _mm512_load_ps(input);
  // Multiply each by the quantization factor.
  val = _mm512_mul_ps(val, quant_mult_reg);
  // Cast to 32-bit int
  return _mm512_cvtps_epi32(val);
}
} // namespace

// Convert 
void AVX_Quantize16(const float *input, int16_t *output, float quant_mult, std::size_t size) {
    assert(size % 16 == 0);
    assert(reinterpret_cast<uintptr_t>(input) % 64 == 0);
    // Fill with the quantization multiplier.
    const __m512 quant_mult_reg = _mm512_set1_ps(quant_mult);
    const float *end = input + size;
    for (; input != end; input += 16, output += 16) {
      // There doesn't seem to be an unmasked version.
      _mm512_mask_cvtsepi32_storeu_epi16(output, 0xffff, QuantizerGrab(input, quant_mult_reg));
    }
}

void AVX_Quantize8(const float *input, int8_t *output, float quant_mult, std::size_t size) {
  assert(size % 16 == 0);
  assert(reinterpret_cast<uintptr_t>(input) % 64 == 0);
//  const __m512i neg127 = _mm512_set1_epi32(-127);
  const __m512 quant_mult_reg = _mm512_set1_ps(quant_mult);
  const float *end = input + size;
  for (; input < end; input += 16, output += 16) {
    __m512i asint = QuantizerGrab(input, quant_mult_reg);
    /* Ban -128.
     * The largest possbile product is -128 * -128 = 2^14. If two of those are
     * summed that's 2^15 which is too large for int16_t. By banning -128 we
     * can accumulate two in int16_t w/o saturation before going to int32_t.
     * But this is ok because apparently the instruction will saturate.
     */
//    asint = _mm512_max_epi32(asint, neg127);
    // There doesn't seem to be an unmasked version.
    _mm512_mask_cvtsepi32_storeu_epi8(output, 0xffff, asint);
  }
}

namespace {

union FloatAccess {
  float as_f[4];
  __m128 as_n;
};
union IntAccess {
  int32_t as_i[4];
  __m128i as_n;
};

// Assuming sum1, sum2, sum3, and sum4 are arrays 32-bit signed integers,
// reduce within each.
// Returns [sum(sum1), sum(sum2), sum(sum3), sum(sum4)]
// TODO: consider doing in 64-bit, allowing 4 more bits of quantization?
inline __m128i Reduce(__m512i sum1, __m512i sum2, __m512i sum3, __m512i sum4) {
  // 1 2 1 2 1 2 1 2 1 2 1 2 1 2 1 2
  __m512i pack12 = _mm512_add_epi32(_mm512_unpackhi_epi32(sum1, sum2), _mm512_unpacklo_epi32(sum1, sum2));
  // 3 4 3 4 3 4 3 4 3 4 3 4 3 4 3 4
  __m512i pack34 = _mm512_add_epi32(_mm512_unpackhi_epi32(sum3, sum4), _mm512_unpacklo_epi32(sum3, sum4));
  // 1 2 3 4 1 2 3 4 1 2 3 4 1 2 3 4
  __m512i pack1234 = _mm512_add_epi32(_mm512_unpackhi_epi64(pack12, pack34), _mm512_unpacklo_epi64(pack12, pack34));
  // Cut the register into halves and sum those.  1 2 3 4 1 2 3 4
  __m256i halves = _mm256_add_epi32(_mm512_castsi512_si256(pack1234), _mm512_extracti64x4_epi64(pack1234, 1));
  // Again: cut the register into halves and sum those. 1 2 3 4
  return _mm_add_epi32(_mm256_castsi256_si128(halves), _mm256_extracti128_si256(halves, 1));
}

// Somewhat inefficient reduce for single __m256i containing int32_t
inline int32_t Reduce(__m256i halves) {
  IntAccess a;
  a.as_n = _mm_add_epi32(_mm256_castsi256_si128(halves), _mm256_extracti128_si256(halves, 1));
  // TODO is there a more efficient way?
  return a.as_i[0] + a.as_i[1] + a.as_i[2] + a.as_i[3];
}

// Somewhat inefficient reduce for single __m512i containing int32_t
inline int32_t Reduce(__m512i sum1) {
  // Fold register over itself.
  return Reduce(_mm256_add_epi32(_mm512_castsi512_si256(sum1), _mm512_extracti64x4_epi64(sum1, 1)));
}

class ScatterPut {
  public:
    explicit ScatterPut(float unquant_mult, int num_B_rows)
      : unquant_mult_(_mm_set1_ps(unquant_mult)),
#ifdef __AVX512VL__
       num_b_rows_scatter_(_mm_set_epi32(num_B_rows * 3 * sizeof(float), num_B_rows * 2 * sizeof(float), num_B_rows * 1 * sizeof(float), num_B_rows * 0 * sizeof(float)))
#else
       num_B_rows_(num_B_rows)
#endif
    {}

    inline void Write(float *base, __m128i reduced) {
      __m128 float_sums = _mm_cvtepi32_ps(reduced);
      float_sums = _mm_mul_ps(float_sums, unquant_mult_);
#ifdef __AVX512VL__
      // The scatter instruction requires avx512vl
      _mm_i32scatter_ps(base, num_b_rows_scatter_, float_sums, 1);
#else
      FloatAccess a;
      // Get floats for each of the sums to write.
      a.as_n = float_sums;
      // Also note that the memory acceses on C are not consecutive, but this is a tradeoff that we have to make.
      // We can't have consecutive accesses of A, B, *and* C. But we access A and B a lot more so it makes
      // sense to do it this way.
      // Scatter to outputs:
      base[0] = a.as_f[0];
      base[num_B_rows] = a.as_f[1];
      base[2*num_B_rows] = a.as_f[2];
      base[3*num_B_rows] = a.as_f[3];
#endif
    }

  private:
    const __m128 unquant_mult_;
#ifdef __AVX512VL__
    const __m128i num_b_rows_scatter_;
#else
    const int num_B_rows_;
#endif
};

} // namespace


// This is an AVX512F implementation of int16_t multiply based on Jacob
// Devlin's SSE code.  The original SSE code was:

// Copyright (c) 2017 Microsoft Corporation

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.


// We are multiplying A * B^T, as opposed to A * B. This is important because it means we can do consecutive memory access on A * B^T which allows to to take the most
// advantage of L1 cache.
// 
// B is typically a weight matrix, so it can be pre-processed offline, and therefore this transpose does not cost anything.
// A is typically an activation minibatch matrix.
// A and B must be 64-byte aligned.
// C should be the usual 4-byte alignment.
void AVX_MatrixMult16(const __m512i * A, const __m512i * B, float * C, float unquant_mult, int num_A_rows, int num_B_rows, int width) {
    assert(width % 32 == 0);
    assert(reinterpret_cast<uintptr_t>(A) % 64 == 0);
    assert(reinterpret_cast<uintptr_t>(B) % 64 == 0);

    ScatterPut put(unquant_mult, num_B_rows);

    const int sse_width = width/32;

    // We do loop unrolling over A. This is *significantly* faster
    // since B can live in the registers. We are assuming that
    // A is a multiple of 4, but we can add extra code to handle values of 1, 2, 3.
    //
    // We could also do loop unrolling over B, which adds some additional speedup.
    // We don't do that for the sake of clarity.
    // 
    // There are other memory access patterns we could do, e.g., put B on the outer loop.
    // The justification is that A is typically small enough that it can live in L1 cache.
    // B is usually a larger weight matrix, so it might not be able to. However, we are using
    // each element of B four times while it's still in a register, so caching is not as important.

    // Round down to a multiple of 4.
    int num_unroll_rows = num_A_rows & ~3;
    for (int i = 0; i < num_unroll_rows; i += 4) {
        const __m512i * A1_row = A + (i+0)*sse_width;
        const __m512i * A2_row = A + (i+1)*sse_width;
        const __m512i * A3_row = A + (i+2)*sse_width;
        const __m512i * A4_row = A + (i+3)*sse_width;

        for (int j = 0; j < num_B_rows; j++) {
            const __m512i * B_row = B + j*sse_width;

            __m512i sum1 = _mm512_setzero_si512();
            __m512i sum2 = _mm512_setzero_si512();
            __m512i sum3 = _mm512_setzero_si512();
            __m512i sum4 = _mm512_setzero_si512();

            // This is just a simple dot product, unrolled four ways.
            for (int k = 0; k < sse_width; k++) {
                __m512i b = *(B_row + k);
                
                __m512i a1 = *(A1_row + k);
                __m512i a2 = *(A2_row + k);
                __m512i a3 = *(A3_row + k);
                __m512i a4 = *(A4_row + k);

                // madd_epi16 does multiply add on 8 16-bit integers and accumulates into a four 32-bit register.
                // E.g.,
                // a1 = [f1, f2, f3, f4, f5, f6, f7, h8] (16-bit ints)
                // b1 = [h1, h2, h3, h4, h5, h6, h7, h8] (16-bit ints)
                // result = [f1*h1 + f2*h2, f3*h3 + f4*h4, f5*h5 + f6*h6, f7*h7 + f8*h8] (32-bit ints)
                // Then add_epi32 just effectively does a += on these 32-bit integers.
                sum1 = _mm512_add_epi32(sum1, _mm512_madd_epi16(b, a1));
                sum2 = _mm512_add_epi32(sum2, _mm512_madd_epi16(b, a2));
                sum3 = _mm512_add_epi32(sum3, _mm512_madd_epi16(b, a3));
                sum4 = _mm512_add_epi32(sum4, _mm512_madd_epi16(b, a4));
            }
            put.Write(C + i * num_B_rows + j, Reduce(sum1, sum2, sum3, sum4));
        }
    }
    // Handle the non-multiples of 4 rows.
    // TODO: efficient version for 3 rows, 2 rows, etc.
    for (int i = num_unroll_rows; i < num_A_rows; ++i) {
      const __m512i * A1_row = A + i * sse_width;
      for (int j = 0; j < num_B_rows; j++) {
        __m512i sum1 = _mm512_setzero_si512();
        for (int k = 0; k < sse_width; k++) {
          const __m512i * B_row = B + j*sse_width;
          __m512i b = *(B_row + k);
          __m512i a1 = *(A1_row + k);
          sum1 = _mm512_add_epi32(sum1, _mm512_madd_epi16(b, a1));
        }
        // TODO is there a more efficient way?
        *(C + (i)*num_B_rows + j) = unquant_mult * static_cast<float>(Reduce(sum1));
      }
    }
}

namespace {

inline __m256i FirstHalf(__m512i val) {
  return _mm512_castsi512_si256(val);
}

inline __m256i SecondHalf(__m512i val) {
  return _mm512_extracti64x4_epi64(val, 1);
}

// Single instruction to concatenate 256s into 512.
inline __m512i Concat(const __m256i first, const __m256i second) {
  return _mm512_inserti64x4(_mm512_castsi256_si512(first), second, 1);
}

inline void Accum(const __m512i ones, __m512i a, const __m512i b, const __m512i b_positive, const __m256i b_second, __m512i &sum) {
  // Apply sign bits.
  __m256i a_first = _mm256_sign_epi8(FirstHalf(a), FirstHalf(b));
  __m256i a_second = _mm256_sign_epi8(SecondHalf(a), b_second);
  // The magic 8-bit multiply then horizontal sum into 16-bit.
  __m512i multiplied = _mm512_maddubs_epi16(b_positive, Concat(a_first, a_second));
  // Now we have 16-bit results that are the sum of two multiplies.
  // Options:
  // - Sum for a few iterations in 16-bit _mm512_adds_epi16
  // - Expand to 32-bit with two _mm512_cvtepi16_epi32 and sum there.
  // - _mm256_hadds_epi16 is a horizontal add and saturate but only does 256-wide.
  // - https://github.com/tesseract-ocr/tesseract/blob/master/src/arch/intsimdmatrixavx2.cpp#L67
  // TODO: try adding to each other then to sum1 for latency reasons.

  // Trick from https://github.com/tesseract-ocr/tesseract/blob/master/src/arch/intsimdmatrixavx2.cpp#L67 under Apache license.
  // This does a horizontal add and expansion into 32-bit.
  multiplied = _mm512_madd_epi16(multiplied, ones);
  sum = _mm512_add_epi32(sum, multiplied);
}

} // namespace

void AVX_MatrixMult8(const __m512i * A, const __m512i * B, float * C, float unquant_mult, int num_A_rows, int num_B_rows, int width) {
  assert(width % 32 == 0);
  assert(reinterpret_cast<uintptr_t>(A) % 64 == 0);
  assert(reinterpret_cast<uintptr_t>(B) % 64 == 0);
  ScatterPut put(unquant_mult, num_B_rows);
  const __m512i ones = _mm512_set1_epi16(1);
  const int sse_width = width/64;
  for (int i = 0; i < num_A_rows; i += 4) {
    const __m512i *A1_row = A + (i+0)*sse_width;
    const __m512i *A2_row = A + (i+1)*sse_width;
    const __m512i *A3_row = A + (i+2)*sse_width;
    const __m512i *A4_row = A + (i+3)*sse_width;
    for (int j = 0; j < num_B_rows; j++) {
      const __m512i *B_row = B + j*sse_width;
      __m512i sum1 = _mm512_setzero_si512();
      __m512i sum2 = _mm512_setzero_si512();
      __m512i sum3 = _mm512_setzero_si512();
      __m512i sum4 = _mm512_setzero_si512();
      for (int k = 0; k < sse_width; k++) {
        __m512i b = *(B_row + k);
        __m256i b_second = SecondHalf(b);
        __m512i b_positive = _mm512_abs_epi8(b);
        Accum(ones, *(A1_row + k), b, b_positive, b_second, sum1);
        Accum(ones, *(A2_row + k), b, b_positive, b_second, sum2);
        Accum(ones, *(A3_row + k), b, b_positive, b_second, sum3);
        Accum(ones, *(A4_row + k), b, b_positive, b_second, sum4);
      }
      put.Write(C + i *num_B_rows + j, Reduce(sum1, sum2, sum3, sum4));
    }
  }
}
