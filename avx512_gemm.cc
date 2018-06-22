#include "avx512_gemm.h"
#include "interleave.h"
#include "multiply.h"

#include <cassert>
#include <cstddef>
#include <emmintrin.h>
#include <immintrin.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <tmmintrin.h>
#include <xmmintrin.h>

namespace intgemm {

#ifdef __AVX512F__
namespace {

// Load from memory, multiply, and convert to int32_t.
inline __m512i QuantizerGrab(const float *input, const __m512 quant_mult_reg) {
  // Multiply each by the quantization factor.
  __m512 val = _mm512_mul_ps(*reinterpret_cast<const __m512*>(input), quant_mult_reg);
  // Cast to 32-bit int
  return _mm512_cvtps_epi32(val);
}

} // namespace


// AVX512 has combined collapse and store instructions:
// _mm512_mask_cvtsepi32_storeu_epi16
// _mm512_mask_cvtsepi32_storeu_epi8
// So conversion in memory uses these, but I also implement a wider version for
// rearranging B.
// 
// Convert to 16-bit signed integers.
void AVX512_16bit::Quantize(const float *input, int16_t *output, float quant_mult, int size) {
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

// Convert to 8-bit signed integers.
void AVX512_8bit::Quantize(const float *input, int8_t *output, float quant_mult, int size) {
  assert(size % 16 == 0);
  assert(reinterpret_cast<uintptr_t>(input) % 64 == 0);
  const __m512i neg127 = _mm512_set1_epi32(-127);
  const __m512 quant_mult_reg = _mm512_set1_ps(quant_mult);
  const float *end = input + size;
  for (; input < end; input += 16, output += 16) {
    __m512i asint = QuantizerGrab(input, quant_mult_reg);
    asint = _mm512_max_epi32(asint, neg127);
    // There doesn't seem to be an unmasked version.
    _mm512_mask_cvtsepi32_storeu_epi8(output, 0xffff, asint);
  }
}

namespace {

// For PrepareB we want to read 8 columns at a time.  When converting 32-bit
// floats to 8-bit values, that's 32 bytes of floats.  But AVX512 is 64 bytes
// wide so it reads off the edge of the tile.  We could expand the tile size
// but then the memory written to won't be contiguous anyway so we'd be doing a
// scatter anyway.  Easier to just read the 8 columns we wanted as 256 bits
// concatenate.
inline __m512 Concat(const __m256 first, const __m256 second) {
  // AVX512DQ but that goes with AVX512BW anyway.
  return _mm512_insertf32x8(_mm512_castps256_ps512(first), second, 1);
}

// Like QuantizerGrab, but allows 32-byte halves (i.e. 8 columns) to be controlled independently.
inline __m512i QuantizerGrabHalves(const float *input0, const float *input1, const __m512 quant_mult_reg) {
  __m512 appended = Concat(*reinterpret_cast<const __m256*>(input0), *reinterpret_cast<const __m256*>(input1));
  appended = _mm512_mul_ps(appended, quant_mult_reg);
  return _mm512_cvtps_epi32(appended);
}

// These are only used for reshaping due to the AVX512 instructions
// _mm512_mask_cvtsepi32_storeu_epi16 and _mm512_mask_cvtsepi32_storeu_epi8
// being used for the quantizer.
class QuantizeTile16 {
  public:
    typedef __m512i Integer;

    explicit QuantizeTile16(float mult) : mult_reg_(_mm512_set1_ps(mult)) {}

    inline __m512i ForReshape(const float *input, int cols) {
      __m512i g0 = QuantizerGrabHalves(input, input + 16 * cols, mult_reg_);
      __m512i g1 = QuantizerGrabHalves(input + 8 * cols, input + 24 * cols, mult_reg_);
      __m512i packed = _mm512_packs_epi32(g0, g1);
      // Permute within 256-bit lanes, so same as AVX2
      return _mm512_permutex_epi64(packed, 0xd8 /* 0, 2, 1, 3 */);
    }

  private:
    const __m512 mult_reg_;
};

class QuantizeTile8 {
  public:
    typedef __m512i Integer;

    explicit QuantizeTile8(float mult) : mult_reg_(_mm512_set1_ps(mult)) {}

    inline __m512i ForReshape(const float *input, int cols) {
      // TODO: try alternative: _mm512_cvtsepi32_epi8 ?
			const __m512i neg127 = _mm512_set1_epi8(-127);
			// In reverse order: grabbing the first 32-bit values from each 128-bit register, then the second 32-bit values, etc.
			const __m512i shuffle_param = _mm512_set_epi32(15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1, 12, 8, 4, 0);

			// 32-bit format.
			__m512i g0 = QuantizerGrabHalves(input, input + 2 * cols, mult_reg_);
			__m512i g1 = QuantizerGrabHalves(input + 16 * cols, input + 18 * cols, mult_reg_);
			__m512i g2 = QuantizerGrabHalves(input + 32 * cols, input + 34 * cols, mult_reg_);
			__m512i g3 = QuantizerGrabHalves(input + 48 * cols, input + 50 * cols, mult_reg_);
			// Pack 32-bit to 16-bit.
			__m512i packed0 = _mm512_packs_epi32(g0, g1);
			__m512i packed1 = _mm512_packs_epi32(g2, g3);
			// Pack 16-bit to 8-bit.
			__m512i packed = _mm512_packs_epi16(packed0, packed1);
			// Ban -128.
			packed = _mm512_max_epi8(packed, neg127);
			// 0 1 2 3 16 17 18 19 32 33 34 35 48 49 50 51 4 5 6 7 20 21 22 23 36 37 38 39 52 53 54 55 8 9 10 11 24 25 26 27 40 41 42 43 56 57 58 59 12 13 14 15 28 29 30 31 44 45 46 47 60 61 62 63
			return _mm512_permutexvar_epi32(shuffle_param, packed);
		}

  private:
    const __m512 mult_reg_;
};

} // namespace

void AVX512_16bit::PrepareB(const float *input, int16_t *output, float quant_mult, int rows, int cols) {
  PrepareBFor16(input, output, QuantizeTile16(quant_mult), rows, cols);
}

void AVX512_8bit::PrepareB(const float *input, int8_t *output, float quant_mult, int rows, int cols) {
  PrepareBFor8(input, output, QuantizeTile8(quant_mult), rows, cols);
}

void AVX512_16bit::Multiply(const int16_t *A, const int16_t *B, float *C, float unquant_mult, int A_rows, int width, int B_cols) {
  // The unquantization is only 256-bit wide because there are 8 results.
  Multiply16<__m512i, __m256> (A, B, C, unquant_mult, A_rows, width, B_cols);
}

/* Another implementation, but unroll over rows of A.
 * This seems to be faster for large matrices with small widths like 4096x128
 * times 128x4096.
 * It's slightly slower for small matrices.
 * And more annoying because A's rows might not be a multiple of a nice number.
 */
/*
namespace {

inline void Accum(const __m512i zeros, __m512i a, const __m512i b, const __m512i b_positive, const __mmask64 neg_mask, __m512i &sum) {
  // Apply sign bits.
  a = _mm512_mask_sub_epi8(a, neg_mask, zeros, a);
  // The magic 8-bit multiply then horizontal sum into 16-bit.
  __m512i multiplied = _mm512_maddubs_epi16(b_positive, a);
  // Now we have 16-bit results that are the sum of two multiplies.
  // Choosing to approximate and do adds.
  // Perhaps every so often we could accumulate by Convert32Sum
  sum = _mm512_adds_epi16(sum, multiplied);
}

// Two sum version.
struct ReducedPair {
  int32_t result[2];
};
inline ReducedPair Reduce16to32(__m512i sum1, __m512i sum2) {
  Convert32Sum(sum1);
  Convert32Sum(sum2);
  // 1 2 1 2 1 2 1 2 1 2 1 2 1 2 1 2
  __m512i pack12 = _mm512_add_epi32(_mm512_unpackhi_epi32(sum1, sum2), _mm512_unpacklo_epi32(sum1, sum2));
  // 1 2 1 2 1 2 1 2
  __m256i halves = _mm256_add_epi32(_mm512_castsi512_si256(pack12), _mm512_extracti64x4_epi64(pack12, 1));
  // 1 2 1 2
  IntAccess a;
  a.as_n = _mm_add_epi32(_mm256_castsi256_si128(halves), _mm256_extracti128_si256(halves, 1));
  ReducedPair ret;
  ret.result[0] = a.as_i[0] + a.as_i[2];
  ret.result[1] = a.as_i[1] + a.as_i[3];
  return ret;
}

inline int32_t Reduce16to32(__m512i sum1) {
  Convert32Sum(sum1);
  // Fold register over itself.
  return Reduce32(_mm256_add_epi32(_mm512_castsi512_si256(sum1), _mm512_extracti64x4_epi64(sum1, 1)));
}

} // namespace

void MatrixMult8(const __m512i * A, const __m512i * B, float * C, float unquant_mult, int num_A_rows, int num_B_rows, int width) {
  assert(width % 32 == 0);
  assert(reinterpret_cast<uintptr_t>(A) % 64 == 0);
  assert(reinterpret_cast<uintptr_t>(B) % 64 == 0);
  ScatterPut put(unquant_mult, num_B_rows);
  const __m512i zeros = _mm512_setzero_si512();

  const int sse_width = width/64;
  int i = 0;
  int mult8rows = num_A_rows & (~7);

  for (; i < mult8rows; i += 8) {
    const __m512i *A1_row = A + (i+0)*sse_width;
    const __m512i *A2_row = A + (i+1)*sse_width;
    const __m512i *A3_row = A + (i+2)*sse_width;
    const __m512i *A4_row = A + (i+3)*sse_width;
    const __m512i *A5_row = A + (i+4)*sse_width;
    const __m512i *A6_row = A + (i+5)*sse_width;
    const __m512i *A7_row = A + (i+6)*sse_width;
    const __m512i *A8_row = A + (i+7)*sse_width;
    for (int j = 0; j < num_B_rows; j++) {
      const __m512i *B_row = B + j*sse_width;
      __m512i sum1 = _mm512_setzero_si512();
      __m512i sum2 = _mm512_setzero_si512();
      __m512i sum3 = _mm512_setzero_si512();
      __m512i sum4 = _mm512_setzero_si512();
      __m512i sum5 = _mm512_setzero_si512();
      __m512i sum6 = _mm512_setzero_si512();
      __m512i sum7 = _mm512_setzero_si512();
      __m512i sum8 = _mm512_setzero_si512();
      for (int k = 0; k < sse_width; k++) {
        __m512i b = *(B_row + k);
        __m512i b_positive = _mm512_abs_epi8(b);
        // Didn't seem to make a difference definining sign bits here vs at top
        __mmask64 neg_mask = _mm512_test_epi8_mask(b, _mm512_set1_epi8(-128));
        Accum(zeros, *(A1_row + k), b, b_positive, neg_mask, sum1);
        Accum(zeros, *(A2_row + k), b, b_positive, neg_mask, sum2);
        Accum(zeros, *(A3_row + k), b, b_positive, neg_mask, sum3);
        Accum(zeros, *(A4_row + k), b, b_positive, neg_mask, sum4);
        Accum(zeros, *(A5_row + k), b, b_positive, neg_mask, sum5);
        Accum(zeros, *(A6_row + k), b, b_positive, neg_mask, sum6);
        Accum(zeros, *(A7_row + k), b, b_positive, neg_mask, sum7);
        Accum(zeros, *(A8_row + k), b, b_positive, neg_mask, sum8);
      }
      put.Write(C + i *num_B_rows + j, Reduce16to32(sum1, sum2, sum3, sum4));
      put.Write(C + (i+4) *num_B_rows + j, Reduce16to32(sum5, sum6, sum7, sum8));
    }
  }

  const __m512i *A1_row = A + (i+0)*sse_width;
  const __m512i *A2_row = A + (i+1)*sse_width;
  const __m512i *A3_row = A + (i+2)*sse_width;
  const __m512i *A4_row = A + (i+3)*sse_width;
  const __m512i *A5_row = A + (i+4)*sse_width;
  const __m512i *A6_row = A + (i+5)*sse_width;
  const __m512i *A7_row = A + (i+6)*sse_width;
  switch (num_A_rows & 7) {
    case 7:
      for (int j = 0; j < num_B_rows; j++) {
        const __m512i *B_row = B + j*sse_width;
        __m512i sum1 = _mm512_setzero_si512();
        __m512i sum2 = _mm512_setzero_si512();
        __m512i sum3 = _mm512_setzero_si512();
        __m512i sum4 = _mm512_setzero_si512();
        __m512i sum5 = _mm512_setzero_si512();
        __m512i sum6 = _mm512_setzero_si512();
        __m512i sum7 = _mm512_setzero_si512();
        for (int k = 0; k < sse_width; k++) {
          __m512i b = *(B_row + k);
          __m512i b_positive = _mm512_abs_epi8(b);
          __mmask64 neg_mask = _mm512_test_epi8_mask(b, _mm512_set1_epi8(-128));
          Accum(zeros, *(A1_row + k), b, b_positive, neg_mask, sum1);
          Accum(zeros, *(A2_row + k), b, b_positive, neg_mask, sum2);
          Accum(zeros, *(A3_row + k), b, b_positive, neg_mask, sum3);
          Accum(zeros, *(A4_row + k), b, b_positive, neg_mask, sum4);
          Accum(zeros, *(A5_row + k), b, b_positive, neg_mask, sum5);
          Accum(zeros, *(A6_row + k), b, b_positive, neg_mask, sum6);
          Accum(zeros, *(A7_row + k), b, b_positive, neg_mask, sum7);
        }
        put.Write(C + i *num_B_rows + j, Reduce16to32(sum1, sum2, sum3, sum4));
        put.Write(C + (i+4) *num_B_rows + j, Reduce16to32(sum5, sum6));
        put.Write(C + (i+6) *num_B_rows + j, Reduce16to32(sum7));
      }
    case 6:
      for (int j = 0; j < num_B_rows; j++) {
        const __m512i *B_row = B + j*sse_width;
        __m512i sum1 = _mm512_setzero_si512();
        __m512i sum2 = _mm512_setzero_si512();
        __m512i sum3 = _mm512_setzero_si512();
        __m512i sum4 = _mm512_setzero_si512();
        __m512i sum5 = _mm512_setzero_si512();
        __m512i sum6 = _mm512_setzero_si512();
        for (int k = 0; k < sse_width; k++) {
          __m512i b = *(B_row + k);
          __m512i b_positive = _mm512_abs_epi8(b);
          __mmask64 neg_mask = _mm512_test_epi8_mask(b, _mm512_set1_epi8(-128));
          Accum(zeros, *(A1_row + k), b, b_positive, neg_mask, sum1);
          Accum(zeros, *(A2_row + k), b, b_positive, neg_mask, sum2);
          Accum(zeros, *(A3_row + k), b, b_positive, neg_mask, sum3);
          Accum(zeros, *(A4_row + k), b, b_positive, neg_mask, sum4);
          Accum(zeros, *(A5_row + k), b, b_positive, neg_mask, sum5);
          Accum(zeros, *(A6_row + k), b, b_positive, neg_mask, sum6);
        }
        put.Write(C + i *num_B_rows + j, Reduce16to32(sum1, sum2, sum3, sum4));
        put.Write(C + (i+4) *num_B_rows + j, Reduce16to32(sum5, sum6));
      }
    case 5:
      for (int j = 0; j < num_B_rows; j++) {
        const __m512i *B_row = B + j*sse_width;
        __m512i sum1 = _mm512_setzero_si512();
        __m512i sum2 = _mm512_setzero_si512();
        __m512i sum3 = _mm512_setzero_si512();
        __m512i sum4 = _mm512_setzero_si512();
        __m512i sum5 = _mm512_setzero_si512();
        for (int k = 0; k < sse_width; k++) {
          __m512i b = *(B_row + k);
          __m512i b_positive = _mm512_abs_epi8(b);
          __mmask64 neg_mask = _mm512_test_epi8_mask(b, _mm512_set1_epi8(-128));
          Accum(zeros, *(A1_row + k), b, b_positive, neg_mask, sum1);
          Accum(zeros, *(A2_row + k), b, b_positive, neg_mask, sum2);
          Accum(zeros, *(A3_row + k), b, b_positive, neg_mask, sum3);
          Accum(zeros, *(A4_row + k), b, b_positive, neg_mask, sum4);
          Accum(zeros, *(A5_row + k), b, b_positive, neg_mask, sum5);
        }
        put.Write(C + i *num_B_rows + j, Reduce16to32(sum1, sum2, sum3, sum4));
        put.Write(C + (i+4) *num_B_rows + j, Reduce16to32(sum5));
      }
    case 4:
      for (int j = 0; j < num_B_rows; j++) {
        const __m512i *B_row = B + j*sse_width;
        __m512i sum1 = _mm512_setzero_si512();
        __m512i sum2 = _mm512_setzero_si512();
        __m512i sum3 = _mm512_setzero_si512();
        __m512i sum4 = _mm512_setzero_si512();
        for (int k = 0; k < sse_width; k++) {
          __m512i b = *(B_row + k);
          __m512i b_positive = _mm512_abs_epi8(b);
          __mmask64 neg_mask = _mm512_test_epi8_mask(b, _mm512_set1_epi8(-128));
          Accum(zeros, *(A1_row + k), b, b_positive, neg_mask, sum1);
          Accum(zeros, *(A2_row + k), b, b_positive, neg_mask, sum2);
          Accum(zeros, *(A3_row + k), b, b_positive, neg_mask, sum3);
          Accum(zeros, *(A4_row + k), b, b_positive, neg_mask, sum4);
        }
        put.Write(C + i *num_B_rows + j, Reduce16to32(sum1, sum2, sum3, sum4));
      }
    case 3:
      for (int j = 0; j < num_B_rows; j++) {
        const __m512i *B_row = B + j*sse_width;
        __m512i sum1 = _mm512_setzero_si512();
        __m512i sum2 = _mm512_setzero_si512();
        __m512i sum3 = _mm512_setzero_si512();
        for (int k = 0; k < sse_width; k++) {
          __m512i b = *(B_row + k);
          __m512i b_positive = _mm512_abs_epi8(b);
          __mmask64 neg_mask = _mm512_test_epi8_mask(b, _mm512_set1_epi8(-128));
          Accum(zeros, *(A1_row + k), b, b_positive, neg_mask, sum1);
          Accum(zeros, *(A2_row + k), b, b_positive, neg_mask, sum2);
          Accum(zeros, *(A3_row + k), b, b_positive, neg_mask, sum3);
        }
        put.Write(C + i *num_B_rows + j, Reduce16to32(sum1, sum2));
        put.Write(C + (i+2) *num_B_rows + j, Reduce16to32(sum3));
      }
    case 2:
      for (int j = 0; j < num_B_rows; j++) {
        const __m512i *B_row = B + j*sse_width;
        __m512i sum1 = _mm512_setzero_si512();
        __m512i sum2 = _mm512_setzero_si512();
        for (int k = 0; k < sse_width; k++) {
          __m512i b = *(B_row + k);
          __m512i b_positive = _mm512_abs_epi8(b);
          __mmask64 neg_mask = _mm512_test_epi8_mask(b, _mm512_set1_epi8(-128));
          Accum(zeros, *(A1_row + k), b, b_positive, neg_mask, sum1);
          Accum(zeros, *(A2_row + k), b, b_positive, neg_mask, sum2);
        }
        put.Write(C + i *num_B_rows + j, Reduce16to32(sum1, sum2));
      }
    case 1:
      for (int j = 0; j < num_B_rows; j++) {
        const __m512i *B_row = B + j*sse_width;
        __m512i sum1 = _mm512_setzero_si512();
        for (int k = 0; k < sse_width; k++) {
          __m512i b = *(B_row + k);
          __m512i b_positive = _mm512_abs_epi8(b);
          __mmask64 neg_mask = _mm512_test_epi8_mask(b, _mm512_set1_epi8(-128));
          Accum(zeros, *(A1_row + k), b, b_positive, neg_mask, sum1);
        }
        put.Write(C + i *num_B_rows + j, Reduce16to32(sum1));
      }
  }
}*/

#endif // __AVX512__
} // namespace intgemm
