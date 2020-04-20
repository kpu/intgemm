#pragma once

#include "intgemm_config.h"
#include "interleave.h"
#include "intrinsics.h"
#include "vec_traits.h"
#include "callbacks.h"

#include <cmath> //sqrt

namespace intgemm {

struct MeanStd {
  float mean;
  float stddev;
};

INTGEMM_SSE2 static inline float MaxFloat32(__m128 a) {
  // Fold to just using the first 64 bits.
  __m128 second_half = _mm_shuffle_ps(a, a, 3 * 4 + 2);
  a = _mm_max_ps(a, second_half);
  // Fold to just using the first 32 bits.
  second_half = _mm_shuffle_ps(a, a, 1);
  a = _mm_max_ps(a, second_half);
  // This casting compiles to nothing.
  return *reinterpret_cast<float*>(&a);
}

INTGEMM_SSE2 static inline dvector_t<CPUType::SSE2, int> PermuteSummer(__m128i pack0123, __m128i pack4567) {
  // No op for 128 bits: already reduced fully.
  return { pack0123, pack4567 };
}

INTGEMM_AVX2 static inline float MaxFloat32(__m256 a) {
  return MaxFloat32(max_ps(_mm256_castps256_ps128(a), _mm256_extractf128_ps(a, 1)));
}

INTGEMM_AVX2 static inline __m256i PermuteSummer(__m256i pack0123, __m256i pack4567) {
  // This instruction generates 1s 2s 3s 4s 5f 6f 7f 8f
  __m256i rev = _mm256_permute2f128_si256(pack0123, pack4567, 0x21);
  // This instruction generates 1f 2f 3f 4f 5s 6s 7s 8s
  __m256i blended = _mm256_blend_epi32(pack0123, pack4567, 0xf0);
  return _mm256_add_epi32(rev, blended);
}

/* https://stackoverflow.com/questions/6996764/fastest-way-to-do-horizontal-float-vector-sum-on-x86 */
INTGEMM_SSSE3 static inline float horizontalSum(__m128 a) {
    __m128 shuf = _mm_movehdup_ps(a);        // broadcast elements 3,1 to 2,0
    __m128 sums = _mm_add_ps(a, shuf);
    shuf        = _mm_movehl_ps(shuf, sums); // high half -> low half
    sums        = _mm_add_ss(sums, shuf);
    return        _mm_cvtss_f32(sums);
}

INTGEMM_AVX2 static inline float horizontalSum(__m256 a) {
    __m128 vlow  = _mm256_castps256_ps128(a);
    __m128 vhigh = _mm256_extractf128_ps(a, 1); // high 128
    vlow  = _mm_add_ps(vlow, vhigh);     // add the low 128
    return horizontalSum(vlow);         // and inline the sse3 version, which is optimal for AVX
}

#ifdef INTGEMM_COMPILER_SUPPORTS_AVX512BW
/* Only INTGEMM_AVX512F is necessary but due to GCC 5.4 bug we have to set INTGEMM_AVX512BW */
INTGEMM_AVX512BW static inline __m256i PermuteSummer(__m512i pack0123, __m512i pack4567) {
  // Form [0th 128-bit register of pack0123, 0st 128-bit register of pack4567, 2nd 128-bit register of pack0123, 2nd 128-bit register of pack4567]
  __m512i mix0 = _mm512_mask_permutex_epi64(pack0123, 0xcc, pack4567, (0 << 4) | (1 << 6));
  // Form [1st 128-bit register of pack0123, 1st 128-bit register of pack4567, 3rd 128-bit register of pack0123, 3rd 128-bit register of pack4567]
  __m512i mix1 = _mm512_mask_permutex_epi64(pack4567, 0x33, pack0123, 2 | (3 << 2));
  __m512i added = _mm512_add_epi32(mix0, mix1);
  // Now we have 0 1 2 3 4 5 6 7 0 1 2 3 4 5 6 7.
  // Fold register over itself.
  return _mm256_add_epi32(_mm512_castsi512_si256(added), _mm512_extracti64x4_epi64(added, 1));
}

// Find the maximum float.
static inline INTGEMM_AVX512F float MaxFloat32(__m512 a) {
  // _mm512_extractf32x8_ps is AVX512DQ but we don't care about masking.
  // So cast to pd, do AVX512F _mm512_extractf64x4_pd, then cast to ps.
  __m256 upper = _mm256_castpd_ps(_mm512_extractf64x4_pd(_mm512_castps_pd(a), 1));
  return MaxFloat32(max_ps(_mm512_castps512_ps256(a), upper));
}

static inline INTGEMM_AVX512F float horizontalSum(__m512 a) {
  __m256 low  = _mm512_castps512_ps256(a);
  __m256 high = _mm256_castpd_ps(_mm512_extractf64x4_pd(_mm512_castps_pd(a),1));
  return horizontalSum(low) + horizontalSum(high);
}

#endif

// Quantize function used for SSSE3 and AVX2.
// Separate function for thread to work around gcc 7 bug that doesn't imbue
// target attributes across #pragma omp parallel.
#define INTGEMM_QUANTIZE_THREAD(target, Register, name) \
target static void QuantizeThread(const float *input, int8_t *output, float quant_mult, std::size_t count) { \
  name::QuantizeTile8 q(quant_mult); \
  _Pragma("omp for") \
  for (std::size_t i = 0; i < count; i += sizeof(Register)) { \
    *reinterpret_cast<Register*>(output + i) = q.Consecutive(input + i); \
  } \
}

#define INTGEMM_QUANTIZE(target, Register, name) \
target static void Quantize(const float *const input, int8_t *const output, float quant_mult, Index size) { \
  assert(reinterpret_cast<uintptr_t>(input) % sizeof(Register) == 0); \
  assert(reinterpret_cast<uintptr_t>(output) % sizeof(Register) == 0); \
  const std::size_t kBatch = sizeof(Register); \
  const std::size_t fast_end = size & ~(kBatch - 1); \
  _Pragma("omp parallel") \
  { \
    QuantizeThread(input, output, quant_mult, fast_end); \
  } \
  std::size_t overhang = size & (kBatch - 1); \
  if (!overhang) return; \
  name::QuantizeTile8 q(quant_mult); \
  /* Each does size(Register) / 32 == kBatch / 4 floats at a time.
   * If we're allowed to read one of them, then we can read the whole register.  */ \
  const float *inputs[4]; \
  std::size_t i; \
  for (i = 0; i < (overhang + (kBatch / 4) - 1) / (kBatch / 4); ++i) { \
    inputs[i] = &input[fast_end + i * (kBatch / 4)]; \
  } \
  /* These will be clipped off. */ \
  for (; i < 4; ++i) { \
    inputs[i] = &input[fast_end]; \
  } \
  Register result = q.Tile(inputs[0], inputs[1], inputs[2], inputs[3]); \
  std::memcpy(output + (size & ~(kBatch - 1)), &result, overhang); \
}

/* Take 4 registers with 32-bit values to be horizontally added.  Reduce them
 * to one register with 32-bit values in the pattern 1 2 3 4 1 2 3 4, leaving
 * the final addition (which crosses 128-bit lanes) to the caller. 
 */
#define INTGEMM_PACK0123(target, Register) \
target inline Register Pack0123(Register sum0, Register sum1, Register sum2, Register sum3) { \
  Interleave32(sum0, sum1); \
  Register pack01 = add_epi32(sum0, sum1); \
  Interleave32(sum2, sum3); \
  Register pack23 = add_epi32(sum2, sum3); \
  Interleave64(pack01, pack23); \
  return add_epi32(pack01, pack23); \
} \

INTGEMM_PACK0123(INTGEMM_SSE2, __m128i)
INTGEMM_PACK0123(INTGEMM_AVX2, __m256i)
#ifdef INTGEMM_COMPILER_SUPPORTS_AVX512BW
/* Only INTGEMM_AVX512F is necessary but due to GCC 5.4 bug we have to set INTGEMM_AVX512BW */
INTGEMM_PACK0123(INTGEMM_AVX512BW, __m512i)
#endif

template <typename Callback>
INTGEMM_SSE2 static inline void RunCallback(Callback& callback_impl, dvector_t<CPUType::SSE2, int> total, Index row_idx, Index col_idx, Index rows, Index cols) {
  callback_impl(total.first, callbacks::OutputBufferInfo(row_idx, col_idx, rows, cols));
  callback_impl(total.second, callbacks::OutputBufferInfo(row_idx, col_idx + 4, rows, cols));
}

template <typename Callback>
INTGEMM_AVX2 static inline void RunCallback(Callback& callback_impl, vector_t<CPUType::AVX2, int> total, Index row_idx, Index col_idx, Index rows, Index cols) {
  callback_impl(total, callbacks::OutputBufferInfo(row_idx, col_idx, rows, cols));
}

// 16-bit multiplier for INTGEMM_SSE2, INTGEMM_AVX2, and AVX512.
// C = A * B * unquant_mult
//
// This has been substantially revised from Jacob Devlin's SSE code which is:
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

// A is a row-major quantized matrix (from PrepareA)
// B is a rearranged quantized matrix (from PrepareB)
// C is output in row-major form.
//
// All of A, B, and C must be in aligned to a multiple of the register size:
// INTGEMM_SSE2: 16 bytes
// INTGEMM_AVX2: 32 bytes
// AVX512: 64 bytes.
//
// A_rows can be anything non-negative.
// width must be a multiple of the register size.
// B_cols must be a multiple of 8.
// Multiply16

// Rewrite that loads of struct to template labdas as soon as c++14 is used
#define INTGEMM_MULTIPLY_INIT_A_LIVES_LOOP_IMPL(target, Register) \
  template <typename Iterator, typename Type> \
  target static void body(const Type* A, Index A_rowidx, Index width, const Register* A_lives[Iterator::total_iterations]) { \
    A_lives[Iterator::template I<0>()] = reinterpret_cast<const Register*>(A + (A_rowidx + Iterator::template I<0>()) * width); \
  }

struct Multiply_InitALivesLoop {
  INTGEMM_MULTIPLY_INIT_A_LIVES_LOOP_IMPL(INTGEMM_SSSE3, __m128i)
  INTGEMM_MULTIPLY_INIT_A_LIVES_LOOP_IMPL(INTGEMM_AVX2, __m256i)
  INTGEMM_MULTIPLY_INIT_A_LIVES_LOOP_IMPL(INTGEMM_AVX512BW, __m512i)
};

#define INTGEMM_MULTIPLY_INIT_SUMS_LOOP_IMPL(target, Register) \
  template <typename Iterator> \
  target static void body(Register sums[Iterator::template N<0>()][Iterator::template N<1>()]) { \
    static constexpr auto Row = Iterator::template I<0>(); \
    static constexpr auto Column = Iterator::template I<1>(); \
    sums[Row][Column] = set1_epi8<Register>(0); \
  }

struct Multiply_InitSumsLoop {
  INTGEMM_MULTIPLY_INIT_SUMS_LOOP_IMPL(INTGEMM_SSSE3, __m128i)
  INTGEMM_MULTIPLY_INIT_SUMS_LOOP_IMPL(INTGEMM_AVX2, __m256i)
  INTGEMM_MULTIPLY_INIT_SUMS_LOOP_IMPL(INTGEMM_AVX512BW, __m512i)
};

template <typename Type>
struct Multiply_TileLoop;

#define INTGEMM_MULTIPLY8_TILE_LOOP_IMPL(target, Register) \
  template <typename Iterator> \
  target static void body(const Register* A_lives[Iterator::template N<0>()], \
                          const Register* b, \
                          Register sums[Iterator::template N<0>()][Iterator::template N<1>()]) { \
    static constexpr auto Row = Iterator::template I<0>(); \
    static constexpr auto Column = Iterator::template I<1>(); \
    sums[Row][Column] = adds_epi16(sums[Row][Column], maddubs_epi16(abs_epi8(*A_lives[Row]), sign_epi8(b[Column], *A_lives[Row]))); \
  }

template <>
struct Multiply_TileLoop<int8_t> {
  INTGEMM_MULTIPLY8_TILE_LOOP_IMPL(INTGEMM_SSSE3, __m128i)
  INTGEMM_MULTIPLY8_TILE_LOOP_IMPL(INTGEMM_AVX2, __m256i)
  // INTGEMM_MULTIPLY8_TILE_LOOP_IMPL(INTGEMM_AVX512BW, __m512i)
};

#define INTGEMM_MULTIPLY16_TILE_LOOP_IMPL(target, Register) \
  template <typename Iterator> \
  target static void body(const Register* A_lives[Iterator::template N<0>()], \
                          const Register* b, \
                          Register sums[Iterator::template N<0>()][Iterator::template N<1>()]) { \
    static constexpr auto Row = Iterator::template I<0>(); \
    static constexpr auto Column = Iterator::template I<1>(); \
    sums[Row][Column] = add_epi32(sums[Row][Column], madd_epi16(*A_lives[Row], b[Column])); \
  }

template <>
struct Multiply_TileLoop<int16_t> {
  INTGEMM_MULTIPLY16_TILE_LOOP_IMPL(INTGEMM_SSSE3, __m128i)
  INTGEMM_MULTIPLY16_TILE_LOOP_IMPL(INTGEMM_AVX2, __m256i)
  INTGEMM_MULTIPLY16_TILE_LOOP_IMPL(INTGEMM_AVX512BW, __m512i)
};

#define INTGEMM_MULTIPLY_INCREASE_A_LIVS_LOOP(target, Register) \
  template <typename Iterator> \
  target static void body(const Register* A_lives[Iterator::total_iterations]) { \
    ++A_lives[Iterator::template I<0>()]; \
  }

struct Multiply_IncreaseALivesLoop {
  INTGEMM_MULTIPLY_INCREASE_A_LIVS_LOOP(INTGEMM_SSSE3, __m128i)
  INTGEMM_MULTIPLY_INCREASE_A_LIVS_LOOP(INTGEMM_AVX2, __m256i)
  INTGEMM_MULTIPLY_INCREASE_A_LIVS_LOOP(INTGEMM_AVX512BW, __m512i)
};

template <typename Type>
struct Multiply_MakeFinalOutputAndRunCallback;

#define INTGEMM_MULTIPLY8_MAKE_FINAL_OUTPUT_AND_RUN_CALLBACK_IMPL(target, Register) \
  template <typename Iterator, typename CallbackImpl> \
  target static void body(Register sums[Iterator::template N<0>()][8 * Iterator::template N<1>()], CallbackImpl callback_impl, Index A_rowidx, Index B_colidx, Index A_rows, Index B_cols) { \
    static constexpr auto Row = Iterator::template I<0>(); \
    static constexpr auto Column8 = Iterator::template I<1>(); \
    sums[Row][8 * Column8 + 0] = madd_epi16(sums[Row][8 * Column8 + 0], set1_epi16<Register>(1)); \
    sums[Row][8 * Column8 + 1] = madd_epi16(sums[Row][8 * Column8 + 1], set1_epi16<Register>(1)); \
    sums[Row][8 * Column8 + 2] = madd_epi16(sums[Row][8 * Column8 + 2], set1_epi16<Register>(1)); \
    sums[Row][8 * Column8 + 3] = madd_epi16(sums[Row][8 * Column8 + 3], set1_epi16<Register>(1)); \
    sums[Row][8 * Column8 + 4] = madd_epi16(sums[Row][8 * Column8 + 4], set1_epi16<Register>(1)); \
    sums[Row][8 * Column8 + 5] = madd_epi16(sums[Row][8 * Column8 + 5], set1_epi16<Register>(1)); \
    sums[Row][8 * Column8 + 6] = madd_epi16(sums[Row][8 * Column8 + 6], set1_epi16<Register>(1)); \
    sums[Row][8 * Column8 + 7] = madd_epi16(sums[Row][8 * Column8 + 7], set1_epi16<Register>(1)); \
    auto pack0123 = Pack0123(sums[Row][8 * Column8 + 0], sums[Row][8 * Column8 + 1], sums[Row][8 * Column8 + 2], sums[Row][8 * Column8 + 3]); \
    auto pack4567 = Pack0123(sums[Row][8 * Column8 + 4], sums[Row][8 * Column8 + 5], sums[Row][8 * Column8 + 6], sums[Row][8 * Column8 + 7]); \
    auto total = PermuteSummer(pack0123, pack4567); \
    RunCallback(callback_impl, total, A_rowidx + Iterator::template I<0>(), B_colidx + 8 * Column8, A_rows, B_cols); \
  }

template <>
struct Multiply_MakeFinalOutputAndRunCallback<int8_t> {
  INTGEMM_MULTIPLY8_MAKE_FINAL_OUTPUT_AND_RUN_CALLBACK_IMPL(INTGEMM_SSSE3, __m128i)
  INTGEMM_MULTIPLY8_MAKE_FINAL_OUTPUT_AND_RUN_CALLBACK_IMPL(INTGEMM_AVX2, __m256i)
  // INTGEMM_MULTIPLY8_MAKE_FINAL_OUTPUT_AND_RUN_CALLBACK_IMPL(INTGEMM_AVX512BW, __m512i)
};

#define INTGEMM_MULTIPLY16_MAKE_FINAL_OUTPUT_AND_RUN_CALLBACK_IMPL(target, Register) \
  template <typename Iterator, typename CallbackImpl> \
  target static void body(const Register sums[Iterator::template N<0>()][8 * Iterator::template N<1>()], CallbackImpl callback_impl, Index A_rowidx, Index B_colidx, Index A_rows, Index B_cols) { \
    static constexpr auto Row = Iterator::template I<0>(); \
    static constexpr auto Column8 = Iterator::template I<1>(); \
    auto pack0123 = Pack0123(sums[Row][8 * Column8 + 0], sums[Row][8 * Column8 + 1], sums[Row][8 * Column8 + 2], sums[Row][8 * Column8 + 3]); \
    auto pack4567 = Pack0123(sums[Row][8 * Column8 + 4], sums[Row][8 * Column8 + 5], sums[Row][8 * Column8 + 6], sums[Row][8 * Column8 + 7]); \
    auto total = PermuteSummer(pack0123, pack4567); \
    RunCallback(callback_impl, total, A_rowidx + Iterator::template I<0>(), B_colidx + 8 * Column8, A_rows, B_cols); \
  }

template <>
struct Multiply_MakeFinalOutputAndRunCallback<int16_t> {
  INTGEMM_MULTIPLY16_MAKE_FINAL_OUTPUT_AND_RUN_CALLBACK_IMPL(INTGEMM_SSSE3, __m128i)
  INTGEMM_MULTIPLY16_MAKE_FINAL_OUTPUT_AND_RUN_CALLBACK_IMPL(INTGEMM_AVX2, __m256i)
  INTGEMM_MULTIPLY16_MAKE_FINAL_OUTPUT_AND_RUN_CALLBACK_IMPL(INTGEMM_AVX512BW, __m512i)
};

#define INTGEMM_MULTIPLY(target, regsiter, cpu_type, integer) \
  template <Index TileRows, Index TileColumnsMultiplier, typename Callback> \
  target static void Multiply(const integer *A, const integer *B, Index A_rows, Index width, Index B_cols, Callback callback) { \
    using Register = regsiter; \
    static constexpr Index TileColumns = 8 * TileColumnsMultiplier; \
    assert(A_rows % TileRows == 0); \
    assert(width % (sizeof(Register) / sizeof(integer)) == 0); \
    assert(B_cols % TileColumns == 0); \
    assert(reinterpret_cast<uintptr_t>(A) % sizeof(Register) == 0); \
    assert(reinterpret_cast<uintptr_t>(B) % sizeof(Register) == 0); \
    \
    const Index simd_width = width / (sizeof(Register) / sizeof(integer)); \
    auto callback_impl = callbacks::CallbackImpl<cpu_type, Callback>(callback); \
    const Register *A_lives[TileRows]; \
    Register sums[TileRows][TileColumns]; \
    \
    /* Process with tile = (TileRows, TileColumns). */ \
    auto *B0_col = reinterpret_cast<const Register*>(B); \
    _Pragma("omp for") \
    for (Index B0_colidx = 0; B0_colidx != B_cols; B0_col += TileColumns * simd_width, B0_colidx += TileColumns) { \
      for (Index A_rowidx = 0; A_rowidx < A_rows; A_rowidx += TileRows) { \
        StaticLoop<Multiply_InitALivesLoop, MakeStaticLoopIterator<TileRows>>(A, A_rowidx, width, A_lives); \
        StaticLoop<Multiply_InitSumsLoop, MakeStaticLoopIterator<TileRows, TileColumns>>(sums); \
        /* Process a tile (use A as the loop variable so the add can be done where gcc likes it for branch prediction. */ \
        auto* B_live = B0_col; \
        for (Index i = 0; i < simd_width; ++i, B_live += TileColumns) { \
          StaticLoop<Multiply_TileLoop<integer>, MakeStaticLoopIterator<TileRows, TileColumns>>(A_lives, B_live, sums); \
          StaticLoop<Multiply_IncreaseALivesLoop, MakeStaticLoopIterator<TileRows>>(A_lives); \
        } \
        StaticLoop<Multiply_MakeFinalOutputAndRunCallback<integer>, MakeStaticLoopIterator<TileRows, TileColumnsMultiplier>>(sums, callback_impl, A_rowidx, B0_colidx, A_rows, B_cols); \
      } \
    } \
  }

//An int8_prepbias version of the above code, using the add 127 technique
#define INTGEMM_PREPAREBIASFOR8(Register, target, cpu_type) \
  template <class Callback> target static void PrepareBias(const int8_t *B, Index width, Index B_cols, Callback callback) { \
  assert(width % (sizeof(Register) / sizeof(int8_t)) == 0); \
  assert(B_cols % 8 == 0); \
  assert(reinterpret_cast<uintptr_t>(B) % sizeof(Register) == 0); \
  const Index simd_width = width / (sizeof(Register) / sizeof(int8_t)); \
  auto callback_impl = callbacks::CallbackImpl<cpu_type, Callback>(callback); \
  const Register a = set1_epi8<Register>(1); \
  _Pragma("omp for") \
  for (Index B0_colidx = 0; B0_colidx < B_cols; B0_colidx += 8) { \
    const Register *B0_col = reinterpret_cast<const Register *>(B) + simd_width * B0_colidx; \
    /*const Register *A_row = reinterpret_cast<const Register*>(A + A_rowidx * width);*/ \
    /* These will be packed 16-bit integers containing sums for each row of B multiplied by the row of A. \
       Iterate over shared (inner) dimension.*/ \
    int k = 0; \
    Register sum0 = maddubs_epi16(a, *(B0_col + k * 8)); \
    Register sum1 = maddubs_epi16(a, *(B0_col + k * 8 + 1)); \
    Register sum2 = maddubs_epi16(a, *(B0_col + k * 8 + 2)); \
    Register sum3 = maddubs_epi16(a, *(B0_col + k * 8 + 3)); \
    Register sum4 = maddubs_epi16(a, *(B0_col + k * 8 + 4)); \
    Register sum5 = maddubs_epi16(a, *(B0_col + k * 8 + 5)); \
    Register sum6 = maddubs_epi16(a, *(B0_col + k * 8 + 6)); \
    Register sum7 = maddubs_epi16(a, *(B0_col + k * 8 + 7)); \
    /* Upcast to 32-bit and horizontally add. Seems a bit faster if this is declared here.*/ \
    Register ones = set1_epi16<Register>(1); \
    sum0 = madd_epi16(sum0, ones); \
    sum1 = madd_epi16(sum1, ones); \
    sum2 = madd_epi16(sum2, ones); \
    sum3 = madd_epi16(sum3, ones); \
    sum4 = madd_epi16(sum4, ones); \
    sum5 = madd_epi16(sum5, ones); \
    sum6 = madd_epi16(sum6, ones); \
    sum7 = madd_epi16(sum7, ones); \
    for (Index k = 1; k < simd_width; ++k) { \
      /*Register a = *(A_row + k);*/ \
      /* Multiply 8-bit, horizontally add to packed 16-bit integers.*/ \
      Register mult0 = maddubs_epi16(a, *(B0_col + k * 8)); \
      Register mult1 = maddubs_epi16(a, *(B0_col + k * 8 + 1)); \
      Register mult2 = maddubs_epi16(a, *(B0_col + k * 8 + 2)); \
      Register mult3 = maddubs_epi16(a, *(B0_col + k * 8 + 3)); \
      Register mult4 = maddubs_epi16(a, *(B0_col + k * 8 + 4)); \
      Register mult5 = maddubs_epi16(a, *(B0_col + k * 8 + 5)); \
      Register mult6 = maddubs_epi16(a, *(B0_col + k * 8 + 6)); \
      Register mult7 = maddubs_epi16(a, *(B0_col + k * 8 + 7)); \
      /* Upcast to 32-bit and horizontally add.*/ \
      mult0 = madd_epi16(mult0, ones); \
      mult1 = madd_epi16(mult1, ones); \
      mult2 = madd_epi16(mult2, ones); \
      mult3 = madd_epi16(mult3, ones); \
      mult4 = madd_epi16(mult4, ones); \
      mult5 = madd_epi16(mult5, ones); \
      mult6 = madd_epi16(mult6, ones); \
      mult7 = madd_epi16(mult7, ones); \
      /*Add in 32bit*/ \
      sum0 = add_epi32(sum0, mult0); \
      sum1 = add_epi32(sum1, mult1); \
      sum2 = add_epi32(sum2, mult2); \
      sum3 = add_epi32(sum3, mult3); \
      sum4 = add_epi32(sum4, mult4); \
      sum5 = add_epi32(sum5, mult5); \
      sum6 = add_epi32(sum6, mult6); \
      sum7 = add_epi32(sum7, mult7); \
      \
    } \
    /* Reduce sums within 128-bit lanes.*/ \
    Register pack0123 = Pack0123(sum0, sum1, sum2, sum3); \
    Register pack4567 = Pack0123(sum4, sum5, sum6, sum7); \
    /*The specific implementation may need to reduce further.*/ \
    auto total = PermuteSummer(pack0123, pack4567); \
    RunCallback(callback_impl, total, 0, B0_colidx, 1, B_cols); \
  } \
} \

//An int8 version of the above code, using the add 127 technique
#define INTGEMM_MULTIPLY8SHIFT(Register, target, cpu_type) \
template <Index TileRows, Index TileColumnsMultiplier, typename Callback> \
target static void Multiply8Shift(const uint8_t *A, const int8_t *B, Index A_rows, Index width, Index B_cols, Callback callback) { \
  assert(width % (sizeof(Register) / sizeof(int8_t)) == 0); \
  assert(B_cols % 8 == 0); \
  assert(reinterpret_cast<uintptr_t>(A) % sizeof(Register) == 0); \
  assert(reinterpret_cast<uintptr_t>(B) % sizeof(Register) == 0); \
  const Index simd_width = width / (sizeof(Register) / sizeof(int8_t)); \
  auto callback_impl = callbacks::CallbackImpl<cpu_type, Callback>(callback); \
  _Pragma("omp for") \
  for (Index B0_colidx = 0; B0_colidx < B_cols; B0_colidx += 8) { \
    const Register *B0_col = reinterpret_cast<const Register *>(B) + simd_width * B0_colidx; \
    /* Process one row of A at a time.  Doesn't seem to be faster to do multiple rows of A at once.*/ \
    for (Index A_rowidx = 0; A_rowidx < A_rows; ++A_rowidx) { \
      const Register *A_row = reinterpret_cast<const Register*>(A + A_rowidx * width); \
      /* These will be packed 16-bit integers containing sums for each row of B multiplied by the row of A. \
         Iterate over shared (inner) dimension.*/ \
      int k = 0; \
      Register a = *(A_row + k); \
      Register sum0 = maddubs_epi16(a, *(B0_col + k * 8)); \
      Register sum1 = maddubs_epi16(a, *(B0_col + k * 8 + 1)); \
      Register sum2 = maddubs_epi16(a, *(B0_col + k * 8 + 2)); \
      Register sum3 = maddubs_epi16(a, *(B0_col + k * 8 + 3)); \
      Register sum4 = maddubs_epi16(a, *(B0_col + k * 8 + 4)); \
      Register sum5 = maddubs_epi16(a, *(B0_col + k * 8 + 5)); \
      Register sum6 = maddubs_epi16(a, *(B0_col + k * 8 + 6)); \
      Register sum7 = maddubs_epi16(a, *(B0_col + k * 8 + 7)); \
      /* Upcast to 32-bit and horizontally add. Seems a bit faster if this is declared here.*/ \
      Register ones = set1_epi16<Register>(1); \
      sum0 = madd_epi16(sum0, ones); \
      sum1 = madd_epi16(sum1, ones); \
      sum2 = madd_epi16(sum2, ones); \
      sum3 = madd_epi16(sum3, ones); \
      sum4 = madd_epi16(sum4, ones); \
      sum5 = madd_epi16(sum5, ones); \
      sum6 = madd_epi16(sum6, ones); \
      sum7 = madd_epi16(sum7, ones); \
      for (Index k = 1; k < simd_width; ++k) { \
        Register a = *(A_row + k); \
        /* Multiply 8-bit, horizontally add to packed 16-bit integers.*/ \
        Register mult0 = maddubs_epi16(a, *(B0_col + k * 8)); \
        Register mult1 = maddubs_epi16(a, *(B0_col + k * 8 + 1)); \
        Register mult2 = maddubs_epi16(a, *(B0_col + k * 8 + 2)); \
        Register mult3 = maddubs_epi16(a, *(B0_col + k * 8 + 3)); \
        Register mult4 = maddubs_epi16(a, *(B0_col + k * 8 + 4)); \
        Register mult5 = maddubs_epi16(a, *(B0_col + k * 8 + 5)); \
        Register mult6 = maddubs_epi16(a, *(B0_col + k * 8 + 6)); \
        Register mult7 = maddubs_epi16(a, *(B0_col + k * 8 + 7)); \
        /* Upcast to 32-bit and horizontally add.*/ \
        mult0 = madd_epi16(mult0, ones); \
        mult1 = madd_epi16(mult1, ones); \
        mult2 = madd_epi16(mult2, ones); \
        mult3 = madd_epi16(mult3, ones); \
        mult4 = madd_epi16(mult4, ones); \
        mult5 = madd_epi16(mult5, ones); \
        mult6 = madd_epi16(mult6, ones); \
        mult7 = madd_epi16(mult7, ones); \
        /*Add in 32bit*/ \
        sum0 = add_epi32(sum0, mult0); \
        sum1 = add_epi32(sum1, mult1); \
        sum2 = add_epi32(sum2, mult2); \
        sum3 = add_epi32(sum3, mult3); \
        sum4 = add_epi32(sum4, mult4); \
        sum5 = add_epi32(sum5, mult5); \
        sum6 = add_epi32(sum6, mult6); \
        sum7 = add_epi32(sum7, mult7); \
         \
      } \
      /* Reduce sums within 128-bit lanes.*/ \
      Register pack0123 = Pack0123(sum0, sum1, sum2, sum3); \
      Register pack4567 = Pack0123(sum4, sum5, sum6, sum7); \
      /*The specific implementation may need to reduce further.*/ \
      auto total = PermuteSummer(pack0123, pack4567); \
      RunCallback(callback_impl, total, A_rowidx, B0_colidx, A_rows, B_cols); \
    } \
  } \
} \

/* Wrap a multiply call in OMP parallelism.  Here it launches threads then
 * inside the implementation there is a pragma omp for.  In gcc >= 8 these
 * could have been the same but older compilers don't imbue target attributes
 * on the hidden function created by pragma omp parallel.
 * 
 * Also, gcc 7 is unable to deduce the function pointer type (for ChooseCPU) if
 * I use typename Backend::Integer directly in the arguments.  As a workaround,
 * have a default template argument Integer then use that so it's resolved.
 */
template <Index TileRows, Index TileColumnsMultiplier, class Backend, class Callback, class Integer = typename Backend::Integer> static inline void OMPParallelWrap(const Integer *A, const Integer *B, Index A_rows, Index width, Index B_cols, Callback callback) {
#pragma omp parallel
  Backend::template Multiply<TileRows, TileColumnsMultiplier, Callback>(A, B, A_rows, width, B_cols, callback);
}

template <Index TileRows, Index TileColumnsMultiplier, class Backend, class Callback> static inline void OMPParallelWrap8Shift(const uint8_t *A, const int8_t *B, Index A_rows, Index width, Index B_cols, Callback callback) {
#pragma omp parallel
  Backend::template Multiply8Shift<TileRows, TileColumnsMultiplier, Callback>(A, B, A_rows, width, B_cols, callback);
}

#define INTGEMM_MAXABSOLUTE(Register, target) \
target static inline float MaxAbsolute(const float *begin_float, const float *end_float) { \
  assert(end_float > begin_float); \
  assert(reinterpret_cast<uintptr_t>(begin_float) % sizeof(Register) == 0); \
  const Register *begin = reinterpret_cast<const Register*>(begin_float); \
  const float *end_reg = end_float - (reinterpret_cast<uintptr_t>(end_float) % sizeof(Register)) / sizeof(float); \
  const Register *end = reinterpret_cast<const Register*>(end_reg); \
  union {float f; int32_t i;} and_convert, float_convert; \
  and_convert.i = 0x7fffffff; \
  Register and_me = set1_ps<Register>(and_convert.f); \
  Register highest = setzero_ps<Register>(); \
  for (; begin < end; ++begin) { \
    Register reg = and_ps(and_me, *begin); \
    highest = max_ps(highest, reg); \
  } \
  float ret = MaxFloat32(highest); \
  /* Overhang: this would be more efficient if done in a single SIMD operation with some zeroing */ \
  for (const float *i = end_reg; i < end_float; ++i) { \
    float_convert.f = *i; \
    float_convert.i &= and_convert.i; \
    ret = std::max(ret, float_convert.f); \
  } \
  return ret; \
} \

#define INTGEMM_GETQUANTIZERSTD(Register, target) \
target static inline MeanStd GetQuantizerStd(const float *begin_float, const float *end_float) { \
  /* Finds a quantizer value that is a certain number of standard deviations of the mean */ \
  assert(end_float > begin_float); \
  assert((end_float - begin_float) % (sizeof(Register) / sizeof(float)) == 0); \
  size_t num_items = end_float - begin_float; \
  const Register *begin = reinterpret_cast<const Register*>(begin_float); \
  const Register *end = reinterpret_cast<const Register*>(end_float); \
  Register squares = set1_ps<Register>(0); \
  Register sums = set1_ps<Register>(0); \
  for (; begin != end; begin++) { \
    squares = add_ps(squares, mul_ps(*begin, *begin)); \
    sums = add_ps(sums, *begin); \
  } \
  float squares_sum = horizontalSum(squares); \
  float normal_sums = horizontalSum(sums); \
  MeanStd ret; \
  ret.mean = normal_sums/num_items; \
  ret.stddev = std::sqrt((squares_sum/num_items) - (ret.mean*ret.mean)); \
  return ret; \
} \

} // namespace intgemm
