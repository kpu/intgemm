#pragma once

#include "intgemm_config.h"
#include "interleave.h"
#include "intrinsics.h"
#include "vec_traits.h"
#include "callbacks.h"

namespace intgemm {

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

#ifdef INTGEMM_COMPILER_SUPPORTS_AVX512
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

#endif

/* Take 4 registers with 32-bit values to be horizontally added.  Reduce them
 * to one register with 32-bit values in the pattern 1 2 3 4 1 2 3 4, leaving
 * the final addition (which crosses 128-bit lanes) to the caller. 
template <class Register> inline Register Pack0123(Register sum0, Register sum1, Register sum2, Register sum3) {
  // 1 2 1 2 1 2 1 2
  Interleave32(sum0, sum1);
  Register pack01 = add_epi32(sum0, sum1);
  // 3 4 3 4 3 4 3 4
  Interleave32(sum2, sum3);
  Register pack23 = add_epi32(sum2, sum3);
  Interleave64(pack01, pack23);
  // 1 2 3 4 1 2 3 4
  return add_epi32(pack01, pack23);
}
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
#ifdef INTGEMM_COMPILER_SUPPORTS_AVX512
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
  target static void body(const Type* A, Index A_rowidx, Index A_rows, Index width, const Register* A_lives[Iterator::total_iterations]) { \
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
    const int simd_width = width / (sizeof(Register) / sizeof(integer)); \
    auto callback_impl = callbacks::CallbackImpl<cpu_type, Callback>(callback); \
    const Register *A_lives[TileRows]; \
    Register sums[TileRows][TileColumns]; \
    \
    /* Process with tile = (TileRows, TileColumns). */ \
    auto *B0_col = reinterpret_cast<const Register*>(B); \
    for (Index B0_colidx = 0; B0_colidx != B_cols; B0_col += TileColumns * simd_width, B0_colidx += TileColumns) { \
      for (Index A_rowidx = 0; A_rowidx < A_rows; A_rowidx += TileRows) { \
        StaticLoop<Multiply_InitALivesLoop, MakeStaticLoopIterator<TileRows>>(A, A_rowidx, A_rows, width, A_lives); \
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
  const int simd_width = width / (sizeof(Register) / sizeof(int8_t)); \
  auto callback_impl = callbacks::CallbackImpl<cpu_type, Callback>(callback); \
  const Register *B0_col = reinterpret_cast<const Register *>(B); \
  const Register a = set1_epi8<Register>(1); \
  for (Index B0_colidx = 0; B0_colidx < B_cols; B0_col += 8 * simd_width, B0_colidx += 8) { \
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
    for (int k = 1; k < simd_width; ++k) { \
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
  const int simd_width = width / (sizeof(Register) / sizeof(int8_t)); \
  auto callback_impl = callbacks::CallbackImpl<cpu_type, Callback>(callback); \
  const Register *B0_col = reinterpret_cast<const Register *>(B); \
  for (Index B0_colidx = 0; B0_colidx < B_cols; B0_col += 8 * simd_width, B0_colidx += 8) { \
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
      for (int k = 1; k < simd_width; ++k) { \
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

#define INTGEMM_MAXABSOLUTE(Register, target) \
target static float MaxAbsolute(const float *begin_float, const float *end_float) { \
  assert(end_float > begin_float); \
  assert((end_float - begin_float) % (sizeof(Register) / sizeof(float)) == 0); \
  const Register *begin = reinterpret_cast<const Register*>(begin_float); \
  const Register *end = reinterpret_cast<const Register*>(end_float); \
  union {float f; int32_t i;} float_convert; \
  float_convert.i = 0x7fffffff; \
  Register and_me = set1_ps<Register>(float_convert.f); \
  Register highest = and_ps(and_me, *begin); \
  for (++begin; begin != end; ++begin) { \
    Register reg = and_ps(and_me, *begin); \
    highest = max_ps(highest, reg); \
  } \
  return MaxFloat32(highest); \
} \

} // namespace intgemm
