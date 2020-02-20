#pragma once

#include "intgemm_config.h"

#ifdef INTGEMM_COMPILER_SUPPORTS_AVX512VNNI
#include "avx512_gemm.h"
#include "types.h"

namespace intgemm {

// Rewrite that loads of struct to template labdas as soon as c++14 is used
struct AVX512VNNI_Multiply_InitALivesLoop {
  template <typename Iterator, typename Type>
  INTGEMM_AVX512VNNI static void body(const Type* A, Index A_rowidx, Index A_rows, Index width, const __m512i* A_lives[Iterator::total_iterations]) {
    A_lives[Iterator::template I<0>()] = reinterpret_cast<const __m512i*>(A + (A_rowidx + Iterator::template I<0>()) * width);
  }
};

struct AVX512VNNI_Multiply_InitSumsLoop {
  template <typename Iterator>
  INTGEMM_AVX512VNNI static void body(__m512i sums[Iterator::template N<0>()][Iterator::template N<1>()]) {
    static constexpr auto Row = Iterator::template I<0>();
    static constexpr auto Column = Iterator::template I<1>();
    sums[Row][Column] = setzero_si<__m512i>();
  }
};

struct AVX512VNNI_Multiply_TileLoop {
  template <typename Iterator>
  INTGEMM_AVX512VNNI static void body(const __m512i* A_lives[Iterator::template N<0>()],
                          const __m512i* B_live,
                          __m512i sums[Iterator::template N<0>()][Iterator::template N<1>()]) {
    static constexpr auto Row = Iterator::template I<0>();
    static constexpr auto Column = Iterator::template I<1>();
    auto neg_mask = _mm512_test_epi8_mask(*A_lives[Row], _mm512_set1_epi8(-128));
    sums[Row][Column] = _mm512_dpbusds_epi32(sums[Row][Column], _mm512_abs_epi8(*A_lives[Row]), _mm512_mask_sub_epi8(B_live[Column], neg_mask, setzero_si<__m512i>(), B_live[Column]));
  }
};

struct AVX512VNNI_Multiply_IncreaseALivesLoop {
  template <typename Iterator>
  INTGEMM_AVX512VNNI static void body(const __m512i* A_lives[Iterator::total_iterations]) {
    ++A_lives[Iterator::template I<0>()];
  }
};

struct AVX512VNNI_Multiply_MakeFinalOutputAndRunCallback {
  template <typename Iterator, typename CallbackImpl>
  INTGEMM_AVX512VNNI static void body(__m512i sums[Iterator::template N<0>()][8 * Iterator::template N<1>()], CallbackImpl callback_impl, Index A_rowidx, Index B_colidx, Index A_rows, Index B_cols) {
    static constexpr auto Row = Iterator::template I<0>();
    static constexpr auto Column8 = Iterator::template I<1>();
    auto pack0123 = Pack0123(sums[Row][8 * Column8 + 0], sums[Row][8 * Column8 + 1], sums[Row][8 * Column8 + 2], sums[Row][8 * Column8 + 3]);
    auto pack4567 = Pack0123(sums[Row][8 * Column8 + 4], sums[Row][8 * Column8 + 5], sums[Row][8 * Column8 + 6], sums[Row][8 * Column8 + 7]);
    auto total = PermuteSummer(pack0123, pack4567);
    RunCallback(callback_impl, total, A_rowidx + Iterator::template I<0>(), B_colidx + 8 * Column8, A_rows, B_cols);
  }
};

struct AVX512VNNI_8bit : public AVX512_8bit {
  template <Index TileRows, Index TileColumnsMultiplier, typename Callback>
  INTGEMM_AVX512VNNI static void Multiply(const int8_t *A, const int8_t *B, Index A_rows, Index width, Index B_cols, Callback callback) {
    static constexpr Index TileColumns = 8 * TileColumnsMultiplier;
    assert(A_rows % TileRows == 0);
    assert(width % sizeof(__m512i) == 0);
    assert(B_cols % TileColumns == 0);
    assert(reinterpret_cast<uintptr_t>(A) % sizeof(__m512i) == 0);
    assert(reinterpret_cast<uintptr_t>(B) % sizeof(__m512i) == 0);

    const int simd_width = width / sizeof(__m512i);
    auto callback_impl = callbacks::CallbackImpl<CPUType::AVX2, Callback>(callback);
    const __m512i *A_lives[TileRows];
    __m512i sums[TileRows][TileColumns];

    /* Process with tile = (TileRows, TileColumns). */
    auto *B0_col = reinterpret_cast<const __m512i*>(B);
    for (Index B0_colidx = 0; B0_colidx != B_cols; B0_col += TileColumns * simd_width, B0_colidx += TileColumns) {
      for (Index A_rowidx = 0; A_rowidx < A_rows; A_rowidx += TileRows) {
        StaticLoop<AVX512VNNI_Multiply_InitALivesLoop, MakeStaticLoopIterator<TileRows>>(A, A_rowidx, A_rows, width, A_lives);
        StaticLoop<AVX512VNNI_Multiply_InitSumsLoop, MakeStaticLoopIterator<TileRows, TileColumns>>(sums);
        /* Process a tile (use A as the loop variable so the add can be done where gcc likes it for branch prediction. */
        auto* B_live = B0_col;
        for (Index i = 0; i < simd_width; ++i, B_live += TileColumns) {
          StaticLoop<AVX512VNNI_Multiply_TileLoop, MakeStaticLoopIterator<TileRows, TileColumns>>(A_lives, B_live, sums);
          StaticLoop<AVX512VNNI_Multiply_IncreaseALivesLoop, MakeStaticLoopIterator<TileRows>>(A_lives);
        }
        StaticLoop<AVX512VNNI_Multiply_MakeFinalOutputAndRunCallback, MakeStaticLoopIterator<TileRows, TileColumnsMultiplier>>(sums, callback_impl, A_rowidx, B0_colidx, A_rows, B_cols);
      }
    }
  }

  template <Index TileRows, Index TileColumnsMultiplier, typename Callback>
  INTGEMM_AVX512VNNI static void Multiply8Shift(const uint8_t *A, const int8_t *B, Index A_rows, Index width, Index B_cols, Callback callback) {
    typedef __m512i __m512i;
    assert(width % sizeof(__m512i) == 0);
    assert(B_cols % 8 == 0);
    assert(reinterpret_cast<uintptr_t>(A) % sizeof(__m512i) == 0);
    assert(reinterpret_cast<uintptr_t>(B) % sizeof(__m512i) == 0);
    auto callback_impl = callbacks::CallbackImpl<CPUType::AVX2, Callback>(callback);
    const int simd_width = width / sizeof(__m512i);
    const __m512i *B0_col = reinterpret_cast<const __m512i*>(B);
    __m512i zeros = setzero_si<__m512i>();
    // Go over 8 columns of B at a time.
    for (Index B0_colidx = 0; B0_colidx != B_cols; B0_col += 8 * simd_width, B0_colidx += 8) {
      // Process one row of A at a time.  Doesn't seem to be faster to do multiple rows of A at once.
      for (Index A_rowidx = 0; A_rowidx < A_rows; ++A_rowidx) {
        // Iterate over shared (inner) dimension.
        const __m512i *A_live = reinterpret_cast<const __m512i *>(A + A_rowidx * width);
        const __m512i *A_end = A_live + simd_width;
        const __m512i *B_live = B0_col;
        // TODO: separate first step.
        __m512i sum0 = zeros, sum1 = zeros, sum2 = zeros, sum3 = zeros, sum4 = zeros, sum5 = zeros, sum6 = zeros, sum7 = zeros;
        for (; A_live != A_end; ++A_live, B_live += 8) {
          __m512i a = *A_live;
          //MultiplyAdd
          sum0 = _mm512_dpbusds_epi32(sum0, a, *B_live);
          sum1 = _mm512_dpbusds_epi32(sum1, a, *(B_live + 1));
          sum2 = _mm512_dpbusds_epi32(sum2, a, *(B_live + 2));
          sum3 = _mm512_dpbusds_epi32(sum3, a, *(B_live + 3));
          sum4 = _mm512_dpbusds_epi32(sum4, a, *(B_live + 4));
          sum5 = _mm512_dpbusds_epi32(sum5, a, *(B_live + 5));
          sum6 = _mm512_dpbusds_epi32(sum6, a, *(B_live + 6));
          sum7 = _mm512_dpbusds_epi32(sum7, a, *(B_live + 7));
        }
        __m512i pack0123 = Pack0123(sum0, sum1, sum2, sum3);
        __m512i pack4567 = Pack0123(sum4, sum5, sum6, sum7);
        auto total = PermuteSummer(pack0123, pack4567);
        callback_impl(total, callbacks::OutputBufferInfo(A_rowidx, B0_colidx, A_rows, B_cols));
      }
    }
  }

  template <typename Callback>
  INTGEMM_AVX512VNNI static void PrepareBias(const int8_t *B, Index width, Index B_cols, Callback callback) {
    typedef __m512i __m512i;
    assert(width % sizeof(__m512i) == 0);
    assert(B_cols % 8 == 0);
    assert(reinterpret_cast<uintptr_t>(B) % sizeof(__m512i) == 0);
    auto callback_impl = callbacks::CallbackImpl<CPUType::AVX2, Callback>(callback);
    const int simd_width = width / sizeof(__m512i);
    const __m512i *B0_col = reinterpret_cast<const __m512i*>(B);
    __m512i zeros = setzero_si<__m512i>();
    const __m512i a = set1_epi8<__m512i>(1);
    // Go over 8 columns of B at a time.
    for (Index B0_colidx = 0; B0_colidx != B_cols; B0_col += 8 * simd_width, B0_colidx += 8) {
      const __m512i *B_live = B0_col; //In order to make the code look as much as possible as the above function
      const __m512i *B_end = B_live + simd_width*8;

      // TODO: separate first step.
      __m512i sum0 = zeros, sum1 = zeros, sum2 = zeros, sum3 = zeros, sum4 = zeros, sum5 = zeros, sum6 = zeros, sum7 = zeros;
      for (; B_live != B_end; B_live += 8) {
        // Retrieve the conveniently consecutive values of B.
        sum0 = _mm512_dpbusds_epi32(sum0, a, *B_live);
        sum1 = _mm512_dpbusds_epi32(sum1, a, *(B_live + 1));
        sum2 = _mm512_dpbusds_epi32(sum2, a, *(B_live + 2));
        sum3 = _mm512_dpbusds_epi32(sum3, a, *(B_live + 3));
        sum4 = _mm512_dpbusds_epi32(sum4, a, *(B_live + 4));
        sum5 = _mm512_dpbusds_epi32(sum5, a, *(B_live + 5));
        sum6 = _mm512_dpbusds_epi32(sum6, a, *(B_live + 6));
        sum7 = _mm512_dpbusds_epi32(sum7, a, *(B_live + 7));
      }
      __m512i pack0123 = Pack0123(sum0, sum1, sum2, sum3);
      __m512i pack4567 = Pack0123(sum4, sum5, sum6, sum7);
      auto total = PermuteSummer(pack0123, pack4567);
      callback_impl(total, callbacks::OutputBufferInfo(0, B0_colidx, 1, B_cols));
    }
  }

  constexpr static const char *const kName = "8-bit AVX512VNNI";

  static const CPUType kUses = CPUType::AVX512VNNI;
};

} // namespace intgemm

#endif
