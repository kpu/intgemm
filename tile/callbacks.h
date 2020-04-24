#pragma once

#include <type_traits>

#include "../types.h"
#include "../kernels.h"

namespace intgemm {

class WriteCallback {
  public:
    struct Config {};

    // TODO: SLOW.  This is here for testing.
    template <Index A_rows, Index B_cols, typename Access> static void Run(Access access, const __m128i *from, const Config&) { Slow<A_rows, B_cols>(access, reinterpret_cast<const typename Access::Content*>(from)); }
    template <Index A_rows, Index B_cols, typename Access> static void Run(Access access, const __m256i *from, const Config&) { Slow<A_rows, B_cols>(access, reinterpret_cast<const typename Access::Content*>(from)); }
    template <Index A_rows, Index B_cols, typename Access> static void Run(Access access, const __m512i *from, const Config&) {
      RunImpl<A_rows, B_cols, B_cols>(access, from);
    }

  private:
    // If there's a full register to write for a column, do that.
    template <Index A_rows, Index B_cols, Index ColRemain, typename Access> INTGEMM_AVX512BW
      static typename std::enable_if<A_rows && B_cols && (ColRemain >= 16)>::type
      RunImpl(Access access, const __m512i *from) {
      _mm512_storeu_si512(&access.Front(), *from);
      RunImpl<A_rows, B_cols, (ColRemain - 16)>(access.Add(0, 16), from + 1);
    }

    // TODO: test this more, also save it somewhere!  Make sure compiler isn't recreating this every time.
    template <Index B_cols, Index Off, typename Access> INTGEMM_AVX512BW static inline __m512i Offsets(Access access) {
      const __m512i coefficients = _mm512_set_epi32(
          (Off + 15) / B_cols, (Off + 14) / B_cols, (Off + 13) / B_cols, (Off + 12) / B_cols,
          (Off + 11) / B_cols, (Off + 10) / B_cols, (Off + 9) / B_cols, (Off + 8) / B_cols,
          (Off + 7) / B_cols, (Off + 6) / B_cols, (Off + 5) / B_cols, (Off + 4) / B_cols,
          (Off + 3) / B_cols, (Off + 2) / B_cols, (Off + 1) / B_cols, Off / B_cols);
      const __m512i row_offsets = _mm512_set_epi32(
          (Off + 15) % B_cols, (Off + 14) % B_cols, (Off + 13) % B_cols, (Off + 12) % B_cols,
          (Off + 11) % B_cols, (Off + 10) % B_cols, (Off + 9) % B_cols, (Off + 8) % B_cols,
          (Off + 7) % B_cols, (Off + 6) % B_cols, (Off + 5) % B_cols, (Off + 4) % B_cols,
          (Off + 3) % B_cols, (Off + 2) % B_cols, (Off + 1) % B_cols, Off % B_cols);

      __m512i cols_reg = _mm512_set1_epi32(access.Cols());
      // Multiply by the number of columns for the offsets.
      const __m512i multiplied = _mm512_mullo_epi32(cols_reg, coefficients);
     // These are the offsets to use if we're perfectly aligned at the beginning of a row.
      return _mm512_add_epi32(row_offsets, multiplied);
    }

    // There is a mix of rows in a register and we need a scatter.
    template <Index A_rows, Index B_cols, Index ColRemain, typename Access> INTGEMM_AVX512BW
      static typename std::enable_if<(A_rows > 1) && ColRemain && (ColRemain < 16)>::type
      RunImpl(Access access, const __m512i *from) {
      __m512i offsets = Offsets<B_cols, B_cols - ColRemain>(access);
      // We might be at the end of the data, in which case a mask is needed.
      constexpr Index remaining = (A_rows - 1) * B_cols + ColRemain;
      // Compilers seem to complain a lot about shifting past the end :-(
      constexpr __mmask16 mask = (remaining >= 16) ? 0xffff : (static_cast<__mmask16>(1 << remaining) - 1);
      _mm512_mask_i32scatter_epi32(&access.Front() - (B_cols - ColRemain), mask, offsets, *from, sizeof(int32_t));
      // We just wrote 16 values: ColRemain, the next row (all or partial), possibly the next etc.
      // 16 - ColRemain of the next row and whatever followed.
      constexpr Index Wrote = ((remaining < 16) ? remaining : 16);
      constexpr Index Position = (B_cols - ColRemain) + Wrote;
      // TODO: more testing on this.
      RunImpl<A_rows - (Position / B_cols), B_cols, B_cols - (Position % B_cols)>(access.Add(Position / B_cols, Position % B_cols - (B_cols - ColRemain)), from + 1);
    }

    // At clean end of column, move to next row.
    template <Index A_rows, Index B_cols, Index ColRemain, typename Access> INTGEMM_AVX512BW
      static typename std::enable_if<A_rows && B_cols && (ColRemain == 0)>::type
      RunImpl(Access access, const __m512i *from) {
      RunImpl<A_rows - 1, B_cols, B_cols>(access.Add(1, -B_cols), from);
    }

    // On the last row, finish the last write with a mask.
    template <Index A_rows, Index B_cols, Index ColRemain, typename Access> INTGEMM_AVX512BW
      static typename std::enable_if<(A_rows == 1) && B_cols && (ColRemain < 16 && ColRemain > 0)>::type
      RunImpl(Access access, const __m512i *from) {
      _mm512_mask_storeu_epi32(&access.Front(), (1 << ColRemain) - 1, *from);
    }

    // Nothing to write.
    template <Index A_rows, Index B_cols, Index ColRemain, typename Access> INTGEMM_AVX512BW
      static typename std::enable_if<!A_rows || !B_cols>::type
      RunImpl(Access, const __m512i *) {}

    template <Index A_rows, Index B_cols, typename Access> static void Slow(Access access, const typename Access::Content *from) {
      for (Index i = 0; i < A_rows; ++i) {
        for (Index j = 0; j < B_cols; ++j) {
          (&access.Front())[i * access.Cols() + j] = from[i * B_cols + j];
        }
      }
    }
};

class UnquantizeAndWriteCallback {
  public:
    struct Config {
      float unquant_mult;
    };

    // TODO: SLOW.  This is here for testing.
    template <Index A_rows, Index B_cols, typename Access> static void Run(Access access, const __m128i *from, const Config& config) {
      Slow<A_rows, B_cols>(access, reinterpret_cast<const typename Access::Content*>(from), config.unquant_mult);
    }

    template <Index A_rows, Index B_cols, typename Access> static void Run(Access access, const __m256i *from, const Config& config) {
      Slow<A_rows, B_cols>(access, reinterpret_cast<const typename Access::Content*>(from), config.unquant_mult);
    }

    template <Index A_rows, Index B_cols, typename Access> static void Run(Access access, const __m512i *from, const Config& config) {
      RunImpl<A_rows, B_cols, B_cols>(access, from, config.unquant_mult);
    }

  private:
    // If there's a full register to write for a column, do that.
    template <Index A_rows, Index B_cols, Index ColRemain, typename Access> INTGEMM_AVX512BW
      static typename std::enable_if<A_rows && B_cols && (ColRemain >= 16)>::type
      RunImpl(Access access, const __m512i *from, float unquant_mult) {
      _mm512_storeu_ps(&access.Front(), kernels::unquantize(*from, set1_ps<__m512>(unquant_mult)));
      RunImpl<A_rows, B_cols, (ColRemain - 16)>(access.Add(0, 16), from + 1, unquant_mult);
    }

    // TODO: test this more, also save it somewhere!  Make sure compiler isn't recreating this every time.
    template <Index B_cols, Index Off, typename Access> INTGEMM_AVX512BW static inline __m512i Offsets(Access access) {
      const __m512i coefficients = _mm512_set_epi32(
          (Off + 15) / B_cols, (Off + 14) / B_cols, (Off + 13) / B_cols, (Off + 12) / B_cols,
          (Off + 11) / B_cols, (Off + 10) / B_cols, (Off + 9) / B_cols, (Off + 8) / B_cols,
          (Off + 7) / B_cols, (Off + 6) / B_cols, (Off + 5) / B_cols, (Off + 4) / B_cols,
          (Off + 3) / B_cols, (Off + 2) / B_cols, (Off + 1) / B_cols, Off / B_cols);
      const __m512i row_offsets = _mm512_set_epi32(
          (Off + 15) % B_cols, (Off + 14) % B_cols, (Off + 13) % B_cols, (Off + 12) % B_cols,
          (Off + 11) % B_cols, (Off + 10) % B_cols, (Off + 9) % B_cols, (Off + 8) % B_cols,
          (Off + 7) % B_cols, (Off + 6) % B_cols, (Off + 5) % B_cols, (Off + 4) % B_cols,
          (Off + 3) % B_cols, (Off + 2) % B_cols, (Off + 1) % B_cols, Off % B_cols);

      __m512i cols_reg = _mm512_set1_epi32(access.Cols());
      // Multiply by the number of columns for the offsets.
      const __m512i multiplied = _mm512_mullo_epi32(cols_reg, coefficients);
     // These are the offsets to use if we're perfectly aligned at the beginning of a row.
      return _mm512_add_epi32(row_offsets, multiplied);
    }

    // There is a mix of rows in a register and we need a scatter.
    template <Index A_rows, Index B_cols, Index ColRemain, typename Access> INTGEMM_AVX512BW
      static typename std::enable_if<(A_rows > 1) && ColRemain && (ColRemain < 16)>::type
      RunImpl(Access access, const __m512i *from, float unquant_mult) {
      __m512i offsets = Offsets<B_cols, B_cols - ColRemain>(access);
      // We might be at the end of the data, in which case a mask is needed.
      constexpr Index remaining = (A_rows - 1) * B_cols + ColRemain;
      // Compilers seem to complain a lot about shifting past the end :-(
      constexpr __mmask16 mask = (remaining >= 16) ? 0xffff : (static_cast<__mmask16>(1 << remaining) - 1);
      _mm512_mask_i32scatter_ps(&access.Front() - (B_cols - ColRemain), mask, offsets, kernels::unquantize(*from, set1_ps<__m512>(unquant_mult)), sizeof(float));
      // We just wrote 16 values: ColRemain, the next row (all or partial), possibly the next etc.
      // 16 - ColRemain of the next row and whatever followed.
      constexpr Index Wrote = ((remaining < 16) ? remaining : 16);
      constexpr Index Position = (B_cols - ColRemain) + Wrote;
      // TODO: more testing on this.
      RunImpl<A_rows - (Position / B_cols), B_cols, B_cols - (Position % B_cols)>(access.Add(Position / B_cols, Position % B_cols - (B_cols - ColRemain)), from + 1, unquant_mult);
    }

    // At clean end of column, move to next row.
    template <Index A_rows, Index B_cols, Index ColRemain, typename Access> INTGEMM_AVX512BW
      static typename std::enable_if<A_rows && B_cols && (ColRemain == 0)>::type
      RunImpl(Access access, const __m512i *from, float unquant_mult) {
      RunImpl<A_rows - 1, B_cols, B_cols>(access.Add(1, -B_cols), from, unquant_mult);
    }

    // On the last row, finish the last write with a mask.
    template <Index A_rows, Index B_cols, Index ColRemain, typename Access> INTGEMM_AVX512BW
      static typename std::enable_if<(A_rows == 1) && B_cols && (ColRemain < 16 && ColRemain > 0)>::type
      RunImpl(Access access, const __m512i *from, float unquant_mult) {
      _mm512_mask_storeu_ps(&access.Front(), (1 << ColRemain) - 1, kernels::unquantize(*from, set1_ps<__m512>(unquant_mult)));
    }

    // Nothing to write.
    template <Index A_rows, Index B_cols, Index ColRemain, typename Access> INTGEMM_AVX512BW
      static typename std::enable_if<!A_rows || !B_cols>::type
      RunImpl(Access, const __m512i *, float) {}

    template <Index A_rows, Index B_cols, typename Access> static void Slow(Access access, const typename Access::Content *from, float unquant_mult) {
      for (Index i = 0; i < A_rows; ++i) {
        for (Index j = 0; j < B_cols; ++j) {
          (&access.Front())[i * access.Cols() + j] = from[i * B_cols + j] * unquant_mult;
        }
      }
    }
};

} // namespace intgemm
