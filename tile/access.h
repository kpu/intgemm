#pragma once

#include <type_traits>

#include "../types.h"

namespace intgemm {

// See also: RegisterRowMajorAccess is RowMajorAccess<Register> but without the
// compiler warning.  That is defined in dot.h.
template <class T> class RowMajorAccess {
  public:
    typedef T Content;

    RowMajorAccess(Content *data, Index cols)
      : data_(data), cols_(cols) {}

    RowMajorAccess<Content> Add(Index row, Index col) const {
      return RowMajorAccess<Content>(data_ + row * cols_ + col, cols_);
    }

    const Content &Front() const { return *data_; }
    Content &Front() { return *data_; }

    // TODO: SLOW.  This is here for testing.
    template <Index A_rows, Index B_cols> void Write(const __m128i *from) { SlowWrite<A_rows, B_cols>(reinterpret_cast<const T*>(from)); }
    template <Index A_rows, Index B_cols> void Write(const __m256i *from) { SlowWrite<A_rows, B_cols>(reinterpret_cast<const T*>(from)); }
    template <Index A_rows, Index B_cols> void Write(const __m512i *from) {
      WriteImpl<A_rows, B_cols, B_cols>(from);
    }

  private:
    // If there's a full register to write for a column, do that.
    template <Index A_rows, Index B_cols, Index CurColumn> INTGEMM_AVX512BW
      typename std::enable_if<A_rows && B_cols && (CurColumn >= 16)>::type
      WriteImpl(const __m512i *from) {
      _mm512_storeu_si512(data_, *from);
      Add(0, 16).template WriteImpl<A_rows, B_cols, (CurColumn - 16)>(from + 1);
    }

    // There is a mix of rows in a register and we need a scatter. Also, we are lucky enough to be at the beginning of a row.
    // TODO case where we are not at the beginning of a register with _mm512_alignr_epi32.
    template <Index A_rows, Index B_cols, Index CurColumn> INTGEMM_AVX512BW
      typename std::enable_if<(A_rows > 1) && CurColumn && (CurColumn < 16) && (CurColumn == B_cols) /* beginning of row */>::type
      WriteImpl(const __m512i *from) {
      // TODO: test this more, also save it somewhere!  Make sure compiler isn't recreating this every time.
      const __m512i coefficients_lo = _mm512_set_epi32(
          15 / B_cols, 14 / B_cols, 13 / B_cols, 12 / B_cols,
          11 / B_cols, 10 / B_cols, 9 / B_cols, 8 / B_cols,
          7 / B_cols, 6 / B_cols, 5 / B_cols, 4 / B_cols,
          3 / B_cols, 2 / B_cols, 1 / B_cols, 0);
      __m512i cols_reg = _mm512_set1_epi32(cols_);
      // Multiply by the number of columns for the offsets.
      const __m512i multiplied = _mm512_mullo_epi32(cols_reg, coefficients_lo);
      // Add row offsets.
      const __m512i row_offsets = _mm512_set_epi32(
          15 % B_cols, 14 % B_cols, 13 % B_cols, 12 % B_cols,
          11 % B_cols, 10 % B_cols, 9 % B_cols, 8 % B_cols,
          7 % B_cols, 6 % B_cols, 5 % B_cols, 4 % B_cols,
          3 % B_cols, 2 % B_cols, 1 % B_cols, 0);
      // These are the offsets to use if we're perfectly aligned: B_cols is a divisor or multiple of 16.
      __m512i offsets = _mm512_add_epi32(row_offsets, multiplied);
      // We might be at the end of the data, in which case a mask is needed.
      constexpr Index remaining = (A_rows - 1) * B_cols + CurColumn;
      _mm512_mask_i32scatter_epi32(data_, (1 << remaining) - 1, offsets, *from, sizeof(int32_t));
      // We just wrote 16 values: CurColumn, the next row (all or partial), possibly the next etc.
      // 16 - CurColumn of the next row and whatever followed.
      constexpr Index WroteMore = ((remaining < 16) ? remaining : 16)  - CurColumn;
      // TODO: testing on this.
      Add(WroteMore / B_cols, WroteMore % B_cols - CurColumn).template WriteImpl<A_rows - WroteMore / B_cols, B_cols, WroteMore % B_cols>(from);
    }

    // At clean end of column, move to next row.
    template <Index A_rows, Index B_cols, Index CurColumn> INTGEMM_AVX512BW
      typename std::enable_if<A_rows && B_cols && (CurColumn == 0)>::type
      WriteImpl(const __m512i *from) {
      Add(1, -B_cols).template WriteImpl<A_rows - 1, B_cols, B_cols>(from);
    }

    // On the last row, finish the last write with a mask.
    template <Index A_rows, Index B_cols, Index CurColumn> INTGEMM_AVX512BW
      typename std::enable_if<(A_rows == 1) && B_cols && (CurColumn < 16 && CurColumn > 0)>::type
      WriteImpl(const __m512i *from) {
      _mm512_mask_storeu_epi32(data_, (1 << CurColumn) - 1, *from);
    }

    // Nothing to write.
    template <Index A_rows, Index B_cols, Index CurColumn> INTGEMM_AVX512BW
      typename std::enable_if<!A_rows || !B_cols>::type
      WriteImpl(const __m512i *) {}

    template <Index A_rows, Index B_cols> void SlowWrite(const T *from) {
      for (Index i = 0; i < A_rows; ++i) {
        for (Index j = 0; j < B_cols; ++j) {
          data_[i * cols_ + j] = from[i * B_cols + j];
        }
      }
    }

    Content *data_;
    Index cols_;
};

template <class T> class ColMajorAccess {
  public:
    typedef T Content;

    ColMajorAccess(Content *data, Index rows)
      : data_(data), rows_(rows) {}

    ColMajorAccess<Content> Add(Index row, Index col) const {
      return ColMajorAccess<Content>(data_ + row + col * rows_, rows_);
    }

    const Content &Front() const { return *data_; }
    Content &Front() { return *data_; }

  private:
    Content *data_;
    Index rows_;
};

template <class AT, class BT, class CT> class Access {
  public:
    typedef AT A;
    typedef BT B;
    typedef CT C;

    typedef typename A::Content AContent;
    typedef typename B::Content BContent;
    typedef typename C::Content CContent;

    Access(A a, B b, C c) : a_(a), b_(b), c_(c) {}

    Access AAdd(Index row, Index col) const {
      return Access(a_.Add(row, col), b_, c_);
    }

    Access BAdd(Index row, Index col) const {
      return Access(a_, b_.Add(row, col), c_);
    }

    Access CAdd(Index row, Index col) const {
      return Access(a_, b_, c_.Add(row, col));
    }

    const A &AAccessor() const { return a_; }
    const B &BAccessor() const { return b_; }
    const C &CAccessor() const { return c_; }
    C &CAccessor() { return c_; }

    AContent &AFront() { return a_.Front(); }
    const AContent &AFront() const { return a_.Front(); }
    BContent &BFront() { return b_.Front(); }
    const BContent &BFront() const { return b_.Front(); }
    CContent &CFront() { return c_.Front(); }
    const CContent &CFront() const { return c_.Front(); }

  private:
    A a_;
    B b_;
    C c_;
};

} // namespace intgemm
