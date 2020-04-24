#pragma once

#include <type_traits>

#include "../types.h"
#include "callbacks.h"

namespace intgemm {

// See also: RegisterRowMajorAccess is RowMajorAccess<Register> but without the
// compiler warning.  That is defined in dot.h.
template <class T, class Callback> class CallbackRowMajorAccess {
  public:
    typedef T Content;

    CallbackRowMajorAccess(Content *data, Index cols, const typename Callback::Config& callback_config = {})
      : data_(data), cols_(cols), callback_config_(callback_config) {}

    CallbackRowMajorAccess<Content, Callback> Add(Index row, Index col) const {
      return CallbackRowMajorAccess<Content, Callback>(data_ + row * cols_ + col, cols_, callback_config_);
    }

    Index Cols() const { return cols_; }

    const Content &Front() const { return *data_; }
    Content &Front() { return *data_; }

    template <Index A_rows, Index B_cols>
    void Write(const __m128i *from) {
      Callback::template Run<A_rows, B_cols>(*this, from, callback_config_);
    }
    template <Index A_rows, Index B_cols>
    void Write(const __m256i *from) {
      Callback::template Run<A_rows, B_cols>(*this, from, callback_config_);
    }
    template <Index A_rows, Index B_cols>
    void Write(const __m512i *from) {
      Callback::template Run<A_rows, B_cols>(*this, from, callback_config_);
    }

  private:
    Content *data_;
    Index cols_;
    typename Callback::Config callback_config_;
};

template <class T>
using RowMajorAccess = CallbackRowMajorAccess<T, WriteCallback>;

template <class T>
using UnquantizeAndWriteRowMajorAccess = CallbackRowMajorAccess<T, UnquantizeAndWriteCallback>;

template <class T>
using UnquantizeAndAddBiasAndWriteRowMajorAccess = CallbackRowMajorAccess<T, UnquantizeAndAddBiasAndWriteCallback>;

template <class T> class ColMajorAccess {
  public:
    typedef T Content;

    ColMajorAccess(Content *data, Index rows)
      : data_(data), rows_(rows) {}

    ColMajorAccess<Content> Add(Index row, Index col) const {
      return ColMajorAccess<Content>(data_ + row + col * rows_, rows_);
    }

    Index Rows() const { return rows_; }

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
