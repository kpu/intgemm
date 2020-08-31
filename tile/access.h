#pragma once

#include <type_traits>

#include "../types.h"

namespace intgemm {

template <class T> class RowMajorAccess {
  public:
    typedef T Content;

    RowMajorAccess(Content *data, Index cols)
      : data_(data), cols_(cols) {}

    RowMajorAccess<Content> Add(Index row, Index col) const {
      return RowMajorAccess<Content>(data_ + row * cols_ + col, cols_);
    }

    template <class R = Content> const R &Front() const {
      return reinterpret_cast<const R&>(*data_);
    }

    template <class R = Content> R &Front() const {
      return reinterpret_cast<const R&>(*data_);
    }

  private:
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

    template <class R = Content> const R &Front() const {
      return reinterpret_cast<const R&>(*data_);
    }

    template <class R = Content> R &Front() const {
      return reinterpret_cast<const R&>(*data_);
    }

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

    template <class R = AContent> const R &AFront() const { return a_.Front<R>(); }
    template <class R = BContent> const R &BFront() const { return b_.Front<R>(); }
    template <class R = CContent> const R &CFront() const { return c_.Front<R>(); }
    template <class R = CContent> R &CFront() { return c_.Front<R>(); }

  private:
    A a_;
    B b_;
    C c_;
};

} // namespace intgemm
