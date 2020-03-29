#pragma once

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
