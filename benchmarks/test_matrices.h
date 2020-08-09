#pragma once

#include "do_not_optimize.h"
#include "../aligned.h"
#include "../tile/access.h"
#include <random>
// Yes both due to debacle.
#include <stdint.h>
#include <cstdint>

namespace intgemm {

template <class T> class DoNotOptimizeAccess {
  public:
    typedef T Content;

    DoNotOptimizeAccess() {}

    DoNotOptimizeAccess Add(Index, Index) const { return *this; }

    template <Index A_rows, Index B_cols> void Write(const __m128i *from) {
      doNotOptimizeAway(from);
    }
    template <Index A_rows, Index B_cols> void Write(const __m256i *from) {
      doNotOptimizeAway(from);
    }
    template <Index A_rows, Index B_cols> void Write(const __m512i *from) {
      doNotOptimizeAway(from);
    }
};

struct TestMatrices8 {
  typedef Access<RowMajorAccess<int8_t>, ColMajorAccess<int8_t>, RowMajorAccess<int32_t> > AccessT;

  explicit TestMatrices8(Tile shape_in) :
    shape(shape_in),
    A(shape.A_rows * shape.inner),
    B(shape.inner * shape.B_cols),
    C(shape.A_rows * shape.B_cols) {

    std::mt19937 gen;
    std::uniform_int_distribution<int8_t> dist(-127,127);
    for (int8_t &it : A) it = dist(gen);
    for (int8_t &it : B) it = dist(gen);
    // C is uninitialized.
  }

  AccessT Accessor() {
    return AccessT(
      RowMajorAccess<int8_t>(A.begin(), shape.inner),
      ColMajorAccess<int8_t>(B.begin(), shape.inner),
      RowMajorAccess<int32_t>(C.begin(), shape.B_cols));
  }

  Tile shape;
  AlignedVector<int8_t> A;
  AlignedVector<int8_t> B;
  // Uninitialized; for using tests to write to.
  AlignedVector<int32_t> C;
};

} // namespace intgemm
