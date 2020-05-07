#pragma once

#include "../aligned.h"
#include "../tile/access.h"
#include <random>
// Yes both due to debacle.
#include <stdint.h>
#include <cstdint>

namespace intgemm {

template <typename AccessA, typename AccessB, typename AccessC>
struct TestMatrices {
  typedef Access<AccessA, AccessB, AccessC> AccessT;

  explicit TestMatrices(Tile shape_in) :
    shape(shape_in),
    A(shape.A_rows * shape.inner),
    B(shape.inner * shape.B_cols),
    C(shape.A_rows * shape.B_cols) {

    std::mt19937 gen;
    std::uniform_int_distribution<typename AccessA::Content> dist(-127,127);
    for (auto &it : A) it = dist(gen);
    for (auto &it : B) it = dist(gen);
    // C is uninitialized.
  }

  AccessT Accessor() {
    return AccessT({A.begin(), shape.inner}, {B.begin(), shape.inner}, {C.begin(), shape.B_cols});
  }

  Tile shape;
  AlignedVector<typename AccessA::Content> A;
  AlignedVector<typename AccessB::Content> B;
  // Uninitialized; for using tests to write to.
  AlignedVector<typename AccessC::Content> C;
};

using TestMatrices8 = TestMatrices<RowMajorAccess<int8_t>, ColMajorAccess<int8_t>, RowMajorAccess<int32_t>>;

} // namespace intgemm
