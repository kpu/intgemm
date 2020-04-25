#pragma once

#include "../aligned.h"
#include "../tile/access.h"
#include <random>
// Yes both due to debacle.
#include <stdint.h>
#include <cstdint>

namespace intgemm {

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

struct TestMatricesUnquantizeAndWriteRowMajorAccess {
  typedef Access<RowMajorAccess<int8_t>, ColMajorAccess<int8_t>, UnquantizeAndWriteRowMajorAccess<float>> AccessT;

  explicit TestMatricesUnquantizeAndWriteRowMajorAccess(Tile shape_in, float unquant_mult) :
    shape(shape_in),
    A(shape.A_rows * shape.inner),
    B(shape.inner * shape.B_cols),
    C(shape.A_rows * shape.B_cols),
    unquant_mult_(unquant_mult) {

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
      UnquantizeAndWriteRowMajorAccess<float>(C.begin(), shape.B_cols, {unquant_mult_}));
  }

  Tile shape;
  AlignedVector<int8_t> A;
  AlignedVector<int8_t> B;
  // Uninitialized; for using tests to write to.
  AlignedVector<float> C;

private:
  float unquant_mult_;
};

struct TestMatricesUnquantizeAndAddBiasAndWriteRowMajorAccess {
  typedef Access<RowMajorAccess<int8_t>, ColMajorAccess<int8_t>, UnquantizeAndAddBiasAndWriteRowMajorAccess<float>> AccessT;

  explicit TestMatricesUnquantizeAndAddBiasAndWriteRowMajorAccess(Tile shape_in, float unquant_mult) :
    shape(shape_in),
    A(shape.A_rows * shape.inner),
    B(shape.inner * shape.B_cols),
    bias(shape.B_cols),
    C(shape.A_rows * shape.B_cols),
    unquant_mult_(unquant_mult) {

    std::mt19937 gen;
    std::uniform_int_distribution<int8_t> dist(-127,127);
    for (int8_t &it : A) it = dist(gen);
    for (int8_t &it : B) it = dist(gen);

    std::uniform_real_distribution<float> distf(-10.0f, 10.0f);
    for (auto& it : bias) it = distf(gen);
    // C is uninitialized.
  }

  AccessT Accessor() {
    return AccessT(
      RowMajorAccess<int8_t>(A.begin(), shape.inner),
      ColMajorAccess<int8_t>(B.begin(), shape.inner),
      UnquantizeAndAddBiasAndWriteRowMajorAccess<float>(C.begin(), shape.B_cols, {unquant_mult_, bias.begin(), C.begin()}));
  }

  Tile shape;
  AlignedVector<int8_t> A;
  AlignedVector<int8_t> B;
  AlignedVector<float> bias;
  // Uninitialized; for using tests to write to.
  AlignedVector<float> C;

private:
  float unquant_mult_;
};

} // namespace intgemm
