#include "../intgemm/aligned.h"
#include "intgemm/intgemm_config.h"
#include "../intgemm/avx512_gemm.h"
#include "../intgemm/sse2_gemm.h"
#include "../intgemm/avx2_gemm.h"
#include "../intgemm/ssse3_gemm.h"
#include "../intgemm/intgemm.h"
#include "../intgemm/stats.h"
#include "../intgemm/callbacks.h"
#include "test_matrices.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <random>

#include "../intgemm/aligned.h"
#include "../intgemm/tile/access.h"
#include "../intgemm/tile/dot.h"
#include "../intgemm/tile/multiply.h"
#include "../intgemm/tile/reduce.h"
#include "../test/test.h"
#include "../intgemm/types.h"

namespace intgemm {
namespace {

struct RandomMatrices {
  RandomMatrices(Index A_rows_in, Index width_in, Index B_cols_in) :
    A_rows(A_rows_in), width(width_in), B_cols(B_cols_in),
    A(A_rows * width), B(width * B_cols) {
    std::mt19937 gen;
    //std::uniform_real_distribution<float> dist(-1.f, 1.f);
    std::uniform_int_distribution<int8_t> dist(0,15);
    gen.seed(45678);

    for (auto& it : A) {
      it = dist(gen);
    }
    for (auto& it : B) {
      it = dist(gen);
    }
  }

  const Index A_rows, width, B_cols;
  AlignedVector<int8_t> A, B;
};

// Replicate the saturation behavior of the Signed8 kernel with 16-bit accumulation.
template <class Access> void Signed8ReferenceMult(Access access, Tile problem) {
  using namespace AVX512VNNI; // To get register
  assert(!(problem.inner % 2));
  for (Index a_row = 0; a_row < problem.A_rows; ++a_row) {
    for (Index b_col = 0; b_col < problem.B_cols; ++b_col) {
      Access acc = access.AAdd(a_row, 0).BAdd(0, b_col).CAdd(a_row, b_col);
      // For VNNI, just do it accurately.
#ifdef INTGEMM_THIS_IS_AVX512VNNI
      acc.CFront() = 0;
      for (Index inner = 0; inner < problem.inner; ++inner) {
        Access innermost = acc.AAdd(0, inner).BAdd(inner, 0);
        acc.CFront() += static_cast<int32_t>(innermost.AFront()) * static_cast<int32_t>(innermost.BFront());
      }
#else
      // For non-VNNI, do the saturation stuff.
      int16_t accumulators[sizeof(Register) / sizeof(int16_t)] = {0};
      for (Index inner = 0; inner < problem.inner; inner += 2) {
        Access innermost = acc.AAdd(0, inner).BAdd(inner, 0);
        int32_t product = static_cast<int32_t>(innermost.AFront()) * static_cast<int32_t>(innermost.BFront());
        innermost = innermost.AAdd(0, 1).BAdd(1, 0);
        product += static_cast<int32_t>(innermost.AFront()) * static_cast<int32_t>(innermost.BFront());
        // Saturate to 16-bit for maddubs.
        if (product > 32767) product = 32767;
        if (product < -32768) product = -32768;
        int16_t &accum = accumulators[(inner / 2) % (sizeof(Register) / sizeof(int16_t))];
        // Saturating accumlation.
        product += static_cast<int32_t>(accum);
        if (product > 32767) product = 32767;
        if (product < -32768) product = -32768;
        accum = static_cast<int16_t>(product);
      }
      acc.CFront() = 0;
      for (Index i = 0; i < sizeof(Register) / sizeof(int16_t); ++i) {
        acc.CFront() += static_cast<int32_t>(accumulators[i]);
      }
#endif
    }
  }
}
/*
void DumpMatrix(int8_t *m, Index rows, Index cols) {
  std::cerr << rows << 'x' << cols << '\n';
  for (Index i = 0; i < rows; ++i) {
    for (Index j = 0; j < cols; ++j) {
      std::cerr << (int16_t)m[i * cols + j] << ' ';
    }
    std::cerr << '\n';
  }
}*/

void DumpMatrix(int32_t *m, Index rows, Index cols) {
  std::cerr << rows << 'x' << cols << '\n';
  for (Index i = 0; i < rows; ++i) {
    for (Index j = 0; j < cols; ++j) {
      std::cerr << m[i * cols + j] << ' ';
    }
    std::cerr << '\n';
  }
}

struct TestMatricesRef : TestMatrices8 {
  TestMatricesRef(Tile shape_in) :
    TestMatrices8(shape_in),
    C_reference(shape.A_rows * shape.B_cols) {

    AccessT ref_access(
        RowMajorAccess<int8_t>(A.begin(), shape.inner),
        ColMajorAccess<int8_t>(B.begin(), shape.inner),
        RowMajorAccess<int32_t>(C_reference.begin(), shape.B_cols));
    Signed8ReferenceMult(ref_access, shape);
  }

  AlignedVector<int32_t> C_reference;
};

} // annonymous namespace
} // namespace intgemm

int main() {
  using namespace intgemm;
  using namespace AVX512VNNI;
  Index width = sizeof(Register);//256;
  Index A_rows = 320;
  Index B_cols = 256;
  Tile shape{1, width, 1};

  std::chrono::duration<double> elapsed_seconds_new = std::chrono::duration<double>::zero();
  std::chrono::duration<double> elapsed_seconds_old = std::chrono::duration<double>::zero();
  

  for (int i = 0; i < 10; i++ ) {
    shape.A_rows = A_rows;
    shape.B_cols = B_cols;
    TestMatricesRef t(shape);
    auto start_new = std::chrono::steady_clock::now();
    Multiply<Signed8, 4, 4>(t.Accessor(), shape);
    auto end_new = std::chrono::steady_clock::now();
    bool correct = !memcmp(t.C_reference.begin(), t.C.begin(), A_rows * B_cols * sizeof(int32_t));
    std::cerr << "Correct: " << correct << std::endl;
    if (!correct) {
      std::cerr << "True" << std::endl;
      DumpMatrix(t.C_reference.begin(), A_rows, B_cols);
      std::cerr << "Ours:" << std::endl;
      DumpMatrix(t.C.begin(), A_rows, B_cols);
    }

    //Since the new code requires RowM x ColM, we can't test using the same matrices. Instead, creae a random matrix with the same shape
    RandomMatrices tradMult(A_rows, width, B_cols);
    AlignedVector<int32_t> C_trad(A_rows*B_cols);
    auto start_trad = std::chrono::steady_clock::now();
    intgemm::Int8Shift::Multiply(tradMult.A.begin(), tradMult.B.begin(), A_rows, width, B_cols, intgemm::callbacks::Write<int32_t>(C_trad.begin()));
    auto end_trad = std::chrono::steady_clock::now();

    if (!correct) {
      std::cerr << "Make sure C is written:" << std::endl;
      DumpMatrix(C_trad.begin(), A_rows, B_cols);
    }
    elapsed_seconds_new += end_new - start_new;
    elapsed_seconds_old += end_trad - start_trad;
  }
  std::cerr << "New time: " << elapsed_seconds_new.count() << "s" << std::endl;
  std::cerr << "Old time: " << elapsed_seconds_old.count() << "s" << std::endl;
  return 0;
}
