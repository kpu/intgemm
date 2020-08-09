#include "../aligned.h"
#include "../stop_watch.h"
#include "test_matrices.h"
#include "../tile/multiply.h"
#include "../tile/dot.h"

#include <algorithm>
#include <chrono>
#include <functional>
#include <limits>
#include <random>
#include <vector>
#include <stdio.h>

#include <unordered_map>

namespace intgemm {
namespace {

struct Result {
  Index A_rows;
  Index B_cols;
  double timing;
  bool operator<(const Result &other) const {
    return timing < other.timing;
  }
};

template <class Accessor, Index A_rows, Index B_cols> static inline Result BenchmarkNoOverhang(Accessor access, Tile shape) {
  if ((shape.A_rows % A_rows) || (shape.B_cols % B_cols))
    return { A_rows, B_cols, std::numeric_limits<double>::infinity() };
  const std::size_t kTries = 100;
  auto start = std::chrono::steady_clock::now();
  typedef AVX512VNNI::UnrollKernel<A_rows, 1, B_cols, AVX512VNNI::Shifted8> Kernel;
  // Burn in.
  // TODO: different arches, guard against old compilers, etc.
  AVX512VNNI::MultiplyNoOverhang<Kernel>(access, shape);
  for (std::size_t t = 0; t < kTries; ++t) {
    AVX512VNNI::MultiplyNoOverhang<Kernel>(access, shape);
  }
  auto end = std::chrono::steady_clock::now();
  double timing = std::chrono::duration<double>(end - start).count() / kTries;
  return Result { A_rows, B_cols, timing };
}

template <class Accessor> void BenchmarkKernels(Accessor access, Tile shape) {
  printf("problem=%4zux%4zux%4zu\n", shape.A_rows, shape.inner, shape.B_cols);
  std::vector<Result> results;
  results.push_back(BenchmarkNoOverhang<Accessor, 1, 1>(access, shape));
  results.push_back(BenchmarkNoOverhang<Accessor, 2, 1>(access, shape));
  results.push_back(BenchmarkNoOverhang<Accessor, 4, 1>(access, shape));
  results.push_back(BenchmarkNoOverhang<Accessor, 8, 1>(access, shape));
  results.push_back(BenchmarkNoOverhang<Accessor, 16, 1>(access, shape));
  results.push_back(BenchmarkNoOverhang<Accessor, 32, 1>(access, shape));
  results.push_back(BenchmarkNoOverhang<Accessor, 1, 2>(access, shape));
  results.push_back(BenchmarkNoOverhang<Accessor, 2, 2>(access, shape));
  results.push_back(BenchmarkNoOverhang<Accessor, 4, 2>(access, shape));
  results.push_back(BenchmarkNoOverhang<Accessor, 8, 2>(access, shape));
  results.push_back(BenchmarkNoOverhang<Accessor, 16, 2>(access, shape));
  results.push_back(BenchmarkNoOverhang<Accessor, 1, 4>(access, shape));
  results.push_back(BenchmarkNoOverhang<Accessor, 2, 4>(access, shape));
  results.push_back(BenchmarkNoOverhang<Accessor, 4, 4>(access, shape));
  results.push_back(BenchmarkNoOverhang<Accessor, 8, 4>(access, shape));
  results.push_back(BenchmarkNoOverhang<Accessor, 16, 4>(access, shape));
  results.push_back(BenchmarkNoOverhang<Accessor, 1, 8>(access, shape));
  results.push_back(BenchmarkNoOverhang<Accessor, 2, 8>(access, shape));
  results.push_back(BenchmarkNoOverhang<Accessor, 4, 8>(access, shape));
  results.push_back(BenchmarkNoOverhang<Accessor, 8, 8>(access, shape));
  results.push_back(BenchmarkNoOverhang<Accessor, 1, 16>(access, shape));
  results.push_back(BenchmarkNoOverhang<Accessor, 2, 16>(access, shape));
  results.push_back(BenchmarkNoOverhang<Accessor, 4, 16>(access, shape));
  results.push_back(BenchmarkNoOverhang<Accessor, 1, 32>(access, shape));
  results.push_back(BenchmarkNoOverhang<Accessor, 2, 32>(access, shape));
  std::sort(results.begin(), results.end());
  for (const Result &r : results) {
    printf("%9.3fus kernel=%2zux%2zu\n", r.timing * 1000000.0, r.A_rows, r.B_cols);
  }
}

template <class T, unsigned int PackCols, unsigned int RegisterLength> class PackedAccess {
  public:
    typedef T Content;

    PackedAccess(Content *data, Index rows)
      : base_(data), rows_(rows), row_(0), col_(0) {}

    PackedAccess Add(Index row, Index col) const {
      PackedAccess ret(base_, rows_);
      ret.row_ = row_ + row;
      ret.col_ = col_ + col;
      return ret;
    }

    const Content &Front() const {
      const Content *ret = base_ +
        (col_ / PackCols) * rows_ * PackCols +
        (col_ % PackCols) * RegisterLength +
        (row_ / RegisterLength) * RegisterLength * PackCols +
        (row_ % RegisterLength);
      return *ret;
    }

    Content &Front() {
      Content *ret = base_ +
        (col_ / PackCols) * rows_ * PackCols +
        (col_ % PackCols) * RegisterLength +
        (row_ / RegisterLength) * RegisterLength * PackCols +
        (row_ % RegisterLength);
      return *ret;
    }

  private:
    Content *base_;
    Index rows_;

    Index row_, col_;
};

template <unsigned int PackCols> void TryPacked(TestMatrices8 &m, Tile shape) {
  typedef Access<RowMajorAccess<int8_t>, PackedAccess<int8_t, PackCols, 64>, DoNotOptimizeAccess<int32_t> > Packed;
  Packed packed(
      RowMajorAccess<int8_t>(m.A.begin(), shape.inner),
      PackedAccess<int8_t, PackCols, 64>(m.B.begin(), shape.inner),
      DoNotOptimizeAccess<int32_t>());
  printf("packed %u ", PackCols);
  BenchmarkKernels(packed, shape);
}

void Benchmark(Tile shape) {
  TestMatrices8 m(shape);
  typedef Access<RowMajorAccess<int8_t>, ColMajorAccess<int8_t>, DoNotOptimizeAccess<int32_t> > Regular;
  Regular regular(
      RowMajorAccess<int8_t>(m.A.begin(), shape.inner),
      ColMajorAccess<int8_t>(m.B.begin(), shape.inner),
      DoNotOptimizeAccess<int32_t>());
  printf("regular ");
  BenchmarkKernels(regular, shape);
  TryPacked<1>(m, shape);
  TryPacked<2>(m, shape);
  TryPacked<4>(m, shape);
  TryPacked<8>(m, shape);
  TryPacked<16>(m, shape);
  TryPacked<32>(m, shape);
}

} // namespace
} // namespace intgemm

int main() {
  using namespace intgemm;
  intgemm::Tile shapes[] = {
    {8, 256, 256},
    {8, 2048, 256},
    {8, 256, 2048},
    {320, 256, 256},
    {472, 256, 256},
    {248, 256, 256},
    {200, 256, 256},
    // Additional stuff
    {512, 512, 512},
    {1024, 1024, 1024},
    {64, 1024, 1024},
  };
  for (const intgemm::Tile *i = shapes; i < shapes + sizeof(shapes) / sizeof(intgemm::Tile); ++i) {
    intgemm::Benchmark(*i);
  }

/*  intgemm::Tile largest = {0,0,0};
  for (const intgemm::Tile *i = shapes; i < shapes + sizeof(shapes) / sizeof(intgemm::Tile); ++i) {
    largest.A_rows = std::max(largest.A_rows, i->A_rows);
    largest.inner = std::max(largest.inner, i->inner);
    largest.B_cols = std::max(largest.B_cols, i->B_cols);
  }
  intgemm::Memoise memo(largest);
  for (const intgemm::Tile *i = shapes; i < shapes + sizeof(shapes) / sizeof(intgemm::Tile); ++i) {
    memo.Print(*i, 0);
  }*/
}
