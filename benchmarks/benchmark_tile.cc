#include "../aligned.h"
#include "../stop_watch.h"
#include "../test/test_matrices.h"
#include "../tile/multiply.h"
#include "../tile/dot.h"

#include <chrono>
#include <functional>
#include <limits>
#include <random>
#include <vector>
#include <stdio.h>

#include <unordered_map>

namespace intgemm {
namespace {

typedef TestMatrices8::AccessT Accessor;

// What do we do with the overhang?  Include in the bottom, include in right, or separate call?
enum Overhang {
  BOTTOM,
  RIGHT,
  SEPARATE,
  NONE
};

/* Entry in the memoisation table.  It has three levels of validity:
 * Blank:
 *   direct_time == std::numeric_limits<double>::infinity()
 *   recursive_time == std::numeric_limits<double>::infinity()
 * Direct evaluated, but not splits:
 *   direct_time valid
 *   recursive_time == std::numeric_limits<double>::infinity()
 *   This is useful e.g. if the size is being considered for big evaluation.
 * Full:
 *   direct_time valid
 *   recursive_time valid
     It's been evaluated with splitting into big, right, bottom, and bottom right corner etc
 */
struct BestConfig {
  // Big matrix size.
  Index big_A_rows;
  Index big_B_cols;
  // The kernel used for the big matrix size.
  Index kernel_A_rows;
  Index kernel_B_cols;
  Overhang overhang;
  double recursive_time { std::numeric_limits<double>::infinity() };
  double direct_time { std::numeric_limits<double>::infinity() };

  void AbsorbIndirect(BestConfig other) {
    if (other.recursive_time < recursive_time) {
      double direct_save = direct_time;
      *this = other;
      direct_time = direct_save;
    }
  }

  void Zero() {
    big_A_rows = 0;
    big_B_cols = 0;
    kernel_A_rows = 0;
    kernel_B_cols = 0;
    overhang = NONE;
    recursive_time = 0.0;
    direct_time = 0.0;
  }

  char OverhangDescription() const {
    switch (overhang) {
      case BOTTOM:
        return 'B';
      case RIGHT:
        return 'R';
      case SEPARATE:
        return 'S';
      case NONE:
        return 'N';
      default:
        std::cerr << "Bad overhang?" << std::endl;
        return 'F';
    }
  }
};

template <Index A_rows, Index B_cols> static inline double BenchmarkNoOverhang(Accessor access, Tile shape) {
  if ((shape.A_rows % A_rows) || (shape.B_cols % B_cols))
    return std::numeric_limits<double>::infinity();
  const std::size_t kTries = 20;
  auto start = std::chrono::steady_clock::now();
  typedef AVX512VNNI::UnrollKernel<A_rows, 1, B_cols, AVX512VNNI::Shifted8> Kernel;
  // Burn in.
  // TODO: different arches, guard against old compilers, etc.
  AVX512VNNI::MultiplyNoOverhang<Accessor, Kernel>(access, shape);
  for (std::size_t t = 0; t < kTries; ++t) {
    AVX512VNNI::MultiplyNoOverhang<Accessor, Kernel>(access, shape);
  }
  auto end = std::chrono::steady_clock::now();
  return std::chrono::duration<double>(end - start).count() / kTries;
}

template <Index A_rows, Index B_cols> static void BenchmarkAndUpdate(Accessor access, Tile shape, BestConfig &out) {
  double time = BenchmarkNoOverhang<A_rows, B_cols>(access, shape);
  if (time < out.direct_time) {
    out.big_A_rows = shape.A_rows;
    out.big_B_cols = shape.B_cols;
    out.kernel_A_rows = A_rows;
    out.kernel_B_cols = B_cols;
    out.direct_time = time;
    out.overhang = NONE;
  }
}

// Size of inner loop to sweep.
constexpr Index kColsMax = 16;
constexpr Index kRowsMax = 16;

template <std::size_t... Iterator> static inline BestConfig BenchmarkKernels(TestMatrices8::AccessT accessor, Tile shape, index_sequence<Iterator...>) {
  BestConfig ret;
  using unfurl = int[];
  // Could have used return values and built an array but the indices were annoying to handle.
  (void)unfurl {0, (
    BenchmarkAndUpdate<(Iterator / kColsMax) + 1, (Iterator % kColsMax) + 1>(accessor, shape, ret)
    , 0)...
  };
  ret.recursive_time = ret.direct_time;
  return ret;
}

struct TileHash {
  std::size_t operator()(const Tile &t) const noexcept {
    std::hash<Index> hasher;
    std::size_t seed = hasher(t.A_rows);
    seed ^= hasher(t.inner) + 0x9e3779b9 + (seed<<6) + (seed>>2);
    seed ^= hasher(t.B_cols) + 0x9e3779b9 + (seed<<6) + (seed>>2);
    return seed;
  }
};

class Memoise {
  public:
    explicit Memoise(Tile max_problem) :
      matrices_(max_problem) {
      // We'll just assume zero-size matrices take zero time.
      zero_.Zero();
    }

    BestConfig Find(const Tile problem) {
      BestConfig &best = Entry(problem);
      const Index A_rows = problem.A_rows;
      const Index inner = problem.inner;
      const Index B_cols = problem.B_cols;
      if (best.recursive_time != std::numeric_limits<double>::infinity()) return best;
      // No overhang.
      best = BenchmarkKernels(matrices_.Accessor(), problem, make_index_sequence<kColsMax * kRowsMax>());
      for (Index row_overhang = 0; row_overhang < std::min(kRowsMax, A_rows); ++row_overhang) {
        // Don't visit (0,0).
        for (Index col_overhang = (row_overhang == 0 ? 1 : 0); col_overhang < std::min(kColsMax, B_cols); ++col_overhang) {
          // This option does full recursion on breaking up big.
          //BestConfig big = Find(A_rows - row_overhang, B_cols - col_overhang);
          // This option assumes we run big directly.
          BestConfig &big = Entry({A_rows - row_overhang, problem.inner, B_cols - col_overhang});
          if (big.direct_time == std::numeric_limits<double>::infinity()) {
            big = BenchmarkKernels(matrices_.Accessor(), Tile{A_rows - row_overhang, problem.inner, B_cols - col_overhang}, make_index_sequence<kColsMax * kRowsMax>());
          }
          
          BestConfig bottom_long = Find({row_overhang, inner, B_cols});
          BestConfig bottom_short = Find({row_overhang, inner, B_cols - col_overhang});
          BestConfig right_long = Find({A_rows, inner, col_overhang});
          BestConfig right_short = Find({A_rows - row_overhang, inner, col_overhang});
          BestConfig bottom_right = Find({row_overhang, inner, col_overhang});
          BestConfig candidate;
          candidate.big_A_rows = A_rows - row_overhang;
          candidate.big_B_cols = B_cols - col_overhang;
          // Record kernel from big tile
          candidate.kernel_A_rows = big.kernel_A_rows;
          candidate.kernel_B_cols = big.kernel_B_cols;

          candidate.recursive_time = big.direct_time + bottom_long.recursive_time + right_short.recursive_time;
          candidate.overhang = BOTTOM;
          best.AbsorbIndirect(candidate);

          candidate.recursive_time = big.direct_time + bottom_short.recursive_time + right_long.recursive_time;
          candidate.overhang = RIGHT;
          best.AbsorbIndirect(candidate);

          candidate.recursive_time = big.direct_time + bottom_short.recursive_time + right_short.recursive_time + bottom_right.recursive_time;
          candidate.overhang = SEPARATE;
          best.AbsorbIndirect(candidate);
        }
      }
      fprintf(stderr, "%8.3fus %8.3fus problem=%4zux%4zux%4zu big=%4zux%4zu bigkernel=%2zux%2zu overhang=%c\n", best.recursive_time * 1000000.0, best.direct_time * 1000000.0, problem.A_rows, problem.inner, problem.B_cols, best.big_A_rows, best.big_B_cols, best.kernel_A_rows, best.kernel_B_cols, best.OverhangDescription());
      return best;
    }

    void Print(Tile problem, unsigned int nest) {
      BestConfig best = Find(problem);
      printf("%8.3fus %8.3fus problem=%4zux%4zux%4zu big=%4zux%4zu bigkernel=%2zux%2zu overhang=%c", best.recursive_time * 1000000.0, best.direct_time * 1000000.0, problem.A_rows, problem.inner, problem.B_cols, best.big_A_rows, best.big_B_cols, best.kernel_A_rows, best.kernel_B_cols, best.OverhangDescription());
      for (unsigned int i = 0; i < nest; ++i) {
        printf("*");
      }
      printf("\n");

      if (best.overhang != NONE) {
        Print({best.big_A_rows, problem.inner, best.big_B_cols}, nest + 1);
        if (problem.A_rows != best.big_A_rows) {
          Print({problem.A_rows - best.big_A_rows, problem.inner, best.overhang == BOTTOM ? problem.B_cols : best.big_B_cols}, nest + 1);
        }
        if (problem.B_cols != best.big_B_cols) { 
          Print({best.overhang == RIGHT ? problem.A_rows : best.big_A_rows, problem.inner, problem.B_cols - best.big_B_cols}, nest + 1);
        }
        if (best.overhang == SEPARATE) {
          Print({problem.A_rows - best.big_A_rows, problem.inner, problem.B_cols - best.big_B_cols}, nest + 1);
        }
      }
    }
 
  private:
    BestConfig &Entry(Tile size) {
      // Nobody should modify zero because it is already valid.
      if (size.A_rows == 0 || size.B_cols == 0) { return zero_; }
      return table_[size];
    }

    std::unordered_map<Tile, BestConfig, TileHash> table_;

    BestConfig zero_;

    TestMatrices8 matrices_;
};

} // namespace
} // namespace intgemm

int main() {
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
  intgemm::Tile largest = {0,0,0};
  for (const intgemm::Tile *i = shapes; i < shapes + sizeof(shapes) / sizeof(intgemm::Tile); ++i) {
    largest.A_rows = std::max(largest.A_rows, i->A_rows);
    largest.inner = std::max(largest.inner, i->inner);
    largest.B_cols = std::max(largest.B_cols, i->B_cols);
  }
  intgemm::Memoise memo(largest);
  for (const intgemm::Tile *i = shapes; i < shapes + sizeof(shapes) / sizeof(intgemm::Tile); ++i) {
    memo.Print(*i, 0);
  }
}
