#include "test.h"
#include "../utils.h"

namespace intgemm {
namespace {

TEST_CASE("Factorial",) {
  CHECK(factorial(0) == 1);
  CHECK(factorial(1) == 1);
  CHECK(factorial(2) == 2);
  CHECK(factorial(3) == 6);
  CHECK(factorial(4) == 24);

  // Maximum result that fits in unsinged long long
  CHECK(factorial(20) == 2432902008176640000);
}

TEST_CASE("Expi (negative)",) {
  const double eps = 0.0000001;
  CHECK_EPS(expi(-1), 0.3678794411714423, eps);
  CHECK_EPS(expi(-2), 0.1353352832366127, eps);
  CHECK_EPS(expi(-10), 0.0000453999297625, eps);
}

TEST_CASE("Expi (zero)",) {
  const double eps = 0.0000001;
  CHECK_EPS(expi(0), 1.0, eps);
}

TEST_CASE("Expi (positive)",) {
  const double eps = 0.0000001;
  CHECK_EPS(expi(1), 2.7182818284590452, eps);
  CHECK_EPS(expi(2), 7.3890560989306502, eps);
  CHECK_EPS(expi(10), 22026.4657948067165170, eps);
}

struct StaticLoopTest {
  template <typename Iterator>
  static void body(Index& result) {
    result >>= 1;
  }
};

TEST_CASE("Static loop (N = 1)",) {
  Index result = 128;
  StaticLoop<StaticLoopTest, MakeStaticLoopIterator<1>>(result);
  CHECK(result == 64);
}

TEST_CASE("Static loop (N = 7)",) {
  Index result = 128;
  StaticLoop<StaticLoopTest, MakeStaticLoopIterator<7>>(result);
  CHECK(result == 1);
}

struct StaticLoopMultiDimTest {
  template <typename Iterator>
  static void body(Index& result) {
    result = result * 10 + Iterator::template I<0>();
  }
};

TEST_CASE("Static loop with mult-dim iterator (Iterator<1, 1>)",) {
  Index result = 0;
  StaticLoop<StaticLoopMultiDimTest, MakeStaticLoopIterator<1, 1>>(result);
  CHECK(result == 0);
}

TEST_CASE("Static loop with mult-dim iterator (Iterator<1, 5>)",) {
  Index result = 0;
  StaticLoop<StaticLoopMultiDimTest, MakeStaticLoopIterator<1, 5>>(result);
  CHECK(result == 0);
}

TEST_CASE("Static loop with mult-dim iterator (Iterator<5, 2>)",) {
  Index result = 0;
  StaticLoop<StaticLoopMultiDimTest, MakeStaticLoopIterator<5, 2>>(result);
  CHECK(result == 11223344);
}

}
}
