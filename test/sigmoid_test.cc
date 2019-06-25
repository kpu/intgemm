#include "3rd_party/catch.hpp"
#include "aligned.h"
#include "postprocess.h"

#include <numeric>

#define CHECK_FLOAT(actual, expected, epsilon) \
  do { \
    if (fabs((actual) - (expected)) < epsilon) { SUCCEED(); } \
    else { CHECK((actual) == (expected)); } \
  } while(0)

namespace intgemm {

INTGEMM_AVX2 TEST_CASE("Sigmoid AVX2",) {
  if (kCPU < CPUType::AVX2)
    return;

  const float error_tolerance = 0.001f;

  AlignedVector<float> input(8);
  AlignedVector<float> output(8);

  std::iota(input.begin(), input.end(), -4);

  auto postproc = PostprocessImpl<Sigmoid, CPUType::AVX2>(Sigmoid());
  *output.as<__m256>() = postproc.run(*input.as<__m256>(), 0);

  CHECK_FLOAT(output[0], 0.0179862f, error_tolerance); // input = -4
  CHECK_FLOAT(output[1], 0.0474259f, error_tolerance); // input = -3
  CHECK_FLOAT(output[2], 0.1192029f, error_tolerance); // input = -2
  CHECK_FLOAT(output[3], 0.2689414f, error_tolerance); // input = -1
  CHECK_FLOAT(output[4], 0.5f      , error_tolerance); // input =  0
  CHECK_FLOAT(output[5], 0.7310586f, error_tolerance); // input =  1
  CHECK_FLOAT(output[6], 0.8807970f, error_tolerance); // input =  2
  CHECK_FLOAT(output[7], 0.9525740f, error_tolerance); // input =  3
}

}