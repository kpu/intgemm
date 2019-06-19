#include "3rd_party/catch.hpp"
#include "postprocess.h"

#include <numeric>

#define CHECK_FLOAT(actual, expected, epsilon) \
  do { \
    if (fabs((actual) - (expected)) < epsilon) { SUCCEED(); } \
    else { CHECK((actual) == (expected)); } \
  } while(0)

namespace intgemm {

INTGEMM_AVX2 TEST_CASE("Tanh AVX2",) {
  if (kCPU < CPUType::AVX2)
    return;

  const float error_tolerance = 0.001f;

  __m256 input;

  { // fill
    auto raw = reinterpret_cast<float*>(&input);
    int n = -4;
    std::generate(raw, raw + 8, [&n] () { return n++ / 4.f; });
  }

  auto postproc = PostprocessImpl<Tanh, CPUType::AVX2>(Tanh());
  auto output = postproc.run(input, 0);
  auto raw_output = reinterpret_cast<float*>(&output);

  CHECK_FLOAT(raw_output[0], -0.7615942f, error_tolerance); // input = -1
  CHECK_FLOAT(raw_output[1], -0.6351490f, error_tolerance); // input = -0.75
  CHECK_FLOAT(raw_output[2], -0.4621172f, error_tolerance); // input = -0.5
  CHECK_FLOAT(raw_output[3], -0.2449187f, error_tolerance); // input = -0.25
  CHECK_FLOAT(raw_output[4],  0.0f      , error_tolerance); // input =  0
  CHECK_FLOAT(raw_output[5],  0.2449187f, error_tolerance); // input =  0.25
  CHECK_FLOAT(raw_output[6],  0.4621172f, error_tolerance); // input =  0.5
  CHECK_FLOAT(raw_output[7],  0.6351490f, error_tolerance); // input =  0.75
}

}
