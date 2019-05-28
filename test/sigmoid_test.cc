#include "3rd_party/catch.hpp"
#include "cops.h"

#include <numeric>

#define CHECK_FLOAT(actual, expected, epsilon) \
  do { \
    if (fabs((actual) - (expected)) < epsilon) { SUCCEED(); } \
    else { CHECK((actual) == (expected)); } \
  } while(0)

namespace intgemm {

// INTGEMM_AVX2 TEST_CASE("Sigmoid AVX2",) {
//   if (kCPU < CPU_AVX2)
//     return;

//   const unsigned N = 8;
//   const float quantization_mult = 0.5f;
//   const float error_tolerance = 0.0001f;

//   int32_t raw_input[N];
//   std::iota(raw_input, raw_input + N, -4);

//   auto input = *reinterpret_cast<__m256i*>(raw_input);

//   float output[N];
//   std::fill(output, output + N, 42);

//   auto postproc = Sigmoid::OnAVX2(Sigmoid(output, quantization_mult));
//   postproc(0, 1, 0, input);

//   CHECK_FLOAT(output[0], 0.1192029f, error_tolerance); // input = -4
//   CHECK_FLOAT(output[1], 0.1824255f, error_tolerance); // input = -3
//   CHECK_FLOAT(output[2], 0.2689414f, error_tolerance); // input = -2
//   CHECK_FLOAT(output[3], 0.3775407f, error_tolerance); // input = -1
//   CHECK_FLOAT(output[4], 0.5f      , error_tolerance); // input =  0
//   CHECK_FLOAT(output[5], 0.6224593f, error_tolerance); // input =  1
//   CHECK_FLOAT(output[6], 0.7310586f, error_tolerance); // input =  2
//   CHECK_FLOAT(output[7], 0.8175745f, error_tolerance); // input =  3
// }

}

