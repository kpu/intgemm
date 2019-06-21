#include "3rd_party/catch.hpp"
#include "aligned.h"
#include "postprocess.h"

#include <numeric>

namespace intgemm {

INTGEMM_SSE2 TEST_CASE("ReLU SSE2",) {
  if (kCPU < CPUType::SSE2)
    return;

  AlignedVector<float> input(8);
  AlignedVector<float> output(8);
  std::iota(input.begin(), input.end(), -2);

  auto postproc = PostprocessImpl<ReLU, CPUType::SSE2>(ReLU());
  auto output_tmp = postproc.run({input.as<__m128>()[0], input.as<__m128>()[1]}, 0);
  output.as<__m128>()[0] = output_tmp.pack0123;
  output.as<__m128>()[1] = output_tmp.pack4567;

  CHECK(output[0] == 0.f); // input = -2
  CHECK(output[1] == 0.f); // input = -1
  CHECK(output[2] == 0.f); // input =  0
  CHECK(output[3] == 1.f); // input =  1
  CHECK(output[4] == 2.f); // input =  2
  CHECK(output[5] == 3.f); // input =  3
  CHECK(output[6] == 4.f); // input =  4
  CHECK(output[7] == 5.f); // input =  5
}

INTGEMM_AVX2 TEST_CASE("ReLU AVX2",) {
  if (kCPU < CPUType::AVX2)
    return;

  AlignedVector<float> input(8);
  AlignedVector<float> output(8);

  std::iota(input.begin(), input.end(), -4);

  auto postproc = PostprocessImpl<ReLU, CPUType::AVX2>(ReLU());
  *output.as<__m256>() = postproc.run(*input.as<__m256>(), 0);

  CHECK(output[0] == 0.f); // input = -4
  CHECK(output[1] == 0.f); // input = -3
  CHECK(output[2] == 0.f); // input = -2
  CHECK(output[3] == 0.f); // input = -1
  CHECK(output[4] == 0.f); // input =  0
  CHECK(output[5] == 1.f); // input =  1
  CHECK(output[6] == 2.f); // input =  2
  CHECK(output[7] == 3.f); // input =  3
}

#ifndef INTGEMM_NO_AVX512

INTGEMM_AVX512BW TEST_CASE("ReLU AVX512",) {
  if (kCPU < CPUType::AVX512BW)
    return;

  AlignedVector<float> input(16);
  AlignedVector<float> output(16);

  std::iota(input.begin(), input.end(), -8);

  auto postproc = PostprocessImpl<ReLU, CPUType::AVX512BW>(ReLU());
  *output.as<__m512>() = postproc.run(*input.as<__m512>(), 0);

  CHECK(output[0]  == 0.f); // input = -8
  CHECK(output[1]  == 0.f); // input = -7
  CHECK(output[2]  == 0.f); // input = -6
  CHECK(output[3]  == 0.f); // input = -5
  CHECK(output[4]  == 0.f); // input = -4
  CHECK(output[5]  == 0.f); // input = -3
  CHECK(output[6]  == 0.f); // input = -2
  CHECK(output[7]  == 0.f); // input = -1
  CHECK(output[8]  == 0.f); // input =  0
  CHECK(output[9]  == 1.f); // input =  1
  CHECK(output[10] == 2.f); // input =  2
  CHECK(output[11] == 3.f); // input =  3
  CHECK(output[12] == 4.f); // input =  4
  CHECK(output[13] == 5.f); // input =  5
  CHECK(output[14] == 6.f); // input =  6
  CHECK(output[15] == 7.f); // input =  7
}

#endif

}
