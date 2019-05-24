#include "3rd_party/catch.hpp"
#include "postprocess.h"

#include <numeric>

namespace intgemm {

INTGEMM_SSE2 TEST_CASE("ReLU SSE2",) {
  if (kCPU < CPU_SSE2)
    return;

  float raw_input[8];
  std::iota(raw_input, raw_input + 8, -2);

  RegisterPair128 input;
  input.pack0123 = *reinterpret_cast<__m128*>(raw_input);
  input.pack4567 = *reinterpret_cast<__m128*>(raw_input + 4);

  auto postproc = PostprocessImpl<ReLU, CPUType::CPU_SSE2>(ReLU());
  auto output = postproc.run(input);
  auto raw_output = reinterpret_cast<float*>(&output);

  CHECK(raw_output[0] == 0.f); // input = -2
  CHECK(raw_output[1] == 0.f); // input = -1
  CHECK(raw_output[2] == 0.f); // input =  0
  CHECK(raw_output[3] == 1.f); // input =  1
  CHECK(raw_output[4] == 2.f); // input =  2
  CHECK(raw_output[5] == 3.f); // input =  3
  CHECK(raw_output[6] == 4.f); // input =  4
  CHECK(raw_output[7] == 5.f); // input =  5
}

INTGEMM_AVX2 TEST_CASE("ReLU AVX2",) {
  if (kCPU < CPU_AVX2)
    return;

  float raw_input[8];
  std::iota(raw_input, raw_input + 8, -4);

  auto input = *reinterpret_cast<__m256*>(raw_input);
  auto postproc = PostprocessImpl<ReLU, CPUType::CPU_AVX2>(ReLU());
  auto output = postproc.run(input);
  auto raw_output = reinterpret_cast<float*>(&output);

  CHECK(raw_output[0] == 0.f); // input = -4
  CHECK(raw_output[1] == 0.f); // input = -3
  CHECK(raw_output[2] == 0.f); // input = -2
  CHECK(raw_output[3] == 0.f); // input = -1
  CHECK(raw_output[4] == 0.f); // input =  0
  CHECK(raw_output[5] == 1.f); // input =  1
  CHECK(raw_output[6] == 2.f); // input =  2
  CHECK(raw_output[7] == 3.f); // input =  3
}

#ifndef INTGEMM_NO_AVX512

INTGEMM_AVX512BW TEST_CASE("ReLU AVX512",) {
  if (kCPU < CPU_AVX512BW)
    return;

  float raw_input[16];
  std::iota(raw_input, raw_input + 16, -8);

  auto input = *reinterpret_cast<__m512*>(raw_input);
  auto postproc = PostprocessImpl<ReLU, CPUType::CPU_AVX512BW>(ReLU());
  auto output = postproc.run(input);
  auto raw_output = reinterpret_cast<float*>(&output);

  CHECK(raw_output[0]  == 0.f); // input = -8
  CHECK(raw_output[1]  == 0.f); // input = -7
  CHECK(raw_output[2]  == 0.f); // input = -6
  CHECK(raw_output[3]  == 0.f); // input = -5
  CHECK(raw_output[4]  == 0.f); // input = -4
  CHECK(raw_output[5]  == 0.f); // input = -3
  CHECK(raw_output[6]  == 0.f); // input = -2
  CHECK(raw_output[7]  == 0.f); // input = -1
  CHECK(raw_output[8]  == 0.f); // input =  0
  CHECK(raw_output[9]  == 1.f); // input =  1
  CHECK(raw_output[10] == 2.f); // input =  2
  CHECK(raw_output[11] == 3.f); // input =  3
  CHECK(raw_output[12] == 4.f); // input =  4
  CHECK(raw_output[13] == 5.f); // input =  5
  CHECK(raw_output[14] == 6.f); // input =  6
  CHECK(raw_output[15] == 7.f); // input =  7
}

#endif

}
