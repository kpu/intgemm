#include "3rd_party/catch.hpp"
#include "cops.h"

#include <numeric>

namespace intgemm {

INTGEMM_SSE2 TEST_CASE("ReLU SSE2",) {
  if (kCPU < CPU_SSE2)
    return;
  const unsigned N = 4;

  int32_t raw_input[2 * N];
  std::iota(raw_input, raw_input + 2 * N, -2);

  MultiplyResult128 input;
  input.pack0123 = *reinterpret_cast<__m128i*>(raw_input);
  input.pack4567 = *reinterpret_cast<__m128i*>(raw_input + N);

  float output[2 * N];
  std::fill(output, output + 2 * N, 42);

  auto postproc = ReLU::OnSSE2(ReLU(output, 1.f));
  postproc(0, 1, 0, input);

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
  if (kCPU < CPU_AVX2)
    return;

  const unsigned N = 8;

  int32_t raw_input[N];
  std::iota(raw_input, raw_input + N, -4);

  auto input = *reinterpret_cast<__m256i*>(raw_input);

  float output[N];
  std::fill(output, output + N, 42);

  auto postproc = ReLU::OnAVX2(ReLU(output, 1.f));
  postproc(0, 1, 0, input);

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
  if (kCPU < CPU_AVX512BW)
    return;

  const unsigned N = 16;

  int32_t raw_input[N];
  std::iota(raw_input, raw_input + N, -8);

  auto input = *reinterpret_cast<__m512i*>(raw_input);

  float output[N];
  std::fill(output, output + N, 42);

  auto postproc = ReLU::OnAVX512(ReLU(output, 1.f));
  postproc(0, 1, 0, input);

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
