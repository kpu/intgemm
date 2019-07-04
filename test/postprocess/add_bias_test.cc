#include "test/test.h"
#include "aligned.h"
#include "postprocess.h"

#include <numeric>

namespace intgemm {

INTGEMM_SSE2 TEST_CASE("AddBias SSE2",) {
  if (kCPU < CPUType::SSE2)
    return;

  AlignedVector<float> input(8);
  AlignedVector<float> bias(8);
  AlignedVector<float> output(8);

  std::iota(input.begin(), input.end(), -2);
  std::iota(bias.begin(), bias.end(), 0);

  auto postproc = PostprocessImpl<AddBias, CPUType::SSE2>(AddBias(bias.begin()));
  auto output_tmp = postproc.run({input.as<__m128>()[0], input.as<__m128>()[1]}, 0, 0);
  output.as<__m128>()[0] = output_tmp.pack0123;
  output.as<__m128>()[1] = output_tmp.pack4567;

  CHECK(output[0] == -2.f); // input = -2, bias = 0
  CHECK(output[1] ==  0.f); // input = -1, bias = 1
  CHECK(output[2] ==  2.f); // input =  0, bias = 2
  CHECK(output[3] ==  4.f); // input =  1, bias = 3
  CHECK(output[4] ==  6.f); // input =  2, bias = 4
  CHECK(output[5] ==  8.f); // input =  3, bias = 5
  CHECK(output[6] == 10.f); // input =  4, bias = 6
  CHECK(output[7] == 12.f); // input =  5, bias = 7
}

INTGEMM_AVX2 TEST_CASE("AddBias AVX2",) {
  if (kCPU < CPUType::AVX2)
    return;

  AlignedVector<float> input(8);
  AlignedVector<float> bias(8);
  AlignedVector<float> output(8);

  std::iota(input.begin(), input.end(), -4);
  std::iota(bias.begin(), bias.end(), 0);

  auto postproc = PostprocessImpl<AddBias, CPUType::AVX2>(AddBias(bias.begin()));
  *output.as<__m256>() = postproc.run(*input.as<__m256>(), 0, 0);

  CHECK(output[0] == -4.f); // input = -4, bias = 0
  CHECK(output[1] == -2.f); // input = -3, bias = 1
  CHECK(output[2] ==  0.f); // input = -2, bias = 2
  CHECK(output[3] ==  2.f); // input = -1, bias = 3
  CHECK(output[4] ==  4.f); // input =  0, bias = 4
  CHECK(output[5] ==  6.f); // input =  1, bias = 5
  CHECK(output[6] ==  8.f); // input =  2, bias = 6
  CHECK(output[7] == 10.f); // input =  3, bias = 7
}

#ifndef INTGEMM_NO_AVX512

INTGEMM_AVX512BW TEST_CASE("AddBias AVX512",) {
  if (kCPU < CPUType::AVX512BW)
    return;

  AlignedVector<float> input(16);
  AlignedVector<float> bias(16);
  AlignedVector<float> output(16);

  std::iota(input.begin(), input.end(), -8);
  std::iota(bias.begin(), bias.end(), 0);

  auto postproc = PostprocessImpl<AddBias, CPUType::AVX512BW>(AddBias(bias.begin()));
  *output.as<__m512>() = postproc.run(*input.as<__m512>(), 0, 0);

  CHECK(output[0]  == -8.f); // input = -8, bias = 0
  CHECK(output[1]  == -6.f); // input = -7, bias = 1
  CHECK(output[2]  == -4.f); // input = -6, bias = 2
  CHECK(output[3]  == -2.f); // input = -5, bias = 3
  CHECK(output[4]  ==  0.f); // input = -4, bias = 4
  CHECK(output[5]  ==  2.f); // input = -3, bias = 5
  CHECK(output[6]  ==  4.f); // input = -2, bias = 6
  CHECK(output[7]  ==  6.f); // input = -1, bias = 7
  CHECK(output[8]  ==  8.f); // input =  0, bias = 8
  CHECK(output[9]  == 10.f); // input =  1, bias = 9
  CHECK(output[10] == 12.f); // input =  2, bias = 10
  CHECK(output[11] == 14.f); // input =  3, bias = 11
  CHECK(output[12] == 16.f); // input =  4, bias = 12
  CHECK(output[13] == 18.f); // input =  5, bias = 13
  CHECK(output[14] == 20.f); // input =  6, bias = 14
  CHECK(output[15] == 22.f); // input =  7, bias = 15
}

#endif

}
