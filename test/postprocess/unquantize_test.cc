#include "test/test.h"
#include "aligned.h"
#include "postprocess.h"

#include <numeric>

namespace intgemm {

INTGEMM_SSE2 TEST_CASE("Unquantize SSE2",) {
  if (kCPU < CPUType::SSE2)
    return;

  AlignedVector<int32_t> input(8);
  AlignedVector<float> output(8);
  std::iota(input.begin(), input.end(), -2);

  auto postproc = PostprocessImpl<Unquantize, CPUType::SSE2>(Unquantize(0.5f));
  auto output_tmp = postproc.run({input.as<__m128i>()[0], input.as<__m128i>()[1]}, 0);
  output.as<__m128>()[0] = output_tmp.pack0123;
  output.as<__m128>()[1] = output_tmp.pack4567;

  CHECK(output[0] == -1.0f); // input = -2
  CHECK(output[1] == -0.5f); // input = -1
  CHECK(output[2] ==  0.0f); // input =  0
  CHECK(output[3] ==  0.5f); // input =  1
  CHECK(output[4] ==  1.0f); // input =  2
  CHECK(output[5] ==  1.5f); // input =  3
  CHECK(output[6] ==  2.0f); // input =  4
  CHECK(output[7] ==  2.5f); // input =  5
}

INTGEMM_AVX2 TEST_CASE("Unquantize AVX2",) {
  if (kCPU < CPUType::AVX2)
    return;

  AlignedVector<int32_t> input(8);
  AlignedVector<float> output(8);

  std::iota(input.begin(), input.end(), -4);

  auto postproc = PostprocessImpl<Unquantize, CPUType::AVX2>(Unquantize(0.5f));
  *output.as<__m256>() = postproc.run(*input.as<__m256i>(), 0);

  CHECK(output[0] == -2.0f); // input = -4
  CHECK(output[1] == -1.5f); // input = -3
  CHECK(output[2] == -1.0f); // input = -2
  CHECK(output[3] == -0.5f); // input = -1
  CHECK(output[4] ==  0.0f); // input =  0
  CHECK(output[5] ==  0.5f); // input =  1
  CHECK(output[6] ==  1.0f); // input =  2
  CHECK(output[7] ==  1.5f); // input =  3
}

#ifndef INTGEMM_NO_AVX512

INTGEMM_AVX512BW TEST_CASE("Unquantize AVX512",) {
  if (kCPU < CPUType::AVX512BW)
    return;

  AlignedVector<int32_t> input(16);
  AlignedVector<float> output(16);

  std::iota(input.begin(), input.end(), -8);

  auto postproc = PostprocessImpl<Unquantize, CPUType::AVX512BW>(Unquantize(0.5f));
  *output.as<__m512>() = postproc.run(*input.as<__m512i>(), 0);

  CHECK(output[0]  == -4.0f); // input = -8
  CHECK(output[1]  == -3.5f); // input = -7
  CHECK(output[2]  == -3.0f); // input = -6
  CHECK(output[3]  == -2.5f); // input = -5
  CHECK(output[4]  == -2.0f); // input = -4
  CHECK(output[5]  == -1.5f); // input = -3
  CHECK(output[6]  == -1.0f); // input = -2
  CHECK(output[7]  == -0.5f); // input = -1
  CHECK(output[8]  ==  0.0f); // input =  0
  CHECK(output[9]  ==  0.5f); // input =  1
  CHECK(output[10] ==  1.0f); // input =  2
  CHECK(output[11] ==  1.5f); // input =  3
  CHECK(output[12] ==  2.0f); // input =  4
  CHECK(output[13] ==  2.5f); // input =  5
  CHECK(output[14] ==  3.0f); // input =  6
  CHECK(output[15] ==  3.5f); // input =  7
}

#endif

}
