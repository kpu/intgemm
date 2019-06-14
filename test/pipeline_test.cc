#include "3rd_party/catch.hpp"
#include "postprocess.h"

#include <numeric>

namespace intgemm {

INTGEMM_AVX2 TEST_CASE("PostprocessPipeline AVX2", "Unquantize-ReLU") {
  if (kCPU < CPU_AVX2)
    return;

  __m256i input;
  __m256 output;

  auto raw_input = reinterpret_cast<int*>(&input);
  std::iota(raw_input, raw_input + 8, -2);

  auto raw_output = reinterpret_cast<float*>(&output);
  std::fill(raw_output, raw_output + 8, 42);

  auto pipeline = CreatePostprocessPipeline(Unquantize(0.5f), ReLU());
  auto inited_pipeline = InitPostprocessPipeline<CPU_AVX2>(pipeline);
  output = inited_pipeline.run(input, 0);

  CHECK(raw_output[0] == 0.0f); // input = -2
  CHECK(raw_output[1] == 0.0f); // input = -1
  CHECK(raw_output[2] == 0.0f); // input =  0
  CHECK(raw_output[3] == 0.5f); // input =  1
  CHECK(raw_output[4] == 1.0f); // input =  2
  CHECK(raw_output[5] == 1.5f); // input =  3
  CHECK(raw_output[6] == 2.0f); // input =  4
  CHECK(raw_output[7] == 2.5f); // input =  5
}

INTGEMM_AVX2 TEST_CASE("PostprocessPipeline AVX2 on whole buffer", "Unquantize-ReLU") {
  if (kCPU < CPU_AVX2)
    return;

  __m256i input[2];
  __m256 output[2];

  auto raw_input = reinterpret_cast<int*>(input);
  std::iota(raw_input, raw_input + 16, -8);

  auto raw_output = reinterpret_cast<float*>(output);
  std::fill(raw_output, raw_output + 16, 42);

  auto pipeline = CreatePostprocessPipeline(Unquantize(0.5f), ReLU());
  auto inited_pipeline = InitPostprocessPipeline<CPU_AVX2>(pipeline);
  inited_pipeline.run(input, 2, output);

  CHECK(raw_output[0]  == 0.f); // input = -8
  CHECK(raw_output[1]  == 0.f); // input = -7
  CHECK(raw_output[2]  == 0.f); // input = -6
  CHECK(raw_output[3]  == 0.f); // input = -5
  CHECK(raw_output[4]  == 0.f); // input = -4
  CHECK(raw_output[5]  == 0.f); // input = -3
  CHECK(raw_output[6]  == 0.f); // input = -2
  CHECK(raw_output[7]  == 0.f); // input = -1
  CHECK(raw_output[8]  == 0.0f); // input =  0
  CHECK(raw_output[9]  == 0.5f); // input =  1
  CHECK(raw_output[10] == 1.0f); // input =  2
  CHECK(raw_output[11] == 1.5f); // input =  3
  CHECK(raw_output[12] == 2.0f); // input =  4
  CHECK(raw_output[13] == 2.5f); // input =  5
  CHECK(raw_output[14] == 3.0f); // input =  6
  CHECK(raw_output[15] == 3.5f); // input =  7
}

}
