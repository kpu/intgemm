#include "test/test.h"
#include "aligned.h"
#include "postprocess.h"

#include <numeric>

namespace intgemm {

INTGEMM_AVX2 TEST_CASE("PostprocessPipeline AVX2", "Unquantize-ReLU") {
  if (kCPU < CPUType::AVX2)
    return;

  AlignedVector<int32_t> input(8);
  AlignedVector<float> output(8);

  std::iota(input.begin(), input.end(), -2);

  auto pipeline = CreatePostprocessPipeline(Unquantize(0.5f), ReLU());
  auto inited_pipeline = InitPostprocessPipeline<CPUType::AVX2>(pipeline);
  *output.as<__m256>() = inited_pipeline.run(*input.as<__m256i>(), 0, 0);

  CHECK(output[0] == 0.0f); // input = -2
  CHECK(output[1] == 0.0f); // input = -1
  CHECK(output[2] == 0.0f); // input =  0
  CHECK(output[3] == 0.5f); // input =  1
  CHECK(output[4] == 1.0f); // input =  2
  CHECK(output[5] == 1.5f); // input =  3
  CHECK(output[6] == 2.0f); // input =  4
  CHECK(output[7] == 2.5f); // input =  5
}

INTGEMM_AVX2 TEST_CASE("PostprocessPipeline AVX2 on whole buffer", "Unquantize-ReLU") {
  if (kCPU < CPUType::AVX2)
    return;

  AlignedVector<int32_t> input(16);
  AlignedVector<float> output(16);

  std::iota(input.begin(), input.end(), -8);

  auto pipeline = CreatePostprocessPipeline(Unquantize(0.5f), ReLU());
  auto inited_pipeline = InitPostprocessPipeline<CPUType::AVX2>(pipeline);
  inited_pipeline.run(input.as<__m256i>(), 1, 2, output.as<__m256>());

  CHECK(output[0]  == 0.f);  // input = -8
  CHECK(output[1]  == 0.f);  // input = -7
  CHECK(output[2]  == 0.f);  // input = -6
  CHECK(output[3]  == 0.f);  // input = -5
  CHECK(output[4]  == 0.f);  // input = -4
  CHECK(output[5]  == 0.f);  // input = -3
  CHECK(output[6]  == 0.f);  // input = -2
  CHECK(output[7]  == 0.f);  // input = -1
  CHECK(output[8]  == 0.0f); // input =  0
  CHECK(output[9]  == 0.5f); // input =  1
  CHECK(output[10] == 1.0f); // input =  2
  CHECK(output[11] == 1.5f); // input =  3
  CHECK(output[12] == 2.0f); // input =  4
  CHECK(output[13] == 2.5f); // input =  5
  CHECK(output[14] == 3.0f); // input =  6
  CHECK(output[15] == 3.5f); // input =  7
}

}
