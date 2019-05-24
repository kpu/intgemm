#include "3rd_party/catch.hpp"
#include "postprocess.h"

#include <numeric>

namespace intgemm {

INTGEMM_AVX2 TEST_CASE("PostprocessPipeline AVX2", "Unquantize-ReLU") {
  if (kCPU < CPU_AVX2)
    return;

  int raw_input[8];
  std::iota(raw_input, raw_input + 8, -2);

  auto input = *reinterpret_cast<__m256i*>(raw_input);
  auto pipeline = CreatePostprocessPipeline(Unquantize(0.5f), ReLU());
  auto inited_pipeline = InitPostprocessPipeline<CPU_AVX2>(pipeline);
  auto output = inited_pipeline.run(input);

  float* raw_output = reinterpret_cast<float*>(&output);

  CHECK(raw_output[0] == 0.0f); // input = -2
  CHECK(raw_output[1] == 0.0f); // input = -1
  CHECK(raw_output[2] == 0.0f); // input =  0
  CHECK(raw_output[3] == 0.5f); // input =  1
  CHECK(raw_output[4] == 1.0f); // input =  2
  CHECK(raw_output[5] == 1.5f); // input =  3
  CHECK(raw_output[6] == 2.0f); // input =  4
  CHECK(raw_output[7] == 2.5f); // input =  5
}

}
