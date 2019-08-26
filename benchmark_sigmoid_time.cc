#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <ctime>

#include "cops.h"

using namespace intgemm;

struct TestCase {
  int32_t raw_input[8];
  float quantization_mult;
};

TestCase generate_testcase(float min, float max, float quantization_multiplier) {
  static std::random_device rand_dev;
  static std::mt19937 generator(rand_dev());

  TestCase result;
  result.quantization_mult = quantization_multiplier;
  int32_t minimum = round(min / quantization_multiplier);
  int32_t maximum = round(max / quantization_multiplier);

  std::uniform_int_distribution<int> dist(minimum, maximum);
  for (auto& value : result.raw_input)
    value = dist(generator);

  return result;
}

float sigmoid_reference(float x) {
  if (x >= 0)
    return 1.f / (1.f + exp(-x));
  else
    return exp(x) / (1.f + exp(x));
}

INTGEMM_AVX2 int main() {
  const unsigned long long TESTCASES = 100 * 1000000;

  float output[8];
  // Reference
  {
    clock_t begin = std::clock();
    for (unsigned long long  i = 0; i < TESTCASES; ++i)
    {
      TestCase testcase = generate_testcase(-2, 2, 0.01f);
      for (unsigned j = 0; j < 8; ++j)
        output[j] = sigmoid_reference(testcase.raw_input[j] * testcase.quantization_mult);
    }
    clock_t end = clock();
    double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    std::cout << "Reference:         " << elapsed_secs << " s" << std::endl;
  }

  // SigmoidTaylor
  {
    clock_t begin = std::clock();
    for (unsigned long long  i = 0; i < TESTCASES; ++i)
    {
      TestCase testcase = generate_testcase(-2, 2, 0.01f);
      auto input = *reinterpret_cast<const __m256i*>(testcase.raw_input);
      SigmoidTaylor::OnAVX2(SigmoidTaylor(output, testcase.quantization_mult))(0, 1, 0, input);
    }
    clock_t end = clock();
    double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    std::cout << "SigmoidTaylor:     " << elapsed_secs << " s" << std::endl;
  }

  // SigmoidReyoung
  {
    clock_t begin = std::clock();
    for (unsigned long long  i = 0; i < TESTCASES; ++i)
    {
      TestCase testcase = generate_testcase(-2, 2, 0.01f);
      auto input = *reinterpret_cast<const __m256i*>(testcase.raw_input);
      SigmoidReyoung::OnAVX2(SigmoidReyoung(output, testcase.quantization_mult))(0, 1, 0, input);
    }
    clock_t end = clock();
    double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    std::cout << "SigmoidReyoung:    " << elapsed_secs << " s" << std::endl;
  }

  // SigmoidExpTaylor
  {
    clock_t begin = std::clock();
    for (unsigned long long  i = 0; i < TESTCASES; ++i)
    {
      TestCase testcase = generate_testcase(-2, 2, 0.01f);
      auto input = *reinterpret_cast<const __m256i*>(testcase.raw_input);
      SigmoidExpTaylor::OnAVX2(SigmoidExpTaylor(output, testcase.quantization_mult))(0, 1, 0, input);
    }
    clock_t end = clock();
    double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    std::cout << "SigmoidExpTaylor:  " << elapsed_secs << " s" << std::endl;
  }

  return 0;
}
