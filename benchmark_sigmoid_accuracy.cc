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

float sigmoid_reference(float x) {
  if (x >= 0)
    return 1.f / (1.f + exp(-x));
  else
    return exp(x) / (1.f + exp(x));
}

float error_perc_7d(float reference, float expected) {
  return fabs(round(reference * 10000000) - round(expected * 10000000)) / round(reference * 10000000);
}

INTGEMM_AVX2 void run_testcase(const TestCase& testcase) {
  auto input = *reinterpret_cast<const __m256i*>(testcase.raw_input);

  float output_sigmoid_taylor[8];
  float output_sigmoid_reyoung[8];
  float output_sigmoid_exp_tayler[8];

  std::fill(output_sigmoid_taylor, output_sigmoid_taylor + 8, 123454321);
  std::fill(output_sigmoid_reyoung, output_sigmoid_reyoung + 8, 123454321);
  std::fill(output_sigmoid_exp_tayler, output_sigmoid_exp_tayler + 8, 123454321);

  auto sigmoid_taylor = SigmoidTaylor::OnAVX2(SigmoidTaylor(output_sigmoid_taylor, testcase.quantization_mult));
  auto sigmoid_reyoung = SigmoidReyoung::OnAVX2(SigmoidReyoung(output_sigmoid_reyoung, testcase.quantization_mult));
  auto sigmoid_exp_taylor = SigmoidExpTaylor::OnAVX2(SigmoidExpTaylor(output_sigmoid_exp_tayler, testcase.quantization_mult));

  // std::cout.setf(ios::fixed, ios::floatfield);
  std::cout.precision(std::numeric_limits<float>::digits10 + 1);
  for (unsigned i = 0; i < 8; ++i) {
    auto reference = sigmoid_reference(testcase.raw_input[i] * testcase.quantization_mult);
    sigmoid_taylor(0, 1, 0, input);
    sigmoid_reyoung(0, 1, 0, input);
    sigmoid_exp_taylor(0, 1, 0, input);

    std::cout << std::fixed << std::left << std::setw(20) << "Input value: "     << std::left << std::setw(20) << testcase.raw_input[i] * testcase.quantization_mult << " (raw: " << testcase.raw_input[i] << ")" << std::endl;
    std::cout << std::fixed << std::left << std::setw(20) << "Reference: "       << std::left << std::setw(20) << reference << std::endl;
    std::cout << std::fixed << std::left << std::setw(20) << "Sigmoid Taylor: "  << std::left << std::setw(20) << output_sigmoid_taylor[i] << " (" <<  100 * error_perc_7d(reference, output_sigmoid_taylor[i]) << "%)" << std::endl;
    std::cout << std::fixed << std::left << std::setw(20) << "Sigmoid Reyoung: " << std::left << std::setw(20) << output_sigmoid_reyoung[i] << " (" <<  100 * error_perc_7d(reference, output_sigmoid_reyoung[i]) << "%)" << std::endl;
    std::cout << std::fixed << std::left << std::setw(20) << "Sigmoid Exp Taylor: " << std::left << std::setw(20) << output_sigmoid_exp_tayler[i] << " (" <<  100 * error_perc_7d(reference, output_sigmoid_exp_tayler[i]) << "%)" << std::endl;
    std::cout << std::endl;
  }
}

int main() {
  int start = -15;
  int end = 15;
  int vec_per_1 = 5;

  for (int i = start; i <= end; ++i) {
    for (int j = 0; j < vec_per_1; ++j) {
      TestCase testcase;
      testcase.quantization_mult = 0.125f / vec_per_1;
      for (unsigned k = 0; k < 8; ++k)
        testcase.raw_input[k] = i * vec_per_1 * 8 + j * 8 + k;
      run_testcase(testcase);
    }
  }

  return 0;
}
