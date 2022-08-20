#include "../intgemm/aligned.h"
#include "intgemm/intgemm_config.h"
#include "../intgemm/avx512_gemm.h"
#include "../intgemm/sse2_gemm.h"
#include "../intgemm/avx2_gemm.h"
#include "../intgemm/ssse3_gemm.h"
#include "../intgemm/intgemm.h"
#include "../intgemm/stats.h"
#include "../intgemm/callbacks.h"
#include "do_not_optimize.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <random>

namespace intgemm {
namespace {

struct RandomMatrices {
  RandomMatrices(Index A_rows_in, Index width_in, Index B_cols_in) :
    A_rows(A_rows_in), width(width_in), B_cols(B_cols_in),
    A(A_rows * width), B(width * B_cols) {
    std::mt19937 gen;
    std::uniform_int_distribution<int8_t> dist(-127, 127);
    gen.seed(456789);

    for (auto& it : A) {
      it = dist(gen);
    }
    for (auto& it : B) {
      it = dist(gen);
    }
  }

  const Index A_rows, width, B_cols;
  AlignedVector<int8_t> A, B;
};

template <class Routine> double Run(const RandomMatrices &m, Routine &&routine) {
  float unquant_mult = 3.14159f;
  AlignedVector<float> output(m.A_rows * m.B_cols);
  auto start = std::chrono::steady_clock::now();
  routine(m.A.begin(), m.B.begin(), m.A_rows, m.width, m.B_cols, callbacks::UnquantizeAndWrite(unquant_mult, output.begin()));
  doNotOptimizeAway(output.begin());
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

const std::size_t kSamples = 10000;
const float kOutlierThreshold = 0.75;

void Summarize(std::vector<double> &stats, const char *name) {
  // Throw out outliers.
  std::vector<double>::iterator keep = stats.begin() + static_cast<std::size_t>(static_cast<float>(stats.size()) * kOutlierThreshold);
  std::nth_element(stats.begin(), keep, stats.end());
  double avg = 0.0;
  for (std::vector<double>::const_iterator i = stats.begin(); i != keep; ++i) {
    avg += *i;
  }
  avg /= (keep - stats.begin());
  double stddev = 0.0;
  for (std::vector<double>::const_iterator i = stats.begin(); i != keep; ++i) {
    double off = (double)*i - avg;
    stddev += off * off;
  }
  stddev = sqrt(stddev / (keep - stats.begin() - 1));
  std::cout << std::setw(8) << *std::min_element(stats.begin(), stats.end()) << '\t' << std::setw(8) << avg << '\t' << std::setw(8) << stddev << '\t' << name << '\n';
}

} // namespace intgemm
} // namespace

// Program takes no input
int main(int, char ** argv) {
  std::cerr << "Remember to run this on a specific core:\ntaskset --cpu-list 0 " << argv[0] << std::endl;

  using namespace intgemm;
  RandomMatrices matrices[] = {
    {1, 512, 512},
    {1, 512, 2048},
//    {1, 64, 8},
/*    {1, 64, 8},
    {8, 256, 256},
    {8, 2048, 256},
    {8, 256, 2048},
    {320, 256, 256},
    {472, 256, 256},
    {248, 256, 256},
    {200, 256, 256},
    // Additional stuff
    {256, 256, 256},
    {512, 512, 512},
    {1024, 1024, 1024},*/
/*    {4096, 4096, 4096},
    {4096, 4096, 2048},
    {4096, 4096, 1024},
    {4096, 4096, 512},
    {4096, 4096, 256},*/
    /*{4096, 4096, 128}*/
  };
  for (std::size_t i = 0; i < sizeof(matrices) / sizeof(RandomMatrices); ++i) {
    std::cout << "Multiply\t" << matrices[i].A_rows << '\t' << matrices[i].width << '\t' << matrices[i].B_cols << '\t' << "Samples=" << kSamples << '\n' << "Minimum  \tAverage  \tStddev\n";
#ifdef INTGEMM_COMPILER_SUPPORTS_AVX512VNNI
    std::vector<double> baseline, areg, areg_write16;
    for (std::size_t sample = 0; sample < kSamples; ++sample) {
      baseline.push_back(Run(matrices[i], [](const int8_t *A, const int8_t *B, Index A_rows, Index inner, Index B_cols, callbacks::UnquantizeAndWrite callback) {
          intgemm::AVX512VNNI::Kernels8::Multiply8Shift((const uint8_t*)A, B, A_rows, inner, B_cols, callback);
      }));

      areg.push_back(Run(matrices[i], [](const int8_t *A, const int8_t *B, Index, Index, Index B_cols, callbacks::UnquantizeAndWrite callback) {
          intgemm::AVX512VNNI::Kernels8::Multiply8ShiftSingleARowWrite8<callbacks::UnquantizeAndWrite, 512>((const uint8_t*)A, B, B_cols, callback);
      }));

      areg_write16.push_back(Run(matrices[i], [](const int8_t *A, const int8_t *B, Index, Index, Index B_cols, callbacks::UnquantizeAndWrite callback) {
          intgemm::AVX512VNNI::Kernels8::Multiply8ShiftSingleARowWrite16<callbacks::UnquantizeAndWrite, 512>((const uint8_t*)A, B, B_cols, callback);
      }));
    }
    Summarize(baseline, "baseline");
    Summarize(areg, "+ A in register");
    Summarize(areg_write16, "+ Write16");
#endif
  }
  return 0;
}


