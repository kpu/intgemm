#include "aligned.h"
#include "avx512_gemm.h"
#include "avx2_gemm.h"
#include "sse2_gemm.h"
#include "stop_watch.h"

#include <cassert>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <cstdlib>

#include <iostream>

namespace intgemm {

struct RandomMatrices {
  RandomMatrices(int A_rows_in, int width_in, int B_cols_in) :
    A_rows(A_rows_in), width(width_in), B_cols(B_cols_in),
    A(A_rows * width), B(width * B_cols) {
    for (int i = 0; i < A_rows * width; i++) {
        A[i] = ((float)rand()/(float)RAND_MAX)*2.0f - 1.0f;
    }
    
    for (int i = 0; i < B_cols * width; i++) {
        B[i] = ((float)rand()/(float)RAND_MAX)*2.0f - 1.0f;
    }
  }

  const int A_rows, width, B_cols;
  AlignedVector<float> A, B;
};

template <class Backend> void Run(RandomMatrices &m, int repeat = 20) {
  typedef typename Backend::Integer Integer;
  float quant_mult = 127.0 / 2;
  float unquant_mult = 1.0 / (quant_mult * quant_mult);
//  std::cout << Backend::Name() << std::endl;
  AlignedVector<Integer> A_prepared(m.A_rows * m.width);
  {
//    StopWatch w("PrepareA");
    Backend::PrepareA(m.A.get(), A_prepared.get(), quant_mult, m.A_rows, m.width);
  }
  AlignedVector<Integer> B_prepared(m.width * m.B_cols);
  {
//    StopWatch w("PrepareB");
    Backend::PrepareB(m.B.get(), B_prepared.get(), quant_mult, m.width, m.B_cols);
  }
  AlignedVector<float> output(m.A_rows * m.B_cols);
  // Burn in
  Backend::Multiply(A_prepared.get(), B_prepared.get(), output.get(), unquant_mult, m.A_rows, m.width, m.B_cols);
  {
    StopWatch w(Backend::Name(), repeat);
    for (int i = 0; i < repeat; ++i) {
      Backend::Multiply(A_prepared.get(), B_prepared.get(), output.get(), unquant_mult, m.A_rows, m.width, m.B_cols);
    }
  }
}

void Time(int A_rows, int width, int B_cols, int repeat = 20) {
  std::cout << A_rows << '\t' << width << '\t' << B_cols << std::endl;
  RandomMatrices m(A_rows, width, B_cols);
  Run<SSE2_8bit>(m, repeat);
  Run<AVX2_8bit>(m, repeat);
  Run<AVX512_8bit>(m, repeat);
  Run<SSE2_16bit>(m, repeat);
  Run<AVX2_16bit>(m, repeat);
#ifdef __AVX512BW__
  Run<AVX512_16bit>(m, repeat);
#endif
}

} // namespace intgemm

// Program takes no input
int main(int argc, char ** argv) {
    std::srand(45678);
    using namespace intgemm;
    // Top matrix sizes from Marian
    Time(8, 256, 256);
    Time(8, 2048, 256);
    Time(8, 256, 2048);
    Time(320, 256, 256);
    Time(472, 256, 256);
    Time(248, 256, 256);
    Time(200, 256, 256);
    // Additional stuff
    Time(256, 256, 256);
    Time(512, 512, 512);
    Time(1024, 1024, 1024);
    Time(4096, 4096, 4096, 3);
    Time(4096, 4096, 2048, 3);
    Time(4096, 4096, 1024, 3);
    Time(4096, 4096, 512, 3);
    Time(4096, 4096, 256, 3);
    Time(4096, 4096, 128, 3);
    return 0;
}


