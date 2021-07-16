#include "../intgemm/aligned.h"
#include "intgemm/intgemm_config.h"
#include "../intgemm/avx512_gemm.h"
#include "../intgemm/sse2_gemm.h"
#include "../intgemm/avx2_gemm.h"
#include "../intgemm/ssse3_gemm.h"
#include "../intgemm/intgemm.h"
#include "../intgemm/stats.h"
#include "../intgemm/callbacks.h"
#include <random>
#include <iostream>

/************************************************************************************ util ************************************************************************************/
template <class T>
int numDigits(T number) {
    int digits = 0;
    if (number <= 0) {
      digits = 1; // count the minus and take care of the zero case
    }
    while (number) {
        number /= 10;
        digits++;
    }
    return digits;
}

template<class intType>
void printMat(intType * a, size_t rows, size_t cols, std::string name, int digits = 0) {
  std::cerr << name << std::endl;
  for (size_t i = 0; i < rows; i++) {
    for (size_t j = 0; j < cols; j++) {
      int numbah = (int)a[i*cols + j];
      // Pad for nice printing
      int mydigits = digits - numDigits(numbah);
      for (int t = 0; t < mydigits; t++) {
        std::cerr << ' ';
      }
      std::cerr << numbah << " ";
    }
      std::cerr << std::endl;
  }
  std::cerr << std::endl;
}

template<class intType>
void toColMajor(intType *in, intType * out, size_t rows, size_t cols) {
  for (size_t i = 0; i < rows; i++) {
    for (size_t j = 0; j < cols; j++) {
      out[j*rows + i] = in[i*cols + j];
    }
  }
}

namespace intgemm {
template <class Routine>
void prepBtst(Index width, Index B_cols, float * in = nullptr) {
  AlignedVector<float> B(width * B_cols);

  //std::mt19937 gen;
  //std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  if (in != 0) {
    for (Index i = 0; i<width*B_cols; i++) {
      B[i] = in[i];
    }
  } else {
    for (Index i = 0; i<width*B_cols; i++) {
        B[i] = (float)(i%127);
    }
  }

  

  float alpha = 127.0f;
  float quant_mult = 127.0f / alpha;
  //float unquant_mult = 1.0f / (quant_mult*quant_mult);

  printMat(B.begin(), width, B_cols, "Raw Mat", 4);

  AlignedVector<int8_t> B_prep(B.size());
  //AlignedVector<int8_t> B_prep_print(B.size());
  Routine::PrepareB(B.begin(), B_prep.begin(), quant_mult, width, B_cols);
  printMat(B_prep.begin(), B_cols, width, "Prep Mat", 3);


  //toColMajor(B_prep.begin(), B_prep_print.begin(), B_cols, width);
  //printMat(B_prep_print.begin(), B_cols, width, "Prep Mat trans", 3);

}

void padMatrixTst(Index width, Index B_cols) {
    AlignedVector<float> B(width * B_cols);
    std::div_t results = std::div(B_cols, 8);

    for (Index i = 0; i<width*B_cols; i++) {
      B[i] = (float)(i%127);
    }
    auto padded = padMatrix(B.begin(), width, B_cols);
    printMat(B.begin(), width, B_cols, "Raw Mat", 4);
    printMat(padded.begin(), width, 8, "Padded", 4);

    auto shrunk = shrinkMat(B.begin(), width, B_cols);
    printMat(shrunk.begin(), width, results.quot*8, "Remainder", 4);
    prepBtst<SSSE3::Kernels8>(width, 8, padded.begin());
}


template <class Routine>
void smallMultTst(Index A_rows, Index width, Index B_cols) {
  AlignedVector<float> A(A_rows* width);
  AlignedVector<float> B(width * B_cols);
  AlignedVector<float> C(A_rows * B_cols);


  for (Index i = 0; i<width*B_cols; i++) {
      B[i] = (float)(i%127);
  }

  for (Index i = 0; i<A_rows*width; i++) {
      A[i] = (float)(i%127);
  }

  float alpha = 127.0f;
  float quant_mult = 127.0f / alpha;
  float unquant_mult = 1.0f / (quant_mult*quant_mult);

  printMat(A.begin(), A_rows, width, "Raw A", 3);
  printMat(B.begin(), width, B_cols, "Raw B", 3);

  AlignedVector<int8_t> A_prep(A.size());
  AlignedVector<int8_t> B_prep(B.size());

  Routine::PrepareA(A.begin(), A_prep.begin(), quant_mult, A_rows, width); // A is strictly positive here
  Routine::PrepareB(B.begin(), B_prep.begin(), quant_mult, width, B_cols);
  printMat(B_prep.begin(), B_cols, width, "Prep Mat B", 3);

  Routine::Multiply8Shift((uint8_t*)A_prep.begin(), B_prep.begin(), A_rows, width, B_cols, callbacks::UnquantizeAndWrite(unquant_mult, C.begin()));
  printMat(C.begin(), A_rows, B_cols, "Prep Mat C", 5);

}

} // namespace intgemm;
int main() {
    using namespace intgemm;
    //prepBtst<SSSE3::Kernels8>(32, 35);
    //prepBtst<AVX512VNNI::Kernels8>(64, 9);
    //padMatrixTst(32, 35);
    smallMultTst<AVX512VNNI::Kernels8>(1, 64, 2);
}
