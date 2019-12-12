#include "intgemm.h"
#include "aligned.h"
#include <vector>
#include <math.h>
#include <iostream>
#include <iomanip>
#include <random>

using namespace intgemm;

void SlowRefFloat(const float *A, const float *B, float *C, Index A_rows, Index width, Index B_cols, const float *bias) {
  for (Index r = 0; r < A_rows; ++r) {
    for (Index c = 0; c < B_cols; ++c) {
      float sum = 0.0f;
      for (Index k = 0; k < width; ++k) {
        sum += A[r * width + k] * B[k * B_cols + c];
      }
      if (bias) {
        C[r * B_cols + c] = sum + bias[c];
      } else {
        C[r * B_cols + c] = sum;
      }
    }
  }
}

// Compute A*B slowly from integers.
template <class Integer> void SlowRefInt(const Integer *A, const Integer *B, float *C, float unquant_mult, Index A_rows, Index width, Index B_cols, const float *bias) {
  for (Index r = 0; r < A_rows; ++r) {
    for (Index c = 0; c < B_cols; ++c) {
      int32_t sum = 0;
      for (Index k = 0; k < width; ++k) {
        sum += static_cast<int16_t>(A[r * width + k]) * static_cast<int16_t>(B[k * B_cols + c]);
      }
      if (bias) {
        C[r * B_cols + c] = sum * unquant_mult + bias[c];
      } else {
        C[r * B_cols + c] = sum * unquant_mult;
      }
    }
  }
}
void SlowRefInt(const uint8_t *A, const int8_t *B, float *C, float unquant_mult, Index A_rows, Index width, Index B_cols, const float *bias) {
  for (Index r = 0; r < A_rows; ++r) {
    for (Index c = 0; c < B_cols; ++c) {
      int32_t sum = 0;
      for (Index k = 0; k < width; ++k) {
        sum += static_cast<int16_t>(A[r * width + k]) * static_cast<int16_t>(B[k * B_cols + c]);
      }
      if (bias) {
        C[r * B_cols + c] = sum * unquant_mult + bias[c];
      } else {
        C[r * B_cols + c] = sum * unquant_mult;
      }
    }
  }
}

int main(int argc, char ** argv) {
  // Initialize A and B.
  int A_rows = 1;
  int width= 256;
  int B_cols = 8;

  if (argc == 4) {
    A_rows = atoi(argv[1]);
    width= atoi(argv[2]);
    B_cols = atoi(argv[3]);
  }
  std::cout << "A_rows: " << A_rows << " width: " << width << " B_cols: " << B_cols << std::endl;
  AlignedVector<float> A(A_rows * width);
  AlignedVector<float> B(width * B_cols);
  AlignedVector<float> bias(B_cols);
  std::mt19937 gen;
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (auto& it : A) {
    it = dist(gen);
  }
  for (auto& it : B) {
    it = dist(gen);
  }

  //Empty bias to exclude noise from it
  std::uniform_real_distribution<float> dist2(-30.0f, 30.0f);
  for (auto& it : bias) {
    it = 0;//dist2(gen);
  }

  float alpha = 2.0f;
  float quant_mult = 127/alpha;
  float unquant_mult = 1.0/(quant_mult*quant_mult);

  AlignedVector<uint8_t> A_prep(A.size());
  AlignedVector<int8_t> A_prep_old(A.size());
  AlignedVector<int8_t> B_prep(B.size());
  AVX2_8bit::PrepareA(A.begin(), A_prep.begin(), quant_mult, A_rows, width); //Shifted version
  AVX2_8bit::PrepareA(A.begin(), A_prep_old.begin(), quant_mult, A_rows, width); //Non shited version

  AlignedVector<float> test_C(A_rows * B_cols);

  /*
  * Reference float multiplication
  */
  AlignedVector<float> float_C(test_C.size());
  SlowRefFloat(A.begin(), B.begin(), float_C.begin(), A_rows, width, B_cols, bias.begin());

  std::cout << std::setprecision(9);
  std::cout << "Float reference:" << std::endl;
  for (auto& it : float_C) {
    std::cout << it << ' ';
  }
  std::cout << std::endl;

  /*
   * Reference Int Multiplication
   */
  AlignedVector<int8_t> B_quant(B.size());
  AVX2_8bit::Quantize(B.begin(), B_quant.begin(), quant_mult, B.size());
  AlignedVector<float> slowint_C(test_C.size());
  // Taking the original A_preparation which means A would be int8_t
  SlowRefInt(A_prep_old.begin(), B_quant.begin(), slowint_C.begin(), unquant_mult, A_rows, width, B_cols, bias.begin());

  std::cout << "Non shifted int:" << std::endl;
  for (auto& it : slowint_C) {
    std::cout << it << ' ';
  }
  std::cout << std::endl;
  /*
  * Bias preparation for shift
  */
  AlignedVector<int8_t> A_prep2(1*width);
  for (auto& it : A_prep2) {
    it = 1;
  }
  AlignedVector<float> ShiftedBias(B_cols);
  float unquant_mult_forprep = (-1)*(alpha)*(alpha)/(127.0f); //Minus one to invert add_ps later on
  SlowRefInt(A_prep2.begin(), B_quant.begin(), ShiftedBias.begin(), unquant_mult_forprep, 1, width, B_cols, bias.begin());

  // Shifted Int multiplication
  SlowRefInt(A_prep.begin(), B_quant.begin(), slowint_C.begin(), unquant_mult, A_rows, width, B_cols, ShiftedBias.begin());
  std::cout << "Shifted int:" << std::endl;
  for (auto& it : slowint_C) {
    std::cout << it << ' ';
  }
  std::cout << std::endl;


  return 0;
}