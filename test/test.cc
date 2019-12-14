#define CATCH_CONFIG_RUNNER
#include "test.h"

int main(int argc, char ** argv) {
  return Catch::Session().run(argc, argv);
}

namespace intgemm {

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

template void SlowRefInt<int8_t>(const int8_t *A, const int8_t *B, float *C, float unquant_mult, Index A_rows, Index width, Index B_cols, const float *bias);
template void SlowRefInt<int16_t>(const int16_t *A, const int16_t *B, float *C, float unquant_mult, Index A_rows, Index width, Index B_cols, const float *bias);
template void SlowRefInt<int32_t>(const int32_t *A, const int32_t *B, float *C, float unquant_mult, Index A_rows, Index width, Index B_cols, const float *bias);

void Compare(const float *float_ref, const float *int_ref, const float *int_test, std::size_t size, std::string test_info,
 float int_tolerance, float float_tolerance, float MSE_float_tolerance, float MSE_int_tolerance) {
  float int_sum = 0.0, float_sum = 0.0;
  for (std::size_t i = 0; i < size; ++i) {
    float int_diff = int_ref[i] - int_test[i];
    float float_diff = float_ref[i] - int_test[i];
    CHECK_MESSAGE(fabs(int_diff) <= int_tolerance, test_info << "Inaccurate compared to int reference at " << i << ' ' << int_ref[i] << ' ' << int_test[i]);
    CHECK_MESSAGE(fabs(float_diff) <= float_tolerance, test_info << "Inaccurate compared to float reference at " << i << ' ' << float_ref[i] << ' ' << int_test[i]);
    int_sum += int_diff * int_diff;
    float_sum += float_diff * float_diff;
  }
  CHECK_MESSAGE(fabs(sqrt(float_sum / size)) <= MSE_float_tolerance, test_info << "Float MSE = " << sqrt(float_sum / size));
  CHECK_MESSAGE(fabs(sqrt(int_sum / size)) <= MSE_int_tolerance, test_info << "Int MSE = " << sqrt(int_sum / size));
}

} //namespace intgemm
