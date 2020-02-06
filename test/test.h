#pragma once

#include "intgemm_config.h"

#include "../3rd_party/catch.hpp"
#include "../intgemm.h"
#include "../aligned.h"

#include <math.h>
#include <sstream>
#include <iostream>
#include <iomanip>

#define CHECK_MESSAGE(cond, msg) do { INFO(msg); CHECK(cond); } while(0)
#define CHECK_FALSE_MESSAGE(cond, msg) do { INFO(msg); CHECK_FALSE(cond); } while(0)
#define REQUIRE_MESSAGE(cond, msg) do { INFO(msg); REQUIRE(cond); } while(0)
#define REQUIRE_FALSE_MESSAGE(cond, msg) do { INFO(msg); REQUIRE_FALSE(cond); } while(0)

#define CHECK_EPS(actual, expected, epsilon) \
  do { \
    if (fabs((actual) - (expected)) < epsilon) { SUCCEED(); } \
    else { CHECK((actual) == (expected)); } \
  } while(0)

#define KERNEL_TEST_CASE(name) TEST_CASE("Kernel: " name, "[kernel_test]")

namespace intgemm {

template <typename Type>
void Compare(const Type* reference, const Type* actual, Index size) {
  for (Index i = 0; i < size; ++i) {
    INFO("Inaccurate at " << i << ' ' << reference[i] << ' ' << actual[i]);
    CHECK(reference[i] == actual[i]);
  }
}

template <typename Type>
void CompareEps(const Type* reference, const Type* actual, Index size, Type epsilon) {
  for (Index i = 0; i < size; ++i) {
    INFO("Inaccurate at " << i << ' ' << reference[i] << ' ' << actual[i]);
    CHECK(fabs(reference[i] - actual[i]) < epsilon);
  }
}

void CompareMSE(const float *float_ref, const float *int_ref, const float *int_test,
                std::size_t size, std::string test_info, float int_tolerance,
                float float_tolerance, float MSE_float_tolerance, float MSE_int_tolerance);

template <typename Type>
std::string PrintMatrix(const Type *mem, Index rows, Index cols) {
  std::ostringstream out;
  for (Index r = 0; r < rows; ++r) {
    for (Index c = 0; c < cols; ++c) {
      out << std::setw(4) << (int64_t) mem[r * cols + c] << ' ';
    }
    out << '\n';
  }
  return out.str();
}

/*
 * References
 */
namespace references {

// Quantize
template <typename Type>
void Quantize(const float* input, Type* output, float quant_mult, Index size) {
  for (Index i = 0; i < size; ++i) {
    float value = roundf(input[i] * quant_mult);
    value = std::max<float>(std::numeric_limits<Type>::min(), value);
    value = std::min<float>(std::numeric_limits<Type>::max(), value);
    output[i] = value;
  }
}

// Multiply A(float) x B(float)
template <typename LambdaCallback>
void MultiplyFF(const float* A, const float* B, float* C, Index A_rows, Index width, Index B_cols, LambdaCallback callback) {
  for (Index r = 0; r < A_rows; ++r) {
    for (Index c = 0; c < B_cols; ++c) {
      float sum = 0.0f;
      for (Index k = 0; k < width; ++k) {
        sum += A[r * width + k] * B[k * B_cols + c];
      }
      C[r * B_cols + c] = callback(sum, {r, c, A_rows, B_cols});
    }
  }
}

// Multiply A(int) x B(int)
template <typename TypeA, typename TypeB, typename LambdaCallback,
          typename std::enable_if<std::is_integral<TypeA>::value>::type* = nullptr,
          typename std::enable_if<std::is_integral<TypeB>::value>::type* = nullptr>
void Multiply(const TypeA* A, const TypeB* B, float* C, Index A_rows, Index width, Index B_cols, LambdaCallback callback) {
  for (Index r = 0; r < A_rows; ++r) {
    for (Index c = 0; c < B_cols; ++c) {
      int32_t sum = 0;
      for (Index k = 0; k < width; ++k) {
        sum += int32_t(A[r * width + k]) * int32_t(B[k * B_cols + c]);
      }
      C[r * B_cols + c] = callback(sum, {r, c, A_rows, B_cols});
    }
  }
}

// Matrix rearragement
template <typename Type>
void Rearragement(const Type* input, Type* output, int simd, int unroll, Index rows, Index cols) {
  for (Index c = 0; c < cols; c += unroll) {
    for (Index r = 0; r < rows; r += simd) {
      for (Index i = 0; i < unroll; ++i)
        for (Index j = 0; j < simd; ++j)
          output[simd * i + j] = input[cols * r + c + cols * j + i];

      output += unroll * simd;
    }
  }
}

// Transpose
template <typename Type>
void Transpose(const Type* input, Type* output, Index rows, Index cols) {
  for (Index r = 0; r < rows; ++r) {
    for (Index c = 0; c < cols; ++c) {
      output[rows * c + r] = input[cols * r + c];
    }
  }
}

} // namespace references
} // namespace intgemm
