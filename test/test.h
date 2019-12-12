#pragma once

#include "3rd_party/catch.hpp"
#include <sstream>
#include "intgemm.h"
#include "aligned.h"

#include "intgemm_config.h"

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
void SlowRefFloat(const float *A, const float *B, float *C, Index A_rows, Index width, Index B_cols, const float *bias=nullptr);

// Compute A*B slowly from integers.
template <class Integer> void SlowRefInt(const Integer *A, const Integer *B, float *C, float unquant_mult, Index A_rows, Index width, Index B_cols, const float *bias=nullptr);
void SlowRefInt(const uint8_t *A, const int8_t *B, float *C, float unquant_mult, Index A_rows, Index width, Index B_cols, const float *bias=nullptr);

void Compare(const float *float_ref, const float *int_ref, const float *int_test, std::size_t size, std::string test_info,
 float int_tolerance, float float_tolerance, float MSE_float_tolerance, float MSE_int_tolerance);

} //namespace intgemm
