#pragma once

#include "intrinsics.h"

namespace intgemm {

struct RegisterPair128i {
  __m128i pack0123, pack4567;
};

struct RegisterPair128 {
  __m128 pack0123, pack4567;
};

/*
 *
 * Quantize
 *
 */
INTGEMM_SSE2 static inline __m128i quantize(__m128 input, __m128 quantize_mult) {
  return cvtps_epi32(mul_ps(input, quantize_mult));
}
INTGEMM_AVX2 static inline __m256i quantize(__m256 input, __m256 quantize_mult) {
  return cvtps_epi32(mul_ps(input, quantize_mult));
}
#ifndef INTGEMM_NO_AVX512
INTGEMM_AVX512BW static inline __m512i quantize(__m512 input, __m512 quantize_mult) {
  return cvtps_epi32(mul_ps(input, quantize_mult));
}
#endif

/*
 *
 * Unquantize
 *
 */
INTGEMM_SSE2 static inline __m128 unquantize(__m128i input, __m128 unquantize_mult) {
  return mul_ps(cvtepi32_ps(input), unquantize_mult);
}
INTGEMM_AVX2 static inline __m256 unquantize(__m256i input, __m256 unquantize_mult) {
  return mul_ps(cvtepi32_ps(input), unquantize_mult);
}
#ifndef INTGEMM_NO_AVX512
INTGEMM_AVX512BW static inline __m512 unquantize(__m512i input, __m512 unquantize_mult) {
  return mul_ps(cvtepi32_ps(input), unquantize_mult);
}
#endif

/*
 *
 * Calculate floor: float -> float
 *
 */
INTGEMM_SSE2 static inline __m128 floor_ff(__m128 a) {
  return cvtepi32_ps(_mm_cvttps_epi32(a));
}
INTGEMM_AVX2 static inline __m256 floor_ff(__m256 a) {
  return _mm256_floor_ps(a);
}
#ifndef INTGEMM_NO_AVX512
INTGEMM_AVX512BW static inline __m512 floor_ff(__m512 a) {
  return cvtepi32_ps(cvttps_epi32(a)); // TODO: Is there any better way to do that?
}
#endif

/*
 *
 * Calculate approximation of e^x using Taylor series and lookup table
 *
 */

template <typename Register>
Register exp_approx_taylor(Register x) {
  static constexpr int EXP_MIN = -20;
  static constexpr int EXP_MAX = 20;
  static constexpr float EXP_LOOKUP[EXP_MAX - EXP_MIN + 1] = {
    expi(-20), expi(-19), expi(-18), expi(-17), expi(-16), expi(-15),
    expi(-14), expi(-13), expi(-12), expi(-11), expi(-10), expi(-9),
    expi(-8), expi(-7), expi(-6), expi(-5), expi(-4), expi(-3), expi(-2),
    expi(-1), expi(0), expi(1), expi(2), expi(3), expi(4), expi(5),
    expi(6), expi(7), expi(8), expi(9), expi(10), expi(11), expi(12),
    expi(13), expi(14), expi(15), expi(16), expi(17), expi(18), expi(19),
    expi(20),
  };

  static const Register dividers[] = {
    set1_ps<Register>(1.f / factorial(7)),
    set1_ps<Register>(1.f / factorial(6)),
    set1_ps<Register>(1.f / factorial(5)),
    set1_ps<Register>(1.f / factorial(4)),
    set1_ps<Register>(1.f / factorial(3)),
    set1_ps<Register>(1.f / factorial(2)),
    set1_ps<Register>(1.f / factorial(1)),
  };
  static const auto const_one = set1_ps<Register>(1.f);
  static const auto const_min_x = set1_ps<Register>(EXP_MIN);
  static const auto const_max_x = set1_ps<Register>(EXP_MAX);

  x = max_ps(x, const_min_x);
  x = min_ps(x, const_max_x);

  auto a = floor_ff(x);
  auto xa = sub_ps(x, a);

  auto result = mul_ps(dividers[0], xa);

  result = add_ps(result, dividers[1]);
  result = mul_ps(result, xa);
  result = add_ps(result, dividers[2]);
  result = mul_ps(result, xa);
  result = add_ps(result, dividers[3]);
  result = mul_ps(result, xa);
  result = add_ps(result, dividers[4]);
  result = mul_ps(result, xa);
  result = add_ps(result, dividers[5]);
  result = mul_ps(result, xa);
  result = add_ps(result, dividers[6]);
  result = mul_ps(result, xa);

  result = add_ps(result, const_one);

  auto ea = i32gather_ps(EXP_LOOKUP + EXP_MAX, cvtps_epi32(a), 4);
  return mul_ps(ea, result);
}

template INTGEMM_AVX2 static __m256 exp_approx_taylor(__m256 x);
template INTGEMM_AVX512BW static __m512 exp_approx_taylor(__m512 x);

}
