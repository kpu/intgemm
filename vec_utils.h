#pragma once

#include "intrinsics.h"

namespace intgemm {

struct MultiplyResult128 {
  __m128i pack0123, pack4567;
};

//------------------------------------------------------------------------------
//
// Quantize
//
//------------------------------------------------------------------------------
INTGEMM_SSE2 static inline __m128i quantize(__m128 input, __m128 quantize_mult) {
  return cvtps_epi32(mul_ps(input, quantize_mult));
}
INTGEMM_AVX2 static inline __m256i quantize(__m256 input, __m256 quantize_mult){
  return cvtps_epi32(mul_ps(input, quantize_mult));
}
#ifndef INTGEMM_NO_AVX512
INTGEMM_AVX512BW static inline __m512i quantize(__m512 input, __m512 quantize_mult){
  return cvtps_epi32(mul_ps(input, quantize_mult));
}
#endif

//------------------------------------------------------------------------------
//
// Unquantize
//
//------------------------------------------------------------------------------
INTGEMM_SSE2 static inline __m128 unquantize(__m128i input, __m128 unquantize_mult) {
  return mul_ps(cvtepi32_ps(input), unquantize_mult);
}
INTGEMM_AVX2 static inline __m256 unquantize(__m256i input, __m256 unquantize_mult){
  return mul_ps(cvtepi32_ps(input), unquantize_mult);
}
#ifndef INTGEMM_NO_AVX512
INTGEMM_AVX512BW static inline __m512 unquantize(__m512i input, __m512 unquantize_mult){
  return mul_ps(cvtepi32_ps(input), unquantize_mult);
}
#endif

//------------------------------------------------------------------------------
//
// Floor float -> float
//
//------------------------------------------------------------------------------
INTGEMM_SSE2 static inline __m128 floor_ff(__m128 a) {
  return cvtepi32_ps(_mm_cvttps_epi32(a));
}

INTGEMM_AVX2 static inline __m256 floor_ff(__m256 a) {
  return _mm256_floor_ps(a);
}

INTGEMM_AVX512BW static inline __m512 floor_ff(__m512 a) {
  return cvtepi32_ps(cvttps_epi32(a)); // TODO: Check if it can be done better.
}

//------------------------------------------------------------------------------
//
// Calculate approximation of exp(x)
//
// Based on: https://github.com/reyoung/avx_mathfun
//
//------------------------------------------------------------------------------
/*
   AVX implementation of sin, cos, sincos, exp and log

   Based on "sse_mathfun.h", by Julien Pommier
   http://gruntthepeon.free.fr/ssemath/

   Copyright (C) 2012 Giovanni Garberoglio
   Interdisciplinary Laboratory for Computational Science (LISC)
   Fondazione Bruno Kessler and University of Trento
   via Sommarive, 18
   I-38123 Trento (Italy)

  This software is provided 'as-is', without any express or implied
  warranty.  In no event will the authors be held liable for any damages
  arising from the use of this software.

  Permission is granted to anyone to use this software for any purpose,
  including commercial applications, and to alter it and redistribute it
  freely, subject to the following restrictions:

  1. The origin of this software must not be misrepresented; you must not
     claim that you wrote the original software. If you use this software
     in a product, an acknowledgment in the product documentation would be
     appreciated but is not required.
  2. Altered source versions must be plainly marked as such, and must not be
     misrepresented as being the original software.
  3. This notice may not be removed or altered from any source distribution.

  (this is the zlib license)
*/
INTGEMM_AVX2 static inline __m256 exp_approx_reyoung(__m256 x) { // 31-32 instrinsics
  static const auto cephes_LOG2EF = set1_ps<__m256>(1.44269504088896341);
  static const auto cephes_exp_C1 = set1_ps<__m256>(0.693359375);
  static const auto cephes_exp_C2 = set1_ps<__m256>(-2.12194440e-4);
  static const auto cephes_exp_p0 = set1_ps<__m256>(1.9875691500E-4);
  static const auto cephes_exp_p1 = set1_ps<__m256>(1.3981999507E-3);
  static const auto cephes_exp_p2 = set1_ps<__m256>(8.3334519073E-3);
  static const auto cephes_exp_p3 = set1_ps<__m256>(4.1665795894E-2);
  static const auto cephes_exp_p4 = set1_ps<__m256>(1.6666665459E-1);
  static const auto cephes_exp_p5 = set1_ps<__m256>(5.0000001201E-1);

  static const auto const_0x7f = set1_epi32<__m256i>(0x7f);
  static const auto const_exp_hi = set1_ps<__m256>(88.3762626647949f);
  static const auto const_exp_lo = set1_ps<__m256>(-88.3762626647949f);
  static const auto const_half = set1_ps<__m256>(0.5f);
  static const auto const_one = set1_ps<__m256>(1.f);

  x = min_ps(x, const_exp_hi);
  x = max_ps(x, const_exp_lo);

  /* express exp(x) as exp(g + n*log(2)) */
  auto fx = mul_ps(x, cephes_LOG2EF);
  fx = add_ps(fx, const_half);

  /* how to perform a floorf with SSE: just below */
  //imm0 = _mm256_cvttps_epi32(fx);
  //tmp  = _mm256_cvtepi32_ps(imm0);

  auto tmp = floor_ff(fx);

  /* if greater, substract 1 */
  //__m256 mask = _mm256_cmpgt_ps(tmp, fx);
  auto mask = _mm256_cmp_ps(tmp, fx, _CMP_GT_OS);
  mask = and_ps(mask, const_one);
  fx = sub_ps(tmp, mask);

  tmp = mul_ps(fx, cephes_exp_C1);
  auto z = mul_ps(fx, cephes_exp_C2);
  x = sub_ps(x, tmp);
  x = sub_ps(x, z);

  z = mul_ps(x,x);

  auto y = cephes_exp_p0;
  y = mul_ps(y, x);
  y = add_ps(y, cephes_exp_p1);
  y = mul_ps(y, x);
  y = add_ps(y, cephes_exp_p2);
  y = mul_ps(y, x);
  y = add_ps(y, cephes_exp_p3);
  y = mul_ps(y, x);
  y = add_ps(y, cephes_exp_p4);
  y = mul_ps(y, x);
  y = add_ps(y, cephes_exp_p5);
  y = mul_ps(y, z);
  y = add_ps(y, x);
  y = add_ps(y, const_one);

  /* build 2^n */
  auto imm0 = _mm256_cvttps_epi32(fx);
  // another two AVX2 instructions
  imm0 = add_epi32(imm0, const_0x7f);
  imm0 = _mm256_slli_epi32(imm0, 23);
  auto pow2n = _mm256_castsi256_ps(imm0);
  y = mul_ps(y, pow2n);
  return y;
}

//------------------------------------------------------------------------------
//
// Calculate approximation of exp(x) using Taylor series and lookup table
//
//------------------------------------------------------------------------------
namespace { // anonymous namespace
template <unsigned N> constexpr long long factorial() { return N * factorial<N-1>(); }
template <> constexpr long long factorial<0>() { return 1; }

template <typename Register>
Register exp_approx_taylor(Register x) { // 21-22 intrinsics
  const static float LOOKUP[] = {
    /* e^-20 = */ 2.061153622438558e-09f,
    /* e^-19 = */ 5.602796437537268e-09f,
    /* e^-18 = */ 1.522997974471263e-08f,
    /* e^-17 = */ 4.139937718785167e-08f,
    /* e^-16 = */ 1.1253517471925912e-07f,
    /* e^-15 = */ 3.059023205018258e-07f,
    /* e^-14 = */ 8.315287191035679e-07f,
    /* e^-13 = */ 2.2603294069810542e-06f,
    /* e^-12 = */ 6.14421235332821e-06f,
    /* e^-11 = */ 1.670170079024566e-05f,
    /* e^-10 = */ 4.5399929762484854e-05f,
    /* e^-9 =  */ 0.00012340980408667956f,
    /* e^-8 =  */ 0.00033546262790251185f,
    /* e^-7 =  */ 0.0009118819655545162f,
    /* e^-6 =  */ 0.0024787521766663585f,
    /* e^-5 =  */ 0.006737946999085467f,
    /* e^-4 =  */ 0.01831563888873418f,
    /* e^-3 =  */ 0.049787068367863944f,
    /* e^-2 =  */ 0.1353352832366127f,
    /* e^-1 =  */ 0.36787944117144233f,
    /* e^0 =   */ 1.0f,
    /* e^1 =   */ 2.718281828459045f,
    /* e^2 =   */ 7.38905609893065f,
    /* e^3 =   */ 20.085536923187668f,
    /* e^4 =   */ 54.598150033144236f,
    /* e^5 =   */ 148.4131591025766f,
    /* e^6 =   */ 403.4287934927351f,
    /* e^7 =   */ 1096.6331584284585f,
    /* e^8 =   */ 2980.9579870417283f,
    /* e^9 =   */ 8103.083927575384f,
    /* e^10 =  */ 22026.465794806718f,
    /* e^11 =  */ 59874.14171519782f,
    /* e^12 =  */ 162754.79141900392f,
    /* e^13 =  */ 442413.3920089205f,
    /* e^14 =  */ 1202604.2841647768f,
    /* e^15 =  */ 3269017.3724721107f,
    /* e^16 =  */ 8886110.520507872f,
    /* e^17 =  */ 24154952.7535753f,
    /* e^18 =  */ 65659969.13733051f,
    /* e^19 =  */ 178482300.96318725f,
    /* e^20 =  */ 485165195.4097903f,
  };
  const static Register dividers[] = {
    set1_ps<Register>(1.f / factorial<7>()),
    set1_ps<Register>(1.f / factorial<6>()),
    set1_ps<Register>(1.f / factorial<5>()),
    set1_ps<Register>(1.f / factorial<4>()),
    set1_ps<Register>(1.f / factorial<3>()),
    set1_ps<Register>(1.f / factorial<2>()),
    set1_ps<Register>(1.f / factorial<1>()),
  };
  const static auto const_one = set1_ps<Register>(1.f);
  const static auto const_min_x = set1_ps<Register>(-20.f);
  const static auto const_max_x = set1_ps<Register>(20.f);

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

  auto ea = i32gather_ps(LOOKUP + 20, cvtps_epi32(a), 4);
  return mul_ps(ea, result);
}
} // anonymous namespace

template INTGEMM_SSE2 static __m128 exp_approx_taylor(__m128 x);
template INTGEMM_AVX2 static __m256 exp_approx_taylor(__m256 x);
template INTGEMM_AVX512BW static __m512 exp_approx_taylor(__m512 x);

}
