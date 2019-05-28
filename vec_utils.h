#pragma once

#include "intrinsics.h"
#include "meta_math.h"

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
INTGEMM_AVX2 static inline __m256 exp_approx_reyoung(__m256 x) { // 31 instrinsics
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

template <typename Register>
Register exp_approx_taylor(Register x) { // 21-22 intrinsics
  const static float LOOKUP[] = {
    exp<-20>(), exp<-19>(), exp<-18>(), exp<-17>(), exp<-16>(), exp<-15>(),
    exp<-14>(), exp<-13>(), exp<-12>(), exp<-11>(), exp<-10>(), exp<-9>(),
    exp<-8>(), exp<-7>(), exp<-6>(), exp<-5>(), exp<-4>(), exp<-3>(), exp<-2>(),
    exp<-1>(), exp<0>(), exp<1>(), exp<2>(), exp<3>(), exp<4>(), exp<5>(),
    exp<6>(), exp<7>(), exp<8>(), exp<9>(), exp<10>(), exp<11>(), exp<12>(),
    exp<13>(), exp<14>(), exp<15>(), exp<16>(), exp<17>(), exp<18>(), exp<19>(),
    exp<20>(),
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
