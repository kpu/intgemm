#pragma once
#include <immintrin.h>

#include "intrinsics.h"
#include "vec_utils.h"

namespace intgemm {

__m512 KennethSigmoid(__m512 from) {
  const int32_t kLookup = 32;
  const float kTaper = 5.0;
  const float kScaling = (kLookup - 1) / kTaper;
  const __m512 kScaleVec = _mm512_set1_ps(kScaling);
  __mmask16 signs = _mm512_movepi32_mask(_mm512_castps_si512(from));
  __m512 scaled = _mm512_mul_ps(from, kScaleVec);
  __m512i as_int = _mm512_cvt_roundps_epi32(scaled, _MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC);
  __m512i abs = _mm512_abs_epi32(as_int);
  __m512i capped = _mm512_min_epi32(abs, _mm512_set1_epi32(kLookup - 1));
  // Copy this from PrintLookupCode.
  const __m512 kLookup0 = _mm512_set_ps(0x1.d62a48p-1, 0x1.cf893cp-1, 0x1.c7fb5p-1, 0x1.bf69f2p-1, 0x1.b5bfbcp-1, 0x1.aae9aep-1, 0x1.9ed8a2p-1, 0x1.9182e6p-1, 0x1.82e5e6p-1, 0x1.7307d8p-1, 0x1.61f9p-1, 0x1.4fd4a6p-1, 0x1.3cc134p-1, 0x1.28ef9cp-1, 0x1.1499bcp-1, 0x1p-1);
  const __m512 kLookup1 = _mm512_set_ps(0x1.fc92c2p-1, 0x1.fbfa6cp-1, 0x1.fb47ep-1, 0x1.fa76b6p-1, 0x1.f981ccp-1, 0x1.f8633p-1, 0x1.f7140ap-1, 0x1.f58c74p-1, 0x1.f3c35ap-1, 0x1.f1ae64p-1, 0x1.ef41ccp-1, 0x1.ec7042p-1, 0x1.e92adcp-1, 0x1.e561p-1, 0x1.e1006ap-1, 0x1.dbf54p-1);
  __m512i looked_up = _mm512_permutex2var_epi32(_mm512_castps_si512(kLookup0), capped, _mm512_castps_si512(kLookup1));
  __m512 looked_up_f = _mm512_castsi512_ps(looked_up);
  return _mm512_mask_sub_ps(looked_up_f, signs, _mm512_set1_ps(1.0f), looked_up_f);
}

__m512 TaylorSigmoid(__m512 from) {
  static const __m512 coeffs[] = {
    set1_ps<__m512>(31.f / 1451520),
    set1_ps<__m512>(17.f / 80640),
    set1_ps<__m512>(1.f / 480),
    set1_ps<__m512>(1.f / 48),
    set1_ps<__m512>(1.f / 4),
  };
  static const __m512 const_half = set1_ps<__m512>(1.f / 2);

  auto x = from;
  auto x_squared = mul_ps(x, x);

  // Horner's method
  auto sigmoid = mul_ps(x_squared, coeffs[0]);
  sigmoid = sub_ps(sigmoid, coeffs[1]);
  sigmoid = mul_ps(sigmoid, x_squared);
  sigmoid = add_ps(sigmoid, coeffs[2]);
  sigmoid = mul_ps(sigmoid, x_squared);
  sigmoid = sub_ps(sigmoid, coeffs[3]);
  sigmoid = mul_ps(sigmoid, x_squared);
  sigmoid = add_ps(sigmoid, coeffs[4]);
  sigmoid = mul_ps(sigmoid, x);
  sigmoid = add_ps(sigmoid, const_half);
}

__m512 ExpTaylorSigmoid(__m512 x) {
          auto minus_x = sub_ps(_mm512_setzero_ps(), x);
          auto e_x = exp_approx_taylor(x);
          auto e_minus_x = exp_approx_taylor(minus_x);

          auto sigmoid_case1 = _mm512_rcp14_ps(add_ps(_mm512_set1_ps(1.0), e_minus_x));
          auto sigmoid_case2 = mul_ps(e_x, _mm512_rcp14_ps(add_ps(_mm512_set1_ps(1.0), e_x)));


          auto negative_x_mask = _mm512_movepi32_mask(_mm512_castps_si512(x));
          return _mm512_mask_blend_ps(negative_x_mask, sigmoid_case2, sigmoid_case1);
}

__m512 Reyong(__m512 x) {

          auto minus_x = sub_ps(_mm512_setzero_ps(), x);
          auto e_x = exp_approx_taylor(x);
          auto e_minus_x = exp_approx_reyoung(minus_x);

          auto sigmoid_case1 = _mm512_rcp14_ps(add_ps(_mm512_set1_ps(1.0), e_minus_x));
          auto sigmoid_case2 = mul_ps(e_x, _mm512_rcp14_ps(add_ps(_mm512_set1_ps(1.0), e_x)));


          auto negative_x_mask = _mm512_movepi32_mask(_mm512_castps_si512(x));
          return _mm512_mask_blend_ps(negative_x_mask, sigmoid_case2, sigmoid_case1);
i}

} // namespace intgemm
