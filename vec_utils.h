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

}
