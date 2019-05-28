#pragma once

#include "types.h"

#include <emmintrin.h>
#include <immintrin.h>
#include <xmmintrin.h>

/*
 * NOTE: Please keep intrinsics in alphabetical order.
 */
namespace intgemm {

/*
 * Define a bunch of intrinstics as overloaded functions so they work with
 * templates.
 */
template <class Register> static inline Register set1_epi16(int16_t to);
template <class Register> static inline Register set1_epi32(int32_t to);
template <class Register> static inline Register set1_ps(float to);
template <class Register> static inline Register setzero_ps();

/*
 *
 * SSE2
 *
 */
INTGEMM_SSSE3 static inline __m128i abs_epi8(__m128i arg) {
  return _mm_abs_epi8(arg);
}
INTGEMM_SSE2 static inline __m128i add_epi32(__m128i first, __m128i second) {
  return _mm_add_epi32(first, second);
}
INTGEMM_SSE2 static inline __m128i adds_epi16(__m128i first, __m128i second) {
  return _mm_adds_epi16(first, second);
}
INTGEMM_SSE2 static inline __m128 and_ps(__m128 first, __m128 second) {
  return _mm_and_ps(first, second);
}
INTGEMM_SSE2 static inline __m128 cvtepi32_ps(__m128i arg) {
  return _mm_cvtepi32_ps(arg);
}
INTGEMM_SSE2 static inline __m128i cvtps_epi32(__m128 arg) {
  return _mm_cvtps_epi32(arg);
}
INTGEMM_SSE2 static inline __m128i madd_epi16(__m128i first, __m128i second) {
  return _mm_madd_epi16(first, second);
}
INTGEMM_SSSE3 static inline __m128i maddubs_epi16(__m128i first, __m128i second) {
  return _mm_maddubs_epi16(first, second);
}
INTGEMM_SSE2 static inline __m128 max_ps(__m128 first, __m128 second) {
  return _mm_max_ps(first, second);
}
INTGEMM_SSE2 static inline __m128 mul_ps(__m128 a, __m128 b) {
  return _mm_mul_ps(a, b);
}
template <> INTGEMM_SSE2 inline __m128i set1_epi16<__m128i>(int16_t to) {
  return _mm_set1_epi16(to);
}
template <> INTGEMM_SSE2 inline __m128i set1_epi32<__m128i>(int32_t to) {
  return _mm_set1_epi32(to);
}
template <> INTGEMM_SSE2 inline __m128 set1_ps<__m128>(float to) {
  return _mm_set1_ps(to);
}
template <> INTGEMM_SSE2 inline __m128 setzero_ps<__m128>() {
  return _mm_setzero_ps();
}
INTGEMM_SSSE3 static inline __m128i sign_epi8(__m128i first, __m128i second) {
  return _mm_sign_epi8(first, second);
}
INTGEMM_SSE2 static inline void storeu_ps(float* mem_addr, __m128 a) {
  _mm_storeu_ps(mem_addr, a);
}

/*
 *
 * AVX2
 *
 */
INTGEMM_AVX2 static inline __m256i abs_epi8(__m256i arg) {
  return _mm256_abs_epi8(arg);
}
INTGEMM_AVX2 static inline __m256i add_epi32(__m256i first, __m256i second) {
  return _mm256_add_epi32(first, second);
}
INTGEMM_AVX2 static inline __m256i adds_epi16(__m256i first, __m256i second) {
  return _mm256_adds_epi16(first, second);
}
INTGEMM_AVX2 static inline __m256 and_ps(__m256 first, __m256 second) {
  return _mm256_and_ps(first, second);
}
INTGEMM_AVX2 static inline __m256 cvtepi32_ps(__m256i arg) {
  return _mm256_cvtepi32_ps(arg);
}
INTGEMM_AVX2 static inline __m256i cvtps_epi32(__m256 arg) {
  return _mm256_cvtps_epi32(arg);
}
INTGEMM_AVX2 static inline __m256i madd_epi16(__m256i first, __m256i second) {
  return _mm256_madd_epi16(first, second);
}
INTGEMM_AVX2 static inline __m256i maddubs_epi16(__m256i first, __m256i second) {
  return _mm256_maddubs_epi16(first, second);
}
INTGEMM_AVX2 static inline __m256 max_ps(__m256 first, __m256 second) {
  return _mm256_max_ps(first, second);
}
INTGEMM_AVX2 static inline __m256 mul_ps(__m256 a, __m256 b) {
  return _mm256_mul_ps(a, b);
}
template <> INTGEMM_AVX2 inline __m256i set1_epi16<__m256i>(int16_t to) {
  return _mm256_set1_epi16(to);
}
template <> INTGEMM_AVX2 inline __m256i set1_epi32<__m256i>(int32_t to) {
  return _mm256_set1_epi32(to);
}
template <> INTGEMM_AVX2 inline __m256 set1_ps<__m256>(float to) {
  return _mm256_set1_ps(to);
}
template <> INTGEMM_AVX2 inline __m256 setzero_ps<__m256>() {
  return _mm256_setzero_ps();
}
INTGEMM_AVX2 static inline __m256i sign_epi8(__m256i first, __m256i second) {
  return _mm256_sign_epi8(first, second);
}
INTGEMM_AVX2 static inline void storeu_ps(float* mem_addr, __m256 a) {
  _mm256_storeu_ps(mem_addr, a);
}

/*
 *
 * AVX512
 *
 */
#ifndef INTGEMM_NO_AVX512

INTGEMM_AVX512BW static inline __m512i abs_epi8(__m512i arg) {
  return _mm512_abs_epi8(arg);
}
INTGEMM_AVX512BW static inline __m512i add_epi32(__m512i first, __m512i second) {
  return _mm512_add_epi32(first, second);
}
INTGEMM_AVX512BW static inline __m512i adds_epi16(__m512i first, __m512i second) {
  return _mm512_adds_epi16(first, second);
}
INTGEMM_AVX512DQ static inline __m512 and_ps(__m512 first, __m512 second) {
  return _mm512_and_ps(first, second);
}
INTGEMM_AVX512BW static inline __m512 cvtepi32_ps(__m512i arg) {
  return _mm512_cvtepi32_ps(arg);
}
INTGEMM_AVX512BW static inline __m512i cvtps_epi32(__m512 arg) {
  return _mm512_cvtps_epi32(arg);
}
INTGEMM_AVX512BW static inline __m512i madd_epi16(__m512i first, __m512i second) {
  return _mm512_madd_epi16(first, second);
}
INTGEMM_AVX512BW static inline __m512i maddubs_epi16(__m512i first, __m512i second) {
  return _mm512_maddubs_epi16(first, second);
}
INTGEMM_AVX512BW static inline __m512 max_ps(__m512 first, __m512 second) {
  return _mm512_max_ps(first, second);
}
INTGEMM_AVX512BW static inline __m512 mul_ps(__m512 a, __m512 b) {
  return _mm512_mul_ps(a, b);
}
template <> inline INTGEMM_AVX512BW __m512i set1_epi16<__m512i>(int16_t to) {
  return _mm512_set1_epi16(to);
}
template <> inline INTGEMM_AVX512BW __m512i set1_epi32<__m512i>(int32_t to) {
  return _mm512_set1_epi32(to);
}
template <> inline INTGEMM_AVX512BW __m512 set1_ps<__m512>(float to) {
  return _mm512_set1_ps(to);
}
template <> INTGEMM_AVX512BW inline __m512 setzero_ps<__m512>() {
  return _mm512_setzero_ps();
}
/*
 * Missing sign_epi8
 */
INTGEMM_AVX512BW static inline void storeu_ps(float* mem_addr, __m512 a) {
  _mm512_storeu_ps(mem_addr, a);
}

#endif

}
