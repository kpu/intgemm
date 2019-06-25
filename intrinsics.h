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
template <class Register> static inline Register loadu_ps(const float* mem_addr);
template <class Register> static inline Register set1_epi16(int16_t to);
template <class Register> static inline Register set1_epi32(int32_t to);
template <class Register> static inline Register set1_ps(float to);
template <class Register> static inline Register setzero_ps();
template <class Register> static inline Register setzero_si();

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
INTGEMM_SSE2 static inline __m128 add_ps(__m128 a, __m128 b) {
  return _mm_add_ps(a, b);
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
INTGEMM_SSE2 static inline __m128i cvttps_epi32(__m128 a) {
  return _mm_cvttps_epi32(a);
}
INTGEMM_SSE2 static inline __m128 div_ps(__m128 a, __m128 b) {
  return _mm_div_ps(a, b);
}
/*
 * Missing i32gather_ps for SSE2
 */
template <> INTGEMM_SSE2 inline __m128 loadu_ps(const float* mem_addr) {
  return _mm_loadu_ps(mem_addr);
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
INTGEMM_SSE2 static inline __m128 min_ps(__m128 a, __m128 b) {
  return _mm_min_ps(a, b);
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
template <> INTGEMM_SSE2 inline __m128i setzero_si<__m128i>() {
  return _mm_setzero_si128();
}
INTGEMM_SSSE3 static inline __m128i sign_epi8(__m128i first, __m128i second) {
  return _mm_sign_epi8(first, second);
}
INTGEMM_SSE2 static inline void storeu_ps(float* mem_addr, __m128 a) {
  _mm_storeu_ps(mem_addr, a);
}
INTGEMM_SSE2 static inline __m128 sub_ps(__m128 a, __m128 b) {
  return _mm_sub_ps(a, b);
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
INTGEMM_AVX2 static inline __m256 add_ps(__m256 a, __m256 b) {
  return _mm256_add_ps(a, b);
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
INTGEMM_AVX2 static inline __m256i cvttps_epi32(__m256 a) {
  return _mm256_cvttps_epi32(a);
}
INTGEMM_AVX2 static inline __m256 div_ps(__m256 a, __m256 b) {
  return _mm256_div_ps(a, b);
}
INTGEMM_AVX2 static inline __m256 i32gather_ps(float const *base_addr, __m256i vindex, const int scale) {
  return _mm256_i32gather_ps(base_addr, vindex, scale);
}
template <> INTGEMM_AVX2 inline __m256 loadu_ps(const float* mem_addr) {
  return _mm256_loadu_ps(mem_addr);
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
INTGEMM_AVX2 static inline __m256 min_ps(__m256 a, __m256 b) {
  return _mm256_min_ps(a, b);
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
template <> INTGEMM_AVX2 inline __m256i setzero_si<__m256i>() {
  return _mm256_setzero_si256();
}
INTGEMM_AVX2 static inline __m256i sign_epi8(__m256i first, __m256i second) {
  return _mm256_sign_epi8(first, second);
}
INTGEMM_AVX2 static inline void storeu_ps(float* mem_addr, __m256 a) {
  _mm256_storeu_ps(mem_addr, a);
}
INTGEMM_AVX2 static inline __m256 sub_ps(__m256 a, __m256 b) {
  return _mm256_sub_ps(a, b);
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
INTGEMM_AVX512BW static inline __m512 add_ps(__m512 a, __m512 b) {
  return _mm512_add_ps(a, b);
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
INTGEMM_AVX512BW static inline __m512i cvttps_epi32(__m512 a) {
  return _mm512_cvttps_epi32(a);
}
INTGEMM_AVX512BW static inline __m512 div_ps(__m512 a, __m512 b) {
  return _mm512_div_ps(a, b);
}
INTGEMM_AVX512BW static inline __m512 i32gather_ps(float const *base_addr, __m512i vindex, const int scale) {
  return _mm512_i32gather_ps(vindex, base_addr, scale);
}
template <> INTGEMM_AVX512BW inline __m512 loadu_ps(const float* mem_addr) {
  return _mm512_loadu_ps(mem_addr);
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
INTGEMM_AVX512BW static inline __m512 min_ps(__m512 a, __m512 b) {
  return _mm512_min_ps(a, b);
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
template <> INTGEMM_AVX512BW inline __m512i setzero_si<__m512i>() {
  return _mm512_setzero_si512();
}
/*
 * Missing sign_epi8
 */
INTGEMM_AVX512BW static inline void storeu_ps(float* mem_addr, __m512 a) {
  _mm512_storeu_ps(mem_addr, a);
}
INTGEMM_AVX512BW static inline __m512 sub_ps(__m512 a, __m512 b) {
  return _mm512_sub_ps(a, b);
}

#endif

}
