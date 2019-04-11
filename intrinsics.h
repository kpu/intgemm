#pragma once

#include "interleave.h"

#include <cassert>
#include <xmmintrin.h>
#include <emmintrin.h>
#include <immintrin.h>

namespace intgemm {

/* Define a bunch of intrinstics as overloaded functions so they work with
 * templates.
 */
template <class Register> static inline Register set1_epi16(int16_t to);
template <class Register> static inline Register set1_ps(float to);

struct MultiplyResult128 {
  __m128i pack0123, pack4567;
};
SSE2 static inline __m128i add_epi32(__m128i first, __m128i second) {
  return _mm_add_epi32(first, second);
}
SSE2 static inline __m128i adds_epi16(__m128i first, __m128i second) {
  return _mm_adds_epi16(first, second);
}
template <> SSE2 inline __m128i set1_epi16<__m128i>(int16_t to) {
  return _mm_set1_epi16(to);
}
template <> SSE2 inline __m128 set1_ps<__m128>(float to) {
  return _mm_set1_ps(to);
}
SSE2 static inline __m128i madd_epi16(__m128i first, __m128i second) {
  return _mm_madd_epi16(first, second);
}
SSSE3 static inline __m128i maddubs_epi16(__m128i first, __m128i second) {
  return _mm_maddubs_epi16(first, second);
}
SSSE3 static inline __m128i sign_epi8(__m128i first, __m128i second) {
  return _mm_sign_epi8(first, second);
}

SSSE3 static inline __m128i abs_epi8(__m128i arg) {
  return _mm_abs_epi8(arg);
}
SSE2 static inline __m128 max_ps(__m128 first, __m128 second) {
  return _mm_max_ps(first, second);
}
SSE2 static inline __m128 and_ps(__m128 first, __m128 second) {
  return _mm_and_ps(first, second);
}
SSE2 static inline __m128 cvtepi32_ps(__m128i arg) {
  return _mm_cvtepi32_ps(arg);
}
SSE2 static inline __m128 mul_ps (__m128 a, __m128 b) {
  return _mm_mul_ps(a, b);
}

AVX2 static inline __m256i add_epi32(__m256i first, __m256i second) {
  return _mm256_add_epi32(first, second);
}
AVX2 static inline __m256i adds_epi16(__m256i first, __m256i second) {
  return _mm256_adds_epi16(first, second);
}
template <> AVX2 inline __m256i set1_epi16<__m256i>(int16_t to) {
  return _mm256_set1_epi16(to);
}
template <> AVX2 inline __m256 set1_ps<__m256>(float to) {
  return _mm256_set1_ps(to);
}
AVX2 static inline __m256i madd_epi16(__m256i first, __m256i second) {
  return _mm256_madd_epi16(first, second);
}
AVX2 static inline __m256i maddubs_epi16(__m256i first, __m256i second) {
  return _mm256_maddubs_epi16(first, second);
}
AVX2 static inline __m256i sign_epi8(__m256i first, __m256i second) {
  return _mm256_sign_epi8(first, second);
}
AVX2 static inline __m256i abs_epi8(__m256i arg) {
  return _mm256_abs_epi8(arg);
}
AVX2 static inline __m256 max_ps(__m256 first, __m256 second) {
  return _mm256_max_ps(first, second);
}
AVX2 static inline __m256 and_ps(__m256 first, __m256 second) {
  return _mm256_and_ps(first, second);
}
AVX2 static inline __m256 cvtepi32_ps(__m256i arg) {
  return _mm256_cvtepi32_ps(arg);
}
AVX2 static inline __m256 mul_ps (__m256 a, __m256 b) {
  return _mm256_mul_ps(a, b);
}

#ifdef __AVX512BW__
AVX512F static inline __m512i add_epi32(__m512i first, __m512i second) {
  return _mm512_add_epi32(first, second);
}
template <> inline AVX512F __m512i set1_epi16<__m512i>(int16_t to) {
  return _mm512_set1_epi16(to);
}
template <> inline AVX512F __m512 set1_ps<__m512>(float to) {
  return _mm512_set1_ps(to);
}
AVX512F static inline __m512i madd_epi16(__m512i first, __m512i second) {
  return _mm512_madd_epi16(first, second);
}
AVX512F static inline __m512i maddubs_epi16(__m512i first, __m512i second) {
  return _mm512_maddubs_epi16(first, second);
}
AVX512F static inline __m512i abs_epi8(__m512i arg) {
  return _mm512_abs_epi8(arg);
}
AVX512F static inline __m512 max_ps(__m512 first, __m512 second) {
  return _mm512_max_ps(first, second);
}
// Technically __AVX512DQ__
AVX512F static inline __m512 and_ps(__m512 first, __m512 second) {
  return _mm512_and_ps(first, second);
}
#endif
} //namespace