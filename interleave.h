#pragma once

/* This macro defines functions that interleave their arguments like
 * inline void Interleave8(__m256i &first, __m256i &second) {
 *   __m256i temp = _mm256_unpacklo_epi8(first, second);
 *   second = _mm256_unpackhi_epi8(first, second);
 *   first = temp;
 * }
 *
 * Example usage:
 *   INTGEMM_INTERLEAVE(__m128i, )
 *   INTGEMM_INTERLEAVE(__m256i, 256)
 *   INTGEMM_INTERLEAVE(__m512i, 512)
 */

#define INTGEMM_INTERLEAVE(type, prefix) \
 inline void Interleave8(type &first, type &second) { \
  type temp = _mm##prefix##_unpacklo_epi8(first, second); \
  second = _mm##prefix##_unpackhi_epi8(first, second); \
  first = temp; \
} \
inline void Interleave16(type &first, type &second) { \
  type temp = _mm##prefix##_unpacklo_epi16(first, second); \
  second = _mm##prefix##_unpackhi_epi16(first, second); \
  first = temp; \
} \
inline void Interleave32(type &first, type &second) { \
  type temp = _mm##prefix##_unpacklo_epi32(first, second); \
  second = _mm##prefix##_unpackhi_epi32(first, second); \
  first = temp; \
} \
inline void Interleave64(type &first, type &second) { \
  type temp = _mm##prefix##_unpacklo_epi64(first, second); \
  second = _mm##prefix##_unpackhi_epi64(first, second); \
  first = temp; \
}

