// Some compilers don't have AVX2 support.  Test for them.
#include <immintrin.h>

#if defined(_MSC_VER)
#define INTGEMM_AVX2
#else
#define INTGEMM_AVX2 __attribute__ ((target ("avx2")))
#endif

INTGEMM_AVX2 int Test() {
  __m256i value = _mm256_set1_epi32(1);
  value = _mm256_abs_epi8(value);
  return *(int*)&value;
}

int main() {
}
