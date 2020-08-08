// Some compilers don't have AVX512BW support.  Test for them.
#include <immintrin.h>

#if defined(_MSC_VER)
#define INTGEMM_AVX512BW
#elif defined(__INTEL_COMPILER)
#define INTGEMM_AVX512BW __attribute__ ((target ("avx512f")))
#else
#define INTGEMM_AVX512BW __attribute__ ((target ("avx512bw")))
#endif

INTGEMM_AVX512BW int Test() {
  // AVX512BW
  __m512i value = _mm512_set1_epi32(1);
  value = _mm512_maddubs_epi16(value, value);
  return *(int*)&value;
}

int main() {
}
