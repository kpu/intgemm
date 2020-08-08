#include <immintrin.h>

#if defined(_MSC_VER)
#elif defined(__INTEL_COMPILER)
__attribute__ ((target ("avx512f")))
#else
__attribute__ ((target ("avx512f,avx512bw,avx512dq,avx512vnni")))
#endif
bool Foo() {
  // AVX512F
  __m512i value = _mm512_set1_epi32(1);
  // AVX512BW
  value = _mm512_maddubs_epi16(value, value);
  // AVX512DQ
   __m256i value2 = _mm256_set1_epi8(1);
  value = _mm512_inserti32x8(value, value2, 1);
  // AVX512VNNI
  value = _mm512_dpbusd_epi32(value, value, value);
  return *(int*)&value;
}

int main() {
  return Foo();
}
