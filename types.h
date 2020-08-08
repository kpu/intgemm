#pragma once
#include <exception>
#include <immintrin.h>

#if defined(_MSC_VER)
/* MSVC does not appear to have target attributes but is also fine with just
 * using intrinsics anywhere.
 */
  #define INTGEMM_SSE2
  #define INTGEMM_SSSE3
  #define INTGEMM_AVX2
  #define INTGEMM_AVX512F
  #define INTGEMM_AVX512BW
  #define INTGEMM_AVX512DQ
  #define INTGEMM_AVX512VNNI
#else
  /* gcc, clang, and Intel compiler */
  #define INTGEMM_SSE2 __attribute__ ((target ("sse2")))
  #define INTGEMM_SSSE3 __attribute__ ((target ("ssse3")))
  #define INTGEMM_AVX2 __attribute__ ((target ("avx2")))
  #if defined(__INTEL_COMPILER)
    /* Intel compiler might not have AVX512 flavors but lets you use them anyway */
    #define INTGEMM_AVX512F __attribute__ ((target ("avx512f")))
    #define INTGEMM_AVX512BW __attribute__ ((target ("avx512f")))
    #define INTGEMM_AVX512DQ __attribute__ ((target ("avx512f")))
    #define INTGEMM_AVX512VNNI __attribute__ ((target ("avx512f")))
  #else
    /* gcc and clang take lists of all the flavors */
    #define INTGEMM_AVX512F __attribute__ ((target ("avx512f")))
    #define INTGEMM_AVX512BW __attribute__ ((target ("avx512f,avx512bw,avx512dq")))
    #define INTGEMM_AVX512DQ __attribute__ ((target ("avx512f,avx512bw,avx512dq")))
    #define INTGEMM_AVX512VNNI __attribute__ ((target ("avx512f,avx512bw,avx512dq,avx512vnni")))
  #endif
#endif
namespace intgemm {

// This will be thrown if a CPU isn't supported by the routines (16-bit without SSE2 or 8-bit without SSSE3).
class UnsupportedCPU : public std::exception {
  public:
    UnsupportedCPU() {}

    ~UnsupportedCPU() throw() {}

    const char *what() const throw() override {
      return "Integer matrix multiplication has not been efficiently implemented for your CPU.";
    }
};

typedef unsigned int Index;

// If you want to detect the CPU and dispatch yourself, here's what to use:
enum class CPUType {
  UNSUPPORTED = 0,
  SSE2,
  SSSE3,
  AVX2,
  AVX512BW,
  AVX512VNNI
};

// Running CPU type.  This is defined in intgemm.cc (as the dispatcher).
extern const CPUType kCPU;

struct MeanStd {
  float mean;
  float stddev;
};

#ifdef INTGEMM_COMPILER_SUPPORTS_AVX512VNNI
namespace avx512vnni {
typedef __m512i Register;
typedef __m512 FRegister;
} // namespace avx512vnni
#endif
#ifdef INTGEMM_COMPILER_SUPPORTS_AVX512BW
namespace avx512bw {
typedef __m512i Register;
typedef __m512 FRegister;
} // namespace avx512bw
#endif
namespace avx2 {
typedef __m256i Register;
typedef __m256 FRegister;
} // namespace avx2
namespace ssse3 {
typedef __m128i Register;
typedef __m128 FRegister;
} // namespace ssse3
namespace sse2 {
typedef __m128i Register;
typedef __m128 FRegister;
} // namespace sse2

} // namespace intgemm
