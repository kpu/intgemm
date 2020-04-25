#pragma once
#include "intgemm_config.h"
#include <cstddef>
#include <exception>
#include <immintrin.h>

#define INTGEMM_SSE2 __attribute__ ((target ("sse2")))
//#define SSE2_3 __attribute__ ((target ("ssse3"), target("sse2"))) //Not supported by clang
#define INTGEMM_SSSE3 __attribute__ ((target ("ssse3")))
#define INTGEMM_AVX2 __attribute__ ((target ("avx2")))
//#define AVX2_512F __attribute__ ((target ("avx2"), target("avx512f"))) //Not supported by clang
#if defined __INTEL_COMPILER
#define INTGEMM_AVX512F __attribute__ ((target ("avx512f")))
#define INTGEMM_AVX512BW __attribute__ ((target ("avx512f")))
#define INTGEMM_AVX512DQ __attribute__ ((target ("avx512f")))
#define INTGEMM_AVX512VNNI __attribute__ ((target ("avx512f")))
#else
#define INTGEMM_AVX512F __attribute__ ((target ("avx512f")))
#define INTGEMM_AVX512BW __attribute__ ((target ("avx512bw")))
#define INTGEMM_AVX512DQ __attribute__ ((target ("avx512dq")))
#define INTGEMM_AVX512VNNI __attribute__ ((target ("avx512f,avx512bw,avx512dq,avx512vnni")))
#endif
namespace intgemm {

// This will be thrown if a CPU isn't supported by the routines (16-bit without INTGEMM_SSE2 or 8-bit without INTGEMM_SSSE3).
class UnsupportedCPU : public std::exception {
  public:
    UnsupportedCPU() {}

    ~UnsupportedCPU() throw() {}

    const char *what() const throw() override {
      return "Integer matrix multiplication has not been efficiently implemented for your CPU.";
    }
};

typedef std::size_t Index;

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

struct Tile {
  Index A_rows, inner, B_cols;
  constexpr bool empty() const { return !A_rows || !inner || !B_cols; }
  constexpr bool operator==(const Tile other) const {
    return A_rows == other.A_rows && inner == other.inner && B_cols == other.B_cols;
  }
};

#ifdef INTGEMM_COMPILER_SUPPORTS_AVX512VNNI
namespace AVX512VNNI {
typedef __m512i Register;
typedef __m512 FRegister;
} // namespace AVX512VNNI
#endif
#ifdef INTGEMM_COMPILER_SUPPORTS_AVX512BW
namespace AVX512BW {
typedef __m512i Register;
typedef __m512 FRegister;
} // namespace AVX512BW
#endif
namespace AVX2 {
typedef __m256i Register;
typedef __m256 FRegister;
} // namespace AVX2
namespace SSSE3 {
typedef __m128i Register;
typedef __m128 FRegister;
} // namespace SSSE3
namespace SSE2 {
typedef __m128i Register;
typedef __m128 FRegister;
} // namespace SSE2

} // namespace intgemm
