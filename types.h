#pragma once
#include <exception>

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

} // namespace intgemm
