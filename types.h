#pragma once
#include <exception>

#define DEFAULT __attribute__ ((target ("default")))
#define SSE2 __attribute__ ((target ("sse2")))
//#define SSE2_3 __attribute__ ((target ("ssse3"), target("sse2"))) //Not supported by clang
#define SSSE3 __attribute__ ((target ("ssse3")))
#define AVX2 __attribute__ ((target ("avx2")))
//#define AVX2_512F __attribute__ ((target ("avx2"), target("avx512f"))) //Not supported by clang
#if defined __INTEL_COMPILER
#define AVX512F __attribute__ ((target ("avx512f")))
#define AVX512BW __attribute__ ((target ("avx512f")))
#define AVX512DQ __attribute__ ((target ("avx512f")))
#else
#define AVX512F __attribute__ ((target ("avx512f")))
#define AVX512BW __attribute__ ((target ("avx512bw")))
#define AVX512DQ __attribute__ ((target ("avx512dq")))
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
typedef enum {CPU_AVX512BW = 4, CPU_AVX2 = 3, CPU_SSSE3 = 2, CPU_SSE2 = 1, CPU_UNSUPPORTED} CPUType;

// Running CPU type.  This is defined in intgemm.cc (as the dispatcher).
extern const CPUType kCPU;

} // namespace intgemm
