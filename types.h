#pragma once

#define DEFAULT __attribute__ ((target ("default")))
#define SSE2 __attribute__ ((target ("sse2")))
//#define SSE2_3 __attribute__ ((target ("ssse3"), target("sse2"))) //Not supported by clang
#define SSSE3 __attribute__ ((target ("ssse3")))
#define AVX2 __attribute__ ((target ("avx2")))
//#define AVX2_512F __attribute__ ((target ("avx2"), target("avx512f"))) //Not supported by clang
#define AVX512F __attribute__ ((target ("avx512f")))

namespace intgemm {

typedef unsigned int Index;

// If you want to detect the CPU and dispatch yourself, here's what to use:
typedef enum {CPU_AVX512BW = 4, CPU_AVX2 = 3, CPU_SSSE3 = 2, CPU_SSE2 = 1, CPU_UNSUPPORTED} CPUType;

// Running CPU type.  This is defined in intgemm.cc (as the dispatcher).
extern const CPUType kCPU;

} // namespace intgemm
