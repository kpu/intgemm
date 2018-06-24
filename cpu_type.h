#pragma once

namespace intgemm {

// If you want to detect the CPU and dispatch yourself, here's what to use:
typedef enum {CPU_AVX512BW = 4, CPU_AVX2 = 3, CPU_SSSE3 = 2, CPU_SSE2 = 1, CPU_UNSUPPORTED} CPUType;

// See intgemm.h (which is the dispatcher) for the running CPU type.

} // namespace intgemm
