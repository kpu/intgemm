#pragma once

#include "intgemm_config.h"

#ifdef INTGEMM_COMPILER_SUPPORTS_AVX512

#define KERNELS_THIS_IS_AVX512BW
#include "kernels/implementations.inl"
#undef KERNELS_THIS_IS_AVX512BW

namespace intgemm {
namespace kernels {

// Put here kernels supported only by AVX512BW...

}
}

#endif
