#pragma once

#include "intgemm_config.h"

#ifdef INTGEMM_COMPILER_SUPPORTS_AVX512

#define THIS_IS_AVX512BW
#include "callbacks/implementations.inl"
#undef THIS_IS_AVX512BW

namespace intgemm {
namespace callbacks {

// Put here callbacks supported only by AVX512BW...

}
}

#endif
