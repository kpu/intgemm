#pragma once

#ifndef INTGEMM_NO_AVX512

#define THIS_IS_AVX512BW
#include "kernels/implementations.inl"
#undef THIS_IS_AVX512BW

namespace intgemm {
namespace kernels {

// Put here kernels supported only by AVX512BW...

}
}

#endif
