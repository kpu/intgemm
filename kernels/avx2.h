#pragma once

#define KERNELS_THIS_IS_AVX2
#include "kernels/implementations.inl"
#undef KERNELS_THIS_IS_AVX2

namespace intgemm {
namespace kernels {

// Put here kernels supported only by AVX2...

}
}
