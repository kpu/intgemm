#pragma once

#include "../types.h"
#include "../intrinsics.h"
#include "../utils.h"

#ifdef INTGEMM_COMPILER_SUPPORTS_AVX512VNNI
#define INTGEMM_THIS_IS_AVX512VNNI
#include "dot.inl"
#undef INTGEMM_THIS_IS_AVX512VNNI
#endif

#ifdef INTGEMM_COMPILER_SUPPORTS_AVX512
#define INTGEMM_THIS_IS_AVX512BW
#include "dot.inl"
#undef INTGEMM_THIS_IS_AVX512BW
#endif

#define INTGEMM_THIS_IS_AVX2
#include "dot.inl"
#undef INTGEMM_THIS_IS_AVX2

#define INTGEMM_THIS_IS_SSSE3
#include "dot.inl"
#undef INTGEMM_THIS_IS_SSSE3

#define INTGEMM_THIS_IS_SSE2
#include "dot.inl"
#undef INTGEMM_THIS_IS_SSE2
