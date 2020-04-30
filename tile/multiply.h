#pragma once

#include "access.h"
#include "dot.h"
#include "reduce.h"
#include "../callbacks.h"
#include "../types.h"

#include <cassert>

#ifdef INTGEMM_COMPILER_SUPPORTS_AVX512VNNI
#define INTGEMM_THIS_IS_AVX512VNNI
#include "multiply.inl"
#undef INTGEMM_THIS_IS_AVX512VNNI
#endif

#ifdef INTGEMM_COMPILER_SUPPORTS_AVX512BW
#define INTGEMM_THIS_IS_AVX512BW
#include "multiply.inl"
#undef INTGEMM_THIS_IS_AVX512BW
#endif

#define INTGEMM_THIS_IS_AVX2
#include "multiply.inl"
#undef INTGEMM_THIS_IS_AVX2

#define INTGEMM_THIS_IS_SSSE3
#include "multiply.inl"
#undef INTGEMM_THIS_IS_SSSE3

#define INTGEMM_THIS_IS_SSE2
#include "multiply.inl"
#undef INTGEMM_THIS_IS_SSE2
