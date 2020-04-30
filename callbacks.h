#pragma once

#include "callbacks/configs.h"
#include "callbacks/output_buffer_info.h"

#include "intgemm_config.h"
#include "intrinsics.h"
#include "kernels.h"
#include "types.h"
#include "utils.h"
#include "vec_traits.h"

#define CALLBACKS_THIS_IS_SSE2
#include "callbacks/implementations.inl"
#undef CALLBACKS_THIS_IS_SSE2

#define CALLBACKS_THIS_IS_SSSE3
#include "callbacks/implementations.inl"
#undef CALLBACKS_THIS_IS_SSSE3

#define CALLBACKS_THIS_IS_AVX2
#include "callbacks/implementations.inl"
#undef CALLBACKS_THIS_IS_AVX2

#ifdef INTGEMM_COMPILER_SUPPORTS_AVX512BW
#define CALLBACKS_THIS_IS_AVX512BW
#include "callbacks/implementations.inl"
#undef CALLBACKS_THIS_IS_AVX512BW
#endif

#ifdef INTGEMM_COMPILER_SUPPORTS_AVX512VNNI
#define CALLBACKS_THIS_IS_AVX512VNNI
#include "callbacks/implementations.inl"
#undef CALLBACKS_THIS_IS_AVX512VNNI
#endif
