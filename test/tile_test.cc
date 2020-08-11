#include "../aligned.h"
#include "../tile/access.h"
#include "../tile/dot.h"
#include "../tile/multiply.h"
#include "../tile/reduce.h"
#include "test.h"

#include "../benchmarks/test_matrices.h"

#include <numeric>
#include <random>

#ifdef INTGEMM_COMPILER_SUPPORTS_AVX512VNNI
#define INTGEMM_THIS_IS_AVX512VNNI
#include "tile_test.inl"
#undef INTGEMM_THIS_IS_AVX512VNNI
#endif

#ifdef INTGEMM_COMPILER_SUPPORTS_AVX512BW
#define INTGEMM_THIS_IS_AVX512BW
#include "tile_test.inl"
#undef INTGEMM_THIS_IS_AVX512BW
#endif

#define INTGEMM_THIS_IS_AVX2
#include "tile_test.inl"
#undef INTGEMM_THIS_IS_AVX2

#define INTGEMM_THIS_IS_SSSE3
#include "tile_test.inl"
#undef INTGEMM_THIS_IS_SSSE3

#define INTGEMM_THIS_IS_SSE2
#include "tile_test.inl"
#undef INTGEMM_THIS_IS_SSE2
