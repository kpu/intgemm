#include "../tile/access.h"
#include "../tile/dot.h"
#include "../tile/reduce.h"

#include "test.h"
#include "../aligned.h"

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
