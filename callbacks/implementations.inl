#include "callbacks/configs.h"

#include "intrinsics.h"
#include "types.h"
#include "vec_traits.h"

#if defined(THIS_IS_SSE2)
  #define CPU_NAME SSE2
  #define CPU_ATTR INTGEMM_SSE2
#elif defined(THIS_IS_AVX2)
  #define CPU_NAME AVX2
  #define CPU_ATTR INTGEMM_AVX2
#elif defined(THIS_IS_AVX512BW)
  #define CPU_NAME AVX512BW
  #define CPU_ATTR INTGEMM_AVX512BW
#else
  #error "Only SSE2, AVX2 and AVX512BW are supported"
#endif

#define vi vector_t<CPUType::CPU_NAME, int>
#define vf vector_t<CPUType::CPU_NAME, float>
#define vd vector_t<CPUType::CPU_NAME, double>
#define dvi dvector_t<CPUType::CPU_NAME, int>
#define dvf dvector_t<CPUType::CPU_NAME, float>
#define dvd dvector_t<CPUType::CPU_NAME, double>

#if defined(THIS_IS_SSE2)
#define vinput dvector_t<CPUType::SSE2, int>
#else
#define vinput vector_t<CPUType::AVX2, int>
#endif

namespace intgemm {
namespace callbacks {

template <typename CallbackConfig, CPUType CpuType>
class CallbackImpl;

}}

/*
 * Callbacks implementations....
 */
namespace intgemm {
namespace callbacks {

/*
 * Dummy
 */
template <> class CallbackImpl<Dummy, CPUType::CPU_NAME> {
public:
  CPU_ATTR CallbackImpl(const Dummy&) {}
  CPU_ATTR void operator()(vinput, Index, Index, Index, Index, Index) {}
};

}
}

#undef CPU_NAME
#undef CPU_ATTR
#undef vi
#undef vf
#undef vd
#undef dvi
#undef dvf
#undef dvd
#undef vinput
