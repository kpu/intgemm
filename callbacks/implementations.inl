#include "callbacks/configs.h"
#include "callbacks/output_buffer_info.h"

#include "intrinsics.h"
#include "kernels.h"
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

#if defined(THIS_IS_SSE2)
  #define vi vector_t<CPUType::SSE2, int>
  #define vf vector_t<CPUType::SSE2, float>
  #define vd vector_t<CPUType::SSE2, double>
#else
  #define vi vector_t<CPUType::AVX2, int>
  #define vf vector_t<CPUType::AVX2, float>
  #define vd vector_t<CPUType::AVX2, double>
#endif

namespace intgemm {
namespace callbacks {

template <CPUType CpuType, typename CallbackConfig>
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
template <> class CallbackImpl<CPUType::CPU_NAME, Dummy> {
public:
  CPU_ATTR CallbackImpl(const Dummy&) {}
  CPU_ATTR void operator()(vi, const OutputBufferInfo&) {}
};

/*
 * Write
 */
template <typename OutputBufferType>
class CallbackImpl<CPUType::CPU_NAME, Write<OutputBufferType>> {
public:
  CPU_ATTR CallbackImpl(const Write<OutputBufferType>& config) : config(config) {}

  CPU_ATTR void operator()(vi input, const OutputBufferInfo& info) {
    kernels::write(input, config.addr, info.row_idx * info.cols + info.col_idx);
  }

private:
  Write<OutputBufferType> config;
};

/*
 * UnquantizeAndWrite
 */
template <> class CallbackImpl<CPUType::CPU_NAME, UnquantizeAndWrite> {
public:
  CPU_ATTR CallbackImpl(const UnquantizeAndWrite& config) : config(config) {
    unquant_mult = set1_ps<vf>(config.unquant_mult);
  }
  CPU_ATTR void operator()(vi input, const OutputBufferInfo& info) {
    auto result = kernels::unquantize(input, unquant_mult);
    kernels::write(result, config.addr, info.row_idx * info.cols + info.col_idx);
  }

private:
  UnquantizeAndWrite config;
  vf unquant_mult;
};

/*
 * AddBiasAndWrite
 */
template <> class CallbackImpl<CPUType::CPU_NAME, AddBiasAndWrite> {
public:
  CPU_ATTR CallbackImpl(const AddBiasAndWrite& config) : config(config) {}

  CPU_ATTR void operator()(vi input, const OutputBufferInfo& info) {
    auto result = kernels::add_bias(input, config.bias_addr, info.col_idx);
    kernels::write(result, config.output_addr, info.row_idx * info.cols + info.col_idx);
  }

private:
  AddBiasAndWrite config;
};

/*
 * UnquantizeAndAddBiasAndWrite
 */
template <> class CallbackImpl<CPUType::CPU_NAME, UnquantizeAndAddBiasAndWrite> {
public:
  CPU_ATTR CallbackImpl(const UnquantizeAndAddBiasAndWrite& config) : config(config) {
    unquant_mult = set1_ps<vf>(config.unquant_mult);
  }
  CPU_ATTR void operator()(vi input, const OutputBufferInfo& info) {
    auto result = kernels::unquantize(input, unquant_mult);
    result = kernels::add_bias(result, config.bias_addr, info.col_idx);
    kernels::write(result, config.output_addr, info.row_idx * info.cols + info.col_idx);
  }
private:
  UnquantizeAndAddBiasAndWrite config;
  vf unquant_mult;
};

}
}

#undef CPU_NAME
#undef CPU_ATTR
#undef vi
#undef vf
#undef vd
