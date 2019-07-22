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

/*
 * SSRUSigmoidF
 *
 * output = sigmoid_lut(scale(input + bias), scale))
 */
template <> class CallbackImpl<CPUType::CPU_NAME, SSRUSigmoidF<int8_t>> {
public:
  CPU_ATTR CallbackImpl(const SSRUSigmoidF<int8_t>& config) : config(config), buffered_inputs_n(0), buffered_info(0, 0, 0, 0) {
    scale = set1_ps<vf>(config.quant_mult_bf / config.quant_mult_f);
    scale2 = set1_ps<vf>((127.0f / config.sigmoid_lut_range) / config.quant_mult_bf);
  }

  // Workaround. If the buffer size is not aligned to 4xsizeof(vec) then there'll be a problem with tails.
  CPU_ATTR void operator()(vi input, const OutputBufferInfo& info) {
    buffered_inputs[buffered_inputs_n++] = input;
    if (buffered_inputs_n == 1)
      buffered_info = info;
    else if (buffered_inputs_n == 4) {
      callback(buffered_inputs[0], buffered_inputs[1], buffered_inputs[2], buffered_inputs[3], buffered_info);
      buffered_inputs_n = 0;
    }
  }

private:
  SSRUSigmoidF<int8_t> config;
  vf scale;
  vf scale2;

  int buffered_inputs_n;
  vi buffered_inputs[4];
  OutputBufferInfo buffered_info;

  CPU_ATTR void callback(vi input1, vi input2, vi input3, vi input4, const OutputBufferInfo& info) {
    auto result = kernels::downcast32to8(
      kernels::rescale(input1, scale),
      kernels::rescale(input2, scale),
      kernels::rescale(input3, scale),
      kernels::rescale(input4, scale));
    result = kernels::add_bias(result, config.bias_addr, info.col_idx);

    auto tmp = kernels::upcast8to32(result);
    result = kernels::downcast32to8(
      kernels::rescale(tmp.first, scale2),
      kernels::rescale(tmp.second, scale2),
      kernels::rescale(tmp.third, scale2),
      kernels::rescale(tmp.fourth, scale2));

    result = kernels::lookup_8b(result, config.sigmoid_lut);
    kernels::write(result, config.output_addr, info.row_idx * info.cols + info.col_idx);
  }
};

/*
 * SSRUPrecomputedPartOfHighway
 *
 * output = (1 - sigmoid) * input
 */
template <> class CallbackImpl<CPUType::CPU_NAME, SSRUPrecomputedPartOfHighway<int8_t>> {
public:
  CPU_ATTR CallbackImpl(const SSRUPrecomputedPartOfHighway<int8_t>& config) : config(config), buffered_inputs_n(0), buffered_info(0, 0, 0, 0) {
    scale = set1_ps<vf>(config.scale);
  }

  // Workaround. If the buffer size is not aligned to 4xsizeof(vec) then there'll be a problem with tails.
  CPU_ATTR void operator()(vi input, const OutputBufferInfo& info) {
    buffered_inputs[buffered_inputs_n++] = input;
    if (buffered_inputs_n == 1)
      buffered_info = info;
    else if (buffered_inputs_n == 4) {
      callback(buffered_inputs[0], buffered_inputs[1], buffered_inputs[2], buffered_inputs[3], buffered_info);
      buffered_inputs_n = 0;
    }
  }

private:
  SSRUPrecomputedPartOfHighway<int8_t> config;
  vf scale;

  int buffered_inputs_n;
  vi buffered_inputs[4];
  OutputBufferInfo buffered_info;

  CPU_ATTR void callback(vi input1, vi input2, vi input3, vi input4, const OutputBufferInfo& info) {
    // TODO: Use 255 for better resolution (it needs u8 intrinsics)
    static const auto vconst_int8_max = set1_epi8<vi>(127);

    const auto offset = info.row_idx * info.cols + info.col_idx;
    const auto sigmoid = *reinterpret_cast<const vi*>(config.sigmoid_f_addr + offset);

    auto result = kernels::downcast32to8(
      kernels::rescale(input1, scale),
      kernels::rescale(input2, scale),
      kernels::rescale(input3, scale),
      kernels::rescale(input4, scale));
    result = kernels::multiply_sat<int8_t>(sub_epi8(vconst_int8_max, sigmoid), result, 7);
    kernels::write(result, config.output_addr, offset);
  }
};

}
}

#undef CPU_NAME
#undef CPU_ATTR
#undef vi
#undef vf
#undef vd
