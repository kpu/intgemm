/* This file is included multiple times, once per architecture. */
#if defined(CALLBACKS_THIS_IS_SSE2)
  #define CPU_NAME SSE2
  #define CPU_ATTR INTGEMM_SSE2
#elif defined(CALLBACKS_THIS_IS_AVX2)
  #define CPU_NAME AVX2
  #define CPU_ATTR INTGEMM_AVX2
#elif defined(CALLBACKS_THIS_IS_AVX512BW)
  #define CPU_NAME AVX512BW
  #define CPU_ATTR INTGEMM_AVX512BW
#else
  #error "Only SSE2, AVX2 and AVX512BW are supported"
#endif

#if defined(CALLBACKS_THIS_IS_SSE2)
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
 * Sequence
 */
template <typename... Configs>
class CallbackImpl<CPUType::CPU_NAME, std::tuple<Configs...>> {
public:
  CPU_ATTR CallbackImpl(const std::tuple<Configs...>& configs) : callbacks(init_callbacks(configs, make_sequence<sizeof...(Configs)>())) {}

  CPU_ATTR void operator()(vi input, const OutputBufferInfo& info) {
    run_callbacks(input, info, callbacks, make_sequence<sizeof...(Configs)>());
  }

private:
  using CallbacksTupleType = std::tuple<CallbackImpl<CPUType::CPU_NAME, Configs>...>;

  CallbacksTupleType callbacks;

  template <unsigned... Indices>
  CallbacksTupleType init_callbacks(const std::tuple<Configs...>& configs, sequence<Indices...>) {
    return std::make_tuple(CallbackImpl<CPUType::CPU_NAME, typename std::tuple_element<Indices, std::tuple<Configs...>>::type>(std::get<Indices>(configs))...);
  }

#define RUN_CALLBACKS_PIPELINE_IMPL(vtype) \
  template <unsigned FirstIndex> \
  CPU_ATTR static inline void run_callbacks(vtype input, const OutputBufferInfo& info, CallbacksTupleType& tuple, sequence<FirstIndex>) { \
    std::get<FirstIndex>(tuple)(input, info); \
  } \
  template <unsigned FirstIndex, unsigned SecondIndex, unsigned... RestIndices> \
  CPU_ATTR static inline void run_callbacks(vtype input, const OutputBufferInfo& info, CallbacksTupleType& tuple, sequence<FirstIndex, SecondIndex, RestIndices...>) { \
    auto output = std::get<FirstIndex>(tuple)(input, info); \
    run_callbacks(output, info, tuple, sequence<SecondIndex, RestIndices...>()); \
  }

  RUN_CALLBACKS_PIPELINE_IMPL(vi)
  RUN_CALLBACKS_PIPELINE_IMPL(vf)
  RUN_CALLBACKS_PIPELINE_IMPL(vd)

#undef RUN_CALLBACKS_PIPELINE_IMPL
};

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
template <typename Type>
class CallbackImpl<CPUType::CPU_NAME, Write<Type>> {
public:
  CPU_ATTR CallbackImpl(const Write<Type>& config) : config(config) {}

  CPU_ATTR void operator()(vector_t<CPUType::CPU_NAME, Type> input, const OutputBufferInfo& info) {
    kernels::write(input, config.output_addr, info.row_idx * info.cols + info.col_idx);
  }

private:
  Write<Type> config;
};

/*
 * Unquantize
 */
template <> class CallbackImpl<CPUType::CPU_NAME, Unquantize> {
public:
  CPU_ATTR CallbackImpl(const Unquantize& config) : config(config) {
    unquant_mult = set1_ps<vf>(config.unquant_mult);
  }

  CPU_ATTR vf operator()(vi input, const OutputBufferInfo&) {
    return kernels::unquantize(input, unquant_mult);
  }

private:
  vf unquant_mult;
  Unquantize config;
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
    // Workaround gcc 5 internal compiler error that can't read register members in debug.
    vf mult_reg;
#if !defined(__OPTIMIZE__) && (__GNUC__ == 5) && !defined(__clang__) && !defined(__INTEL_COMPILER)
    asm ("vmovdqa %1, %0" : "=x" (mult_reg) : "m" (unquant_mult));
#else
    mult_reg = unquant_mult;
#endif
    auto result = kernels::unquantize(input, mult_reg);
    kernels::write(result, config.output_addr, info.row_idx * info.cols + info.col_idx);
  }

private:
  vf unquant_mult;
  UnquantizeAndWrite config;
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
    // Workaround gcc 5 internal compiler error that can't read register members in debug.
    vf mult_reg;
#if !defined(__OPTIMIZE__) && (__GNUC__ == 5) && !defined(__clang__) && !defined(__INTEL_COMPILER)
    asm ("vmovdqa %1, %0" : "=x" (mult_reg) : "m" (unquant_mult));
#else
    mult_reg = unquant_mult;
#endif
    auto result = kernels::unquantize(input, mult_reg);
    result = kernels::add_bias(result, config.bias_addr, info.col_idx);
    kernels::write(result, config.output_addr, info.row_idx * info.cols + info.col_idx);
  }
private:
  vf unquant_mult;
  UnquantizeAndAddBiasAndWrite config;
};

}
}

#undef CPU_NAME
#undef CPU_ATTR
#undef vi
#undef vf
#undef vd
