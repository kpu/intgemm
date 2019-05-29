#pragma once

#include "intrinsics.h"
#include "types.h"

#include <tuple>

namespace intgemm {

template <typename... Stages>
using PostprocessPipeline = std::tuple<Stages...>;

template <typename... Stages>
constexpr std::tuple<Stages...> CreatePostprocessPipeline(const Stages&... stages) {
  return std::tuple<Stages...>(stages...);
}

template <typename Postprocess, CPUType CpuType>
class PostprocessImpl;

namespace { // anonymous namespace

template <std::size_t... I>
struct integer_seq {};

template <std::size_t N, std::size_t... I>
struct integer_seq_from_one_s : integer_seq_from_one_s<N - 1, N - 1, I...> {};

template <std::size_t... I>
struct integer_seq_from_one_s<1, I...> : integer_seq<I...> {};

template <typename... Types>
using integer_seq_from_one = integer_seq_from_one_s<sizeof...(Types) + 1>;

template <typename Stage>
struct remove_first_stage_type_s { using type = std::tuple<>;};

template <typename FirstStage, typename... RestStages>
struct remove_first_stage_type_s<std::tuple<FirstStage, RestStages...>> { using type = std::tuple<RestStages...>; };

template <typename... Stages>
using remove_first_stage_type = typename remove_first_stage_type_s<Stages...>::type;

template <typename FirstStage, typename... RestStages>
struct first_stage_type_s { using type = FirstStage; };

template <typename Stage>
struct first_stage_type_s<Stage> { using type = Stage; };

template <typename... Stages>
using first_stage_type = typename first_stage_type_s<Stages...>::type;

template <typename FirstStage, typename... RestStages>
struct last_stage_type_s { using type = typename last_stage_type_s<RestStages...>::type; };

template <typename Stage>
struct last_stage_type_s<Stage> { using type = Stage; };

template <typename... Stages>
using last_stage_type = typename last_stage_type_s<Stages...>::type;

template <typename... Stages>
using input_register_type = typename first_stage_type<Stages...>::InputRegister;

template <typename... Stages>
using output_register_type = typename last_stage_type<Stages...>::OutputRegister;

template <typename Tuple, typename std::size_t...I>
constexpr remove_first_stage_type<Tuple> ShiftPostprocessPipelineImpl(const Tuple& pipeline, integer_seq<I...>) {
  return CreatePostprocessPipeline(std::get<I>(pipeline)...);
}

template <typename FirstStage, typename... RestStages>
constexpr std::tuple<RestStages...> ShiftPostprocessPipeline(const std::tuple<FirstStage, RestStages...>& pipeline) {
  return ShiftPostprocessPipelineImpl(pipeline, integer_seq_from_one<std::tuple<FirstStage, RestStages...>>());
}

template <CPUType CpuType, typename Stage>
constexpr std::tuple<PostprocessImpl<Stage, CpuType>> InitPostprocessPipelineImpl(std::tuple<Stage> pipeline) {
  return std::tuple<PostprocessImpl<Stage, CpuType>>(PostprocessImpl<Stage, CpuType>(std::get<0>(pipeline)));
}

template <CPUType CpuType, typename FirstStage, typename... RestStages>
constexpr std::tuple<PostprocessImpl<FirstStage, CpuType>, PostprocessImpl<RestStages, CpuType>...> InitPostprocessPipelineImpl(std::tuple<FirstStage, RestStages...> pipeline) {
  return std::tuple_cat(
    std::tuple<PostprocessImpl<FirstStage, CpuType>>(PostprocessImpl<FirstStage, CpuType>(std::get<0>(pipeline))),
    InitPostprocessPipelineImpl<CpuType, RestStages...>(ShiftPostprocessPipeline(pipeline))
  );
}

template <CPUType CpuType>
struct RunPostprocessPipelineImpl;

#define RUN_POSTPROCESS_PIPELINE_IMPL_INSERT_IMPL(attribute, cpu_type) \
  template <>                                                                                   \
  struct RunPostprocessPipelineImpl<cpu_type> {                                                 \
    template <typename Stage>                                                                   \
    attribute static constexpr output_register_type<Stage>                                      \
    run(std::tuple<Stage> pipeline, input_register_type<Stage> input, Index offset) {           \
      return std::get<0>(pipeline).run(input, offset);                                          \
    }                                                                                           \
    template <typename... Stages>                                                               \
    attribute static constexpr output_register_type<Stages...>                                  \
    run(std::tuple<Stages...> pipeline, input_register_type<Stages...> input, Index offset) {   \
      return run(                                                                               \
        ShiftPostprocessPipeline(pipeline),                                                     \
        std::get<0>(pipeline).run(input, offset), offset);                                      \
    }                                                                                           \
  };

RUN_POSTPROCESS_PIPELINE_IMPL_INSERT_IMPL(INTGEMM_SSE2, CPUType::CPU_SSE2)
RUN_POSTPROCESS_PIPELINE_IMPL_INSERT_IMPL(INTGEMM_SSSE3, CPUType::CPU_SSSE3)
RUN_POSTPROCESS_PIPELINE_IMPL_INSERT_IMPL(INTGEMM_AVX2, CPUType::CPU_AVX2)
RUN_POSTPROCESS_PIPELINE_IMPL_INSERT_IMPL(INTGEMM_AVX512BW, CPUType::CPU_AVX512BW)

} // anonymous namespace

template <CPUType CpuType, typename... Stages>
class InitedPostprocessPipeline {};

template <CPUType CpuType, typename... Stages>
constexpr InitedPostprocessPipeline<CpuType, Stages...> InitPostprocessPipeline(std::tuple<Stages...> pipeline) {
  return InitedPostprocessPipeline<CpuType, Stages...>(pipeline);
}

#define INITED_POSTPROCESS_PIPELINE_INSERT_IMPL(attribute, cpu_type) \
  template <typename... Stages>                                                            \
  class InitedPostprocessPipeline<cpu_type, Stages...> {                                   \
  public:                                                                                  \
    using InputRegister = input_register_type<PostprocessImpl<Stages, cpu_type>...>;       \
    using OutputRegister = output_register_type<PostprocessImpl<Stages, cpu_type>...>;     \
    InitedPostprocessPipeline(std::tuple<Stages...> pipeline)                              \
        : inited_pipeline(InitPostprocessPipelineImpl<cpu_type, Stages...>(pipeline)) {}   \
    attribute inline OutputRegister run(InputRegister input, Index offset) {               \
      return RunPostprocessPipelineImpl<cpu_type>::run(inited_pipeline, input, offset);    \
    }                                                                                      \
  private:                                                                                 \
    const std::tuple<PostprocessImpl<Stages, cpu_type>...> inited_pipeline;                \
  };

INITED_POSTPROCESS_PIPELINE_INSERT_IMPL(INTGEMM_SSE2, CPUType::CPU_SSE2)
INITED_POSTPROCESS_PIPELINE_INSERT_IMPL(INTGEMM_SSSE3, CPUType::CPU_SSSE3)
INITED_POSTPROCESS_PIPELINE_INSERT_IMPL(INTGEMM_AVX2, CPUType::CPU_AVX2)
INITED_POSTPROCESS_PIPELINE_INSERT_IMPL(INTGEMM_AVX512BW, CPUType::CPU_AVX512BW)

}
