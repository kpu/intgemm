#include "intgemm.h"
#include "stats.h"

namespace intgemm {

float Unsupported_MaxAbsolute(const float * /*begin*/, const float * /*end*/) {
  throw UnsupportedCPU();
}

MeanStd Unsupported_VectorMeanStd(const float * /*begin*/, const float * /*end*/, bool /*absolute*/) {
  throw UnsupportedCPU();
}

void (*Int16::Quantize)(const float *input, int16_t *output, float quant_mult, Index size) = ChooseCPU(AVX512BW::Kernels16::Quantize, AVX512BW::Kernels16::Quantize, AVX2::Kernels16::Quantize, SSE2::Kernels16::Quantize, SSE2::Kernels16::Quantize, Unsupported_16bit::Quantize);

void (*Int16::PrepareB)(const float *input, int16_t *output, float quant_mult, Index rows, Index cols) = ChooseCPU(AVX512BW::Kernels16::PrepareB, AVX512BW::Kernels16::PrepareB, AVX2::Kernels16::PrepareB, SSE2::Kernels16::PrepareB, SSE2::Kernels16::PrepareB, Unsupported_16bit::PrepareB);

void (*Int16::PrepareBQuantizedTransposed)(const int16_t *input, int16_t *output, Index inner, Index B_untransposed_cols) = ChooseCPU(AVX512BW::Kernels16::PrepareBQuantizedTransposed, AVX512BW::Kernels16::PrepareBQuantizedTransposed, AVX2::Kernels16::PrepareBQuantizedTransposed, SSE2::Kernels16::PrepareBQuantizedTransposed, SSE2::Kernels16::PrepareBQuantizedTransposed, Unsupported_16bit::PrepareBQuantizedTransposed);

void (*Int16::PrepareBTransposed)(const float *input, int16_t *output, float quant_mult, Index inner, Index B_untransposed_cols) = ChooseCPU(AVX512BW::Kernels16::PrepareBTransposed, AVX512BW::Kernels16::PrepareBTransposed, AVX2::Kernels16::PrepareBTransposed, SSE2::Kernels16::PrepareBTransposed, SSE2::Kernels16::PrepareBTransposed, Unsupported_16bit::PrepareBTransposed);

void (*Int16::SelectColumnsB)(const int16_t *input, int16_t *output, Index rows, const Index *cols_begin, const Index *cols_end) = ChooseCPU(AVX512BW::Kernels16::SelectColumnsB, AVX512BW::Kernels16::SelectColumnsB, AVX2::Kernels16::SelectColumnsB, SSE2::Kernels16::SelectColumnsB, SSE2::Kernels16::SelectColumnsB, Unsupported_16bit::SelectColumnsB);

const char *const Int16::kName = ChooseCPU(AVX512BW::Kernels16::kName, AVX512BW::Kernels16::kName, AVX2::Kernels16::kName, SSE2::Kernels16::kName, SSE2::Kernels16::kName, Unsupported_16bit::kName);

void (*Int8::Quantize)(const float *input, int8_t *output, float quant_mult, Index size) = ChooseCPU(AVX512VNNI::Kernels8::Quantize, AVX512BW::Kernels8::Quantize, AVX2::Kernels8::Quantize, SSSE3::Kernels8::Quantize, Unsupported_8bit::Quantize, Unsupported_8bit::Quantize);

void (*Int8::QuantizeU)(const float *input, uint8_t *output, float quant_mult, Index size) = ChooseCPU(AVX512VNNI::Kernels8::QuantizeU, AVX512BW::Kernels8::QuantizeU, AVX2::Kernels8::QuantizeU, SSSE3::Kernels8::QuantizeU, Unsupported_8bit::QuantizeU, Unsupported_8bit::QuantizeU);

void (*Int8::PrepareB)(const float *input, int8_t *output, float quant_mult, Index rows, Index cols) = ChooseCPU(AVX512VNNI::Kernels8::PrepareB, AVX512BW::Kernels8::PrepareB, AVX2::Kernels8::PrepareB, SSSE3::Kernels8::PrepareB, Unsupported_8bit::PrepareB, Unsupported_8bit::PrepareB);

void (*Int8::PrepareBQuantizedTransposed)(const int8_t *input, int8_t *output, Index inner, Index B_untransposed_cols) = ChooseCPU(AVX512BW::Kernels8::PrepareBQuantizedTransposed, AVX512BW::Kernels8::PrepareBQuantizedTransposed, AVX2::Kernels8::PrepareBQuantizedTransposed, SSSE3::Kernels8::PrepareBQuantizedTransposed, Unsupported_8bit::PrepareBQuantizedTransposed, Unsupported_8bit::PrepareBQuantizedTransposed);

void (*Int8::PrepareBTransposed)(const float *input, int8_t *output, float quant_mult, Index inner, Index B_untransposed_cols) = ChooseCPU(AVX512BW::Kernels8::PrepareBTransposed, AVX512BW::Kernels8::PrepareBTransposed, AVX2::Kernels8::PrepareBTransposed, SSSE3::Kernels8::PrepareBTransposed, Unsupported_8bit::PrepareBTransposed, Unsupported_8bit::PrepareBTransposed);

void (*Int8::SelectColumnsB)(const int8_t *input, int8_t *output, Index rows, const Index *cols_begin, const Index *cols_end) = ChooseCPU(AVX512VNNI::Kernels8::SelectColumnsB, AVX512BW::Kernels8::SelectColumnsB, AVX2::Kernels8::SelectColumnsB, SSSE3::Kernels8::SelectColumnsB, Unsupported_8bit::SelectColumnsB, Unsupported_8bit::SelectColumnsB);

const char *const Int8::kName = ChooseCPU(AVX512VNNI::Kernels8::kName, AVX512BW::Kernels8::kName, AVX2::Kernels8::kName, SSSE3::Kernels8::kName, Unsupported_8bit::kName, Unsupported_8bit::kName);

void (*Int8Shift::QuantizeU)(const float *input, uint8_t *output, float quant_mult, Index size) = ChooseCPU(AVX512VNNI::Kernels8::QuantizeU, AVX512BW::Kernels8::QuantizeU, AVX2::Kernels8::QuantizeU, SSSE3::Kernels8::QuantizeU, Unsupported_8bit::QuantizeU, Unsupported_8bit::QuantizeU);

const char *const Int8Shift::kName = ChooseCPU(AVX512VNNI::Kernels8::kName, AVX512BW::Kernels8::kName, AVX2::Kernels8::kName, SSSE3::Kernels8::kName, Unsupported_8bit::kName, Unsupported_8bit::kName);

const CPUType kCPU = ChooseCPU(CPUType::AVX512VNNI, CPUType::AVX512BW, CPUType::AVX2, CPUType::SSSE3, CPUType::SSE2, CPUType::UNSUPPORTED);

#if !defined(INTGEMM_COMPILER_SUPPORTS_AVX2)
namespace AVX2{
using SSE2::MaxAbsolute;
using SSE2::VectorMeanStd;
} // namespace AVX2
#endif
#if !defined(INTGEMM_COMPILER_SUPPORTS_AVX512BW)
namespace AVX512BW {
using AVX2::MaxAbsolute;
using AVX2::VectorMeanStd;
} // namespace AVX512BW
#endif

float (*MaxAbsolute)(const float *begin, const float *end) = ChooseCPU(AVX512BW::MaxAbsolute, AVX512BW::MaxAbsolute, AVX2::MaxAbsolute, SSE2::MaxAbsolute, SSE2::MaxAbsolute, Unsupported_MaxAbsolute);

MeanStd (*VectorMeanStd)(const float *begin, const float *end, bool absolute) = ChooseCPU(AVX512BW::VectorMeanStd, AVX512BW::VectorMeanStd, AVX2::VectorMeanStd, SSE2::VectorMeanStd, SSE2::VectorMeanStd, Unsupported_VectorMeanStd);

constexpr const char *const Unsupported_16bit::kName;
constexpr const char *const Unsupported_8bit::kName;
constexpr const char *const SSE2::Kernels16::kName;
constexpr const char *const SSSE3::Kernels8::kName;
#ifdef INTGEMM_COMPILER_SUPPORTS_AVX2
constexpr const char *const AVX2::Kernels8::kName;
constexpr const char *const AVX2::Kernels16::kName;
#endif
#ifdef INTGEMM_COMPILER_SUPPORTS_AVX512BW
constexpr const char *const AVX512BW::Kernels8::kName;
constexpr const char *const AVX512BW::Kernels16::kName;
#endif
#ifdef INTGEMM_COMPILER_SUPPORTS_AVX512VNNI
constexpr const char *const AVX512VNNI::Kernels8::kName;
#endif

}
