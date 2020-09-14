#include "intgemm.h"
#include "stats.h"

namespace intgemm {

float Unsupported_MaxAbsolute(const float * /*begin*/, const float * /*end*/) {
  throw UnsupportedCPU();
}

MeanStd Unsupported_VectorMeanStd(const float * /*begin*/, const float * /*end*/, bool /*absolute*/) {
  throw UnsupportedCPU();
}

void (*Int16::Quantize)(const float *input, int16_t *output, float quant_mult, Index size) = ChooseCPU(avx512bw::Kernels16::Quantize, avx512bw::Kernels16::Quantize, avx2::Kernels16::Quantize, sse2::Kernels16::Quantize, sse2::Kernels16::Quantize, Unsupported_16bit::Quantize);

void (*Int16::PrepareB)(const float *input, int16_t *output, float quant_mult, Index rows, Index cols) = ChooseCPU(avx512bw::Kernels16::PrepareB, avx512bw::Kernels16::PrepareB, avx2::Kernels16::PrepareB, sse2::Kernels16::PrepareB, sse2::Kernels16::PrepareB, Unsupported_16bit::PrepareB);

void (*Int16::PrepareBQuantizedTransposed)(const int16_t *input, int16_t *output, Index inner, Index B_untransposed_cols) = ChooseCPU(avx512bw::Kernels16::PrepareBQuantizedTransposed, avx512bw::Kernels16::PrepareBQuantizedTransposed, avx2::Kernels16::PrepareBQuantizedTransposed, sse2::Kernels16::PrepareBQuantizedTransposed, sse2::Kernels16::PrepareBQuantizedTransposed, Unsupported_16bit::PrepareBQuantizedTransposed);

void (*Int16::PrepareBTransposed)(const float *input, int16_t *output, float quant_mult, Index inner, Index B_untransposed_cols) = ChooseCPU(avx512bw::Kernels16::PrepareBTransposed, avx512bw::Kernels16::PrepareBTransposed, avx2::Kernels16::PrepareBTransposed, sse2::Kernels16::PrepareBTransposed, sse2::Kernels16::PrepareBTransposed, Unsupported_16bit::PrepareBTransposed);

void (*Int16::SelectColumnsB)(const int16_t *input, int16_t *output, Index rows, const Index *cols_begin, const Index *cols_end) = ChooseCPU(avx512bw::Kernels16::SelectColumnsB, avx512bw::Kernels16::SelectColumnsB, avx2::Kernels16::SelectColumnsB, sse2::Kernels16::SelectColumnsB, sse2::Kernels16::SelectColumnsB, Unsupported_16bit::SelectColumnsB);

const char *const Int16::kName = ChooseCPU(avx512bw::Kernels16::kName, avx512bw::Kernels16::kName, avx2::Kernels16::kName, sse2::Kernels16::kName, sse2::Kernels16::kName, Unsupported_16bit::kName);

void (*Int8::Quantize)(const float *input, int8_t *output, float quant_mult, Index size) = ChooseCPU(avx512vnni::Kernels8::Quantize, avx512bw::Kernels8::Quantize, avx2::Kernels8::Quantize, ssse3::Kernels8::Quantize, Unsupported_8bit::Quantize, Unsupported_8bit::Quantize);

void (*Int8::QuantizeU)(const float *input, uint8_t *output, float quant_mult, Index size) = ChooseCPU(avx512vnni::Kernels8::QuantizeU, avx512bw::Kernels8::QuantizeU, avx2::Kernels8::QuantizeU, ssse3::Kernels8::QuantizeU, Unsupported_8bit::QuantizeU, Unsupported_8bit::QuantizeU);

void (*Int8::PrepareB)(const float *input, int8_t *output, float quant_mult, Index rows, Index cols) = ChooseCPU(avx512vnni::Kernels8::PrepareB, avx512bw::Kernels8::PrepareB, avx2::Kernels8::PrepareB, ssse3::Kernels8::PrepareB, Unsupported_8bit::PrepareB, Unsupported_8bit::PrepareB);

void (*Int8::PrepareBQuantizedTransposed)(const int8_t *input, int8_t *output, Index inner, Index B_untransposed_cols) = ChooseCPU(avx512bw::Kernels8::PrepareBQuantizedTransposed, avx512bw::Kernels8::PrepareBQuantizedTransposed, avx2::Kernels8::PrepareBQuantizedTransposed, ssse3::Kernels8::PrepareBQuantizedTransposed, Unsupported_8bit::PrepareBQuantizedTransposed, Unsupported_8bit::PrepareBQuantizedTransposed);

void (*Int8::PrepareBTransposed)(const float *input, int8_t *output, float quant_mult, Index inner, Index B_untransposed_cols) = ChooseCPU(avx512bw::Kernels8::PrepareBTransposed, avx512bw::Kernels8::PrepareBTransposed, avx2::Kernels8::PrepareBTransposed, ssse3::Kernels8::PrepareBTransposed, Unsupported_8bit::PrepareBTransposed, Unsupported_8bit::PrepareBTransposed);

void (*Int8::SelectColumnsB)(const int8_t *input, int8_t *output, Index rows, const Index *cols_begin, const Index *cols_end) = ChooseCPU(avx512vnni::Kernels8::SelectColumnsB, avx512bw::Kernels8::SelectColumnsB, avx2::Kernels8::SelectColumnsB, ssse3::Kernels8::SelectColumnsB, Unsupported_8bit::SelectColumnsB, Unsupported_8bit::SelectColumnsB);

const char *const Int8::kName = ChooseCPU(avx512vnni::Kernels8::kName, avx512bw::Kernels8::kName, avx2::Kernels8::kName, ssse3::Kernels8::kName, Unsupported_8bit::kName, Unsupported_8bit::kName);

void (*Int8Shift::QuantizeU)(const float *input, uint8_t *output, float quant_mult, Index size) = ChooseCPU(avx512vnni::Kernels8::QuantizeU, avx512bw::Kernels8::QuantizeU, avx2::Kernels8::QuantizeU, ssse3::Kernels8::QuantizeU, Unsupported_8bit::QuantizeU, Unsupported_8bit::QuantizeU);

const char *const Int8Shift::kName = ChooseCPU(avx512vnni::Kernels8::kName, avx512bw::Kernels8::kName, avx2::Kernels8::kName, ssse3::Kernels8::kName, Unsupported_8bit::kName, Unsupported_8bit::kName);

const CPUType kCPU = ChooseCPU(CPUType::AVX512VNNI, CPUType::AVX512BW, CPUType::AVX2, CPUType::SSSE3, CPUType::SSE2, CPUType::UNSUPPORTED);

#if !defined(INTGEMM_COMPILER_SUPPORTS_AVX512BW)
namespace avx512bw {
using avx2::MaxAbsolute;
using avx2::VectorMeanStd;
} // namespace avx512bw
#endif

float (*MaxAbsolute)(const float *begin, const float *end) = ChooseCPU(avx512bw::MaxAbsolute, avx512bw::MaxAbsolute, avx2::MaxAbsolute, sse2::MaxAbsolute, sse2::MaxAbsolute, Unsupported_MaxAbsolute);

MeanStd (*VectorMeanStd)(const float *begin, const float *end, bool absolute) = ChooseCPU(avx512bw::VectorMeanStd, avx512bw::VectorMeanStd, avx2::VectorMeanStd, sse2::VectorMeanStd, sse2::VectorMeanStd, Unsupported_VectorMeanStd);

constexpr const char *const Unsupported_16bit::kName;
constexpr const char *const Unsupported_8bit::kName;
constexpr const char *const sse2::Kernels16::kName;
constexpr const char *const ssse3::Kernels8::kName;
constexpr const char *const avx2::Kernels8::kName;
constexpr const char *const avx2::Kernels16::kName;
#ifdef INTGEMM_COMPILER_SUPPORTS_AVX512BW
constexpr const char *const avx512bw::Kernels8::kName;
constexpr const char *const avx512bw::Kernels16::kName;
#endif
#ifdef INTGEMM_COMPILER_SUPPORTS_AVX512VNNI
constexpr const char *const avx512vnni::Kernels8::kName;
#endif

}
