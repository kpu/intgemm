#pragma once
/* Main interface for integer matrix multiplication.
 *
 * We are computing C = A * B with an optional scaling factor.
 *
 * A is typically activations.
 * Rows a multiple of 1 (no restriction)
 * Columns a multiple of 64 for 8-bit or 32 for 16-bit.
 * Use PrepareA to prepare A for multiplication.  This is meant to be fast.
 *
 * B is typically fixed model parameters.
 * Rows a multiple of 64 for 8-bit or 32 for 16-bit.
 * Columns a multiple of: 8
 * Use PrepareB to prepare B for multiplication.  This is slower, with the
 * intention that it will be prepared once and remembered.
 *
 * C is row major.
 *
 * Once both A and B are prepared, call Multiply.
 *
 * All memory (A, B, and C in float or prepared form) must be 64-byte aligned.
 * It's easy to write code that works on your CPU with lower alignment, but
 * breaks on AVX512.
 *
 * When preparing, you provide a quantization multiplier.  Values will be
 * multiplied by this then rounded to an integer.
 * For 16-bit neural networks, Jacob Devlin recommends 1024.0.
 * For 8-bit, use 127 / largest absolute value.
 *
 * Note that quantization saturates.  However, 16-bit does accumulation in
 * 32-bit which can overflow if you use too big of a multiplier.
 *
 * The multiply routine expects an unquantization multiplier.
 * This should be unquant_mult = 1.0 / (A_quant_mult * B_quant_mult).
 * Where A_quant_mult is what you passed to PrepareA and B_quant_mult is what you
 * passed to PrepareB.
 *
 * Feel free to multiply in a scaling factor to compute C = \lambda A * B by
 * passing unquant_mult = \lambda / (A_quant_mult * B_quant_mult).
 */

#include <cstdint>

#include "intgemm/intgemm_config.h"
#include "types.h"
#include "sse2_gemm.h"
#include "ssse3_gemm.h"
#include "avx2_gemm.h"
#include "avx512_gemm.h"
#include "avx512vnni_gemm.h"

#if defined(__INTEL_COMPILER)
#include <immintrin.h>
#elif defined(_MSC_VER)
#include <intrin.h>
#elif defined(__GNUC__) || defined(__clang__)
#include <cpuid.h>
#endif

/* Dispatch to functions based on runtime CPUID.  This adds one call-by-variable to each call. */

namespace intgemm {

struct Unsupported_16bit {
  static void Quantize(const float *, int16_t *, float, Index) {
    throw UnsupportedCPU();
  }
  static void PrepareB(const float *, int16_t *, float, Index, Index) {
    throw UnsupportedCPU();
  }
  static void PrepareBQuantizedTransposed(const int16_t *, int16_t *, Index, Index) {
    throw UnsupportedCPU();
  }
  static void PrepareBTransposed(const float *, int16_t *, float, Index, Index) {
    throw UnsupportedCPU();
  }
  static void SelectColumnsB(const int16_t *, int16_t *, Index, const Index *, const Index *) {
    throw UnsupportedCPU();
  }
  template <typename Callback>
  static void Multiply(const int16_t *, const int16_t *, Index, Index, Index, Callback) {
    throw UnsupportedCPU();
  }
  constexpr static const char *const kName = "16-bit Unsupported";
};

struct Unsupported_8bit {
  static void Quantize(const float *, int8_t *, float, Index) {
    throw UnsupportedCPU();
  }
  static void QuantizeU(const float *, uint8_t *, float, Index) {
    throw UnsupportedCPU();
  }
  static void PrepareA(const float *, int8_t *, float, Index, Index) {
    throw UnsupportedCPU();
  }
  static void PrepareBQuantizedTransposed(const int8_t *, int8_t *, Index, Index) {
    throw UnsupportedCPU();
  }
  static void PrepareBTransposed(const float *, int8_t *, float, Index, Index) {
    throw UnsupportedCPU();
  }
  static void PrepareB(const float *, int8_t *, float, Index, Index) {
    throw UnsupportedCPU();
  }
  template<class Callback>
  static void PrepareBias(const int8_t *, Index, Index, Callback) {
    throw UnsupportedCPU();
  }
  static void SelectColumnsB(const int8_t *, int8_t *, Index, const Index *, const Index *) {
    throw UnsupportedCPU();
  }
  template <typename Callback>
  static void Multiply(const int8_t *, const int8_t *, Index, Index, Index, Callback) {
    throw UnsupportedCPU();
  }
  template<class Callback>
  static void Multiply8Shift(const uint8_t *, const int8_t *, Index, Index, Index, Callback) {
    throw UnsupportedCPU();
  }

  constexpr static const char *const kName = "8-bit Unsupported";
};

#ifndef INTGEMM_COMPILER_SUPPORTS_AVX512VNNI
// These won't ever be called in this capacity, but it does let the code below compile.
namespace avx512vnni {
typedef Unsupported_8bit Kernels8;
} // namespace avx512vnni
#endif
#ifndef INTGEMM_COMPILER_SUPPORTS_AVX512BW
namespace avx512bw {
typedef Unsupported_8bit Kernels8;
typedef Unsupported_16bit Kernels16;
} // namespace avx512bw
#endif

/* Returns:
 * axx512vnni if the CPU supports AVX512VNNI
 *
 * avx512bw if the CPU supports AVX512BW
 *
 * avx2 if the CPU supports AVX2
 *
 * ssse3 if the CPU supports SSSE3 (this distinction from SSE2 matters for 8-bit)
 *
 * sse2 if the CPU supports SSE2
 *
 * unsupported otherwise
 */
template <class T> T ChooseCPU(T
#ifdef INTGEMM_COMPILER_SUPPORTS_AVX512VNNI
    avx512vnni
#endif
    , T
#ifdef INTGEMM_COMPILER_SUPPORTS_AVX512BW
    avx512bw
#endif
    , T avx2, T ssse3, T sse2, T unsupported) {
#if defined(__INTEL_COMPILER)
#  ifdef INTGEMM_COMPILER_SUPPORTS_AVX512VNNI
  if (_may_i_use_cpu_feature(_FEATURE_AVX512_VNNI)) return avx512vnni;
#  endif
#  ifdef INTGEMM_COMPILER_SUPPORTS_AVX512BW
  if (_may_i_use_cpu_feature(_FEATURE_AVX512BW)) return avx512bw;
#  endif
  if (_may_i_use_cpu_feature(_FEATURE_AVX2)) return avx2;
  if (_may_i_use_cpu_feature(_FEATURE_SSSE3)) return ssse3;
  if (_may_i_use_cpu_feature(_FEATURE_SSE2)) return sse2;
  return unsupported;
#else
// Everybody except Intel compiler.
#  if defined(_MSC_VER)
  int regs[4];
  int &eax = regs[0], &ebx = regs[1], &ecx = regs[2], &edx = regs[3];
  __cpuid(regs, 0);
  int m = eax;
#  else
  /* gcc and clang.
   * If intgemm is compiled by gcc 6.4.1 then dlopened into an executable
   * compiled by gcc 7.3.0, there will be a undefined symbol __cpu_info.
   * Work around this by calling the intrinsics more directly instead of
   * __builtin_cpu_supports.
   *
   * clang 6.0.0-1ubuntu2 supports vnni but doesn't have
   *   __builtin_cpu_supports("avx512vnni")
   * so use the hand-coded CPUID for clang.
   */
  unsigned int m = __get_cpuid_max(0, 0);
  unsigned int eax, ebx, ecx, edx;
#  endif
  if (m >= 7) {
#  if defined(_MSC_VER)
    __cpuid(regs, 7);
#  else
    __cpuid_count(7, 0, eax, ebx, ecx, edx);
#  endif
#  ifdef INTGEMM_COMPILER_SUPPORTS_AVX512VNNI
    if (ecx & (1 << 11)) return avx512vnni;
#  endif
#  ifdef INTGEMM_COMPILER_SUPPORTS_AVX512BW
    if (ebx & (1 << 30)) return avx512bw;
#  endif
    if (ebx & (1 << 5)) return avx2;
  }
  if (m >= 1) {
#  if defined(_MSC_VER)
    __cpuid(regs, 1);
#  else
    __cpuid_count(1, 0, eax, ebx, ecx, edx);
#  endif
    if (ecx & (1 << 9)) return ssse3;
    if (edx & (1 << 26)) return sse2;
  }
  return unsupported;
#endif
}

struct TileInfo {
  const Index a_rows;
  const Index a_cols;
  const Index b_rows;
  const Index b_cols;
};

/*
 * 8-bit matrix multiplication
 */
struct Int8 {
  using Integer = int8_t;

  // A's size must be a multiple of 1x64, B's size must be a multiple of 64x8.
  static constexpr TileInfo tile_info{1, 64, 64, 8};

  // Currently A is prepared by quantization but this could theoretically change.
  // A's columns must be a multiple of 8.
  // The number of rows is anything.
  static inline void PrepareA(const float *input, int8_t *output, float quant_mult, Index rows, Index cols) {
    Quantize(input, output, quant_mult, rows * cols);
  }

  // Multiply floats by quant_mult then convert to 8-bit integers with saturation.
  static void (*Quantize)(const float *input, int8_t *output, float quant_mult, Index size);

  // Multiply floats by quant_mult then convert to 8-bit integers with saturation.
  // A version that adds 127 to each number, making sure that all numbers are positive
  static void (*QuantizeU)(const float *input, uint8_t *output, float quant_mult, Index size);

  // Warning: the output of PrepareB depends on the CPU.
  // It will match the Multiply function on the same CPU though.
  static void (*PrepareB)(const float *input, int8_t *output, float quant_mult, Index rows, Index cols);

  // Convert from a B that was already transposed (routine not provided) and
  // quantized (e.g. with Quantize) to the CPU-dependent format used for
  // Multiply.  This is useful for storing a quantized model on disk then in a
  // CPU-independent fashion.
  static void (*PrepareBQuantizedTransposed)(const int8_t *input, int8_t *output, Index inner, Index B_untransposed_cols);

  // Convert from a B that was already transposed (routine not provided) to
  // the CPU-dependent format used for Multiply.  This is useful for storing
  // a quantized model on disk then in a CPU-independent fashion.
  static void (*PrepareBTransposed)(const float *input, int8_t *output, float quant_mul, Index inner, Index B_untransposed_cols);

  // Select columns from a prepared B matrix.  The number of selected columns must be a multiple of 8.
  static void (*SelectColumnsB)(const int8_t *input, int8_t *output, Index rows, const Index *cols_begin, const Index *cols_end);

  // Multiply C = A * B, presuming A and B have been prepared.
  template <typename Callback>
  static void Multiply(const int8_t *A, const int8_t *B, Index A_rows, Index width, Index B_cols, Callback callback) {
    MultiplyImpl<Callback>::run(A, B, A_rows, width, B_cols, callback);
  }

  static const char *const kName;

private:
  template <typename Callback>
  struct MultiplyImpl {
    static void (*run)(const int8_t *A, const int8_t *B, Index A_rows, Index width, Index B_cols, Callback callback);
  };
};

template <typename Callback>
void (*Int8::MultiplyImpl<Callback>::run)(const int8_t *A, const int8_t *B, Index A_rows, Index width, Index B_cols, Callback callback) = ChooseCPU(OMPParallelWrap<Callback, avx512vnni::Kernels8>, OMPParallelWrap<Callback, avx512bw::Kernels8>, OMPParallelWrap<Callback, avx2::Kernels8>, OMPParallelWrap<Callback, ssse3::Kernels8>, Unsupported_8bit::Multiply<Callback>, Unsupported_8bit::Multiply<Callback>);

/*
 * 8-bit matrix multiplication with shifting A by 127
 */
struct Int8Shift {
  using Integer = int8_t;

  // A's size must be a multiple of 1x64, B's size must be a multiple of 64x8.
  static constexpr TileInfo tile_info{1, 64, 64, 8};

  // Identical to the Int8 Version, except it adds 127 to each number, making sure that all numbers are positive.
  static inline void PrepareA(const float *input, int8_t *output, float quant_mult, Index rows, Index cols) {
    QuantizeU(input, reinterpret_cast<uint8_t *>(output), quant_mult, rows * cols);
  }

  // Multiply floats by quant_mult then convert to 8-bit integers with saturation.
  // A version that adds 127 to each number, making sure that all numbers are positive
  static void (*QuantizeU)(const float *input, uint8_t *output, float quant_mult, Index size);
  
  // Warning: the output of PrepareB depends on the CPU.
  // It will match the Multiply function on the same CPU though.
  static void PrepareB(const float *input, int8_t *output, float quant_mult, Index rows, Index cols) {
    Int8::PrepareB(input, output, quant_mult, rows, cols);
  }

  // Select columns from a prepared B matrix.  The number of selected columns must be a multiple of 8. 
  static void SelectColumnsB(const int8_t *input, int8_t *output, Index rows, const Index *cols_begin, const Index *cols_end) {
    Int8::SelectColumnsB(input, output, rows, cols_begin, cols_end);
  }

  // A slightly faster version compared to the Int8 one (assuming a bias is used) because of better handling of the sign bit
  // Multiply C = A * B + Bias, presuming A, B and Bias have all been prepared (for A, PrepareAnew should be used
  template<class Callback>
  static void Multiply(const int8_t *A, const int8_t *B, Index A_rows, Index width, Index B_cols, Callback callback) {
    MultiplyImpl<Callback>::run((const uint8_t *)A, B, A_rows, width, B_cols, callback);
  }

  // This function prepares the bias for the Multiply routine that does unsigned * signed multiplication.
  // The function takes:
  // a preparedB matrix, width, B_cols and
  // the callback UnquantizeAndAddBiasAndWrite(unquant_mult, Bias_matrix, Bias_matrix)
  // unquant_mult is computed by (-1)*(alpha)*(alpha)/(127.0f);
  template<class Callback>
  static void PrepareBias(const int8_t *B, Index width, Index B_cols, Callback callback) {
    PrepareBiasImpl<Callback>::run(B, width, B_cols, callback);
  }
  
  static const char *const kName;

private:
  template <typename Callback>
  struct MultiplyImpl {
    static void (*run)(const uint8_t *A, const int8_t *B, Index A_rows, Index width, Index B_cols, Callback callback);
  };

  template <typename Callback>
  struct PrepareBiasImpl {
    static void (*run)(const int8_t *B, Index width, Index B_cols, Callback callback);
  };
};

template <class Callback>
void (*Int8Shift::MultiplyImpl<Callback>::run)(const uint8_t *A, const int8_t *B, Index A_rows, Index width, Index B_cols, Callback callback) = ChooseCPU(
    OMPParallelWrap8Shift<Callback, avx512vnni::Kernels8>,
    OMPParallelWrap8Shift<Callback, avx512bw::Kernels8>,
    OMPParallelWrap8Shift<Callback, avx2::Kernels8>,
    OMPParallelWrap8Shift<Callback, ssse3::Kernels8>, 
    Unsupported_8bit::Multiply8Shift<Callback>, Unsupported_8bit::Multiply8Shift<Callback>);

template <class Callback>
void (*Int8Shift::PrepareBiasImpl<Callback>::run)(const int8_t *B, Index width, Index B_cols, Callback callback) = ChooseCPU(avx512vnni::Kernels8::PrepareBias<Callback>, avx512bw::Kernels8::PrepareBias<Callback>, avx2::Kernels8::PrepareBias<Callback>, ssse3::Kernels8::PrepareBias<Callback>, ssse3::Kernels8::PrepareBias<Callback>, Unsupported_8bit::PrepareBias);

/*
 * 16-bit matrix multiplication
 */
struct Int16 {
  using Integer = int16_t;

  // A's size must be a multiple of 1x32, B's size must be a multiple of 32x8.
  static constexpr TileInfo tile_info{1, 32, 32, 8};

  // Currently A is prepared by quantization but this could theoretically change.
  // A's columns must be a multiple of 8.
  // The number of rows is anything.
  static inline void PrepareA(const float *input, int16_t *output, float quant_mult, Index rows, Index cols) {
    Quantize(input, output, quant_mult, rows * cols);
  }

  // Multiply floats by quant_mult then convert to 16-bit integers with saturation.
  // input
  static void (*Quantize)(const float *input, int16_t *output, float quant_mult, Index size);

  // Warning: the output of PrepareB depends on the CPU.
  // It will match the Multiply function on the same CPU though.
  static void (*PrepareB)(const float *input, int16_t *output, float quant_mult, Index rows, Index cols);

  // Convert from a B that was already transposed (routine not provided) and
  // quantized (e.g. with Quantize) to the CPU-dependent format used for
  // Multiply.  This is useful for storing a quantized model on disk then in a
  // CPU-independent fashion.
  static void (*PrepareBQuantizedTransposed)(const int16_t *input, int16_t *output, Index inner, Index B_untransposed_cols);

  // Convert from a B that was already transposed (routine not provided) to
  // the CPU-dependent format used for Multiply.  This is useful for storing
  // a quantized model on disk then in a CPU-independent fashion.
  static void (*PrepareBTransposed)(const float *input, int16_t *output, float quant_mul, Index inner, Index B_untransposed_cols);

  // Select columns from a prepared B matrix.  The number of selected columns must be a multiple of 8. 
  static void (*SelectColumnsB)(const int16_t *input, int16_t *output, Index rows, const Index *cols_begin, const Index *cols_end);

  // Multiply C = A * B, presuming A and B have been prepared.
  template <typename Callback>
  static void Multiply(const int16_t *A, const int16_t *B, Index A_rows, Index width, Index B_cols, Callback callback) {
    MultiplyImpl<Callback>::run(A, B, A_rows, width, B_cols, callback);
  }

  static const char *const kName;

private:
  template <typename Callback>
  struct MultiplyImpl {
    static void (*run)(const int16_t *A, const int16_t *B, Index A_rows, Index width, Index B_cols, Callback callback);
  };
};

template <typename Callback>
void (*Int16::MultiplyImpl<Callback>::run)(const int16_t *A, const int16_t *B, Index A_rows, Index width, Index B_cols, Callback callback) = ChooseCPU(OMPParallelWrap<Callback, avx512bw::Kernels16> /*TODO VNNI 16-bit. */, OMPParallelWrap<Callback, avx512bw::Kernels16>, OMPParallelWrap<Callback, avx2::Kernels16>, OMPParallelWrap<Callback, sse2::Kernels16>, OMPParallelWrap<Callback, sse2::Kernels16>, Unsupported_16bit::Multiply<Callback>);

extern const CPUType kCPU;

// Get the maximum absolute value of an array of floats. The number of floats must be a multiple of 16 and 64-byte aligned.
extern float (*MaxAbsolute)(const float *begin, const float *end);

// Get a Quantization value that is equant to the mean of the data +N standard deviations. Use 2 by default
extern MeanStd (*VectorMeanStd)(const float *begin, const float *end, bool);

/* Returns the Mean and the Standard deviation of a vector. 
 * If "absolute" is set to true, it computes the mean and the standard deviation of the absolute values of the vector */
static inline MeanStd GetVectorMeanStd(const float * begin, const float * end, bool absolute=false) {
  return VectorMeanStd(begin, end, absolute);
}


} // namespace intgemm
