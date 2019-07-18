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

// Yes, both headers due to the debacle about int32_t
#include <cstdint>
#include <stdint.h>

#include "types.h"
#include "sse2_gemm.h"
#include "ssse3_gemm.h"
#include "avx2_gemm.h"
#include "cops.h"
#ifndef INTGEMM_NO_AVX512
#include "avx512_gemm.h"
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
  static void SelectColumnsB(const int16_t *, int16_t *, Index, const Index *, const Index *) {
    throw UnsupportedCPU();
  }
  template<class WriteC>
  static void Multiply(const int16_t *, const int16_t *, WriteC, Index, Index, Index) {
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
  static void PrepareB(const float *, int8_t *, float, Index, Index) {
    throw UnsupportedCPU();
  }
  static void PrepareBiasFor8(const float *input, float *bias, float alpha, Index rows, Index cols) {
    throw UnsupportedCPU();
  }
  static void SelectColumnsB(const int8_t *, int8_t *, Index, const Index *, const Index *) {
    throw UnsupportedCPU();
  }
  template<class WriteC>
  static void Multiply(const int8_t *, const int8_t *, WriteC, Index, Index, Index) {
    throw UnsupportedCPU();
  }
  template<class WriteC>
  static void Multiply8new(const uint8_t *, const int8_t *, WriteC, Index, Index, Index) {
    throw UnsupportedCPU();
  }
  constexpr static const char *const kName = "8-bit Unsupported";
};

#ifdef INTGEMM_NO_AVX512
// These won't ever be called in this capacity, but it does let the code below compile.
typedef Unsupported_16bit AVX512_16bit;
typedef Unsupported_8bit AVX512_8bit;
namespace avx512f {
float MaxAbsolute(const float *begin, const float *end) {
  throw UnsupportedCPU();
}
} //namespace
#endif

/* Returns:
 * avx512 if the CPU supports AVX512F (though really it should be AVX512BW, but
 * cloud providers lie).  TODO: don't catch Knights processors with this.
 *
 * avx2 if the CPU supports AVX2
 * 
 * ssse3 if the CPU supports SSSE3 (this distinction from SSE2 matters for 8-bit)
 * 
 * sse2 if the CPU supports SSE2
 *
 * unsupported otherwise
 */
template <class T> T ChooseCPU(T avx512, T avx2, T ssse3, T sse2, T unsupported) {
  // TODO: don't catch Knights processors here!
#ifndef INTGEMM_NO_AVX512
  if (__builtin_cpu_supports("avx512f")) {
    return avx512;
  }
#endif
  if (__builtin_cpu_supports("avx2")) {
    return avx2;
  } else if (__builtin_cpu_supports("ssse3")) {
    return ssse3;
  } else if (__builtin_cpu_supports("sse2")) {
    return sse2;
  } else {
    return unsupported;
  }
}

/* 16-bit matrix multiplication. */
template<class WriteC>
class Int16Mult {
public:
  // Multiply C = A * B, presuming A and B have been prepared.
  static void (*Multiply)(const int16_t *A, const int16_t *B, WriteC functor, Index A_rows, Index width, Index B_cols);
};

template <class WriteC>
void (*Int16Mult<WriteC>::Multiply)(const int16_t *A, const int16_t *B, WriteC functor, Index A_rows, Index width, Index B_cols) = ChooseCPU(AVX512_16bit::Multiply<WriteC>, AVX2_16bit::Multiply<WriteC>, SSE2_16bit::Multiply<WriteC>, SSE2_16bit::Multiply<WriteC>, Unsupported_16bit::Multiply);

struct Int16 {
  typedef int16_t Integer;

  // A's size must be a multiple of 1x32.
  static const Index kATileRow = 1;
  static const Index kATileCol = 32;
  // B's size must be a multiple of 32x8.
  static const Index kBTileRow = 32;
  static const Index kBTileCol = 8;

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

  // Select columns from a prepared B matrix.  The number of selected columns must be a multiple of 8. 
  static void (*SelectColumnsB)(const int16_t *input, int16_t *output, Index rows, const Index *cols_begin, const Index *cols_end);

  // Multiply C = A * B, presuming A and B have been prepared.
  template<class WriteC>
  static void Multiply(const int16_t *A, const int16_t *B, WriteC functor, Index A_rows, Index width, Index B_cols) {
    Int16Mult<WriteC>::Multiply(A, B, functor, A_rows, width, B_cols);
  }

  static const char *const kName;
};

/* 8-bit matrix multiplication */
template<class WriteC>
class Int8Mult {
public:
  // Multiply C = A * B, presuming A and B have been prepared.
  static void (*Multiply)(const int8_t *A, const int8_t *B, WriteC functor, Index A_rows, Index width, Index B_cols);
  static void (*Multiply8new)(const uint8_t *A, const int8_t *B, WriteC functor, Index A_rows, Index width, Index B_cols);
};

template <class WriteC>
void (*Int8Mult<WriteC>::Multiply)(const int8_t *A, const int8_t *B, WriteC functor, Index A_rows, Index width, Index B_cols) = ChooseCPU(AVX512_8bit::Multiply<WriteC>, AVX2_8bit::Multiply<WriteC>, SSSE3_8bit::Multiply<WriteC>, SSSE3_8bit::Multiply<WriteC>, Unsupported_8bit::Multiply);

template <class WriteC>
void (*Int8Mult<WriteC>::Multiply8new)(const uint8_t *A, const int8_t *B, WriteC functor, Index A_rows, Index width, Index B_cols) = ChooseCPU(AVX512_8bit::Multiply8new<WriteC>, AVX2_8bit::Multiply8new<WriteC>, SSSE3_8bit::Multiply8new<WriteC>, SSSE3_8bit::Multiply8new<WriteC>, Unsupported_8bit::Multiply8new);


struct Int8 {
  typedef int8_t Integer;

  // A's size must be a multiple of 1x64.
  static const Index kATileRow = 1;
  static const Index kATileCol = 64;
  // B's size must be a multiple of 64x8.
  static const Index kBTileRow = 64;
  static const Index kBTileCol = 8;

  // Currently A is prepared by quantization but this could theoretically change.
  // A's columns must be a multiple of 8.
  // The number of rows is anything.
  static inline void PrepareA(const float *input, int8_t *output, float quant_mult, Index rows, Index cols) {
    Quantize(input, output, quant_mult, rows * cols);
  }

  static inline void PrepareANew(const float *input, int8_t *output, float quant_mult, Index rows, Index cols) {
    QuantizeU(input, reinterpret_cast<uint8_t *>(output), quant_mult, rows * cols);
  }

  // Multiply floats by quant_mult then convert to 8-bit integers with saturation.
  static void (*Quantize)(const float *input, int8_t *output, float quant_mult, Index size);

  // Multiply floats by quant_mult then convert to 8-bit integers with saturation.
  static void (*QuantizeU)(const float *input, uint8_t *output, float quant_mult, Index size);

  // PrepareB
  static void (*PrepareBiasFor8)(const float *input, float *bias, float alpha, Index rows, Index cols);
  
  // Warning: the output of PrepareB depends on the CPU.
  // It will match the Multiply function on the same CPU though.
  static void (*PrepareB)(const float *input, int8_t *output, float quant_mult, Index rows, Index cols);

  // Select columns from a prepared B matrix.  The number of selected columns must be a multiple of 8. 
  static void (*SelectColumnsB)(const int8_t *input, int8_t *output, Index rows, const Index *cols_begin, const Index *cols_end);

  // Multiply C = A * B, presuming A and B have been prepared.
  template<class WriteC>
  static void Multiply(const int8_t *A, const int8_t *B, WriteC functor, Index A_rows, Index width, Index B_cols) {
    Int8Mult<WriteC>::Multiply(A, B, functor, A_rows, width, B_cols);
  }

  template<class WriteC>
  static void Multiply8new(const int8_t *A, const int8_t *B, WriteC functor, Index A_rows, Index width, Index B_cols) {
    Int8Mult<WriteC>::Multiply8new((const uint8_t *)A, B, functor, A_rows, width, B_cols);
  }
  
  static const char *const kName;
};

extern const CPUType kCPU;

// Get the maximum absolute value of an array of floats. The number of floats must be a multiple of 16 and 64-byte aligned.
extern float (*MaxAbsolute)(const float *begin, const float *end);


} // namespace intgemm
