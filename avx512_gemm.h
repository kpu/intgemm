#pragma once
#include <stdint.h>

namespace intgemm {

struct AVX512_16bit {
  typedef int16_t Integer;

  // Currently A is prepared by quantization but this could theoretically change.
  static inline void PrepareA(const float *input, int16_t *output, float quant_mult, int rows, int cols) {
    Quantize(input, output, quant_mult, rows * cols);
  }

  static void Quantize(const float *input, int16_t *output, float quant_mult, int size);

  // Tile size for B; B must be a multiple of this block size.
  static const int kBTileRow = 32;
  static const int kBTileCol = 8;

  static void PrepareB(const float *input, int16_t *output, float quant_mult, int rows, int cols);

  static void Multiply(const int16_t *A, const int16_t *B, float *C, float unquant_mult, int A_rows, int width, int B_cols);

  static inline const char *Name() { return "AVX512 16-bit"; }
};

struct AVX512_8bit {
  typedef int8_t Integer;

  // Currently A is prepared by quantization but this could theoretically change.
  static inline void PrepareA(const float *input, int8_t *output, float quant_mult, int rows, int cols) {
    Quantize(input, output, quant_mult, rows * cols);
  }

  static void Quantize(const float *input, int8_t *output, float quant_mult, int size);

  // Tile size for B; B must be a multiple of this block size.
  static const int kBTileRow = 64;
  static const int kBTileCol = 8;

  static void PrepareB(const float *input, int8_t *output, float quant_mult, int rows, int cols);

  static void Multiply(const int8_t *A, const int8_t *B, float *C, float unquant_mult, int A_rows, int width, int B_cols);

  static inline const char *Name() { return "AVX512 8-bit"; }
};

} // namespace intgemm
