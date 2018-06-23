#pragma once
#include <stdint.h>

#include <exception>

/* Dispatch to functions based on runtime CPUID.  This adds one call-by-variable to each call. */

namespace intgemm {

// This will be thrown if a CPU isn't supported yet.
class UnsupportedCPU : public std::exception {
  public:
    UnsupportedCPU();

    ~UnsupportedCPU();

    const char *what() const throw();
};

struct Dispatch_16bit {
  typedef int16_t Integer;

  // Currently A is prepared by quantization but this could theoretically change.
  // A's columns must be a multiple of 8.
  // The number of rows is anything.
  static inline void PrepareA(const float *input, int16_t *output, float quant_mult, int rows, int cols) {
    Quantize(input, output, quant_mult, rows * cols);
  }

  static void (*Quantize)(const float *input, int16_t *output, float quant_mult, int size);

  // B's size must be a multiple of this to run on all CPU backends.
  static const int kBTileRow = 32;
  static const int kBTileCol = 8;

  // Warning: the output of PrepareB depends on the CPU.
  // It will match the Multiply function on the same CPU though.
  static void (*PrepareB)(const float *input, int16_t *output, float quant_mult, int rows, int cols);

  static void (*Multiply)(const int16_t *A, const int16_t *B, float *C, float unquant_mult, int A_rows, int width, int B_cols);

  static const char *Name() { return "Dispatch 16-bit"; }
};

struct Dispatch_8bit {
  typedef int8_t Integer;

  // Currently A is prepared by quantization but this could theoretically change.
  // A's columns must be a multiple of 8.
  // The number of rows is anything.
  static inline void PrepareA(const float *input, int8_t *output, float quant_mult, int rows, int cols) {
    Quantize(input, output, quant_mult, rows * cols);
  }

  static void (*Quantize)(const float *input, int8_t *output, float quant_mult, int size);

  // B's size must be a multiple of this to run on all CPU backends.
  static const int kBTileRow = 64;
  static const int kBTileCol = 8;
  
  // Warning: the output of PrepareB depends on the CPU.
  // It will match the Multiply function on the same CPU though.
  static void (*PrepareB)(const float *input, int8_t *output, float quant_mult, int rows, int cols);

  static void (*Multiply)(const int8_t *A, const int8_t *B, float *C, float unquant_mult, int A_rows, int width, int B_cols);
  
  static const char *Name() { return "Dispatch 8-bit"; }
};

} // namespace intgemm
