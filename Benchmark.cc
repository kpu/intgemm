#include "avx512_gemm.h"
#include "avx2_gemm.h"
#include "SSE_Matrix_Mult.h"
#include "Quantize.h"
#include "StopWatch.h"

#include <cassert>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <cstdlib>

#include <iostream>

namespace intgemm {

void Time(int num_A_rows, int num_B_rows, int width, int repeat = 20) {
    std::cout << num_A_rows << '\t' << num_B_rows << '\t' << width << '\n';
    float * A = static_cast<float*>(aligned_alloc(64, sizeof(float) * num_A_rows * width));
    float * B = static_cast<float*>(aligned_alloc(64, sizeof(float) * num_B_rows * width));
    
    for (int i = 0; i < num_A_rows*width; i++) {
        A[i] = ((float)rand()/(float)RAND_MAX)*2.0f - 1.0f;
    }
    
    for (int i = 0; i < num_B_rows*width; i++) {
        B[i] = ((float)rand()/(float)RAND_MAX)*2.0f - 1.0f;
    }

    __m256i * quant_A = static_cast<__m256i *>(aligned_alloc(64, num_A_rows*width * 2));
    __m256i * quant_B = static_cast<__m256i *>(aligned_alloc(64, num_B_rows*width * 2));
    float *AVX_C = static_cast<float*>(aligned_alloc(64, num_A_rows * num_B_rows * sizeof(float)));

    // We quantize with 10 bits of precision. This works well "universally". 
    // See the top of this file for more info on why.
    float quant_mult = 1000.0;
    // If we quantize to n bits and then multiply the values together, the result will be quantized to n^2 bits.
    // So we must divide by 1.0/(n^2) to get back the original value.
    float unquant_mult = 1.0/(quant_mult*quant_mult);

    intgemm::AVX2::Quantize16(B, (int16_t*)quant_B, quant_mult, num_B_rows * width);
    intgemm::AVX2::Quantize16(A, (int16_t*)quant_A, quant_mult, num_A_rows * width);

    AVX2::MatrixMult16((const __m256i *)quant_A, (const __m256i *)quant_B, AVX_C, unquant_mult, num_A_rows, num_B_rows, width);
    {
      StopWatch w("16-bit", repeat);
      for (int i = 0; i < repeat; ++i)
        AVX2::MatrixMult16((const __m256i *)quant_A, (const __m256i *)quant_B, AVX_C, unquant_mult, num_A_rows, num_B_rows, width);
    }


    // Moving on to 8-bit.
    quant_mult = 64;
    unquant_mult = 1.0/(quant_mult*quant_mult);

    {
//      StopWatch w("Quantize8 B");
      intgemm::AVX2::Quantize8(B, (int8_t*)quant_B, quant_mult, num_B_rows * width);
    }
    {
//      StopWatch w("Quantize8 A");
      intgemm::AVX2::Quantize8(A, (int8_t*)quant_A, quant_mult, num_A_rows * width);
    }

    AVX2::MatrixMult8((const __m256i *)quant_A, (const __m256i *)quant_B, AVX_C, unquant_mult, num_A_rows, num_B_rows, width);
    {
      StopWatch w("8-bit Base", repeat);
      for (int i = 0; i < repeat; ++i)
        AVX2::MatrixMult8((const __m256i *)quant_A, (const __m256i *)quant_B, AVX_C, unquant_mult, num_A_rows, num_B_rows, width);
    }
    AVX2::MatrixMult8Contrast((const __m256i *)quant_A, (const __m256i *)quant_B, AVX_C, unquant_mult, num_A_rows, num_B_rows, width);
    {
      StopWatch w("8-bit Contig", repeat);
      for (int i = 0; i < repeat; ++i)
        AVX2::MatrixMult8Contrast((const __m256i *)quant_A, (const __m256i *)quant_B, AVX_C, unquant_mult, num_A_rows, num_B_rows, width);
    }
    AVX2::MatrixMult8ASM((const __m256i *)quant_A, (const __m256i *)quant_B, AVX_C, unquant_mult, num_A_rows, num_B_rows, width);
    {
      StopWatch w("8-bit Cont+ASM", repeat);
      for (int i = 0; i < repeat; ++i)
        AVX2::MatrixMult8ASM((const __m256i *)quant_A, (const __m256i *)quant_B, AVX_C, unquant_mult, num_A_rows, num_B_rows, width);
    }

    free(A);
    free(B);
    free(quant_A);
    free(quant_B);
    free(AVX_C);
}

} // namespace intgemm

// Program takes no input
int main(int argc, char ** argv) {
    std::srand(45678);
    using namespace intgemm;
    // Top matrix sizes from Marian
    Time(8, 256, 256);
    Time(8, 256, 2048);
    Time(8, 2048, 256);
    Time(320, 256, 256);
    Time(472, 256, 256);
    Time(248, 256, 256);
    Time(200, 256, 256);
    // Additional stuff
    Time(256, 256, 256);
    Time(512, 512, 512);
    Time(1024, 1024, 1024);
    Time(4096, 4096, 4096, 3);
    Time(4096, 4096, 2048, 3);
    Time(4096, 4096, 1024, 3);
    Time(4096, 4096, 512, 3);
    Time(4096, 4096, 256, 3);
    Time(4096, 4096, 128, 3);
    return 0;
}


