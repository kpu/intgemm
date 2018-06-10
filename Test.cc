// Based on https://arxiv.org/abs/1705.01991

// Copyright (c) 2017 Microsoft Corporation

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "AVX_Matrix_Mult.h"
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

// Compute A*B^T very naively.
void SlowRefFloat(const float * A, const float * B, float * C, int num_A_rows, int num_B_rows, int width) {
    for (int i = 0; i < num_A_rows; i++) {
        const float * A_row = A + i*width;
        float * C_row = C + i*num_B_rows;
        for (int j = 0; j < num_B_rows; j++) {
            const float * B_row = B + j*width;
            float sum = 0.0f;
            for (int k = 0; k < width; k++) {
                sum += A_row[k]*B_row[k];
            }
            C_row[j] = sum;
        }
    }
}

void SlowRef16(const int16_t * A, const int16_t * B, float * C, float quant_mult, int num_A_rows, int num_B_rows, int width) {
    for (int i = 0; i < num_A_rows; i++) {
        const int16_t * A_row = A + i*width;
        float * C_row = C + i*num_B_rows;
        for (int j = 0; j < num_B_rows; j++) {
            const int16_t * B_row = B + j*width;
            int32_t sum = 0.0f;
            for (int k = 0; k < width; k++) {
                sum += A_row[k]*B_row[k];
            }
            C_row[j] = sum * quant_mult;
        }
    }
}

void SlowRef8(const int8_t * A, const int8_t * B, float * C, float unquant_mult, int num_A_rows, int num_B_rows, int width) {
    for (int i = 0; i < num_A_rows; i++) {
        const int8_t *A_row = A + i * width;
        float *C_row = C + i * num_B_rows;
        for (int j = 0; j < num_B_rows; j++) {
            const int8_t *B_row = B + j*width;
            int32_t sum = 0;
            for (int k = 0; k < width; k++) {
                sum += static_cast<int32_t>(A_row[k])*static_cast<int32_t>(B_row[k]);
            }
            C_row[j] = sum * unquant_mult;
        }
    }
}

void Compare(const float *float_ref, const float *int_ref, const float *int_test, std::size_t size) {
  float int_sum = 0.0, float_sum = 0.0;
  for (std::size_t i = 0; i < size; ++i) {
    float int_diff = int_ref[i] - int_test[i];
    float float_diff = float_ref[i] - int_test[i];
/*    if (fabs(int_diff) > .1 || fabs(float_diff) > 1) {
      std::cerr << "Inaccurate at " << i << ' ' << float_ref[i] << ' ' << int_ref[i] << ' ' << int_test[i] << '\n';
    }*/
    int_sum += int_diff * int_diff;
    float_sum += float_diff * float_diff;
  }
  std::cerr << "Float MSE = " << sqrt(float_sum / size) << "\tInt MSE = " << sqrt(int_sum / size) << std::endl;
}

void Time(int num_A_rows, int num_B_rows, int width, int repeat = 10) {
    std::cout << num_A_rows << '\t' << num_B_rows << '\t' << width << '\n';
    float * A = static_cast<float*>(aligned_alloc(64, sizeof(float) * num_A_rows * width));
    float * B = static_cast<float*>(aligned_alloc(64, sizeof(float) * num_B_rows * width));
    
    for (int i = 0; i < num_A_rows*width; i++) {
        A[i] = ((float)rand()/(float)RAND_MAX)*2.0f - 1.0f;
    }
    
    for (int i = 0; i < num_B_rows*width; i++) {
        B[i] = ((float)rand()/(float)RAND_MAX)*2.0f - 1.0f;
    }

    float *float_C = new float[num_A_rows*num_B_rows];
    SlowRefFloat(A, B, float_C, num_A_rows, num_B_rows, width);
    
    // Each __m512i fits 8 16-bit integers, so we assume the width is a multiple of 8.
    // We could pad with 0 in the general case.
    __m512i * quant_A = static_cast<__m512i *>(aligned_alloc(64, num_A_rows*width * 2));
    __m512i * quant_B = static_cast<__m512i *>(aligned_alloc(64, num_B_rows*width * 2));

    // We quantize with 10 bits of precision. This works well "universally". 
    // See the top of this file for more info on why.
    float quant_mult = 1000.0;
    // If we quantize to n bits and then multiply the values together, the result will be quantized to n^2 bits.
    // So we must divide by 1.0/(n^2) to get back the original value.
    float unquant_mult = 1.0/(quant_mult*quant_mult);

    // The weight matrix should be quantized before starting decoding, since it is known beforehand.
    AVX512::Quantize16(B, (int16_t*)quant_B, quant_mult, num_B_rows * width);
    // The activation matrix must be quantized on-the-fly.
    AVX512::Quantize16(A, (int16_t*)quant_A, quant_mult, num_A_rows * width);
    float * AVX_C = new float[num_A_rows*num_B_rows];
    memset(AVX_C, 0, sizeof(float) * num_A_rows*num_B_rows);
    // Burn in.
    AVX512::MatrixMult16(quant_A, quant_B, AVX_C, unquant_mult, num_A_rows, num_B_rows, width);
    {
      StopWatch w("16-bit");
      for (int i = 0; i < repeat; ++i)
        AVX512::MatrixMult16(quant_A, quant_B, AVX_C, unquant_mult, num_A_rows, num_B_rows, width);
    }

    float *ref_C = new float[num_A_rows*num_B_rows];
    SlowRef16((const int16_t*)quant_A, (const int16_t*)quant_B, ref_C, unquant_mult, num_A_rows, num_B_rows, width);
    Compare(float_C, ref_C, AVX_C, num_A_rows*num_B_rows);

    // Moving on to 8-bit.
    quant_mult = 64;
    unquant_mult = 1.0/(quant_mult*quant_mult);

    {
      StopWatch w("Quantize8 B");
      intgemm::AVX512::Quantize8(B, (int8_t*)quant_B, quant_mult, num_B_rows * width);
    }
    {
      StopWatch w("Quantize8 A");
      intgemm::AVX512::Quantize8(A, (int8_t*)quant_A, quant_mult, num_A_rows * width);
    }

    AVX512::MatrixMult8((const __m512i *)quant_B, (const __m512i *)quant_A, AVX_C, unquant_mult, num_B_rows, num_A_rows, width);
    {
      StopWatch w("8-bitr", repeat);
      for (int i = 0; i < repeat; ++i)
        AVX512::MatrixMult8((const __m512i *)quant_B, (const __m512i *)quant_A, AVX_C, unquant_mult, num_B_rows, num_A_rows, width);
    }
    AVX512::MatrixMult8((const __m512i *)quant_A, (const __m512i *)quant_B, AVX_C, unquant_mult, num_A_rows, num_B_rows, width);
    {
      StopWatch w("8-bit", repeat);
      for (int i = 0; i < repeat; ++i)
        AVX512::MatrixMult8((const __m512i *)quant_A, (const __m512i *)quant_B, AVX_C, unquant_mult, num_A_rows, num_B_rows, width);
    }


    SlowRef8((const int8_t*)quant_A, (const int8_t*)quant_B, ref_C, unquant_mult, num_A_rows, num_B_rows, width);
    Compare(float_C, ref_C, AVX_C, num_A_rows*num_B_rows);

    free(A);
    free(B);
    free(quant_A);
    free(quant_B);
    delete [] AVX_C;
    delete [] ref_C;
    delete [] float_C;
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
    Time(4096, 4096, 4096, 1);
    return 0;
}


