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
#include "StopWatch.h"

#include <cassert>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <cstdlib>

#include <iostream>

// Compute A*B^T very naively.
void SlowRef_Float(const float * A, const float * B, float * C, int num_A_rows, int num_B_rows, int width)
{
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

void SlowRef_16(const int16_t * A, const int16_t * B, float * C, float quant_mult, int num_A_rows, int num_B_rows, int width)
{
    for (int i = 0; i < num_A_rows; i++) {
        const int16_t * A_row = A + i*width;
        float * C_row = C + i*num_B_rows;
        for (int j = 0; j < num_B_rows; j++) {
            const int16_t * B_row = B + j*width;
            float sum = 0.0f;
            for (int k = 0; k < width; k++) {
                sum += A_row[k]*B_row[k];
            }
            C_row[j] = sum * quant_mult;
        }
    }
}

void Time(int num_A_rows, int num_B_rows, int width) {
    std::cerr << num_A_rows << '\t' << num_B_rows << '\t' << width << '\t';
    float * A = static_cast<float*>(aligned_alloc(64, sizeof(float) * num_A_rows * width));
    float * B = static_cast<float*>(aligned_alloc(64, sizeof(float) * num_B_rows * width));
    
    for (int i = 0; i < num_A_rows*width; i++) {
        A[i] = ((float)rand()/(float)RAND_MAX)*2.0f - 1.0f;
    }
    
    for (int i = 0; i < num_B_rows*width; i++) {
        B[i] = ((float)rand()/(float)RAND_MAX)*2.0f - 1.0f;
    }
    
    // Each __m512i fits 8 16-bit integers, so we assume the width is a multiple of 8.
    // We could pad with 0 in the general case.
    __m512i * quant_A = static_cast<__m512i *>(aligned_alloc(64, num_A_rows*width * 2));
    __m512i * quant_B = static_cast<__m512i *>(aligned_alloc(64, num_B_rows*width * 2));

    // We quantize with 10 bits of precision. This works well "universally". 
    // See the top of this file for more info on why.
    //double quant_mult = pow(2.0, 10.0);
    double quant_mult = 1000.0;
    
    // If we quantize to n bits and then multiply the values together, the result will be quantized to n^2 bits.
    // So we must divide by 1.0/(n^2) to get back the original value.
    double unquant_mult = 1.0/(quant_mult*quant_mult);

    // The weight matrix should be quantized before starting decoding, since it is known beforehand.
    AVX_Quantize16(B, (int16_t*)quant_B, (float)quant_mult, num_B_rows * width);
    // The activation matrix must be quantized on-the-fly.
    AVX_Quantize16(A, (int16_t*)quant_A, (float)quant_mult, num_A_rows * width);
    float * AVX_C = new float[num_A_rows*num_B_rows];
    memset(AVX_C, 0, sizeof(float) * num_A_rows*num_B_rows);
    // Burn in
    AVX_MatrixMult16(quant_A, quant_B, AVX_C, (float)unquant_mult, num_A_rows, num_B_rows, width);
    {
      StopWatch w("AVX");
      for (int i = 0; i < 10; ++i)
        AVX_MatrixMult16(quant_A, quant_B, AVX_C, (float)unquant_mult, num_A_rows, num_B_rows, width);
    }

    float * ref_C = new float[num_A_rows*num_B_rows];
    SlowRef_16((const int16_t*)quant_A, (const int16_t*)quant_B, ref_C, (float)unquant_mult, num_A_rows, num_B_rows, width);

    float *float_C = new float[num_A_rows*num_B_rows];
    SlowRef_Float(A, B, float_C, num_A_rows, num_B_rows, width);

    for (int i = 0; i < num_A_rows * num_B_rows; ++i) {
      float diff16 = fabs(ref_C[i] - AVX_C[i]);
      float difff = fabs(float_C[i] - AVX_C[i]);
      if (diff16 > 0.0001 || difff > 1) {
        std::cerr << "Bug at " << i << ' ' << ref_C[i] << ' ' << AVX_C[i] << ' ' << float_C[i] << '\n';
      }
    }

    free(A);
    free(B);
    free(quant_A);
    free(quant_B);
    delete [] AVX_C;
    delete [] ref_C;
    delete [] float_C;
}

// Program takes no input
int main(int argc, char ** argv) {
    std::srand(45678);
    Time(4, 64, 64);
    Time(512, 1024, 1024);
    Time(1024, 2048, 1024);
    Time(64, 64, 64);
    Time(64, 64, 512);
    return 0;
}


