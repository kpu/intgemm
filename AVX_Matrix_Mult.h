// This is an AVX512F implementation of int16_t multiply based on Jacob
// Devlin's SSE code.  The original SSE code was:

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

#pragma once
#include <immintrin.h>
#include <cstddef>

// ***************************************
// ************** IMPORTANT **************
// ***************************************
// The biggest "gotcha" when using this type of multiplication is dealing with overflow related to quantization.
// It is NOT enough to simply ensure that A and B fit into 16 bit integers. If A and B are quantized with $n$ bits,
// the result of multiplying them together will be quantized to $n^2$ bits. So if they are near the boundary of the 16-bit
// mark, then the result will be near 32-bits and overflow. However, if we use, say, n = 10 bits, then the product is 20 bits.
// This gives us 12 bits left over for the accumulation. So as long as the width of the common dimension is less than 2^12 = 4096, it is
// *impossible* to overflow. If we used, say, n = 12 bits, then we have 32-(12*2) = 8 bits left over. So we *could* overflow if width > 2^8.
//
// So, the tradeoff is between quantization precision and possibility of overflow. A good general value is 10 bits, since this gives high precision
// (precision is 1/2^10 ~= 0.001, which is more than what's needed for almost all neural nets), and cannot overflow unless the matrix width is > 4096. 

// This quantizes floating point values into fixed-point 16-bit integers. Effectively, we are performing an SSE version of
// float x = ...;
// int16_t y = (int16_t)(quant_mult*x);
// 
// Except that the casting is saturated. However, you should always ensure that the input fits into a fixed range anyways.
// I.e., you should ensure that quant_mult*x fits into the range [-2^15, 2^15].
// This should always be possible because the value you're quantizing will either be NN weights or NN activations, both of
// which can be clipped to a fixed range during training.

// input must be 64-byte aligned.
void AVX_Quantize16(const float *input, int16_t *output, float quant_mult, std::size_t size);
void AVX_Quantize8(const float *input, int32_t* output, float quant_mult, std::size_t size);

// We are multiplying A * B^T, as opposed to A * B. This is important because it means we can do consecutive memory access on A * B^T which allows to to take the most
// advantage of L1 cache.
// 
// B is typically a weight matrix, so it can be pre-processed offline, and therefore this transpose does not cost anything.
// A is typically an activation minibatch matrix.
// A and B must be 64-byte aligned.
// C should be the usual 4-byte alignment.
void AVX_MatrixMult16(const __m512i * A, const __m512i * B, float * C, float unquant_mult, int num_A_rows, int num_B_rows, int width);
