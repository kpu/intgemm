// This is an AVX512F implementation 16-bit and 8-bit multiply based on Jacob
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

namespace intgemm {
#ifdef __AVX512F__
namespace AVX512 {

// We are multiplying A * B^T, as opposed to A * B. This is important because it means we can do consecutive memory access on A * B^T which allows to to take the most
// advantage of L1 cache.
// 
// B is typically a weight matrix, so it can be pre-processed offline, and therefore this transpose does not cost anything.
// A is typically an activation minibatch matrix.
// A and B must be 64-byte aligned.
// C should be the usual 4-byte alignment.
void MatrixMult16(const __m512i * A, const __m512i * B, float * C, float unquant_mult, int num_A_rows, int num_B_rows, int width);
void MatrixMult8(const __m512i * A, const __m512i * B, float * C, float unquant_mult, int num_A_rows, int num_B_rows, int width);
void MatrixMult8Contrast(const __m512i * A, const __m512i * B, float * C, float unquant_mult, int num_A_rows, int num_B_rows, int width);

} // namespace AVX512
#endif // __AVX512F__
} // namespace intgemm
