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

#include <cassert>
#include <emmintrin.h>
#include <immintrin.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <tmmintrin.h>
#include <xmmintrin.h>

#include "stopwatch.h"

/*#include <iostream>
#include <iomanip>

void Print(__m512i value) {
  int32_t out[16];
  _mm512_storeu_si512(&out, value);
  for (int i = 0; i < 16; ++i) {
    std::cout << std::setw(2) << out[i] << ' ';
  }
  std::cout << '\n';
}

void Print(__m256i value) {
  int32_t out[8];
  _mm256_storeu_si256((__m256i*)&out, value);
  for (int i = 0; i < 8; ++i) {
    std::cout << std::setw(2) << out[i] << ' ';
  }
  std::cout << '\n';
}

void Print(__m128i value) {
  int32_t out[4];
  _mm_storeu_si128((__m128i*)&out, value);
  for (int i = 0; i < 4; ++i) {
    std::cout << std::setw(2) << out[i] << ' ';
  }
  std::cout << '\n';
}*/



// This is a reference implementation of 16-bit matrix multiplication described in "Sharp Models on Dull Hardware: Fast and Accurate Neural Machine Translation Decoding on the CPU".
// This model is not as fast as the one in the paper, becuase it uses SSE2 instead of AVX2. AVX2 instructions are only available on more modern CPUs (Haswell or later).
// The only difference between SSE2 and AVX2 is that SSE operates on 128-bit vectors and AVX2 operates on 256-bit vecetors. So AVX2 can fit 16 16-bit integers intead of 8 8-bit integers.
// The algorithm is the same, you just replace these instructions with their 256-bit counterpart, i.e., _mm256_add_epi32, _mm256_madd_epi16, _mm256_hadd_epi32, ...
// Additional improvements can also be made from unrolling the for loop over num_B_rows in SSE_MatrixMult, which is not done here for clarity.

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

void SSE_Quantize(const float * input, __m128i * output, float quant_mult, int num_rows, int width) {
    assert(width % 8 == 0);
    
    int num_input_chunks = width/8;
    
    // Fill an SSE float with 4 copies of the quant mult
    __m128 sse_quant_mult = _mm_set_ps(quant_mult, quant_mult, quant_mult, quant_mult);
    
    for (int i = 0; i < num_rows; i++) {
        const float * input_row = input + i*width;
        __m128i * output_row = output + i*num_input_chunks;
        for (int j = 0; j < num_input_chunks; j++) {
            const float * x = input_row + j*8;
            // Process 8 floats at once, since each __m128i can contain 8 16-bit integers.
            
            // Load floats floats into SSE registers.
            __m128 f_0 = _mm_loadu_ps(x);
            __m128 f_1 = _mm_loadu_ps(x + 4);
            
            // Multiply by quantization factor (e.g., if quant_mult = 1000.0, 0.34291 --> 342.21)
            __m128 m_0 = _mm_mul_ps(f_0, sse_quant_mult);
            __m128 m_1 = _mm_mul_ps(f_1, sse_quant_mult);
            
            // Cast float to 32-bit int (e.g., 342.21 --> 342)
            __m128i i_0 = _mm_cvtps_epi32(m_0);
            __m128i i_1 = _mm_cvtps_epi32(m_1);
            
            // Cast 32-bit int to 16-bit int. You must ensure that these fit into the 16-bit range
            // by clipping values during training.
            *(output_row + j) = _mm_packs_epi32(i_0, i_1);
        }
    }
}

void AVX_Quantize(const float * input, __m256i * output, float quant_mult, int size) {
    assert(size % 16 == 0);
    assert(reinterpret_cast<uintptr_t>(input) % 64 == 0);
    assert(reinterpret_cast<uintptr_t>(output) % 64 == 0);
    // Annoyingly, _mm512_packs_epi32 requires AVX512BW which isn't supported
    // on my target.  Therefore I use _mm256_packs_epi32.

    // Fill with the quantization multiplier.
    const __m512 quant_mult_reg = _mm512_set1_ps(quant_mult);
    const float *end = input + size;

    for (; input != end; input += 16, output += 1) {
      // Load 16 floats
      __m512 val = _mm512_load_ps(input);
      // Multiply each by the quantization factor.
      val = _mm512_mul_ps(val, quant_mult_reg);
      // Cast to 32-bit int
      __m512i as_int =  _mm512_cvtps_epi32(val);
      // Pack into 16-bit ints with saturation.
      // I would do this with two AVX512 registers, but _mm512_packs_epi32 isn't supported on the WNMT machine.
      *output = _mm256_packs_epi32(_mm512_castsi512_si256(as_int), _mm512_extracti64x4_epi64(as_int, 1));
    }    
}

// Assuming sum1, sum2, sum3, and sum4 are arrays 32-bit signed integers,
// reduce within each.
// Returns [sum(sum1), sum(sum2), sum(sum3), sum(sum4)]
// TODO: consider doing in 64-bit, allowing 4 more bits of quantization?
inline __m128i Reduce(__m512i sum1, __m512i sum2, __m512i sum3, __m512i sum4) {
  // 1 2 1 2 1 2 1 2 1 2 1 2 1 2 1 2
  __m512i pack12 = _mm512_add_epi32(_mm512_unpackhi_epi32(sum1, sum2), _mm512_unpacklo_epi32(sum1, sum2));
  // 3 4 3 4 3 4 3 4 3 4 3 4 3 4 3 4
  __m512i pack34 = _mm512_add_epi32(_mm512_unpackhi_epi32(sum3, sum4), _mm512_unpacklo_epi32(sum3, sum4));
  // 1 2 3 4 1 2 3 4 1 2 3 4 1 2 3 4
  __m512i pack1234 = _mm512_add_epi32(_mm512_unpackhi_epi64(pack12, pack34), _mm512_unpacklo_epi64(pack12, pack34));
  // Cut the register into halves and sum those.  1 2 3 4 1 2 3 4
  __m256i halves = _mm256_add_epi32(_mm512_castsi512_si256(pack1234), _mm512_extracti64x4_epi64(pack1234, 1));
  // Again: cut the register into halves and sum those. 1 2 3 4
  __m128i ret = _mm_add_epi32(_mm256_castsi256_si128(halves), _mm256_extracti128_si256(halves, 1));
  return ret;
}

union FloatAccess {
  float as_f[4];
  __m128 as_n;
};

// We are multiplying A * B^T, as opposed to A * B. This is important because it means we can do consecutive memory access on A * B^T which allows to to take the most
// advantage of L1 cache.
// 
// B is typically a weight matrix, so it can be pre-processed offline, and therefore this transpose does not cost anything.
// A is typically an activation minibatch matrix.
// A and B must be 64-byte aligned.
// C should be the usual 4-byte alignment.
void AVX_MatrixMult(const __m512i * A, const __m512i * B, float * C, float unquant_mult, int num_A_rows, int num_B_rows, int width)
{
    assert(num_A_rows % 4 == 0);
    assert(width % 32 == 0);
    assert(reinterpret_cast<uintptr_t>(A) % 64 == 0);
    assert(reinterpret_cast<uintptr_t>(B) % 64 == 0);

    const __m128 unquant_mult_sse = _mm_set1_ps(unquant_mult);

    const int sse_width = width/32;

    // We do loop unrolling over A. This is *significantly* faster
    // since B can live in the registers. We are assuming that
    // A is a multiple of 4, but we can add extra code to handle values of 1, 2, 3.
    //
    // We could also do loop unrolling over B, which adds some additional speedup.
    // We don't do that for the sake of clarity.
    // 
    // There are other memory access patterns we could do, e.g., put B on the outer loop.
    // The justification is that A is typically small enough that it can live in L1 cache.
    // B is usually a larger weight matrix, so it might not be able to. However, we are using
    // each element of B four times while it's still in a register, so caching is not as important.
    for (int i = 0; i < num_A_rows; i += 4) {
        const __m512i * A1_row = A + (i+0)*sse_width;
        const __m512i * A2_row = A + (i+1)*sse_width;
        const __m512i * A3_row = A + (i+2)*sse_width;
        const __m512i * A4_row = A + (i+3)*sse_width;

        for (int j = 0; j < num_B_rows; j++) {
            const __m512i * B_row = B + j*sse_width;

            __m512i sum1 = _mm512_setzero_si512();
            __m512i sum2 = _mm512_setzero_si512();
            __m512i sum3 = _mm512_setzero_si512();
            __m512i sum4 = _mm512_setzero_si512();

            // This is just a simple dot product, unrolled four ways.
            for (int k = 0; k < sse_width; k++) {
                __m512i b = *(B_row + k);
                
                __m512i a1 = *(A1_row + k);
                __m512i a2 = *(A2_row + k);
                __m512i a3 = *(A3_row + k);
                __m512i a4 = *(A4_row + k);

                // madd_epi16 does multiply add on 8 16-bit integers and accumulates into a four 32-bit register.
                // E.g.,
                // a1 = [f1, f2, f3, f4, f5, f6, f7, h8] (16-bit ints)
                // b1 = [h1, h2, h3, h4, h5, h6, h7, h8] (16-bit ints)
                // result = [f1*h1 + f2*h2, f3*h3 + f4*h4, f5*h5 + f6*h6, f7*h7 + f8*h8] (32-bit ints)
                // Then add_epi32 just effectively does a += on these 32-bit integers.
                sum1 = _mm512_add_epi32(sum1, _mm512_madd_epi16(b, a1));
                sum2 = _mm512_add_epi32(sum2, _mm512_madd_epi16(b, a2));
                sum3 = _mm512_add_epi32(sum3, _mm512_madd_epi16(b, a3));
                sum4 = _mm512_add_epi32(sum4, _mm512_madd_epi16(b, a4));
            }
            FloatAccess a;
            // Get floats for each of the sums to write.
            a.as_n = _mm_cvtepi32_ps(Reduce(sum1, sum2, sum3, sum4));
            // Undo quantization scaling.
            a.as_n = _mm_mul_ps(a.as_n, unquant_mult_sse);
            // Also note that the memory acceses on C are not consecutive, but this is a tradeoff that we have to make.
            // We can't have consecutive accesses of A, B, *and* C. But we access A and B a lot more so it makes
            // sense to do it this way.
            // Scatter to outputs:
            *(C + (i+0)*num_B_rows + j) = a.as_f[0];
            *(C + (i+1)*num_B_rows + j) = a.as_f[1];
            *(C + (i+2)*num_B_rows + j) = a.as_f[2];
            *(C + (i+3)*num_B_rows + j) = a.as_f[3];
            /* Sadly the scatter instruction requires avx512vl
             * _mm_i32scatter_ps(C + i * num_B_rows + j, num_b_rows_scatter, float_sums, sizeof(float));
             */
        }
    }
}

// We are multiplying A * B^T, as opposed to A * B. This is important because it means we can do consecutive memory access on A * B^T which allows to to take the most
// advantage of L1 cache.
// 
// B is typically a weight matrix, so it can be pre-processed offline, and therefore this transpose does not cost anything.
// A is typically an activation minibatch matrix.
void SSE_MatrixMult(const __m128i * A, const __m128i * B, float * C, float unquant_mult, int num_A_rows, int num_B_rows, int width)
{
    assert(num_A_rows % 4 == 0);
    assert(width % 8 == 0);

    int sse_width = width/8;

    // We do loop unrolling over A. This is *significantly* faster
    // since B can live in the registers. We are assuming that
    // A is a multiple of 4, but we can add extra code to handle values of 1, 2, 3.
    //
    // We could also do loop unrolling over B, which adds some additional speedup.
    // We don't do that for the sake of clarity.
    // 
    // There are other memory access patterns we could do, e.g., put B on the outer loop.
    // The justification is that A is typically small enough that it can live in L1 cache.
    // B is usually a larger weight matrix, so it might not be able to. However, we are using
    // each element of B four times while it's still in a register, so caching is not as important.
    for (int i = 0; i < num_A_rows; i += 4) {
        const __m128i * A1_row = A + (i+0)*sse_width;
        const __m128i * A2_row = A + (i+1)*sse_width;
        const __m128i * A3_row = A + (i+2)*sse_width;
        const __m128i * A4_row = A + (i+3)*sse_width;

        for (int j = 0; j < num_B_rows; j++) {
            const __m128i * B_row = B + j*sse_width;

            __m128i sum1 = _mm_setzero_si128();
            __m128i sum2 = _mm_setzero_si128();
            __m128i sum3 = _mm_setzero_si128();
            __m128i sum4 = _mm_setzero_si128();

            // This is just a simple dot product, unrolled four ways.
            for (int k = 0; k < sse_width; k++) {
                __m128i b = *(B_row + k);
                
                __m128i a1 = *(A1_row + k);
                __m128i a2 = *(A2_row + k);
                __m128i a3 = *(A3_row + k);
                __m128i a4 = *(A4_row + k);

                // _mm_madd_epi16 does multiply add on 8 16-bit integers and accumulates into a four 32-bit register.
                // E.g.,
                // a1 = [f1, f2, f3, f4, f5, f6, f7, h8] (16-bit ints)
                // b1 = [h1, h2, h3, h4, h5, h6, h7, h8] (16-bit ints)
                // result = [f1*h1 + f2*h2, f3*h3 + f4*h4, f5*h5 + f6*h6, f7*h7 + f8*h8] (32-bit ints)
                // Then _mm_add_epi32 just effectively does a += on these 32-bit integers.

                sum1 = _mm_add_epi32(sum1, _mm_madd_epi16(b, a1));
                sum2 = _mm_add_epi32(sum2, _mm_madd_epi16(b, a2));
                sum3 = _mm_add_epi32(sum3, _mm_madd_epi16(b, a3));
                sum4 = _mm_add_epi32(sum4, _mm_madd_epi16(b, a4));
            }
            
            // We now have each sum spread across 4 32-bit ints in SSE register, e.g., 
            // sum1 = [r1, r2, r3, r4]. We need to compute r1 + r2 + r3 + r4.
            //
            // This uses 'horizontal add' to do that efficiently. The first add gets us
            // [r1 + r2, r2 + r3, r1 + r2, r2 + r3]
            // Then the second gets us.
            // [r1 + r2 + r2 + r3, r2 + r3 + r1 + r2, r1 + r2 + r2 + r3, r2 + r3 + r1 + r2]
            // E.g., each 32-bit in contains the full sum.
            sum1 = _mm_hadd_epi32(sum1, sum1);
            sum1 = _mm_hadd_epi32(sum1, sum1);
            sum2 = _mm_hadd_epi32(sum2, sum2);
            sum2 = _mm_hadd_epi32(sum2, sum2);
            sum3 = _mm_hadd_epi32(sum3, sum3);
            sum3 = _mm_hadd_epi32(sum3, sum3);
            sum4 = _mm_hadd_epi32(sum4, sum4);
            sum4 = _mm_hadd_epi32(sum4, sum4);
            
            float * C1 = C + (i+0)*num_B_rows + j;
            float * C2 = C + (i+1)*num_B_rows + j;
            float * C3 = C + (i+2)*num_B_rows + j;
            float * C4 = C + (i+3)*num_B_rows + j;

            // Now that we have the full sum in each 32-bit register, we convert them to an integer with _mm_cvtepi32_ps
            // and take the first one with _mm_store_ss.
            // We don't use an SSE instruction to unquantize, although we could.
            // It doesn't really matter since most of the computation is in the above
            // loop over the width.
            // 
            // Also note that the memory acceses on C are not consecutive, but this is a tradeoff that we have to make.
            // We can't have consecutive accesses of A, B, *and* C. But we access A and B a lot more so it makes
            // sense to do it this way.
            _mm_store_ss(C1, _mm_cvtepi32_ps(sum1));
            *(C1) *= unquant_mult;
            
            _mm_store_ss(C2, _mm_cvtepi32_ps(sum2));
            *(C2) *= unquant_mult;

            _mm_store_ss(C3, _mm_cvtepi32_ps(sum3));
            *(C3) *= unquant_mult;
            
            _mm_store_ss(C4, _mm_cvtepi32_ps(sum4));
            *(C4) *= unquant_mult;
        }
    }
}

// Comptue A*B^T very naively.
void SlowRef_MatrixMult(const float * A, const float * B, float * C, int num_A_rows, int num_B_rows, int width)
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

// Program takes no input
int main(int argc, char ** argv) {
    srand(45678);

    // A is usually an activation matrix, B is usually a weight matrix.
    // We actually comptue A * B^T. num_B_rows is the rows in B^T. 
    int num_A_rows = 1024;
    int num_B_rows = 512;
    // This is the shared dimension.
    int width = 1024;

    printf("Computing matrix multiplication: %d x %d x %d\n", num_A_rows, width, num_B_rows);
    
    assert(num_A_rows % 4 == 0);
    assert(width % 8 == 0);

    float * A = static_cast<float*>(aligned_alloc(64, sizeof(float) * num_A_rows * width));
    float * B = static_cast<float*>(aligned_alloc(64, sizeof(float) * num_B_rows * width));
    
    for (int i = 0; i < num_A_rows*width; i++) {
        A[i] = ((float)rand()/(float)RAND_MAX)*2.0f - 1.0f;
    }
    
    for (int i = 0; i < num_B_rows*width; i++) {
        B[i] = ((float)rand()/(float)RAND_MAX)*2.0f - 1.0f;
    }
    
    // C will thus be num_A_rows x num_B_rows
    float * ref_C = new float[num_A_rows*num_B_rows];
    memset(ref_C, 0, sizeof(float)*num_A_rows*num_B_rows);
    SlowRef_MatrixMult(A, B, ref_C, num_A_rows, num_B_rows, width);

    // The quantized version of C is never explicity created. We de-quantize on the fly
    // to avoid extraneous memory accesses.
    float * SSE_C = new float[num_A_rows*num_B_rows];
    float * AVX_C = new float[num_A_rows*num_B_rows];
    
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
    {
      StopWatch w("SSE quantization");
      // The weight matrix should be quantized before starting decoding, since it is known beforehand.
      SSE_Quantize(B, (__m128i *)quant_B, (float)quant_mult, num_B_rows, width);
      // The activation matrix must be quantized on-the-fly.
      SSE_Quantize(A, (__m128i *)quant_A, (float)quant_mult, num_A_rows, width);
    }
    {
      StopWatch w("SSE matrix multiply");  
      SSE_MatrixMult((__m128i*)quant_A, (__m128i*)quant_B, SSE_C, (float)unquant_mult, num_A_rows, num_B_rows, width);
    }

    {
      StopWatch w("AVX quantization");
      // The weight matrix should be quantized before starting decoding, since it is known beforehand.
      AVX_Quantize(B, (__m256i *)quant_B, (float)quant_mult, num_B_rows * width);
      // The activation matrix must be quantized on-the-fly.
      AVX_Quantize(A, (__m256i *)quant_A, (float)quant_mult, num_A_rows * width);
    }
    {
      StopWatch w("AVX multiply");
      AVX_MatrixMult(quant_A, quant_B, AVX_C, (float)unquant_mult, num_A_rows, num_B_rows, width);
    }
    
    double max_diff = 0.0;
    double mean_diff = 0.0;
    for (int i = 0; i < num_A_rows; i++) {
        for (int j = 0; j < num_B_rows; j++) {
            float r = SSE_C[i*num_B_rows + j];
            float f = AVX_C[i*num_B_rows + j];
            double diff = fabs(r-f);
            if (diff > max_diff) {
                max_diff = diff;
            }
            mean_diff += diff;
        }
    }
    
    mean_diff /= (double)num_A_rows*(double)num_B_rows;

    printf("Diff between AVX512 and SSE:\n");
    printf("  Mean = %g\n", mean_diff);
    printf("  Max = %g\n", max_diff);
    
    return 0;
}


