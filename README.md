[![Build SSE](https://img.shields.io/jenkins/s/http/vali.inf.ed.ac.uk/jenkins/view/intgemm/job/intgemm-SSE.svg?label=SSE)](http://vali.inf.ed.ac.uk/jenkins/job/intgemm-SSE/)
[![Build AVX2](https://img.shields.io/jenkins/s/http/vali.inf.ed.ac.uk/jenkins/view/intgemm/job/intgemm-AVX2.svg?label=AVX2)](http://vali.inf.ed.ac.uk/jenkins/job/intgemm-AVX2/)
[![Build AVX512BW](https://img.shields.io/jenkins/s/http/vali.inf.ed.ac.uk/jenkins/view/intgemm/job/intgemm-AVX512BW.svg?label=AVX512BW)](http://vali.inf.ed.ac.uk/jenkins/job/intgemm-AVX512BW/)

# Integer Matrix Multiplication

This repository implements 8-bit and 16-bit matrix multiplication:

C = A * B

It's designed with neural network inference in mind: A is typically activations, B is typically fixed parameters, and C is activations for the next layer.

A can have any number of rows.  Typically this is a batch size.
The shared dimension, A's columns and B's rows, must be a multiple of 32 (for 16-bit) or 64 (for 8-bit).
B's columns must be a multiple of 8.

## Accuracy
16-bit multiplication accumulates into 32-bit integers WITHOUT SATURATION (because there is no 32-bit add with saturation). If width is too large (i.e. >2048) or many 16-bit values are large, there is substantial risk of overflow.  Choose a smaller quantization multiplier to scale things down or implement periodic upcasting to 64-bit for me.

8-bit multiplication accumulates into 16-bit integers with saturation.  This saturates for larger widths (~1024) and is worst on SSSE3 because it accumulates in fewer values.  It's possible to upcast to 32-bit every so often, but this has not been implemented yet.

## Usage

A full example appears in [example.cc](example.cc).

Both A and B should be prepared before multiplication.
```C++
#include "intgemm.h"

/* Not shown: allocate 64-byte aligned memory with e.g. aligned_alloc.
 * A is A_rows x width.
 * B is width x B_cols.
 */
/* Prepare A for multiplication.  This might be offline or on the fly. */
intgemm::Int16::PrepareA(A.begin(), A_prepared.begin(), quant_mult, A_rows, width);
/* Prepare B for multiplication.  This is typically done offline. */
intgemm::Int16::PrepareB(B.begin(), B_prepared.begin(), quant_mult, width, B_cols);
/* Multiply and produce results in C */
intgemm::Int16::Multiply(A_prepared.begin(), B_prepared.begin(), A_rows, width, B_cols, intgemm::callbacks::UnquantizeAndWrite(1.0 / (quant_mult * quant_mult), C.begin()));
```
For 8-bit, use `Int8` instead of `Int16`.

When repesented as floats, all of A, B, and C are in row-major format.

~~You can write your own PostProcessing functions on C and use them as a template argument to `Multiply`. For details, see [cops.h](cops.h).~~

For 8-bit, you can make use a of a slightly faster implementation, assuming you can determine tha quantization multipliers and prepare the biases offline:

```C++
#include "intgemm.h"

/* Not shown: allocate 64-byte aligned memory with e.g. aligned_alloc.
 * A is A_rows x width.
 * B is width x B_cols.
 * If you want to make use of the slightly faster 8bit codepath (assuming you can cache biases and quantization multipliers)
 * This routine only supports C = A*B + Bias
 * In practise it computes C = (A+127)*B + Bias - |127|*B
 * Prepare A and B first:
 */
float alpha = 25;
float quant_mult = 127/alpha;
intgemm::Int8::PrepareANew(A.begin(), A_prepared.begin(), quant_mult, A_rows, width);
intgemm::Int8::PrepareB(B.begin(), B_prepared.begin(), quant_mult, width, B_cols);
/* Prepare the bias (inplace) */
float unquant_mult_forprep = (-1)*(alpha)*(alpha)/(127.0f);
intgemm::Int8::PrepareBiasFor8(1, B_prepared.begin(), 1, width, B_cols, callbacks::UnquantizeAndAddBiasAndWrite(unquant_mult_forprep, inputBias.begin(), inputBias.begin()));
/* Multiply */
intgemm::Int8::Multiply8new(A_prepared.begin(), B_prepared.begin(), A_rows, width, B_cols, callbacks::UnquantizeAndAddBiasAndWrite(unquant_mult_forprep, bias.begin(), C.begin()));
```

## Quantization
Floating-point values are multiplied by a user-specified constant then rounded to an integer.  

In 16 bit, Jacob Devlin recommends 1024.0 for neural networks to prevent the aforementioned overflow.

In 8 bit, use 127.0 / the largest value.  Quantization will saturate so it's possible to use larger multipliers to obtain clipping.

## Acknowledgments
The original 16-bit SSE2 code came from:

Sharp Models on Dull Hardware: Fast and Accurate Neural Machine Translation Decoding on the CPU by Jacob Devlin
https://arxiv.org/abs/1705.01991

Under a license:

Copyright (c) 2017 Microsoft Corporation

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

