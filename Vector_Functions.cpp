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

void Slow_Add(const float * a, const float * b, float * y, int n) {
    for (int i = 0; i < n; i++) {
        y[i] = a[i] + b[i];
    }
}

void SSE_Add(const float * a, const float * b, float * y, int n) {
    int i = 0;
    for (; i < n; i += 4) {
        __m128 ai = _mm_loadu_ps(a + i);
        __m128 bi = _mm_loadu_ps(b + i);
        __m128 yi = _mm_add_ps(ai, bi);
        _mm_store_ps(y + i, yi);
    }
    for (; i < n; i++) {
        y[i] = a[i] + b[i];
    }
}

void Slow_Mult(const float * a, const float * b, float * y, int n) {
    for (int i = 0; i < n; i++) {
        y[i] = a[i] * b[i];
    }
}

void SSE_Mult(const float * a, const float * b, float * y, int n) {
    int i = 0;
    for (; i < n; i += 4) {
        __m128 ai = _mm_loadu_ps(a + i);
        __m128 bi = _mm_loadu_ps(b + i);
        __m128 yi = _mm_mul_ps(ai, bi);
        _mm_store_ps(y + i, yi);
    }
    for (; i < n; i++) {
        y[i] = a[i]*b[i];
    }
}

class TanhTable {
    float m_multiplier;
    float m_add;
    float * m_table;
    int32_t m_num_bins;

public:
    TanhTable(float min_x, float max_x, int num_bins) {
        m_num_bins = num_bins;
        
        m_multiplier = (float)m_num_bins/(float)(max_x-min_x);
        m_add = (-min_x*(float)m_num_bins/(max_x-min_x));
        
        m_table = new float[m_num_bins+1];
        for (int32_t i = 0; i <= m_num_bins; i++) {
            float x = ((float)i/(float)(m_num_bins))*(max_x-min_x) + min_x;
            m_table[i] = Compute(x);
        }
    }
    
    inline float Lookup(float x) const {
        int32_t index = (int32_t)(x*m_multiplier + m_add);
        if (index < 0) {
            index = 0;
        }
        else if (index > m_num_bins) {
            index = m_num_bins;
        }
        return m_table[index];
    }
    
    inline float Compute(float x) const
    {
        float e2x = expf(2.0f*x);
        float y = (e2x - 1.0f)/(e2x + 1.0f);
        return y;
    }
};

int main(int argc, char ** argv) {
    srand(45678);
    
    // Check tanh
    float min_x = -8.0;
    float max_x = 8.0;
    int num_bins = 8192;
    
    TanhTable tanh_table(min_x, max_x, num_bins);
    
    int num_items = 1000;
    double max_diff = 0.0;
    double mean_diff = 0.0;
    for (int i = 0; i < num_items; i++) {
        float x = (float)rand()/(float)RAND_MAX*20.0 - 10.0;
        float ref_y = tanh_table.Compute(x);
        float table_y = tanh_table.Lookup(x);
        double diff = fabs(ref_y-table_y);
        if (diff > max_diff) {
            max_diff = diff;
        }
        mean_diff += diff;
    }
    mean_diff /= (double)num_items;

    printf("Diff between lookup table and actual tanh:\n");
    printf("  Mean = %g\n", mean_diff);
    printf("  Max = %g\n", max_diff);
}
