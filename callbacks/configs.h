#pragma once

namespace intgemm {
namespace callbacks {

struct Dummy {
};

template <typename OutputBufferType>
struct Write {
  OutputBufferType* addr;

  Write(OutputBufferType* addr) : addr(addr) {}
};

struct UnquantizeAndWrite {
  float unquant_mult;
  float* addr;

  UnquantizeAndWrite(float unquant_mult, float* addr) : unquant_mult(unquant_mult), addr(addr) {}
};

struct AddBiasAndWrite {
  const int* bias_addr;
  int* output_addr;

  AddBiasAndWrite(const int* bias_addr, int* output_addr) :  bias_addr(bias_addr), output_addr(output_addr) {}
};

struct UnquantizeAndAddBiasAndWrite {
  float unquant_mult;
  const float* bias_addr;
  float* output_addr;

  UnquantizeAndAddBiasAndWrite(float unquant_mult, const float* bias_addr, float* output_addr) : unquant_mult(unquant_mult), bias_addr(bias_addr), output_addr(output_addr) {}
};

template <typename Type>
struct SSRUSigmoidF {
  const Type* bias_addr;
  const Type* sigmoid_lut;
  float quant_mult_f;
  float quant_mult_bf;
  float sigmoid_lut_range;
  Type* output_addr;

  SSRUSigmoidF(const Type* bias_addr, const Type* sigmoid_lut, float quant_mult_f, float quant_mult_bf, float sigmoid_lut_range,  Type* output_addr) : bias_addr(bias_addr), sigmoid_lut(sigmoid_lut), quant_mult_f(quant_mult_f), quant_mult_bf(quant_mult_bf), sigmoid_lut_range(sigmoid_lut_range), output_addr(output_addr) {}
};

template <typename Type>
struct SSRUPrecomputedPartOfHighway {
  const Type* sigmoid_f_addr;
  float scale;
  Type* output_addr;

  SSRUPrecomputedPartOfHighway(const Type* sigmoid_f_addr, float scale, Type* output_addr) : sigmoid_f_addr(sigmoid_f_addr), scale(scale), output_addr(output_addr) {}
};

}
}
