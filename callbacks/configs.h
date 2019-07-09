#pragma once

namespace intgemm {
namespace callbacks {

struct Dummy {
};

struct UnquantizeAndWrite {
  float unquant_mult;
  float* addr;

  UnquantizeAndWrite(float unquant_mult, float* addr) : unquant_mult(unquant_mult), addr(addr) {}
};

}
}
