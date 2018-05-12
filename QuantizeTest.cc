#include "AVX_Matrix_Mult.h"

#include <iostream>

int main() {
  float input[16] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
  int16_t output[16];
  AVX_Quantize16(input, output, 1, 16);
  for (int i = 0; i < 16; ++i) {
    if (output[i] != i) {
      std::cerr << "Failure at i " << '\n';
      return 1;
    }
  }
  return 0;
}
