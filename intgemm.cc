#include "intgemm.h"

namespace intgemm {

Int8::Interface Int8::interface = ChooseCPU<Int8::Interface>(
  Interface::create<AVX512VNNI_8bit>(),
  Interface::create<AVX512_8bit>(),
  Interface::create<AVX2_8bit>(),
  Interface::create<SSSE3_8bit>(),
  Interface::create<Unsupported_8bit>(),
  Interface::create<Unsupported_8bit>()
);

Int8Shift::Interface Int8Shift::interface = ChooseCPU<Int8Shift::Interface>(
  Interface::create<AVX512VNNI_8bit>(),
  Interface::create<AVX512_8bit>(),
  Interface::create<AVX2_8bit>(),
  Interface::create<SSSE3_8bit>(),
  Interface::create<Unsupported_8bit>(),
  Interface::create<Unsupported_8bit>()
);

Int16::Interface Int16::interface = ChooseCPU<Int16::Interface>(
  Interface::create<AVX512_16bit>(),
  Interface::create<AVX512_16bit>(),
  Interface::create<AVX2_16bit>(),
  Interface::create<SSE2_16bit>(),
  Interface::create<SSE2_16bit>(),
  Interface::create<Unsupported_16bit>()
);

const CPUType kCPU = ChooseCPU(CPUType::AVX512VNNI, CPUType::AVX512BW, CPUType::AVX2, CPUType::SSSE3, CPUType::SSE2, CPUType::UNSUPPORTED);

float Unsupported_MaxAbsolute(const float *begin, const float *end) {
  throw UnsupportedCPU();
}

float (*MaxAbsolute)(const float *begin, const float *end) = ChooseCPU(avx512f::MaxAbsolute, avx512f::MaxAbsolute, avx2::MaxAbsolute, sse2::MaxAbsolute, sse2::MaxAbsolute, Unsupported_MaxAbsolute);

}
