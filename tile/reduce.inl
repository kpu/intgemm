#if defined(INTGEMM_THIS_IS_AVX512BW)
#define INTGEMM_ARCH AVX512BW
#define INTGEMM_TARGET INTGEMM_AVX512BW
#elif defined(INTGEMM_THIS_IS_AVX2)
#define INTGEMM_ARCH AVX2
#define INTGEMM_TARGET INTGEMM_AVX2
#elif defined(INTGEMM_THIS_IS_SSE2)
#define INTGEMM_ARCH SSE2
#define INTGEMM_TARGET INTGEMM_SSE2
#else
#error "Included without expected architecture"
#endif

namespace intgemm {
namespace INTGEMM_ARCH {

template <class Op> struct Pack64Even {
  template <class Iterator> INTGEMM_TARGET static inline void body(Register *regs) {
    const Index i = Iterator::template I<0>();
    Register hi = unpackhi_epi64(regs[2 * i], regs[2 * i + 1]);
    Register lo = unpacklo_epi64(regs[2 * i], regs[2 * i + 1]);
    regs[i] = Op::Run(hi, lo);
  }
};
template <Index Valid, class Op> INTGEMM_TARGET static inline void Pack64(Register *regs) {
  StaticLoop<Pack64Even<Op>, MakeStaticLoopIterator<Valid / 2>>(regs);
  if (Valid & 1) {
    // For the odd case, shuffle to form 0 g where g is garbage and 0 is accumlated.
    Register shuffled = shuffle_epi32(regs[Valid - 1], 0xB0 /* CDAA */);
    regs[Valid / 2] = Op::Run(shuffled, regs[Valid - 1]);
  }
  // Now [0, (Valid + 1) / 2) contains registers to pack with 128-bit interleaving.
}

template <class Op> struct Pack32Even {
  template <class Iterator> INTGEMM_TARGET static inline void body(Register *regs) {
    const Index i = Iterator::template I<0>();
    Register hi = unpackhi_epi32(regs[2 * i], regs[2 * i + 1]);
    Register lo = unpacklo_epi32(regs[2 * i], regs[2 * i + 1]);
    regs[i] = Op::Run(hi, lo);
  }
};

template <Index Valid, class Op> INTGEMM_TARGET static inline void Pack32(Register *regs) {
  StaticLoop<Pack32Even<Op>, MakeStaticLoopIterator<Valid / 2>>(regs);
  if (Valid & 1) {
    // For the odd case, shuffle to form 0 g 0 g where g is garbage and 0 is accumlated.
    Register shuffled = shuffle_epi32(regs[Valid - 1], 0x4C /* BADA */);
    regs[Valid / 2] = Op::Run(shuffled, regs[Valid - 1]);
  }
  // Now [0, (Valid + 1) / 2) contains registers to pack with 64-bit interleaving.
  Pack64<(Valid + 1) / 2, Op>(regs);
}

} // namespace INTGEMM_ARCH
} // namespace intgemm

#undef INTGEMM_ARCH
#undef INTGEMM_TARGET
