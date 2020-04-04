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

struct RegisterPair { Register hi; Register lo; };

template <class Op, class Folder> struct ReduceEvens {
  template <class Iterator> INTGEMM_TARGET static inline void body(Register *regs) {
    const Index i = Iterator::template I<0>();
    RegisterPair ret = Folder::Even(regs[2 * i], regs[2 * i + 1]);
    regs[i] = Op::Run(ret.hi, ret.lo);
  }
};

template <Index Valid, class Op, class Folder> INTGEMM_TARGET static inline void GenericReduce(Register *regs) {
  StaticLoop<ReduceEvens<Op, Folder>, MakeStaticLoopIterator<Valid / 2>>(regs);
  if (Valid & 1) {
    auto values = Folder::Odd(regs[Valid - 1]);
    regs[Valid / 2] = Folder::OddUpcast(Op::Run(values.lo, values.hi));
  }
}

struct Reduce32Folder {
  INTGEMM_TARGET static inline RegisterPair Even(Register first, Register second) {
    return RegisterPair { unpackhi_epi32(first, second), unpacklo_epi32(first, second) };
  }
  INTGEMM_TARGET static inline RegisterPair Odd(Register reg) {
    // For the odd case, shuffle to form 0 g 0 g where g is garbage and 0 is accumlated.
    return RegisterPair { reg, shuffle_epi32(reg, 0x31) };
  }
  INTGEMM_TARGET static inline Register OddUpcast(Register reg) { return reg; }
};

struct Reduce64Folder {
  INTGEMM_TARGET static inline RegisterPair Even(Register first, Register second) {
    return RegisterPair { unpackhi_epi64(first, second), unpacklo_epi64(first, second) };
  }
  INTGEMM_TARGET static inline RegisterPair Odd(Register reg) {
    // For the odd case, shuffle to form 0 g where g is garbage and 0 is accumlated.
    return RegisterPair { reg, shuffle_epi32(reg, 3 * 4 + 2) };
  }
  INTGEMM_TARGET static inline Register OddUpcast(Register reg) { return reg; }
};

#ifdef INTGEMM_THIS_IS_AVX2
struct Reduce128Folder {
  INTGEMM_TARGET static inline RegisterPair Even(Register first, Register second) {
    return RegisterPair {
      // This instruction generates 0s 1s 2s 3s 4f 5f 6f 7f
      _mm256_permute2f128_si256(first, second, 0x21),
      // This instruction generates 0f 1f 2f 3f 4s 5s 6s 7s
      _mm256_blend_epi32(first, second, 0xf0)
    };
  }
  INTGEMM_TARGET static inline SSE2::RegisterPair Odd(Register reg) {
    return SSE2::RegisterPair { _mm256_extracti128_si256(reg, 1), _mm256_castsi256_si128(reg) };
  }
  INTGEMM_TARGET static inline Register OddUpcast(SSE2::Register reg) { return _mm256_castsi128_si256(reg); }
};
#endif

#ifdef INTGEMM_THIS_IS_AVX512BW
struct Reduce128Folder {
  INTGEMM_TARGET static inline RegisterPair Even(Register first, Register second) {
    // TODO can this be optimized with a blend and a shuffle instruction?
    return RegisterPair {
      // Form [0th 128-bit of first, 0th 128-bit second, 2nd 128-bit of first, 2nd 128-bit of second]
      _mm512_mask_permutex_epi64(first, 0xcc, second, (0 << 4) | (1 << 6)),
      // Form [1st 128-bit of first, 1st 128-bit of second, 3rd 128-bit of first, 3rd 128-bit of second]
      _mm512_mask_permutex_epi64(second, 0x33, first, 2 | (3 << 2))
    };
  }
};

struct Reduce256Folder {
  INTGEMM_TARGET static inline RegisterPair Even(Register first, Register second) {
    return RegisterPair {
      // This instruction generates first[2] first[3] second[0] second[1]
      _mm512_shuffle_i64x2(first, second, 2 | (3 << 2) | (0 << 4) | (1 << 6)),
      // This instruction generates first[0] first[1] second[2] second[3]
      _mm512_mask_blend_epi64(0xf0, first, second)
    };
  }
};

template <class Op> struct ReduceFours {
  // Collapse 4 AVX512 registers at once, interleaving 128-bit fields.
  template <class Iterator> INTGEMM_TARGET static inline void body(Register *regs) {
    const Index i = Iterator::template I<0>();
    const Register *in = regs + i * 4;
    // Do 256-bit interleaving first because it's slightly cheaper.
    RegisterPair mix0pair = Reduce256Folder::Even(in[0], in[2]);
    RegisterPair mix1pair = Reduce256Folder::Even(in[1], in[3]);
    // 0 0 2 2
    Register mix0 = Op::Run(mix0pair.hi, mix0pair.lo);
    // 1 1 3 3
    Register mix1 = Op::Run(mix1pair.hi, mix1pair.lo);
    mix0pair = Reduce128Folder::Even(mix0, mix1);
    regs[i] = Op::Run(mix0pair.hi, mix0pair.lo);
  }
};

// non-type partial specialization ‘ReduceOverhang<0, Op>’ is not allowed
template <Index Valid, class Op> struct ReduceOverhang;

template <class Op> struct ReduceOverhang<0, Op> {
  INTGEMM_TARGET static inline void Run(const Register *, Register &) {}
};
template <class Op> struct ReduceOverhang<1, Op> {
  // Overhang of 1 register: fold it overself to SSE2.
  INTGEMM_TARGET static inline void Run(const Register *regs, Register &to) {
    AVX2::Register folded = Op::Run(_mm512_castsi512_si256(regs[0]), _mm512_extracti64x4_epi64(regs[0], 1));
    SSE2::RegisterPair pair = AVX2::Reduce128Folder::Odd(folded);
    SSE2::Register more = Op::Run(pair.hi, pair.lo);
    to = _mm512_castsi128_si512(more);
  }
};
template <class Op> struct ReduceOverhang<2, Op> {
  // Overhang of 2 registers: fold to AVX2.
  INTGEMM_TARGET static inline void Run(const Register *regs, Register &to) {
    RegisterPair mixpair = Reduce128Folder::Even(regs[0], regs[1]);
    Register mix = Op::Run(mixpair.hi, mixpair.lo);
    AVX2::Register folded = Op::Run(_mm512_castsi512_si256(mix), _mm512_extracti64x4_epi64(mix, 1));
    to = _mm512_castsi256_si512(folded);
  }
};
template <class Op> struct ReduceOverhang<3, Op> {
  INTGEMM_TARGET static inline void Run(const Register *regs, Register &to) {
    RegisterPair mix0pair = Reduce256Folder::Even(regs[0], regs[2]);
    Register mix0022 = Op::Run(mix0pair.hi, mix0pair.lo);
    // mix0022 128-bit bit blocks: 0 0 2 2

    AVX2::Register fold11 = Op::Run(_mm512_castsi512_si256(regs[1]), _mm512_extracti64x4_epi64(regs[1], 1));
    // fold11 128-bit blocks: 1 1

    RegisterPair finish = Reduce128Folder::Even(mix0022, _mm512_castsi256_si512(fold11));
    to = Op::Run(finish.hi, finish.lo);
  }
};

#endif

template <Index Valid, class Op> INTGEMM_TARGET static inline void Reduce32(Register *regs) {
  GenericReduce<Valid, Op, Reduce32Folder>(regs);
  GenericReduce<(Valid + 1) / 2, Op, Reduce64Folder>(regs);
  // SSE2 is done.
#if defined(INTGEMM_THIS_IS_AVX2)
  GenericReduce<(Valid + 3) / 4, Op, Reduce128Folder>(regs);
#elif defined(INTGEMM_THIS_IS_AVX512BW)
  // Special handling for AVX512BW because we need to fold twice and it can actually go all the way down to SSE2.
  constexpr Index remaining = (Valid + 3) / 4;
  // Handle registers a multiple of 4.
  StaticLoop<ReduceFours<Op>, MakeStaticLoopIterator<(remaining / 4)>>(regs);
  ReduceOverhang<remaining & 3, Op>::Run(regs + (remaining & ~3), *(regs + remaining / 4));
#endif
}

} // namespace INTGEMM_ARCH
} // namespace intgemm

#undef INTGEMM_ARCH
#undef INTGEMM_TARGET
