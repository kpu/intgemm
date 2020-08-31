// This file is included multiple times from dot.h for different architectures.
// Do not #include headers here.  Put them in dot.h.  The reason is that other
// headers should not see these macros (and are within rights to use them too).
#if defined(INTGEMM_THIS_IS_AVX512VNNI)
#define INTGEMM_ARCH AVX512VNNI
#define INTGEMM_TARGET INTGEMM_AVX512VNNI
#elif defined(INTGEMM_THIS_IS_AVX512BW)
#define INTGEMM_ARCH AVX512BW
#define INTGEMM_TARGET INTGEMM_AVX512BW
#elif defined(INTGEMM_THIS_IS_AVX2)
#define INTGEMM_ARCH AVX2
#define INTGEMM_TARGET INTGEMM_AVX2
#elif defined(INTGEMM_THIS_IS_SSSE3)
#define INTGEMM_ARCH SSSE3
#define INTGEMM_TARGET INTGEMM_SSSE3
#elif defined(INTGEMM_THIS_IS_SSE2)
#define INTGEMM_ARCH SSE2
#define INTGEMM_TARGET INTGEMM_SSE2
#endif

namespace intgemm {
namespace INTGEMM_ARCH {

struct DotBase {
  static constexpr Tile kTile { 1, sizeof(Register), 1 };

  template <class From, class To> INTGEMM_TARGET static inline void PrepareB(From from, To to) {
    to.Front<Register>() = from.Front<Register>();
  }
};

/* gcc _mm512_dpbusds_epi32 is slow because it inserts spurious vmovdqa64 instructions.
 * Simple test program:
 * #include <immintrin.h>
 *
 * __m512i Foo(const __m512i *a, const __m512i b0, const __m512i b1, std::size_t count) {
 *   register __m512i c0 = _mm512_setzero_epi32();
 *   register __m512i c1 = _mm512_setzero_epi32();
 *   for (std::size_t i = 0; i < count; ++i) {
 *     c0 = _mm512_dpbusds_epi32(c0, a[i], b0);
 *     c1 = _mm512_dpbusds_epi32(c1, a[i], b1);
 *   }
 *   // Do not optimize away
 *   return _mm512_sub_epi32(c0, c1);
 * }
 * Then with g++ (Gentoo 9.2.0-r2 p3) 9.2.0 run as
 *   g++ -mavx512vnni -O3 example.cc -S
 * We get some inefficient asm:
 * .L3:
 * vmovdqa64 (%rdi), %zmm6
 * vmovdqa64 %zmm3, %zmm0
 * vmovdqa64 %zmm4, %zmm2
 * addq  $64, %rdi
 * vpdpbusds %zmm5, %zmm6, %zmm0
 * vpdpbusds %zmm1, %zmm6, %zmm2
 * vmovdqa64 %zmm0, %zmm3
 * vmovdqa64 %zmm2, %zmm4
 * cmpq  %rdi, %rax
 * jne .L3
 *
 * Why does it copy from zmm3 to zmm0, then copy zmm0 to zmm3 each loop????
 * So for gcc instead of
 *   c = _mm512_dpbusds_epi32(c, a, b);
 * I use:
 *   asm ("vpdpbusds %2, %1, %0" : "+x"(c) : "x"(a), "mx"(b));
 * and that works better in the test program.
 *
 * clang 9.0.1 deals with this fine.
 */
#ifdef INTGEMM_THIS_IS_AVX512VNNI
INTGEMM_TARGET static inline void VNNI8(Register a, Register b, Register &c) {
#if defined(__GNUC__) && !defined(__clang__) && !defined(__INTEL_COMPILER)
    asm ("vpdpbusds %2, %1, %0" : "+x"(c) : "x"(a), "mx"(b));
#else
    c = _mm512_dpbusds_epi32(c, a, b);
#endif
}
#endif

// 8-bit integer multiplication unsigned A * signed B.
#if !defined(INTGEMM_THIS_IS_SSE2) // No int8 on SSE2.
struct Shifted8 : DotBase {
  template <class Access> INTGEMM_TARGET static inline void Run(Access access) {
    const Register &a = access.AFront<Register>();
    const Register &b = access.BFront<Register>();
#ifdef INTGEMM_THIS_IS_AVX512VNNI
    VNNI8(a, b, access.CFront<Register>());
#else
    const Register ones = set1_epi16<Register>(1);
    Register mult = maddubs_epi16(a, b);
    mult = madd_epi16(mult, ones);
    access.CFront<Register>() = add_epi32(access.CFront<Register>(), mult);
#endif
  }

  struct Packed {
    typedef uint8_t A;
    typedef int8_t B;
    typedef int32_t C;
  };
};

// 8-bit integer multiplication signed A * signed B.  Slower.
struct Signed8 : DotBase {
  template <class Access> INTGEMM_TARGET static inline void Run(Access access) {
    const Register &a = access.ARead<Register>();
    const Register &b = access.BRead<Register>();
    const Register a_positive = abs_epi8(a);
    // b_signed = b * sign(a)
#if defined(INTGEMM_THIS_IS_AVX512VNNI) || defined(INTGEMM_THIS_IS_AVX512BW)
    // AVX512 doesn't have sign. Get a mask of negative a values.
    __mmask64 neg_mask = _mm512_test_epi8_mask(a, _mm512_set1_epi8(-128));
    const Register zeros = setzero_si<Register>();
    // Negate by subtracting from zero with the mask.
    const Register b_signed = _mm512_mask_sub_epi8(b, neg_mask, zeros, b);
#else
    // Not AVX512.
    const Register b_signed = sign_epi8(b, a);
#endif

    // c += |a| * b_signed
#if defined(INTGEMM_THIS_IS_AVX512VNNI)
    VNNI8(a_positive, b_signed, access.CFront<Register>());
#else
    Register mult = maddubs_epi16(a_positive, b_signed);
    access.CFront<Register>() = adds_epi16(access.CFront<Register>(), mult);
#endif
  }

  struct Packed {
    typedef int8_t A;
    typedef int8_t B;
#if defined(INTGEMM_THIS_IS_AVX512VNNI)
    typedef int32_t C;
#else
    typedef int16_t C;
#endif
  };
};
#endif // No int8 on SSE2.

// Signed 16-bit integer multiplication.
struct Signed16 : DotBase {
  template <class Access> INTGEMM_TARGET static inline void Run(Access access) {
    const Register &a = access.AFront<Register>();
    const Register &b = access.BFront<Register>();
#if defined(INTGEMM_THIS_IS_AVX512VNNI)
    access.CFront<Register>() = _mm512_dpwssds_epi32(access.CFront<Register>(), a, b);
#else
    Register mult = madd_epi16(a, b);
    access.CFront<Register>() = add_epi32(access.CFront<Register>(), mult);
#endif
  }

  struct Packed {
    typedef int16_t A;
    typedef int16_t B;
    typedef int32_t C;
  };
};

class SingleAccess {
  public:
    explicit SingleAccess(const Register &reg) : reg_(reg) {}

    // template but only has one option.
    template <class R = Content> const Register &Front() const {
      return reg_;
    }

  private:
    const Register reg_;
};

template <class Backend> struct Rotate {
  template <class Acc> INTGEMM_TARGET static inline void Run(Acc access) {
    // Could loop over A but really just put the loop around this.
    static_assert(Backend::kTile.A_rows == 1);
    Register a = access.AFront<Register>();
    Run128(a, access);
    // Create permuations.
#ifdef INTGEMM_THIS_IS_AVX2
    // Flip 128-bit registers
    Run128(_mm256_permute4x64_epi64(a, 2 | (3 << 2) | (0 << 4) | (1 << 6)), access.BAdd(0, 4));
#elif defined(INTGEMM_THIS_IS_AVX512VNNI) || defined(INTGEMM_THIS_IS_AVX512BW)
    // TODO consider permuting same register to save registers.
    // 1 0 3 2
    Run128(_mm512_permutex_epi64(a, 2 | (3 << 2) | (0 << 4) | (1 << 6)), access.BAdd(0, 4));
    // 2 3 0 1
    Run128(_mm512_shuffle_i64x2(a, a, 2 | (3 << 2) | (0 << 4) | (1 << 6)), access.BAdd(0, 8));
    // 3 2 1 0 could also be derived using _mm512_permutex_epi64 on the above line.
    Run128(_mm512_shuffle_i64x2(a, a, 3 | (2 << 2) | (1 << 4) | (0 << 6)), access.BAdd(0, 12));
#endif
  }

  template <class From, class To> INTGEMM_TARGET static inline void PrepareB(From from, To to) {
  }

  typedef typename Backend::Packed Packed;

  private:
    template <class Acc> INTGEMM_TARGET static inline void Run128(Register a, Acc access) {
      typedef Access<SingleAccess, typename Acc::B, typename Acc::C> Internal;
      Backend::Run(Internal(SingleAccess(a), acc.BAccessor(), acc.CAccessor()));
      // Shift everything left.  TODO: consider sequence vs all derived from same register.
      a = shuffle_epi32(a, 1 | (2 << 2) | (3 << 4) | (0 << 6));
      Backend::Run(Internal(SingleAccess(a), acc.BAccessor().Add(0, 1), acc.CAccessor());
      a = shuffle_epi32(a, 1 | (2 << 2) | (3 << 4) | (0 << 6));
      Backend::Run(Internal(SingleAccess(a), acc.BAccessor().Add(0, 2), acc.CAccessor());
      a = shuffle_epi32(a, 1 | (2 << 2) | (3 << 4) | (0 << 6));
      Backend::Run(Internal(SingleAccess(a), acc.BAccessor().Add(0, 3), acc.CAccessor()));
    }
};

// Statically unroll a kernel into a larger tile.
// Can't have Tile as a value until C++20.
template <Index A_rows, Index inner, Index B_cols, class Backend> struct UnrollKernel {
  template <class Access> INTGEMM_FLATTEN INTGEMM_TARGET static inline void Run(Access access) {
    body(access, make_index_sequence<A_rows * inner * B_cols>());
  }
  static constexpr Tile kTile { A_rows * Backend::kTile.A_rows, inner * Backend::kTile.inner, B_cols * Backend::kTile.B_cols };
  typedef typename Backend::Packed Packed;

 private:
  template <class Access, size_t... Index> INTGEMM_FLATTEN INTGEMM_TARGET static inline void body(
      Access access,
      index_sequence<Index...>) {
    // for each inner computed as (Index / A_rows / B_cols)
    //   for each A_row computed as (Index % (A_rows * B_cols)) / B_cols
    //     for each B_col computed as (Index % B_cols)
    unordered_unfurl((
       Backend::Run(access
            .AAdd((Index % (A_rows * B_cols)) / B_cols * Backend::kTile.A_rows, (Index / A_rows / B_cols) * Backend::kTile.inner)
            .BAdd((Index / A_rows / B_cols) * Backend::kTile.inner, (Index % B_cols) * Backend::kTile.B_cols)
            .CAdd((Index % (A_rows * B_cols)) / B_cols * Backend::kTile.A_rows, (Index % B_cols) * Backend::kTile.B_cols))
       // Backend returns void, so use a tuple to make 0.
       , 0)...);
  }
};

} // namespace INTGEMM_ARCH
} // namespace

#undef INTGEMM_TARGET
#undef INTGEMM_ARCH
