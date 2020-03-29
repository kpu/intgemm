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

/* When Register is used as a template argument, gcc warns
 * warning: ignoring attributes on template argument ‘Register’ {aka ‘__vector(8) long long int’} [-Wignored-attributes]
 * So here is a class that doesn't warn.
 */
class RegisterRowMajorAccess {
  public:
    typedef Register Content;

    RegisterRowMajorAccess(Content *data, Index cols)
      : data_(data), cols_(cols) {}

    RegisterRowMajorAccess Add(Index row, Index col) const {
      return RegisterRowMajorAccess(data_ + row * cols_ + col, cols_);
    }

    const Content &Front() const { return *data_; }
    Content &Front() { return *data_; }

  private:
    Content *data_;
    Index cols_;
};

// 8-bit integer multiplication unsigned A * signed B.
#if !defined(INTGEMM_THIS_IS_SSE2) // No int8 on SSE2.
struct Shifted8 {
  template <class Access> INTGEMM_TARGET static inline void Run(Access access) {
    const Register &a = reinterpret_cast<const Register&>(access.AFront());
    const Register &b = reinterpret_cast<const Register&>(access.BFront());
#ifdef INTGEMM_THIS_IS_AVX512VNNI
    access.CFront() = _mm512_dpbusds_epi32(access.CFront(), a, b);
#else
    const Register ones = set1_epi16<Register>(1);
    Register mult = maddubs_epi16(a, b);
    mult = madd_epi16(mult, ones);
    access.CFront() = add_epi32(access.CFront(), mult);
#endif
  }

  static constexpr Tile kTile { 1, sizeof(Register), 1 };

  struct Packed {
    typedef uint8_t A;
    typedef int8_t B;
    typedef int32_t C;
  };
};

// 8-bit integer multiplication signed A * signed B.  Slower.
struct Signed8 {
  template <class Access> INTGEMM_TARGET static inline void Run(Access access) {
    const Register &a = reinterpret_cast<const Register&>(access.AFront());
    const Register &b = reinterpret_cast<const Register&>(access.BFront());
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
    access.CFront() = _mm512_dpbusds_epi32(access.CFront(), a_positive, b_signed);
#else
    Register mult = maddubs_epi16(a_positive, b_signed);
    access.CFront() = adds_epi16(access.CFront(), mult);
#endif
  }

  static constexpr Tile kTile { 1, sizeof(Register), 1 };

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
struct Signed16 {
  template <class Access> INTGEMM_TARGET static inline void Run(Access access) {
    const Register &a = reinterpret_cast<const Register&>(access.AFront());
    const Register &b = reinterpret_cast<const Register&>(access.BFront());
#if defined(INTGEMM_THIS_IS_AVX512VNNI)
    access.CFront() = _mm512_dpwssds_epi32(access.CFront(), a, b);
#else
    Register mult = madd_epi16(a, b);
    access.CFront() = add_epi32(access.CFront(), mult);
#endif
  }

  static constexpr Tile kTile { 1, sizeof(Register), 1 };

  struct Packed {
    typedef int16_t A;
    typedef int16_t B;
    typedef int32_t C;
  };
};

/* These would normally be outside arch namespaces but gcc refuses to inline
 * functions with target attributes into functions without target attributes.
 * So they all need target attributes. */
template <class Backend> struct MatrixTileBody {
  template <typename Iterator, typename Access> INTGEMM_TARGET static inline void body(Access access) {
    Backend::Run(access
        .AAdd(Iterator::template I<0>() * Backend::kTile.A_rows, 0)
        .BAdd(0, Iterator::template I<1>() * Backend::kTile.B_cols)
        .CAdd(Iterator::template I<0>() * Backend::kTile.A_rows, Iterator::template I<1>() * Backend::kTile.B_cols));
  }
};
// Wrap a tile to statically loop over rows of A and columns of B.
template <Index A_rows, Index B_cols, class Backend> struct MatrixTile {
  template <class Access> INTGEMM_TARGET __attribute__((flatten)) static inline void Run(Access access) {
    StaticLoop<MatrixTileBody<Backend>, MakeStaticLoopIterator<A_rows, B_cols>>(access);
  }
  static constexpr Tile kTile { A_rows * Backend::kTile.A_rows, Backend::kTile.inner, B_cols * Backend::kTile.B_cols };
  typedef typename Backend::Packed Packed;
};

template <class Backend> struct InnerTileBody {
  template <typename Iterator, typename Access> INTGEMM_TARGET static inline void body(Access access) {
    Backend::Run(access
        .AAdd(0, Iterator::template I<0>() * Backend::kTile.inner)
        .BAdd(Iterator::template I<0>() * Backend::kTile.inner, 0));
  }
};
// Wrap a tile to statically loop over the inner dimension.
template <Index inner, class Backend> struct InnerTile {
  template <class Access> INTGEMM_TARGET __attribute__((flatten)) static inline void Run(Access access) {
    StaticLoop<InnerTileBody<Backend>, MakeStaticLoopIterator<inner>>(access);
  }
  static constexpr Tile kTile { Backend::kTile.A_rows, inner * Backend::kTile.inner, Backend::kTile.B_cols };
  typedef typename Backend::Packed Packed;
};

} // namespace INTGEMM_ARCH
} // namespace

#undef INTGEMM_TARGET
#undef INTGEMM_ARCH
