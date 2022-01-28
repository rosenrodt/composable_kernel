#ifndef CK_THREADWISE_GEMM_DLOPS_V3_HPP
#define CK_THREADWISE_GEMM_DLOPS_V3_HPP

#include "common_header.hpp"
#include "math.hpp"

namespace ck {

// C[M, N] += transpose(A[K, M]) * B[K, N]
//   Element of matrix can be vectorized data
// Assume:
//   1. AThreadDesc_E1_K_E2, BThreadDesc_E1_N_Ho_Wo_E2, CThreadDesc_K_N_Ho_Wo are known at
//   compile-time
//   2. AOriginIdx, BOriginIdx, COriginIdx are known at compile-time
template <typename FloatA,
          typename FloatB,
          typename FloatC,
          typename AThreadDesc_E1_K_E2,
          typename BThreadDesc_E1_N_Ho_Wo_E2,
          typename CThreadDesc_K_N_Ho_Wo,
          typename enable_if<AThreadDesc_E1_K_E2::IsKnownAtCompileTime() &&
                                 BThreadDesc_E1_N_Ho_Wo_E2::IsKnownAtCompileTime() &&
                                 CThreadDesc_K_N_Ho_Wo::IsKnownAtCompileTime(),
                             bool>::type = false>
struct ThreadwiseGemmDlops_km_kn_mn_v3
{

    template <typename ABuffer,
              typename AOriginIdx,
              typename BBuffer,
              typename BOriginIdx,
              typename CBuffer,
              typename COriginIdx>
    __device__ static void Run(const ABuffer& a_buf,
                               AOriginIdx,
                               const BBuffer& b_buf,
                               BOriginIdx,
                               CBuffer& c_buf,
                               COriginIdx)
    {

        static_assert(AThreadDesc_E1_K_E2::IsKnownAtCompileTime() &&
                          BThreadDesc_E1_N_Ho_Wo_E2::IsKnownAtCompileTime() &&
                          CThreadDesc_K_N_Ho_Wo::IsKnownAtCompileTime(),
                      "wrong! Desc should be known at compile-time");

        static_assert(is_known_at_compile_time<remove_cvref_t<AOriginIdx>>::value &&
                          is_known_at_compile_time<remove_cvref_t<BOriginIdx>>::value &&
                          is_known_at_compile_time<remove_cvref_t<COriginIdx>>::value,
                      "wrong! AOriginIdx, BOriginIdx, COringinIdx should be known at compile-time");

        static_assert(
            is_same<remove_cvref_t<typename ABuffer::type>, remove_cvref_t<FloatA>>::value &&
            is_same<remove_cvref_t<typename BBuffer::type>, remove_cvref_t<FloatB>>::value &&
            is_same<remove_cvref_t<typename CBuffer::type>, remove_cvref_t<FloatC>>::value &&
            "wrong! inconsistent type");

        constexpr auto I0 = Number<0>{};
        constexpr auto I1 = Number<1>{};
        constexpr auto I2 = Number<2>{};
        constexpr auto I3 = Number<3>{};

#if 0
        constexpr auto E1 = AThreadDesc_E1_K_E2{}.GetLength(I0);
        constexpr auto K  = AThreadDesc_E1_K_E2{}.GetLength(I1);
        constexpr auto E2 = AThreadDesc_E1_K_E2{}.GetLength(I2);

        constexpr auto Ho = BThreadDesc_E1_N_Ho_Wo_E2{}.GetLength(I2);
        constexpr auto Wo = BThreadDesc_E1_N_Ho_Wo_E2{}.GetLength(I3);

        constexpr auto a_origin_idx = to_multi_index(AOriginIdx{});
        constexpr auto b_origin_idx = to_multi_index(BOriginIdx{});
        constexpr auto c_origin_idx = to_multi_index(COriginIdx{});

        static_for<0, K, 1>{}([&](auto k) {
            static_for<0, Ho, 1>{}([&](auto h) {
                static_for<0, Wo, 1>{}([&](auto w) {
                    static_for<0, E1, 1>{}([&](auto e1) {
                        vector_type<FloatA, E2> a_vec;
                        vector_type<FloatB, E2> b_vec;

                        static_for<0, E2, 1>{}([&](auto e2) {
                            constexpr index_t a_offset = AThreadDesc_E1_K_E2{}.CalculateOffset(
                                a_origin_idx + make_tuple(e1, k, e2));

                            a_vec.template AsType<FloatA>()(Number<e2>{}) =
                                a_buf[Number<a_offset>{}];

                            constexpr index_t b_offset =
                                BThreadDesc_E1_N_Ho_Wo_E2{}.CalculateOffset(
                                    b_origin_idx + make_tuple(e1, 0, h, w, e2));

                            b_vec.template AsType<FloatB>()(Number<e2>{}) =
                                b_buf[Number<b_offset>{}];
                        });

                        constexpr index_t c_offset = CThreadDesc_K_N_Ho_Wo{}.CalculateOffset(
                            c_origin_idx + make_tuple(k, 0, h, w));

                        using a_vec_t = typename vector_type<FloatA, E2>::type;
                        using b_vec_t = typename vector_type<FloatB, E2>::type;

                        inner_product<a_vec_t, b_vec_t, FloatC>(
                            a_vec.template AsType<a_vec_t>()[Number<0>{}],
                            b_vec.template AsType<b_vec_t>()[Number<0>{}],
                            c_buf(Number<c_offset>{}));
                    });
                });
            });
        });
#else
        constexpr auto E1 = AThreadDesc_E1_K_E2{}.GetLength(I0);
        constexpr auto E2 = AThreadDesc_E1_K_E2{}.GetLength(I2);

        constexpr auto a_origin_idx = to_multi_index(AOriginIdx{});
        constexpr auto b_origin_idx = to_multi_index(BOriginIdx{});
        constexpr auto c_origin_idx = to_multi_index(COriginIdx{});

        constexpr auto a_lengths_sub = generate_sequence_v2(
            [&](auto i) { return AThreadDesc_E1_K_E2{}.GetLength(Number<i + 1>{}); },
            Number<AThreadDesc_E1_K_E2{}.GetNumOfVisibleDimension() - 2>{});

        constexpr auto b_lengths_sub = generate_sequence_v2(
            [&](auto i) { return BThreadDesc_E1_N_Ho_Wo_E2{}.GetLength(Number<i + 1>{}); },
            Number<BThreadDesc_E1_N_Ho_Wo_E2{}.GetNumOfVisibleDimension() - 2>{});

        static_assert(AThreadDesc_E1_K_E2{}.GetNumOfVisibleDimension() == 3, "");
        static_assert(BThreadDesc_E1_N_Ho_Wo_E2{}.GetNumOfVisibleDimension() == 5, "");

        static_assert(a_lengths_sub.Size() == 1, "");
        static_assert(b_lengths_sub.Size() == 3, "");

        static_for<0, E1, 1>{}([&](auto e1) {
            static_ford<decltype(a_lengths_sub)>{}([&](auto a_idx_sub) {
                static_ford<decltype(b_lengths_sub)>{}([&](auto b_idx_sub) {
                    vector_type<FloatA, E2> a_vec;
                    vector_type<FloatB, E2> b_vec;

                    static_for<0, E2, 1>{}([&](auto e2) {
                        constexpr auto a_idx = generate_tuple(
                            [&](auto i) {
                                if constexpr(i == 0)
                                {
                                    return Number<e1>{};
                                }
                                else if constexpr(i ==
                                                  AThreadDesc_E1_K_E2{}.GetNumOfVisibleDimension() -
                                                      1)
                                {
                                    return Number<e2>{};
                                }
                                else
                                {
                                    return a_idx_sub[i - 1];
                                }
                            },
                            Number<AThreadDesc_E1_K_E2{}.GetNumOfVisibleDimension()>{});

                        constexpr index_t a_offset =
                            AThreadDesc_E1_K_E2{}.CalculateOffset(a_origin_idx + a_idx);

                        a_vec.template AsType<FloatA>()(Number<e2>{}) = a_buf[Number<a_offset>{}];

                        constexpr auto b_idx = generate_tuple(
                            [&](auto i) {
                                if constexpr(i == 0)
                                {
                                    return Number<e1>{};
                                }
                                else if constexpr(i == BThreadDesc_E1_N_Ho_Wo_E2{}
                                                               .GetNumOfVisibleDimension() -
                                                           1)
                                {
                                    return Number<e2>{};
                                }
                                else
                                {
                                    return b_idx_sub[i - 1];
                                }
                            },
                            Number<BThreadDesc_E1_N_Ho_Wo_E2{}.GetNumOfVisibleDimension()>{});

                        constexpr index_t b_offset =
                            BThreadDesc_E1_N_Ho_Wo_E2{}.CalculateOffset(b_origin_idx + b_idx);

                        b_vec.template AsType<FloatB>()(Number<e2>{}) = b_buf[Number<b_offset>{}];
                    });

                    // create c_idx = {a_idx_sub, b_idx_sub}
                    constexpr auto c_idx = generate_tuple(
                        [&](auto i) {
                            if constexpr(i < a_idx_sub.Size())
                            {
                                return a_idx_sub[i];
                            }
                            else
                            {
                                return b_idx_sub[Number<i - a_idx_sub.Size()>{}];
                            }
                        },
                        Number<CThreadDesc_K_N_Ho_Wo{}.GetNumOfVisibleDimension()>{});

                    constexpr index_t c_offset =
                        CThreadDesc_K_N_Ho_Wo{}.CalculateOffset(c_origin_idx + c_idx);

                    using a_vec_t = typename vector_type<FloatA, E2>::type;
                    using b_vec_t = typename vector_type<FloatB, E2>::type;

                    inner_product<a_vec_t, b_vec_t, FloatC>(
                        a_vec.template AsType<a_vec_t>()[Number<0>{}],
                        b_vec.template AsType<b_vec_t>()[Number<0>{}],
                        c_buf(Number<c_offset>{}));
                });
            });
        });

#endif
    }
};

} // namespace ck
#endif
