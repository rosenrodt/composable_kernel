#ifndef CK_THREADWISE_TENSOR_CONTRACTION_V3_HPP
#define CK_THREADWISE_TENSOR_CONTRACTION_V3_HPP

#include "common_header.hpp"
#include "math.hpp"

namespace ck {

// C[M0, M1, ..., N0, N1, ...] += transpose(A[E, M0, M1, ...]) * B[E, N0, N1, ...]
template <typename ADesc,
          typename BDesc,
          typename CDesc,
          typename ALengths,
          typename BLengths,
          typename CLengths,
          index_t AVectorDim,
          index_t AVectorSize,
          index_t BVectorDim,
          index_t BVectorSize,
          typename std::enable_if<ADesc::IsKnownAtCompileTime() && BDesc::IsKnownAtCompileTime() &&
                                      CDesc::IsKnownAtCompileTime(),
                                  bool>::type = false>
struct ThreadwiseContraction_em_en_mn_v1
{
    template <typename FloatA, typename FloatB, typename FloatC>
    __device__ static void Run_source(const FloatA* p_a, const FloatB* p_b, FloatC* p_c)
    {
        static_assert(ADesc::IsKnownAtCompileTime() && BDesc::IsKnownAtCompileTime() &&
                          CDesc::IsKnownAtCompileTime(),
                      "wrong! Desc should be known at compile-time");

        constexpr auto a_lengths = ALengths{};
        constexpr auto b_lengths = BLengths{};
        constexpr auto c_lengths = CLengths{};

        constexpr index_t a_vec_dim  = AVectorDim;
        constexpr index_t a_vec_size = AVectorSize;

        constexpr index_t b_vec_dim  = BVectorDim;
        constexpr index_t b_vec_size = BVectorSize;

        static_assert(a_lengths[0] == b_lengths[0], "E Dim is different in a/b_lengths!");

        // extract e
        constexpr auto E = a_lengths[0];

        static_assert(a_lengths[a_vec_dim] % a_vec_size == 0, "");

        // remove e dim
        constexpr auto a_lengths_sub = generate_sequence_v2(
            [&](auto i) {
                if constexpr(i == a_vec_dim - 1)
                {
                    return Number<a_lengths[i + 1] / a_vec_size>{};
                }
                else
                {
                    return Number<a_lengths[i + 1]>{};
                }
            },
            Number<a_lengths.Size() - 1>{});

        static_assert(b_lengths[b_vec_dim] % b_vec_size == 0, "");

        // remove e dim
        constexpr auto b_lengths_sub = generate_sequence_v2(
            [&](auto i) {
                if constexpr(i == b_vec_dim - 1)
                {
                    return Number<b_lengths[i + 1] / b_vec_size>{};
                }
                else
                {
                    return Number<b_lengths[i + 1]>{};
                }
            },
            Number<b_lengths.Size() - 1>{});

        static_for<0, E, 1>{}([&](auto e) {
            static_ford<decltype(a_lengths_sub)>{}([&](auto a_idx_sub) {
                static_ford<decltype(b_lengths_sub)>{}([&](auto b_idx_sub) {
                    // add e idx
                    constexpr auto a_idx = generate_tuple(
                        [&](auto i) {
                            if constexpr(i == 0)
                            {
                                return Number<e>{};
                            }
                            else if constexpr(i == a_vec_dim)
                            {
                                return a_idx_sub[i - 1] * a_vec_size;
                            }
                            else
                            {
                                return a_idx_sub[i - 1];
                            }
                        },
                        Number<a_lengths.Size()>{});

                    // add e idx
                    constexpr auto b_idx = generate_tuple(
                        [&](auto i) {
                            if constexpr(i == 0)
                            {
                                return Number<e>{};
                            }
                            else if constexpr(i == b_vec_dim)
                            {
                                return b_idx_sub[i - 1] * b_vec_size;
                            }
                            else
                            {
                                return b_idx_sub[i - 1];
                            }
                        },
                        Number<b_lengths.Size()>{});

                    static_assert(c_lengths.Size() == (a_lengths_sub.Size() + b_lengths_sub.Size()),
                                  "c_size != a_sub_size + b_sub_size!");

                    // create c_idx = {a_idx_sub, b_idx_sub}
                    constexpr auto c_idx = generate_tuple(
                        [&](auto i) {
                            if constexpr(i < a_idx_sub.Size())
                            {
                                return a_idx[Number<i + 1>{}];
                            }
                            else
                            {
                                return b_idx[Number<i - a_idx_sub.Size() + 1>{}];
                            }
                        },
                        Number<c_lengths.Size()>{});

                    constexpr auto a_idx_off = [a_idx](index_t off) {
                        return generate_tuple(
                            [&](auto i) {
                                if constexpr(i == a_vec_dim)
                                    return a_idx[i] + off;
                                else
                                    return a_idx[i];
                            },
                            Number<a_idx.Size()>{});
                    };

                    constexpr auto b_idx_off = [b_idx](index_t off) {
                        return generate_tuple(
                            [&](auto i) {
                                if constexpr(i == b_vec_dim)
                                    return b_idx[i] + off;
                                else
                                    return b_idx[i];
                            },
                            Number<b_idx.Size()>{});
                    };

                    constexpr auto c_vec_dim_a = a_vec_dim - 1;
                    constexpr auto c_vec_dim_b = a_lengths_sub.Size() + (b_vec_dim - 1);

                    constexpr auto c_idx_off = [c_idx](index_t off_a, index_t off_b) {
                        return generate_tuple(
                            [&](auto i) {
                                if constexpr(i == c_vec_dim_a)
                                    return c_idx[i] + off_a;
                                else if constexpr(i == c_vec_dim_b)
                                    return c_idx[i] + off_b;
                                else
                                    return c_idx[i];
                            },
                            Number<c_idx.Size()>{});
                    };

                });
            });
        });
    }

    template <typename FloatA, typename FloatB, typename FloatC>
    __device__ static void Run(const FloatA* p_a, const FloatB* p_b, FloatC* p_c)
    {
        Run_source(p_a, p_b, p_c);
    }
};

} // namespace ck
#endif
