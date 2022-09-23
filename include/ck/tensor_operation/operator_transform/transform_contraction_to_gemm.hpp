// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/tensor_operation/gpu/device/matrix_padder.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"

namespace ck {
namespace tensor_operation {

// assume C[G0, G1, ..., M0, M1, M2, ..., N0, N1, N2...]
template <index_t MPerBlock, index_t NPerBlock, index_t NumDimG, index_t NumDimM, index_t NumDimN>
static auto MakeGridDescriptorPair(const std::vector<index_t>& gs_ms_ns_lengths_vec,
                                   const std::vector<index_t>& gs_ms_ns_strides_vec)
{
    if(!(gs_ms_ns_lengths_vec.size() == NumDimG + NumDimM + NumDimN &&
         gs_ms_ns_strides_vec.size() == NumDimG + NumDimM + NumDimN))
    {
        throw std::runtime_error("wrong! dimension must match input lengths");
    }

    const auto to_tuple = [&](auto& vec, auto start, auto end) {
        return generate_tuple([&](auto i) { return vec[start + i]; }, Number<end - start>{});
    };

    const auto gs_ms_ns_lengths =
        to_tuple(gs_ms_ns_lengths_vec, Number<0>{}, Number<NumDimG + NumDimM + NumDimN>{});
    const auto gs_ms_ns_strides =
        to_tuple(gs_ms_ns_strides_vec, Number<0>{}, Number<NumDimG + NumDimM + NumDimN>{});

    // dimension Ids for G0, G1, ...
    constexpr auto gDimIds = typename arithmetic_sequence_gen<0, NumDimG, 1>::type{};

    // dimension Ids for M0, M1, ...
    constexpr auto mDimIds =
        typename arithmetic_sequence_gen<NumDimG, NumDimG + NumDimM, 1>::type{};

    // dimension Ids for N0, N1, ...
    constexpr auto nDimIds =
        typename arithmetic_sequence_gen<NumDimG + NumDimM, NumDimG + NumDimM + NumDimN, 1>::type{};

    // lengths for G0, G1, ...
    const auto gLengths = get_container_subset(gs_ms_ns_lengths, gDimIds);

    // lengths for M0, M1, ...
    const auto mLengths = get_container_subset(gs_ms_ns_lengths, mDimIds);

    // lengths for N0, N1, ...
    const auto nLengths = get_container_subset(gs_ms_ns_lengths, nDimIds);

    // naive tensor C[G0, G1, ..., M0, M1, M2, ..., N0, N1, N2...]
    const auto grid_desc_gs_ms_ns =
        make_naive_tensor_descriptor(gs_ms_ns_lengths, gs_ms_ns_strides);

    // transformed tensor C[G = G0 * G1 * ..., MRaw = M0 * M1 * M2 * ... , NRaw = N0 * N1 *
    // N2 * ...]
    // Note: This does not require padding as it only provides G offset calculation. Technically
    // descriptor for only G is needed. Here we opt for backward compatibility purpose to return
    // G_M_N
    const auto grid_desc_g_mraw_nraw =
        transform_tensor_descriptor(grid_desc_gs_ms_ns,
                                    make_tuple(make_merge_transform(gLengths),
                                               make_merge_transform(mLengths),
                                               make_merge_transform(nLengths)),
                                    make_tuple(gDimIds, mDimIds, nDimIds),
                                    make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}));

    const auto c_ms_ns_lengths =
        to_tuple(gs_ms_ns_lengths_vec, Number<NumDimG>{}, Number<NumDimG + NumDimM + NumDimN>{});
    const auto c_ms_ns_strides =
        to_tuple(gs_ms_ns_strides_vec, Number<NumDimG>{}, Number<NumDimG + NumDimM + NumDimN>{});

    // transformed tensor C[MRaw = M0 * M1 * M2 * ... , NRaw = N0 * N1 *
    // N2 * ...]
    const auto grid_desc_ms_ns = make_naive_tensor_descriptor(c_ms_ns_lengths, c_ms_ns_strides);

    const auto grid_desc_mraw_nraw = transform_tensor_descriptor(
        grid_desc_ms_ns,
        make_tuple(make_merge_transform(mLengths), make_merge_transform(nLengths)),
        make_tuple(mDimIds - Number<NumDimG>{}, nDimIds - Number<NumDimG>{}),
        make_tuple(Sequence<0>{}, Sequence<1>{}));

    return std::make_pair(grid_desc_g_mraw_nraw, grid_desc_mraw_nraw);
}


    // C
    static auto MakeCGridDescriptorPair(const std::vector<index_t>& c_gs_ms_os_lengths_vec,
                                        const std::vector<index_t>& c_gs_ms_os_strides_vec)
    {
        return MakeGridDescriptorPair<MPerBlock, OPerBlock, NumDimG, NumDimM, NumDimO>(
            c_gs_ms_os_lengths_vec, c_gs_ms_os_strides_vec);
    }

    static auto MakeCGridDescriptor_G_M_N(const std::vector<index_t>& c_gs_ms_os_lengths_vec,
                                          const std::vector<index_t>& c_gs_ms_os_strides_vec)
    {
        return MakeCGridDescriptorPair(c_gs_ms_os_lengths_vec, c_gs_ms_os_strides_vec).first;
    }
    static auto MakeCGridDescriptor_M_N(const std::vector<index_t>& c_gs_ms_os_lengths_vec,
                                        const std::vector<index_t>& c_gs_ms_os_strides_vec)
    {
        return matrix_padder.PadCDescriptor_M_N(
            MakeCGridDescriptorPair(c_gs_ms_os_lengths_vec, c_gs_ms_os_strides_vec).second);
    }

    template <typename AGridDesc_M_K, typename Number>
    __host__ __device__ static constexpr auto
    MakeDefaultAGridDescriptor_AK0_M_AK1(const AGridDesc_M_K& a_grid_desc_m_k, const Number& AK1)
    {
        const auto M = a_grid_desc_m_k.GetLength(I0);
        const auto K = a_grid_desc_m_k.GetLength(I1);

        const auto AK0 = K / AK1;

        return transform_tensor_descriptor(a_grid_desc_m_k,
                                           make_tuple(make_unmerge_transform(make_tuple(AK0, AK1)),
                                                      make_pass_through_transform(M)),
                                           make_tuple(Sequence<1>{}, Sequence<0>{}),
                                           make_tuple(Sequence<0, 2>{}, Sequence<1>{}));
    }

    template <typename BGridDesc_N_K, typename Number>
    __host__ __device__ static constexpr auto
    MakeDefaultBGridDescriptor_BK0_N_BK1(const BGridDesc_N_K& b_grid_desc_n_k, const Number& BK1)
    {
        const auto N = b_grid_desc_n_k.GetLength(I0);
        const auto K = b_grid_desc_n_k.GetLength(I1);

        const auto BK0 = K / BK1;

        return transform_tensor_descriptor(b_grid_desc_n_k,
                                           make_tuple(make_unmerge_transform(make_tuple(BK0, BK1)),
                                                      make_pass_through_transform(N)),
                                           make_tuple(Sequence<1>{}, Sequence<0>{}),
                                           make_tuple(Sequence<0, 2>{}, Sequence<1>{}));
    }

    template <typename B1GridDesc_N_K, typename Number>
    __host__ __device__ static constexpr auto
    MakeDefaultB1GridDescriptor_BK0_N_BK1(const B1GridDesc_N_K& b1_grid_desc_n_k, const Number& B1K1)
    {
        const auto N = b1_grid_desc_n_k.GetLength(I0);
        const auto K = b1_grid_desc_n_k.GetLength(I1);

        const auto B1K0 = K / B1K1;

        return transform_tensor_descriptor(
            b1_grid_desc_n_k,
            make_tuple(make_unmerge_transform(make_tuple(B1K0, B1K1)),
                       make_pass_through_transform(N)),
            make_tuple(Sequence<1>{}, Sequence<0>{}),
            make_tuple(Sequence<0, 2>{}, Sequence<1>{}));
    }

};

} // namespace tensor_operation
} // namespace ck
