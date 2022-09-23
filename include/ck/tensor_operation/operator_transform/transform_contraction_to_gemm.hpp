// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/tensor_operation/gpu/device/matrix_padder.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"

namespace ck {
namespace tensor_operation {

template <
    typename NumDims_G_M_N_K_O, // Sequence<>
    typename PerBlock_M_N_K_O, // Sequence<>
    device::GemmSpecialization GemmSpec>
struct TransformBatchedContractionContractionToBatchedGemmGemm
{
    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};
    static constexpr auto I3 = Number<3>{};
    static constexpr auto I4 = Number<4>{};

    static constexpr index_t NumDimG = NumDims_G_M_N_K_O::At(I0);
    static constexpr index_t NumDimM = NumDims_G_M_N_K_O::At(I1);
    static constexpr index_t NumDimN = NumDims_G_M_N_K_O::At(I2);
    static constexpr index_t NumDimK = NumDims_G_M_N_K_O::At(I3);
    static constexpr index_t NumDimO = NumDims_G_M_N_K_O::At(I4);

    static constexpr index_t MPerBlock = PerBlock_M_N_K_O::At(I0);
    static constexpr index_t NPerBlock = PerBlock_M_N_K_O::At(I1);
    static constexpr index_t KPerBlock = PerBlock_M_N_K_O::At(I2);
    static constexpr index_t OPerBlock = PerBlock_M_N_K_O::At(I3);

    static constexpr auto matrix_padder =
        device::GemmGemmPadder<GemmSpec, index_t, index_t, index_t, index_t>{
            MPerBlock, NPerBlock, KPerBlock, OPerBlock};

    // assume C[G0, G1, ..., M0, M1, M2, ..., N0, N1, N2...]
    static auto MakeCGridDescriptorPair(const std::vector<index_t>& c_gs_ms_os_lengths_vec,
                                        const std::vector<index_t>& c_gs_ms_os_strides_vec)
    {
        if(!(c_gs_ms_os_lengths_vec.size() == NumDimG + NumDimM + NumDimO &&
             c_gs_ms_os_strides_vec.size() == NumDimG + NumDimM + NumDimO))
        {
            throw std::runtime_error("wrong! dimension must match input lengths");
        }

        const auto to_tuple = [&](auto& vec, auto start, auto end) {
            return generate_tuple([&](auto i) { return vec[start + i]; }, Number<end - start>{});
        };

        const auto c_gs_ms_os_lengths =
            to_tuple(c_gs_ms_os_lengths_vec, Number<0>{}, Number<NumDimG + NumDimM + NumDimO>{});
        const auto c_gs_ms_os_strides =
            to_tuple(c_gs_ms_os_strides_vec, Number<0>{}, Number<NumDimG + NumDimM + NumDimO>{});

        // dimension Ids for G0, G1, ...
        constexpr auto gDimIds = typename arithmetic_sequence_gen<0, NumDimG, 1>::type{};

        // dimension Ids for M0, M1, ...
        constexpr auto mDimIds =
            typename arithmetic_sequence_gen<NumDimG, NumDimG + NumDimM, 1>::type{};

        // dimension Ids for O0, O1, ...
        constexpr auto oDimIds = typename arithmetic_sequence_gen<NumDimG + NumDimM,
                                                                  NumDimG + NumDimM + NumDimO,
                                                                  1>::type{};

        // lengths for G0, G1, ...
        const auto gLengths = get_container_subset(c_gs_ms_os_lengths, gDimIds);

        // lengths for M0, M1, ...
        const auto mLengths = get_container_subset(c_gs_ms_os_lengths, mDimIds);

        // lengths for O0, O1, ...
        const auto oLengths = get_container_subset(c_gs_ms_os_lengths, oDimIds);

        // naive tensor C[G0, G1, ..., M0, M1, M2, ..., O0, O1, O2...]
        const auto c_grid_desc_gs_ms_os =
            make_naive_tensor_descriptor(c_gs_ms_os_lengths, c_gs_ms_os_strides);

        // transformed tensor C[G = G0 * G1 * ..., MRaw = M0 * M1 * M2 * ... , ORaw = O0 * O1 *
        // O2 * ...]
        // Note: This does not require padding as it only provides G offset calculation. Technically
        // descriptor for only G is needed. Here we opt for backward compatibility purpose to return
        // G_M_N
        const auto c_grid_desc_g_mraw_oraw =
            transform_tensor_descriptor(c_grid_desc_gs_ms_os,
                                        make_tuple(make_merge_transform(gLengths),
                                                   make_merge_transform(mLengths),
                                                   make_merge_transform(oLengths)),
                                        make_tuple(gDimIds, mDimIds, oDimIds),
                                        make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}));

        const auto c_ms_os_lengths =
            to_tuple(c_gs_ms_os_lengths_vec, Number<NumDimG>{}, Number<NumDimG + NumDimM + NumDimO>{});
        const auto c_ms_os_strides =
            to_tuple(c_gs_ms_os_strides_vec, Number<NumDimG>{}, Number<NumDimG + NumDimM + NumDimO>{});

        const auto c_grid_desc_ms_os =
            make_naive_tensor_descriptor(c_ms_os_lengths, c_ms_os_strides);

        const auto c_grid_desc_mraw_oraw = transform_tensor_descriptor(
            c_grid_desc_ms_os,
            make_tuple(make_merge_transform(mLengths), make_merge_transform(oLengths)),
            make_tuple(mDimIds - Number<NumDimG>{}, oDimIds - Number<NumDimG>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}));

        const auto c_grid_desc_m_o = matrix_padder.PadCDescriptor_M_N(c_grid_desc_mraw_oraw);

        return std::make_pair(c_grid_desc_g_mraw_oraw, c_grid_desc_m_o);
    }

    static auto MakeCGridDescriptor_G_M_N(const std::vector<index_t>& c_gs_ms_os_lengths_vec,
                                          const std::vector<index_t>& c_gs_ms_os_strides_vec)
    {
        return MakeCGridDescriptorPair(c_gs_ms_os_lengths_vec, c_gs_ms_os_strides_vec).first;
    }
    static auto MakeCGridDescriptor_M_N(const std::vector<index_t>& c_gs_ms_os_lengths_vec,
                                        const std::vector<index_t>& c_gs_ms_os_strides_vec)
    {
        return MakeCGridDescriptorPair(c_gs_ms_os_lengths_vec, c_gs_ms_os_strides_vec).second;
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
