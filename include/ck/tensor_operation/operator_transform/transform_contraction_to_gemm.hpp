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
    static constexpr index_t NumDimG = NumDims_G_M_N_K_O::At(Number<0>{});
    static constexpr index_t NumDimM = NumDims_G_M_N_K_O::At(Number<1>{});
    static constexpr index_t NumDimN = NumDims_G_M_N_K_O::At(Number<2>{});
    static constexpr index_t NumDimK = NumDims_G_M_N_K_O::At(Number<3>{});
    static constexpr index_t NumDimO = NumDims_G_M_N_K_O::At(Number<4>{});

    static constexpr index_t MPerBlock = PerBlock_M_N_K_O::At(Number<0>{});
    static constexpr index_t NPerBlock = PerBlock_M_N_K_O::At(Number<1>{});
    static constexpr index_t KPerBlock = PerBlock_M_N_K_O::At(Number<2>{});
    static constexpr index_t OPerBlock = PerBlock_M_N_K_O::At(Number<3>{});

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
};

} // namespace tensor_operation
} // namespace ck
