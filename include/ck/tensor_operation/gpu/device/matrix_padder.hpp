// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

template <bool PadM,
          bool PadN,
          typename TensorDesc_MRaw_NRaw,
          typename MPerBlockType,
          typename NPerBlockType>
__host__ __device__ constexpr auto
PadTensorDescriptor_M_N(const TensorDesc_MRaw_NRaw& tensor_desc_mraw_nraw,
                        MPerBlockType MPerBlock,
                        NPerBlockType NPerBlock)
{
    const auto MRaw = tensor_desc_mraw_nraw.GetLength(Number<0>{});
    const auto NRaw = tensor_desc_mraw_nraw.GetLength(Number<1>{});

    const auto M = math::integer_divide_ceil(MRaw, MPerBlock) * MPerBlock;
    const auto N = math::integer_divide_ceil(NRaw, NPerBlock) * NPerBlock;

    const auto MPad = M - MRaw;
    const auto NPad = N - NRaw;

    const auto MTransform = conditional_expr<PadM>(make_right_pad_transform(MRaw, MPad),
                                                   make_pass_through_transform(MRaw));
    const auto NTransform = conditional_expr<PadN>(make_right_pad_transform(NRaw, NPad),
                                                   make_pass_through_transform(NRaw));

    return transform_tensor_descriptor(tensor_desc_mraw_nraw,
                                       make_tuple(MTransform, NTransform),
                                       make_tuple(Sequence<0>{}, Sequence<1>{}),
                                       make_tuple(Sequence<0>{}, Sequence<1>{}));
}

// M/N/KPerTileType could be index_t or Number<>
template <GemmSpecialization GemmSpec,
          typename MPerTileType,
          typename NPerTileType,
          typename KPerTileType>
struct MatrixPadder
{
    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};
    static constexpr auto I3 = Number<3>{};

    static constexpr bool PadM =
        (GemmSpec == GemmSpecialization::MPadding || GemmSpec == GemmSpecialization::MNPadding ||
         GemmSpec == GemmSpecialization::MKPadding || GemmSpec == GemmSpecialization::MNKPadding);
    static constexpr bool PadN =
        (GemmSpec == GemmSpecialization::NPadding || GemmSpec == GemmSpecialization::MNPadding ||
         GemmSpec == GemmSpecialization::NKPadding || GemmSpec == GemmSpecialization::MNKPadding);
    static constexpr bool PadK =
        (GemmSpec == GemmSpecialization::KPadding || GemmSpec == GemmSpecialization::MKPadding ||
         GemmSpec == GemmSpecialization::NKPadding || GemmSpec == GemmSpecialization::MNKPadding);

    template <typename ADesc_MRaw_KRaw>
    __host__ __device__ constexpr auto
    PadADescriptor_M_K(const ADesc_MRaw_KRaw& a_desc_mraw_kraw) const
    {
        return PadTensorDescriptor_M_N<PadM, PadK>(a_desc_mraw_kraw, MPerTile_, KPerTile_);
    }

    template <typename BDesc_NRaw_KRaw>
    __host__ __device__ constexpr auto
    PadBDescriptor_N_K(const BDesc_NRaw_KRaw& b_desc_nraw_kraw) const
    {
        return PadTensorDescriptor_M_N<PadN, PadK>(b_desc_nraw_kraw, NPerTile_, KPerTile_);
    }

    template <typename CDesc_MRaw_NRaw>
    __host__ __device__ constexpr auto
    PadCDescriptor_M_N(const CDesc_MRaw_NRaw& c_desc_mraw_nraw) const
    {
        return PadTensorDescriptor_M_N<PadM, PadN>(c_desc_mraw_nraw, MPerTile_, NPerTile_);
    }

    MPerTileType MPerTile_;
    NPerTileType NPerTile_;
    KPerTileType KPerTile_;
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
