// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <iostream>
#include <vector>

#include "device_base.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

struct CPermuteDesc_G0_G1_M_O
{
    ck::index_t G0_, G1_, M_, O_;
    ck::index_t stride_G0_, stride_G1_, stride_M_, stride_O_;
};

template <typename ALayout,
          typename B0Layout,
          typename B1Layout,
          typename CLayout,
          typename ADataType,
          typename B0DataType,
          typename B1DataType,
          typename CDataType,
          typename CPermuteDesc_G0_G1_M_O,
          typename AElementwiseOperation,
          typename B0ElementwiseOperation,
          typename Acc0ElementwiseOperation,
          typename B1ElementwiseOperation,
          typename CElementwiseOperation>
struct DeviceBatchedGemmSoftmaxGemm : public BaseOperator
{
    virtual std::unique_ptr<BaseArgument>
    MakeArgumentPointer(const void* p_a,
                        const void* p_b0,
                        const void* p_b1,
                        void* p_c,
                        ck::index_t M,
                        ck::index_t N,
                        ck::index_t K,
                        ck::index_t O,
                        ck::index_t StrideA,
                        ck::index_t StrideB0,
                        ck::index_t StrideB1,
                        ck::index_t BatchStrideA,
                        ck::index_t BatchStrideB0,
                        ck::index_t BatchStrideB1,
                        CPermuteDesc_G0_G1_M_O c_permute_desc,
                        AElementwiseOperation a_element_op,
                        B0ElementwiseOperation b0_element_op,
                        Acc0ElementwiseOperation acc0_element_op,
                        B1ElementwiseOperation b1_element_op,
                        CElementwiseOperation c_element_op) = 0;

    virtual std::unique_ptr<BaseInvoker> MakeInvokerPointer() = 0;
};

template <typename ALayout,
          typename B0Layout,
          typename B1Layout,
          typename CLayout,
          typename ADataType,
          typename B0DataType,
          typename B1DataType,
          typename CDataType,
          typename CPermuteDesc_G0_G1_M_O,
          typename AElementwiseOperation,
          typename B0ElementwiseOperation,
          typename Acc0ElementwiseOperation,
          typename B1ElementwiseOperation,
          typename CElementwiseOperation>
using DeviceBatchedGemmSoftmaxGemmPtr =
    std::unique_ptr<DeviceBatchedGemmSoftmaxGemm<ALayout,
                                                 B0Layout,
                                                 B1Layout,
                                                 CLayout,
                                                 ADataType,
                                                 B0DataType,
                                                 B1DataType,
                                                 CDataType,
                                                 CPermuteDesc_G0_G1_M_O,
                                                 AElementwiseOperation,
                                                 B0ElementwiseOperation,
                                                 Acc0ElementwiseOperation,
                                                 B1ElementwiseOperation,
                                                 CElementwiseOperation>>;

} // namespace device
} // namespace tensor_operation
} // namespace ck
