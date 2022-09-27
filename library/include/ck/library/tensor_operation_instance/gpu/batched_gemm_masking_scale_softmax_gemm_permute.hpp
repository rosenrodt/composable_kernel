// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <cstdlib>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_batched_gemm_softmax_gemm_permute.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/tensor_operation_instance/device_operation_instance_factory.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

void add_device_batched_gemm_masking_scale_softmax_gemm_permute_xdl_cshuffle_f16_f16_f16_f16_gmk_gnk_gno_gmo_instances(
    std::vector<std::unique_ptr<DeviceBatchedGemmSoftmaxGemmPermute<2,
                                                                    1,
                                                                    1,
                                                                    1,
                                                                    1,
                                                                    F16,
                                                                    F16,
                                                                    F16,
                                                                    F16,
                                                                    ck::Tuple<>,
                                                                    ck::Tuple<>,
                                                                    PassThrough,
                                                                    PassThrough,
                                                                    Scale,
                                                                    PassThrough,
                                                                    PassThrough,
                                                                    true>>>& instances);

void add_device_batched_gemm_scale_softmax_gemm_permute_xdl_cshuffle_f16_f16_f16_f16_gmk_gnk_gno_gmo_instances(
    std::vector<std::unique_ptr<DeviceBatchedGemmSoftmaxGemmPermute<2,
                                                                    1,
                                                                    1,
                                                                    1,
                                                                    1,
                                                                    F16,
                                                                    F16,
                                                                    F16,
                                                                    F16,
                                                                    ck::Tuple<>,
                                                                    ck::Tuple<>,
                                                                    PassThrough,
                                                                    PassThrough,
                                                                    Scale,
                                                                    PassThrough,
                                                                    PassThrough,
                                                                    false>>>& instances);

template <typename ADataType,
          typename B0DataType,
          typename B1DataType,
          typename CDataType,
          bool MaskOutUpperTriangle>
struct DeviceOperationInstanceFactory<
    ck::tensor_operation::device::DeviceBatchedGemmSoftmaxGemmPermute<2,
                                                                      1,
                                                                      1,
                                                                      1,
                                                                      1,
                                                                      ADataType,
                                                                      B0DataType,
                                                                      B1DataType,
                                                                      CDataType,
                                                                      ck::Tuple<>,
                                                                      ck::Tuple<>,
                                                                      PassThrough,
                                                                      PassThrough,
                                                                      Scale,
                                                                      PassThrough,
                                                                      PassThrough,
                                                                      MaskOutUpperTriangle>>
{
    using DeviceOp = DeviceBatchedGemmSoftmaxGemmPermute<2,
                                                         1,
                                                         1,
                                                         1,
                                                         1,
                                                         ADataType,
                                                         B0DataType,
                                                         B1DataType,
                                                         CDataType,
                                                         ck::Tuple<>,
                                                         ck::Tuple<>,
                                                         PassThrough,
                                                         PassThrough,
                                                         Scale,
                                                         PassThrough,
                                                         PassThrough,
                                                         MaskOutUpperTriangle>;

    static auto GetInstances()
    {
        std::vector<std::unique_ptr<DeviceOp>> op_ptrs;

        if constexpr(is_same_v<ADataType, half_t> && is_same_v<B0DataType, half_t> &&
                     is_same_v<B1DataType, half_t> && is_same_v<CDataType, half_t>)
        {
            if constexpr(MaskOutUpperTriangle)
            {
                // TODO ANT: batched_gemm_softmax_gemm_ab0b1cpermute
                add_device_batched_gemm_masking_scale_softmax_gemm_permute_xdl_cshuffle_f16_f16_f16_f16_gmk_gnk_gno_gmo_instances(
                    op_ptrs);
            }
            else
            {
                add_device_batched_gemm_scale_softmax_gemm_permute_xdl_cshuffle_f16_f16_f16_f16_gmk_gnk_gno_gmo_instances(
                    op_ptrs);
            }
        }
        return op_ptrs;
    }
};

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
