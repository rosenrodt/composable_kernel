// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/data_type.hpp"
#include "ck/utility/reduction_common.hpp"
#include "ck/utility/reduction_operator.hpp"
#include "ck/utility/reduction_functions_accumulate.hpp"
#include "ck/tensor_operation/gpu/block/reduction_functions_blockwise.hpp"
#include "ck/tensor_operation/gpu/thread/reduction_functions_threadwise.hpp"

namespace ck {

// TODO: how to capture different use cases like "load + softmax" and "gemm + softmax"? obviously
//       static buffer will be two different classes with their own accessors
template <index_t BlockSize,
          typename AccDataType,
          typename ThreadMap_M_K, // thread_id to m_k
          typename ThreadClusterDesc_M_K,
          typename ThreadSliceDesc_M_K>
struct BlockwiseSoftmax_V1
{
    static constexpr auto I0         = Number<0>{};
    static constexpr auto I1         = Number<1>{};
    static constexpr index_t MRepeat = ThreadSliceDesc_M_K{}.GetLength(I0);
    static constexpr index_t KRepeat = ThreadSliceDesc_M_K{}.GetLength(I1);

    using ThreadSliceDesc_M = decltype(make_naive_tensor_descriptor_packed(
        make_tuple(ThreadSliceDesc_M_K{}.GetLength(I0))));

    using ThreadwiseMaxReduce = ThreadwiseReduction<AccDataType,
                                                    ThreadSliceDesc_M_K,
                                                    ThreadSliceDesc_M,
                                                    reduce::Max,
                                                    false>;

    using ThreadClusterLengths_M_K = decltype(ThreadClusterDesc_M_K{}.GetLengths());

    using BlockwiseMaxReduce = PartitionedBlockwiseReduction2<AccDataType,
                                                              BlockSize,
                                                              ThreadClusterLengths_M_K,
                                                              ThreadMap_M_K,
                                                              reduce::Max,
                                                              false>;

    using BlockwiseSumReduce = PartitionedBlockwiseReduction2<AccDataType,
                                                              BlockSize,
                                                              ThreadClusterLengths_M_K,
                                                              ThreadMap_M_K,
                                                              reduce::Add,
                                                              false>;

    using ThreadwiseSumReduce = ThreadwiseReduction<AccDataType,
                                                    ThreadSliceDesc_M_K,
                                                    ThreadSliceDesc_M,
                                                    reduce::Add,
                                                    false>;

    using BufferType = StaticBuffer<AddressSpaceEnum::Vgpr, AccDataType, MRepeat, true>;

    template <typename CThreadBuffer, typename WorkspaceBuffer>
    __host__ __device__ void Run(CThreadBuffer& in_thread_buf, WorkspaceBuffer& reduce_work_buf)
    {
        // find max value
        static_for<0, MRepeat, 1>{}([&](auto I) {
            max_value_buf(I) = reduce::Max::template GetIdentityValue<AccDataType>();
        });
        ThreadwiseMaxReduce::Reduce(in_thread_buf, max_value_buf);
        static_for<0, MRepeat, 1>{}([&](auto I) {
            BlockwiseMaxReduce::Reduce(reduce_work_buf, max_value_buf(I));
            block_sync_lds();
        });

        // calculate exp for elements, P=exp(s-max)
        static_for<0, MRepeat, 1>{}([&](auto iM) {
            static_for<0, KRepeat, 1>{}([&](auto iK) {
                auto offset = Number<ThreadClusterDesc_M_K{}.CalculateOffset(make_tuple(iM, iK))>{};
                in_thread_buf(offset) = math::exp(in_thread_buf[offset] - max_value_buf(iM));
            });
        });

        // sum data
        static_for<0, MRepeat, 1>{}([&](auto I) {
            sum_value_buf(I) = reduce::Add::template GetIdentityValue<AccDataType>();
        });
        ThreadwiseSumReduce::Reduce(in_thread_buf, sum_value_buf);
        static_for<0, MRepeat, 1>{}([&](auto I) {
            BlockwiseSumReduce::Reduce(reduce_work_buf, sum_value_buf(I));
            block_sync_lds();
        });
    }

    BufferType max_value_buf;
    BufferType sum_value_buf;
};

} // namespace ck
