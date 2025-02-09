/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2020 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/
#ifndef CK_REDUCTION_FUNCTIONS_BLOCKWISE_HPP
#define CK_REDUCTION_FUNCTIONS_BLOCKWISE_HPP

#include "data_type.hpp"

#include "reduction_common.hpp"
#include "reduction_operator.hpp"
#include "reduction_functions_accumulate.hpp"

#include "cluster_descriptor.hpp"

namespace ck {

template <typename AccDataType,
          index_t BlockSize,
          typename ThreadClusterLengths_M_K,
          typename ThreadClusterArrangeOrder,
          typename OpReduce,
          bool PropagateNan>
struct PartitionedBlockwiseReduction
{
    static_assert(BlockSize == ThreadClusterLengths_M_K::At(0) * ThreadClusterLengths_M_K::At(1),
                  "The product of cluster lengths should be same as BlockSize!");

    static constexpr auto BufferLength_M = ThreadClusterLengths_M_K::At(0);
    static constexpr auto BufferLength_K = ThreadClusterLengths_M_K::At(1);

    static_assert(BufferLength_K > 1, "Parallel reduction need work on at least two elements");

    static constexpr auto block_buf_desc_m_k = make_naive_tensor_descriptor_packed(
        make_tuple(Number<BufferLength_M>{}, Number<BufferLength_K>{}));

    static constexpr auto thread_cluster_desc =
        make_cluster_descriptor(ThreadClusterLengths_M_K{}, ThreadClusterArrangeOrder{});

    using Accumulation = detail::AccumulateWithNanCheck<PropagateNan, OpReduce, AccDataType>;

    template <typename BufferType>
    __device__ static void Reduce(BufferType& block_buffer, AccDataType& accuData)
    {
        constexpr auto cluster_len_shift = get_shift<BufferLength_K>();

        const auto thread_cluster_idx =
            thread_cluster_desc.CalculateBottomIndex(make_multi_index(get_thread_local_1d_id()));

        const auto thread_m_cluster_id = thread_cluster_idx[Number<0>{}];
        const auto thread_k_cluster_id = thread_cluster_idx[Number<1>{}];

        static_for<0, cluster_len_shift, 1>{}([&](auto I) {
            constexpr index_t indOffset = 1 << (cluster_len_shift - 1 - I());

            if(thread_k_cluster_id < indOffset)
            {
                index_t offset1 = block_buf_desc_m_k.CalculateOffset(thread_cluster_idx);
                index_t offset2 = block_buf_desc_m_k.CalculateOffset(thread_cluster_idx +
                                                                     make_tuple(0, indOffset));

                AccDataType opData1 = type_convert<AccDataType>(block_buffer[offset1]);
                AccDataType opData2 = type_convert<AccDataType>(block_buffer[offset2]);
                Accumulation::Calculate(opData1, opData2);
                block_buffer(offset1) = type_convert<AccDataType>(opData1);
            }

            __syncthreads();
        });

        index_t offset = block_buf_desc_m_k.CalculateOffset(make_tuple(thread_m_cluster_id, 0));

        accuData = type_convert<AccDataType>(block_buffer[offset]);
    };
};

template <typename AccDataType,
          typename IndexDataType,
          index_t BlockSize,
          typename ThreadClusterLengths_M_K,
          typename ThreadClusterArrangeOrder,
          typename OpReduce,
          bool PropagateNan>
struct PartitionedBlockwiseReductionWithIndex
{
    static_assert(BlockSize == ThreadClusterLengths_M_K::At(0) * ThreadClusterLengths_M_K::At(1),
                  "The product of cluster lengths should be same as BlockSize!");

    static constexpr auto BufferLength_M = ThreadClusterLengths_M_K::At(0);
    static constexpr auto BufferLength_K = ThreadClusterLengths_M_K::At(1);

    static_assert(BufferLength_K > 1, "Parallel reduction need work on at least two elements");

    static constexpr auto block_buf_desc_m_k = make_naive_tensor_descriptor_packed(
        make_tuple(Number<BufferLength_M>{}, Number<BufferLength_K>{}));

    static constexpr auto thread_cluster_desc =
        make_cluster_descriptor(ThreadClusterLengths_M_K{}, ThreadClusterArrangeOrder{});

    using Accumulation =
        detail::AccumulateWithIndexAndNanCheck<PropagateNan, OpReduce, AccDataType, IndexDataType>;

    // This interface accumulates on both data values and indices
    template <typename BufferType, typename IdxBufferType>
    __device__ static void Reduce(BufferType& block_val_buffer,
                                  IdxBufferType& block_idx_buffer,
                                  AccDataType& accuData,
                                  IndexDataType& accuIndex)
    {
        constexpr auto cluster_len_shift = get_shift<BufferLength_K>();

        const auto thread_cluster_idx =
            thread_cluster_desc.CalculateBottomIndex(make_multi_index(get_thread_local_1d_id()));

        const auto thread_m_cluster_id = thread_cluster_idx[Number<0>{}];
        const auto thread_k_cluster_id = thread_cluster_idx[Number<1>{}];

        static_for<0, cluster_len_shift, 1>{}([&](auto I) {
            constexpr index_t indOffset = 1 << I();

            if(thread_k_cluster_id % (indOffset * 2) == 0)
            {
                index_t offset1 = block_buf_desc_m_k.CalculateOffset(thread_cluster_idx);
                index_t offset2 = block_buf_desc_m_k.CalculateOffset(thread_cluster_idx +
                                                                     make_tuple(0, indOffset));

                AccDataType opData1      = type_convert<AccDataType>(block_val_buffer[offset1]);
                AccDataType opData2      = type_convert<AccDataType>(block_val_buffer[offset2]);
                IndexDataType currIndex1 = block_idx_buffer[offset1];
                IndexDataType currIndex2 = block_idx_buffer[offset2];

                Accumulation::Calculate(opData1, opData2, currIndex1, currIndex2);
                block_val_buffer(offset1) = type_convert<AccDataType>(opData1);
                block_idx_buffer(offset1) = currIndex1;
            }

            __syncthreads();
        });

        index_t offset = block_buf_desc_m_k.CalculateOffset(make_tuple(thread_m_cluster_id, 0));

        accuData  = type_convert<AccDataType>(block_val_buffer[offset]);
        accuIndex = block_idx_buffer[offset];
    }
};

}; // end of namespace ck

#endif
