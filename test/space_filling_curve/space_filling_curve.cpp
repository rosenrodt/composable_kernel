// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include <vector>
#include <iostream>
#include <numeric>
#include <cassert>

#include "ck/ck.hpp"
#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_space_filling_curve.hpp"
#include "ck/tensor_operation/gpu/block/blockwise_gemm_xdlops.hpp"
#include <hip/hip_runtime.h>

using namespace ck;

void traverse_using_space_filling_curve();
void acc_tile_visitor();

int main(int argc, char** argv)
{
    (void)argc;
    (void)argv;

    traverse_using_space_filling_curve();
    acc_tile_visitor();

    return 0;
}

#if 1
__global__ void acc_tile_visitor_kernel(int* p)
{
    constexpr auto block_desc =
        make_naive_tensor_descriptor_packed(make_tuple(Number<8>{}, Number<128>{}, Number<4>{}));

    auto blockwise_gemm = BlockwiseGemmXdlops_k0mk1_k0nk1_m0n0m1n1m2m3m4n2_Selector<
        256,                       // BlockSize
        half_t,                    // FloatAB
        float,                     // FloatGemmAcc
        decltype(block_desc),      // decltype(a_block_desc_ak0_m_ak1)
        decltype(block_desc),      // decltype(b_block_desc_bk0_n_bk1)
        32,                        // MPerXdl
        32,                        // NPerXdl
        2,                         // MXdlPerWave
        2,                         // NXdlPerWave
        8,                         // KPack
        LoopScheduler::Default>(); // LoopSched

    // 8d thread_desc in thread scope
    constexpr auto c_thread_lengths =
        blockwise_gemm.GetCThreadDescriptor_M0_N0_M1_N1_M2_M3_M4_N2().GetLengths();

    // 8d block_desc in block scope
    constexpr auto c_block_lengths =
        blockwise_gemm.GetCBlockDescriptor_M0_N0_M1_N1_M2_M3_M4_N2().GetLengths();
    constexpr auto m0 = c_block_lengths.At(Number<0>{});
    constexpr auto n0 = c_block_lengths.At(Number<1>{});
    constexpr auto m1 = c_block_lengths.At(Number<2>{});
    constexpr auto n1 = c_block_lengths.At(Number<3>{});
    constexpr auto m2 = c_block_lengths.At(Number<4>{});
    constexpr auto m3 = c_block_lengths.At(Number<5>{});
    constexpr auto m4 = c_block_lengths.At(Number<6>{});
    constexpr auto n2 = c_block_lengths.At(Number<7>{});

    // 8d thread_cluster in block scope
    constexpr auto c_cluster_lengths = c_block_lengths / c_thread_lengths;

    // c_block_lengths.bar();   // ck::Sequence<2, 2, 2, 2, 4, 2, 4, 32>
    // c_thread_lengths.foo();  // ck::Sequence<2, 2, 1, 1, 4, 1, 4, 1>
    // c_cluster_lengths.faz(); // ck::Sequence<1, 1, 2, 2, 1, 2, 1, 32>
    (void)c_block_lengths;
    (void)c_cluster_lengths;

    using ThreadIterator = SpaceFillingCurve<decltype(c_thread_lengths),
                                                Sequence<0, 1, 2, 3, 4, 5, 6, 7>,
                                                Sequence<1, 1, 1, 1, 1, 1, 1, 1>,
                                                false>;
    // using BlockIterator = SpaceFillingCurve<decltype(c_block_lengths),
    //                                                  Sequence<0, 1, 2, 3, 4, 5, 6, 7>,
    //                                                  Sequence<1, 1, 2, 2, 1, 2, 1, 32>>;

    auto c_idx = blockwise_gemm.CalculateCThreadOriginDataIndex_8D(
        Number<0>{}, Number<0>{}, Number<0>{}, Number<0>{});

    // auto thread_slice_desc = blockwise_gemm.GetCThreadDescriptor_M0_N0_M1_N1_M2_M3_M4_N2();
    constexpr auto block_slice_8d_to_m_n_adaptor = make_single_stage_tensor_adaptor(
        make_tuple(make_unmerge_transform(make_tuple(m0, m1, m2, m3, m4)),
                   make_unmerge_transform(make_tuple(n0, n1, n2))),
        make_tuple(Sequence<0>{}, Sequence<1>{}),
        make_tuple(Sequence<0, 2, 4, 5, 6>{}, Sequence<1, 3, 7>{}));

    static_for<0, /* ThreadIterator::GetNumOfAccess() */ Number<8>{}, 1>{}([&](auto i) {
        constexpr auto idx = ThreadIterator::GetIndex(i);
        auto blockwise_idx = ThreadIterator::GetIndex(i) + c_idx;
        auto blockwise_m_n = block_slice_8d_to_m_n_adaptor.CalculateBottomIndex(blockwise_idx);
        printf("tid = %zd, i = %d, idx = (%d, %d, %d, %d, %d, %d, %d, %d), blockwise_idx = (%d, %d, %d, %d, "
               "%d, %d, %d, %d), (m, n) = (%d, %d)\n",
               hipThreadIdx_x,
               i.value,
               idx[Number<0>{}],
               idx[Number<1>{}],
               idx[Number<2>{}],
               idx[Number<3>{}],
               idx[Number<4>{}],
               idx[Number<5>{}],
               idx[Number<6>{}],
               idx[Number<7>{}],
               blockwise_idx[Number<0>{}],
               blockwise_idx[Number<1>{}],
               blockwise_idx[Number<2>{}],
               blockwise_idx[Number<3>{}],
               blockwise_idx[Number<4>{}],
               blockwise_idx[Number<5>{}],
               blockwise_idx[Number<6>{}],
               blockwise_idx[Number<7>{}],
               blockwise_m_n[Number<0>{}],
               blockwise_m_n[Number<1>{}]);
    });

    // 8d to 1d vgpr offset
    // 8d to m_local, n_local

    p[0] = 1;
}
#endif

void acc_tile_visitor()
{
    printf("acc_tile_visitor\n");
    int* p_dev;
    std::ignore = hipMalloc(&p_dev, 4);
    acc_tile_visitor_kernel<<<1, 256>>>(p_dev);
    std::ignore = hipFree(p_dev);
}

void traverse_using_space_filling_curve()
{
    constexpr auto I0 = Number<0>{};
    constexpr auto I1 = Number<1>{};
    constexpr auto I2 = Number<2>{};

    using TensorLengths     = Sequence<16, 10, 9>;
    using DimAccessOrder    = Sequence<2, 0, 1>;
    using ScalarsPerAccess  = Sequence<4, 2, 3>;
    using SpaceFillingCurve = SpaceFillingCurve<TensorLengths, DimAccessOrder, ScalarsPerAccess>;

    constexpr auto expected = make_tuple(make_tuple(0, 0, 0),
                                         make_tuple(0, 2, 0),
                                         make_tuple(0, 4, 0),
                                         make_tuple(0, 6, 0),
                                         make_tuple(0, 8, 0),
                                         make_tuple(4, 8, 0),
                                         make_tuple(4, 6, 0),
                                         make_tuple(4, 4, 0),
                                         make_tuple(4, 2, 0),
                                         make_tuple(4, 0, 0),
                                         make_tuple(8, 0, 0),
                                         make_tuple(8, 2, 0),
                                         make_tuple(8, 4, 0),
                                         make_tuple(8, 6, 0),
                                         make_tuple(8, 8, 0),
                                         make_tuple(12, 8, 0),
                                         make_tuple(12, 6, 0),
                                         make_tuple(12, 4, 0),
                                         make_tuple(12, 2, 0),
                                         make_tuple(12, 0, 0),
                                         make_tuple(12, 0, 3),
                                         make_tuple(12, 2, 3),
                                         make_tuple(12, 4, 3),
                                         make_tuple(12, 6, 3),
                                         make_tuple(12, 8, 3),
                                         make_tuple(8, 8, 3),
                                         make_tuple(8, 6, 3),
                                         make_tuple(8, 4, 3),
                                         make_tuple(8, 2, 3),
                                         make_tuple(8, 0, 3),
                                         make_tuple(4, 0, 3),
                                         make_tuple(4, 2, 3),
                                         make_tuple(4, 4, 3),
                                         make_tuple(4, 6, 3),
                                         make_tuple(4, 8, 3),
                                         make_tuple(0, 8, 3),
                                         make_tuple(0, 6, 3),
                                         make_tuple(0, 4, 3),
                                         make_tuple(0, 2, 3),
                                         make_tuple(0, 0, 3),
                                         make_tuple(0, 0, 6),
                                         make_tuple(0, 2, 6),
                                         make_tuple(0, 4, 6),
                                         make_tuple(0, 6, 6),
                                         make_tuple(0, 8, 6),
                                         make_tuple(4, 8, 6),
                                         make_tuple(4, 6, 6),
                                         make_tuple(4, 4, 6),
                                         make_tuple(4, 2, 6),
                                         make_tuple(4, 0, 6),
                                         make_tuple(8, 0, 6),
                                         make_tuple(8, 2, 6),
                                         make_tuple(8, 4, 6),
                                         make_tuple(8, 6, 6),
                                         make_tuple(8, 8, 6),
                                         make_tuple(12, 8, 6),
                                         make_tuple(12, 6, 6),
                                         make_tuple(12, 4, 6),
                                         make_tuple(12, 2, 6),
                                         make_tuple(12, 0, 6));

    constexpr index_t num_access = SpaceFillingCurve::GetNumOfAccess();

    static_assert(num_access == reduce_on_sequence(TensorLengths{} / ScalarsPerAccess{},
                                                   math::multiplies{},
                                                   Number<1>{}));

    static_for<1, num_access, 1>{}([&](auto i) {
        constexpr auto idx_curr = SpaceFillingCurve::GetIndex(i);

        static_assert(idx_curr[I0] == expected[i][I0]);
        static_assert(idx_curr[I1] == expected[i][I1]);
        static_assert(idx_curr[I2] == expected[i][I2]);

        constexpr auto backward_step = SpaceFillingCurve::GetBackwardStep(i);
        constexpr auto expected_step = expected[i - I1] - expected[i];
        static_assert(backward_step[I0] == expected_step[I0]);
        static_assert(backward_step[I1] == expected_step[I1]);
        static_assert(backward_step[I2] == expected_step[I2]);
    });

    static_for<0, num_access - 1, 1>{}([&](auto i) {
        constexpr auto idx_curr = SpaceFillingCurve::GetIndex(i);

        static_assert(idx_curr[I0] == expected[i][I0]);
        static_assert(idx_curr[I1] == expected[i][I1]);
        static_assert(idx_curr[I2] == expected[i][I2]);

        constexpr auto forward_step  = SpaceFillingCurve::GetForwardStep(i);
        constexpr auto expected_step = expected[i + I1] - expected[i];
        static_assert(forward_step[I0] == expected_step[I0]);
        static_assert(forward_step[I1] == expected_step[I1]);
        static_assert(forward_step[I2] == expected_step[I2]);
    });
}
