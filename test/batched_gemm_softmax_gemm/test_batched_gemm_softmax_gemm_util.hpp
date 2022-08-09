// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include <iostream>

#include <vector>
#include "profiler/include/profile_batched_gemm_softmax_gemm_impl.hpp"

using ADataType = ck::half_t;
using B0DataType = ck::half_t;
using B1DataType = ck::half_t;
using CDataType = ck::half_t;

template <typename Tuple>
struct TestGemmSoftmaxGemm : public ::testing::Test
{
    using ADataType  = std::tuple_element_t<0, Tuple>;
    using B0DataType = std::tuple_element_t<1, Tuple>;
    using B1DataType = std::tuple_element_t<2, Tuple>;
    using CDataType  = std::tuple_element_t<3, Tuple>;
    using ALayout    = std::tuple_element_t<4, Tuple>;
    using B0Layout   = std::tuple_element_t<5, Tuple>;
    using B1Layout   = std::tuple_element_t<6, Tuple>;
    using CLayout    = std::tuple_element_t<7, Tuple>;

    // std::vector<std::vector<int>> lengths_ = {{128, 128, 32, 128, 1}, {1024, 1024, 1024, 1024, 4}};
    std::vector<std::vector<int>> lengths_ = {{128, 128, 32, 128, 1}};

    void RunSingle(int M, int N, int K, int O, int BatchCount)
    {
        bool pass = ck::profiler::profile_batched_gemm_softmax_gemm_impl<ADataType,
                                                                         B0DataType,
                                                                         B1DataType,
                                                                         CDataType,
                                                                         ALayout,
                                                                         B0Layout,
                                                                         B1Layout,
                                                                         CLayout>(
            true, 1, false, true, M, N, K, O, BatchCount);

        EXPECT_TRUE(pass);
    }

    void Run()
    {
        for(auto lengths : this->lengths_)
        {
            int M          = lengths[0];
            int N          = lengths[1];
            int K          = lengths[2];
            int O          = lengths[3];
            int BatchCount = lengths[4];

            this->RunSingle(M, N, K, O, BatchCount);
        }
    }
};
