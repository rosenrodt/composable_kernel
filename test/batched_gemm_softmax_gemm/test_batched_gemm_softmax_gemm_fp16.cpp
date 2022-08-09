// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include "gtest/gtest.h"
#include "test_batched_gemm_softmax_gemm_util.hpp"

template <ck::index_t N>
using I = ck::Number<N>;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;
using F16 = ck::half_t;

template <typename Tuple>
class TestGemmSoftmaxGemmFP16 : public TestGemmSoftmaxGemm<Tuple>
{
};

// clang-format off
using KernelTypes = ::testing::Types<
    std::tuple<F16, F16, F16, F16, Row, Col, Row, Row>
    >;
// clang-format on

TYPED_TEST_SUITE(TestGemmSoftmaxGemmFP16, KernelTypes);
TYPED_TEST(TestGemmSoftmaxGemmFP16, Test_FP16) { this->Run(); }
