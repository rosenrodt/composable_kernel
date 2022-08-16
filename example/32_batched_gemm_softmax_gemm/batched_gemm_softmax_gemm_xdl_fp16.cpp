// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

/*
Gemm + Gemm fused operation. Computes C_m_o = A_m_k * B0_k_n * B1_n_o
                                              |------------|
                                                   Gemm0
                                              |---------------------|
                                                       Gemm1
*/

#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/device_batched_gemm_softmax_gemm_xdl_cshuffle.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_batched_gemm.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_softmax.hpp"

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using F16 = ck::half_t;
using F32 = float;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;

using ADataType        = F16;
using B0DataType       = F16;
using B1DataType       = F16;
using AccDataType      = F32;
using CShuffleDataType = F32;
using CDataType        = F16;

using ALayout  = Row;
using B0Layout = Col;
using B1Layout = Row;
using CLayout  = Row;

using AElementOp    = PassThrough;
using B0ElementOp   = PassThrough;
using Acc0ElementOp = PassThrough;
using B1ElementOp   = PassThrough;
using CElementOp    = PassThrough;

static constexpr auto GemmDefault = ck::tensor_operation::device::GemmSpecialization::Default;

using DeviceGemmInstance = ck::tensor_operation::device::DeviceBatchedGemmSoftmaxGemm_Xdl_CShuffle<
    ALayout,
    B0Layout,
    B1Layout,
    CLayout,
    ADataType,
    B0DataType,
    B1DataType,
    CDataType,
    AccDataType,
    CShuffleDataType,
    AElementOp,
    B0ElementOp,
    Acc0ElementOp,
    B1ElementOp,
    CElementOp,
    GemmDefault,
    1,
    256,
    128,         // MPerBlock
    128,         // NPerBlock
    32,          // KPerBlock
    64,          // Gemm1NPerBlock
    32,          // Gemm1KPerBlock
    8,           // AK1
    8,           // BK1
    2,           // B1K1
    32,          // MPerXDL
    32,          // NPerXDL
    1,           // MXdlPerWave
    4,           // NXdlPerWave
    2,           // Gemm1NXdlPerWave
    S<4, 64, 1>, // ABlockTransfer
    S<1, 0, 2>,
    S<1, 0, 2>,
    2,
    8,
    8,
    true,
    S<4, 64, 1>, // BBlockTransfer
    S<1, 0, 2>,
    S<1, 0, 2>,
    2,
    8,
    8,
    true,
    S<16, 16, 1>, // B1BlockTransfer
    S<0, 2, 1>,
    S<0, 2, 1>,
    1,
    4,
    2,
    false,
    1,              // CShuffleMXdlPerWavePerShuffle
    2,              // CShuffleNXdlPerWavePerShuffle
    S<1, 32, 1, 8>, // CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock
    8>;             // CShuffleBlockTransferScalarPerVector_NPerBlock

// Ref Gemm0: fp16 in, fp32 out
using ReferenceGemm0Instance = ck::tensor_operation::host::ReferenceBatchedGemm<ADataType,
                                                                                B0DataType,
                                                                                AccDataType,
                                                                                AccDataType,
                                                                                AElementOp,
                                                                                B0ElementOp,
                                                                                CElementOp>;

// Ref Softmax: fp32 in, fp16 out
using ReferenceSoftmaxInstance =
    ck::tensor_operation::host::ReferenceSoftmax<AccDataType, ADataType, AccDataType>;

// Ref Gemm1: fp16 in, fp16 out
using ReferenceGemm1Instance = ck::tensor_operation::host::ReferenceBatchedGemm<ADataType,
                                                                                B1DataType,
                                                                                CDataType,
                                                                                AccDataType,
                                                                                AElementOp,
                                                                                B1ElementOp,
                                                                                CElementOp>;

int main(int argc, char* argv[])
{
    bool do_verification = true;
    int init_method      = 1;
    bool time_kernel     = false;

    // GEMM shape
    ck::index_t M  = 1024;
    ck::index_t N  = 1024;
    ck::index_t K  = 64;
    ck::index_t O  = 128;
    ck::index_t G0 = 3;
    ck::index_t G1 = 7;

    if(argc == 1)
    {
        // use default case
    }
    else if(argc == 4)
    {
        do_verification = std::stoi(argv[1]);
        init_method     = std::stoi(argv[2]);
        time_kernel     = std::stoi(argv[3]);
    }
    else if(argc == 10)
    {
        do_verification = std::stoi(argv[1]);
        init_method     = std::stoi(argv[2]);
        time_kernel     = std::stoi(argv[3]);

        M = std::stoi(argv[4]);
        N = std::stoi(argv[5]);
        K = std::stoi(argv[6]);
        O = std::stoi(argv[7]);

        G0 = std::stoi(argv[8]);
        G1 = std::stoi(argv[9]);
    }
    else
    {
        printf("arg1: verification (0=no, 1=yes)\n");
        printf("arg2: initialization (0=no init, 1=integer value, 2=decimal value)\n");
        printf("arg3: time kernel (0=no, 1=yes)\n");
        printf("arg4 to 10: M, N, K, O, G0, G1\n");
        exit(0);
    }

    const int StrideA  = ck::is_same_v<ALayout, Row> ? K : M;
    const int StrideB0 = ck::is_same_v<B0Layout, Row> ? N : K;
    const int StrideB1 = ck::is_same_v<B1Layout, Row> ? O : N;
    const int StrideC  = ck::is_same_v<CLayout, Row> ? O : M;

    const int BatchStrideA  = (ck::is_same_v<ALayout, Col> ? K : M) * StrideA;
    const int BatchStrideB0 = (ck::is_same_v<B0Layout, Col> ? N : K) * StrideB0;
    const int BatchStrideB1 = (ck::is_same_v<B1Layout, Col> ? O : N) * StrideB1;
    const int BatchStrideC  = (ck::is_same_v<CLayout, Col> ? O : M) * StrideC;

    const int BatchCount = G0 * G1;

    // output layout row major C - [G0, M, G1, O]
    //               col major C - [G0, O, G1, M]
    const int StrideC_G0 = G1 * BatchStrideC;
    const int StrideC_G1 = StrideC;
    const int StrideC_M  = ck::is_same_v<CLayout, Col> ? 1 : G1 * StrideC;
    const int StrideC_O  = ck::is_same_v<CLayout, Row> ? 1 : G1 * StrideC;

    ck::tensor_operation::device::CPermuteDesc_G0_G1_M_O c_permute_desc{
        G0, G1, M, O, StrideC_G0, StrideC_G1, StrideC_M, StrideC_O};

    auto f_host_c_tensor_descriptor = [](std::size_t G0_,
                                         std::size_t G1_,
                                         std::size_t M_,
                                         std::size_t N_,
                                         std::size_t stride_G0_,
                                         std::size_t stride_G1_,
                                         std::size_t stride_M_,
                                         std::size_t stride_N_) {
        return HostTensorDescriptor(
            std::vector<std::size_t>({G0_, G1_, M_, N_}),
            std::vector<std::size_t>({stride_G0_, stride_G1_, stride_M_, stride_N_}));
    };

    auto f_host_tensor_descriptor = [](std::size_t batch_count,
                                       std::size_t row,
                                       std::size_t col,
                                       std::size_t stride,
                                       std::size_t batch_stride,
                                       auto layout) {
        if(std::is_same<decltype(layout), Row>::value)
        {
            return HostTensorDescriptor(std::vector<std::size_t>({batch_count, row, col}),
                                        std::vector<std::size_t>({batch_stride, stride, 1}));
        }
        else
        {
            return HostTensorDescriptor(std::vector<std::size_t>({batch_count, row, col}),
                                        std::vector<std::size_t>({batch_stride, 1, stride}));
        }
    };

    // C_m_o = A_m_k * B0_k_n * B1_n_o
    Tensor<ADataType> a_g_m_k(
        f_host_tensor_descriptor(BatchCount, M, K, StrideA, BatchStrideA, ALayout{}));
    Tensor<B0DataType> b0_g_k_n(
        f_host_tensor_descriptor(BatchCount, K, N, StrideB0, BatchStrideB0, B0Layout{}));
    Tensor<B1DataType> b1_g_n_o(
        f_host_tensor_descriptor(BatchCount, N, O, StrideB1, BatchStrideB1, B1Layout{}));
    Tensor<CDataType> c_g0_g1_m_o_host_result(
        f_host_c_tensor_descriptor(G0, G1, M, O, StrideC_G0, StrideC_G1, StrideC_M, StrideC_O));
    Tensor<CDataType> c_g0_g1_m_o_device_result(
        f_host_c_tensor_descriptor(G0, G1, M, O, StrideC_G0, StrideC_G1, StrideC_M, StrideC_O));

    std::cout << "a_g_m_k: " << a_g_m_k.mDesc << std::endl;
    std::cout << "b0_g_k_n: " << b0_g_k_n.mDesc << std::endl;
    std::cout << "b1_g_n_o: " << b1_g_n_o.mDesc << std::endl;
    std::cout << "c_g0_g1_m_o: " << c_g0_g1_m_o_host_result.mDesc << std::endl;

    switch(init_method)
    {
    case 0: break;
    case 1:
        a_g_m_k.GenerateTensorValue(GeneratorTensor_2<ADataType>{-5, 5});
        b0_g_k_n.GenerateTensorValue(GeneratorTensor_2<B0DataType>{-5, 5});
        b1_g_n_o.GenerateTensorValue(GeneratorTensor_2<B1DataType>{-5, 5});
        break;
    case 2:
        a_g_m_k.GenerateTensorValue(GeneratorTensor_3<ADataType>{0.0, 1.0});
        b0_g_k_n.GenerateTensorValue(GeneratorTensor_3<B0DataType>{0.0, 1.0});
        b1_g_n_o.GenerateTensorValue(GeneratorTensor_3<B1DataType>{-0.5, 0.5});
        break;
    case 3:
        a_g_m_k.GenerateTensorValue(GeneratorTensor_2<ADataType>{-2, 2});
        b0_g_k_n.GenerateTensorValue(GeneratorTensor_Diagonal<B0DataType>{});
        b1_g_n_o.GenerateTensorValue(GeneratorTensor_Diagonal<B1DataType>{});
        break;
    default:
        a_g_m_k.GenerateTensorValue(GeneratorTensor_1<ADataType>{1});
        b0_g_k_n.GenerateTensorValue(GeneratorTensor_Sequential<1>{});
        b1_g_n_o.GenerateTensorValue(GeneratorTensor_Diagonal<B1DataType>{});
    }

    DeviceMem a_g_m_k_device_buf(sizeof(ADataType) * a_g_m_k.mDesc.GetElementSize());
    DeviceMem b0_g_k_n_device_buf(sizeof(B0DataType) * b0_g_k_n.mDesc.GetElementSize());
    DeviceMem b1_g_n_o_device_buf(sizeof(B1DataType) * b1_g_n_o.mDesc.GetElementSize());
    DeviceMem c_g0_g1_m_o_device_buf(sizeof(CDataType) *
                                     c_g0_g1_m_o_device_result.mDesc.GetElementSize());

    a_g_m_k_device_buf.ToDevice(a_g_m_k.mData.data());
    b0_g_k_n_device_buf.ToDevice(b0_g_k_n.mData.data());
    b1_g_n_o_device_buf.ToDevice(b1_g_n_o.mData.data());

    auto a_element_op    = AElementOp{};
    auto b0_element_op   = B0ElementOp{};
    auto acc0_element_op = Acc0ElementOp{};
    auto b1_element_op   = B1ElementOp{};
    auto c_element_op    = CElementOp{};

    // do GEMM
    auto gemm    = DeviceGemmInstance{};
    auto invoker = gemm.MakeInvoker();
    auto argument =
        gemm.MakeArgument(static_cast<ADataType*>(a_g_m_k_device_buf.GetDeviceBuffer()),
                          static_cast<B0DataType*>(b0_g_k_n_device_buf.GetDeviceBuffer()),
                          static_cast<B1DataType*>(b1_g_n_o_device_buf.GetDeviceBuffer()),
                          static_cast<CDataType*>(c_g0_g1_m_o_device_buf.GetDeviceBuffer()),
                          M,
                          N,
                          K,
                          O,
                          StrideA,
                          StrideB0,
                          StrideB1,
                          BatchStrideA,
                          BatchStrideB0,
                          BatchStrideB1,
                          c_permute_desc,
                          a_element_op,
                          b0_element_op,
                          acc0_element_op,
                          b1_element_op,
                          c_element_op);

    if(!gemm.IsSupportedArgument(argument))
    {
        std::cout << gemm.GetTypeString() << " does not support this problem" << std::endl;

        return 0;
    }

    float ave_time = invoker.Run(argument, StreamConfig{nullptr, time_kernel});

    std::size_t flop      = (size_t(M) * N * K * 2 + size_t(M) * N * O * 2) * BatchCount;
    std::size_t num_btype = (sizeof(ADataType) * M * K + sizeof(B0DataType) * K * N +
                             sizeof(B1DataType) * N * O + sizeof(CDataType) * M * O) *
                            BatchCount;

    float tflops = static_cast<float>(flop) / 1.E9 / ave_time;

    float gb_per_sec = num_btype / 1.E6 / ave_time;

    std::cout << "Perf: " << ave_time << " ms, " << tflops << " TFlops, " << gb_per_sec << " GB/s, "
              << gemm.GetTypeString() << std::endl;

    c_g0_g1_m_o_device_buf.FromDevice(c_g0_g1_m_o_device_result.mData.data());

    if(do_verification)
    {
        // Output of Gemm0 is input A of Gemm1
        Tensor<AccDataType> acc0_g_m_n(f_host_tensor_descriptor(BatchCount, M, N, N, M * N, Row{}));

        Tensor<ADataType> a1_g_m_n(f_host_tensor_descriptor(BatchCount, M, N, N, M * N, Row{}));

        Tensor<CDataType> c_g_m_o_host_result(
            f_host_tensor_descriptor(BatchCount, M, O, O, M * O, Row{}));

        auto ref_gemm0          = ReferenceGemm0Instance{};
        auto ref_gemm0_invoker  = ref_gemm0.MakeInvoker();
        auto ref_gemm0_argument = ref_gemm0.MakeArgument(
            a_g_m_k, b0_g_k_n, acc0_g_m_n, a_element_op, b0_element_op, PassThrough{});

        ref_gemm0_invoker.Run(ref_gemm0_argument);

        auto ref_softmax          = ReferenceSoftmaxInstance{};
        auto ref_softmax_invoker  = ref_softmax.MakeInvoker();
        auto ref_softmax_argument = ref_softmax.MakeArgument(acc0_g_m_n, a1_g_m_n, 1, 0, {2});

        ref_softmax_invoker.Run(ref_softmax_argument);

        auto ref_gemm1          = ReferenceGemm1Instance{};
        auto ref_gemm1_invoker  = ref_gemm1.MakeInvoker();
        auto ref_gemm1_argument = ref_gemm1.MakeArgument(
            a1_g_m_n, b1_g_n_o, c_g_m_o_host_result, PassThrough{}, b1_element_op, c_element_op);

        ref_gemm1_invoker.Run(ref_gemm1_argument);

        for(int g0 = 0; g0 < G0; g0++)
        {
            for(int g1 = 0; g1 < G1; g1++)
            {
                for(int m = 0; m < M; m++)
                {
                    for(int o = 0; o < O; o++)
                    {
                        int g = g0 * G1 + g1;

                        c_g0_g1_m_o_host_result(g0, g1, m, o) = c_g_m_o_host_result(g, m, o);
                    }
                }
            }
        }

        return ck::utils::check_err(c_g0_g1_m_o_device_result.mData, c_g0_g1_m_o_host_result.mData)
                   ? 0
                   : 1;
    }

    return 0;
}
