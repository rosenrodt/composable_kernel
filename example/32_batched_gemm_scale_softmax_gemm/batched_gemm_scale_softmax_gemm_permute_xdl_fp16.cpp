// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

/*
Gemm + Softmax + Gemm fused operation. Computes C_g_m_o = Softmax(A_g_m_k * B0_g_k_n) * B1_g_n_o
                                                                  |-----------------|
                                                                          Gemm0
                                                          |-------------------------------------|
                                                                          Gemm1
*/

#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/tensor_specialization.hpp"
#include "ck/tensor_operation/gpu/device/device_batched_gemm_softmax_gemm_permute_xdl_cshuffle.hpp"
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

// using ALayout  = Row;
// using B0Layout = Col;
// using B1Layout = Row;

// using CPermuteNumDims_G_M_O =
//     S<2, 1, 1>; // "using CLayout = Row" has been replaced by CPermuteNumDims_G_M_O

static constexpr ck::index_t NumDimG = 2;
static constexpr ck::index_t NumDimM = 1;
static constexpr ck::index_t NumDimN = 1;
static constexpr ck::index_t NumDimK = 1;
static constexpr ck::index_t NumDimO = 1;

using AElementOp    = PassThrough;
using B0ElementOp   = PassThrough;
using Acc0ElementOp = ck::tensor_operation::element_wise::Scale;
using B1ElementOp   = PassThrough;
using CElementOp    = PassThrough;

// build/bin/example_batched_gemm_scale_softmax_gemm_permute_xdl_fp16 1 4 1 128 128 32 64 1 1 1 2>&1 | less
// MNKOPadding error, Default OK
//
// static constexpr auto GemmSpec = ck::tensor_operation::device::GemmSpecialization::MNKOPadding;
static constexpr auto GemmSpec = ck::tensor_operation::device::GemmSpecialization::Default;

using DeviceGemmInstance =
    ck::tensor_operation::device::DeviceBatchedGemmSoftmaxGemmPermute_Xdl_CShuffle<
        NumDimG,
        NumDimM,
        NumDimN,
        NumDimK,
        NumDimO,
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
        GemmSpec,
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
        8,              // CShuffleBlockTransferScalarPerVector_NPerBlock
        false>;         // MaskOutUpperTriangle

// Ref Gemm0: fp16 in, fp32 out
using ReferenceGemm0Instance = ck::tensor_operation::host::ReferenceBatchedGemm<ADataType,
                                                                                B0DataType,
                                                                                AccDataType,
                                                                                AccDataType,
                                                                                AElementOp,
                                                                                B0ElementOp,
                                                                                Acc0ElementOp>;

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

#if 0
    // GEMM shape for A/B0/B1/C
    // C_g_m_o = A_g_m_k * B0_g_k_n * B1_g_n_o
    ck::index_t M = 120;
    ck::index_t N = 1000;
    ck::index_t K = 64;
    ck::index_t O = 128;
    float alpha   = 1;

    // Output shape C[G0, M, G1, O]. Batch dim, outer dim, inner dim must match GEMM shape
    // C_g0_g1_m_o = reshape(C_g_m_o, [g0, g1, m, o])
    // C_g0_m_g1_o = permute(C_g0_g1_m_o, [0, 2, 1, 3])
    ck::index_t G0 = 7;
    ck::index_t G1 = 13;
#else
    ck::index_t M = 128;
    ck::index_t N = 128;
    ck::index_t K = 32;
    ck::index_t O = 64;
    float alpha   = 1;
    ck::index_t G0 = 3;
    ck::index_t G1 = 7;
#endif

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
    else if(argc == 11)
    {
        do_verification = std::stoi(argv[1]);
        init_method     = std::stoi(argv[2]);
        time_kernel     = std::stoi(argv[3]);

        M  = std::stoi(argv[4]);
        N  = std::stoi(argv[5]);
        K  = std::stoi(argv[6]);
        O  = std::stoi(argv[7]);
        G0 = std::stoi(argv[8]);
        G1 = std::stoi(argv[9]);

        alpha = std::stof(argv[10]);
    }
    else
    {
        printf("arg1: verification (0=no, 1=yes)\n");
        printf("arg2: initialization (0=no init, 1=integer value, 2=decimal value)\n");
        printf("arg3: time kernel (0=no, 1=yes)\n");
        printf("arg4 to 11: M, N, K, O, G0, G1\n");
        printf("arg10: scale (alpha)\n");
        exit(0);
    }

#if 1
    // A layout [G0, M, G1, K]
    std::vector<ck::index_t> a_gs_ms_ks_lengths{G0, G1, M, K};
    std::vector<ck::index_t> a_gs_ms_ks_strides{M * G1 * K, K, G1 * K, 1};

    // B0 layout [G0, N, G1, K]
    std::vector<ck::index_t> b0_gs_ns_ks_lengths{G0, G1, N, K};
    std::vector<ck::index_t> b0_gs_ns_ks_strides{N * G1 * K, K, G1 * K, 1};

    // B1 layout [G0, N, G1, O]
    std::vector<ck::index_t> b1_gs_os_ns_lengths{G0, G1, O, N};
    std::vector<ck::index_t> b1_gs_os_ns_strides{N * G1 * O, O, 1, G1 * O};

    // C layout [G0, M, G1, O]
    std::vector<ck::index_t> c_gs_ms_os_lengths{G0, G1, M, O};
    std::vector<ck::index_t> c_gs_ms_os_strides{M * G1 * O, O, G1 * O, 1};
#else
    // A layout [G0, G1, M, K]
    std::vector<ck::index_t> a_gs_ms_ks_lengths{G0, G1, M, K};
    std::vector<ck::index_t> a_gs_ms_ks_strides{G1 * M * K, M * K, K, 1};

    // B0 layout [G0, G1, N, K]
    std::vector<ck::index_t> b0_gs_ns_ks_lengths{G0, G1, N, K};
    std::vector<ck::index_t> b0_gs_ns_ks_strides{G1 * N * K, N * K, K, 1};

    // B1 layout [G0, G1, N, O]
    std::vector<ck::index_t> b1_gs_os_ns_lengths{G0, G1, O, N};
    std::vector<ck::index_t> b1_gs_os_ns_strides{G1 * N * O, N * O, 1, O};

    // C layout [G0, G1, M, O]
    std::vector<ck::index_t> c_gs_ms_os_lengths{G0, G1, M, O};
    std::vector<ck::index_t> c_gs_ms_os_strides{G1 * M * O, M * O, O, 1};
#endif

    Tensor<ADataType> a_gs_ms_ks(a_gs_ms_ks_lengths, a_gs_ms_ks_strides);
    Tensor<B0DataType> b0_gs_ns_ks(b0_gs_ns_ks_lengths, b0_gs_ns_ks_strides);
    Tensor<B1DataType> b1_gs_os_ns(b1_gs_os_ns_lengths, b1_gs_os_ns_strides);
    Tensor<CDataType> c_gs_ms_os_host_result(c_gs_ms_os_lengths, c_gs_ms_os_strides);
    Tensor<CDataType> c_gs_ms_os_device_result(c_gs_ms_os_lengths, c_gs_ms_os_strides);

    std::cout << "a_gs_ms_ks: " << a_gs_ms_ks.mDesc << std::endl;
    std::cout << "b0_gs_ns_ks: " << b0_gs_ns_ks.mDesc << std::endl;
    std::cout << "b1_gs_os_ns: " << b1_gs_os_ns.mDesc << std::endl;
    std::cout << "c_gs_ms_os: " << c_gs_ms_os_host_result.mDesc << std::endl;

    switch(init_method)
    {
    case 0: break;
    case 1:
        a_gs_ms_ks.GenerateTensorValue(GeneratorTensor_2<ADataType>{-5, 5});
        b0_gs_ns_ks.GenerateTensorValue(GeneratorTensor_2<B0DataType>{-5, 5});
        b1_gs_os_ns.GenerateTensorValue(GeneratorTensor_2<B1DataType>{-5, 5});
        break;
    case 2:
        a_gs_ms_ks.GenerateTensorValue(GeneratorTensor_3<ADataType>{0.0, 1.0});
        b0_gs_ns_ks.GenerateTensorValue(GeneratorTensor_3<B0DataType>{0.0, 1.0});
        b1_gs_os_ns.GenerateTensorValue(GeneratorTensor_3<B1DataType>{-0.5, 0.5});
        break;
    case 3:
        a_gs_ms_ks.GenerateTensorValue(GeneratorTensor_2<ADataType>{-2, 2});
        b0_gs_ns_ks.GenerateTensorValue(GeneratorTensor_Diagonal<B0DataType>{});
        b1_gs_os_ns.GenerateTensorValue(GeneratorTensor_Diagonal<B1DataType>{});
        break;
    default:
        a_gs_ms_ks.GenerateTensorValue(GeneratorTensor_Sequential<2>{});
        b0_gs_ns_ks.GenerateTensorValue(GeneratorTensor_Diagonal<B0DataType>{});
        b1_gs_os_ns.GenerateTensorValue(GeneratorTensor_Diagonal<B1DataType>{});
    }

    LogRangeAsType<float>(std::cout << "a_gs_ms_ks.mData: " , std::vector<float>(a_gs_ms_ks.mData.begin(), a_gs_ms_ks.mData.begin() + 512), ",") << std::endl;
    LogRangeAsType<float>(std::cout << "b0_gs_ns_ks.mData: " , std::vector<float>(b0_gs_ns_ks.mData.begin(), b0_gs_ns_ks.mData.begin() + 512), ",") << std::endl;
    LogRangeAsType<float>(std::cout << "b1_gs_os_ns.mData: " , std::vector<float>(b1_gs_os_ns.mData.begin(), b1_gs_os_ns.mData.begin() + 512), ",") << std::endl;

    DeviceMem a_device_buf(sizeof(ADataType) * a_gs_ms_ks.mDesc.GetElementSpaceSize());
    DeviceMem b0_device_buf(sizeof(B0DataType) * b0_gs_ns_ks.mDesc.GetElementSpaceSize());
    DeviceMem b1_device_buf(sizeof(B1DataType) * b1_gs_os_ns.mDesc.GetElementSpaceSize());
    DeviceMem c_device_buf(sizeof(CDataType) *
                           c_gs_ms_os_device_result.mDesc.GetElementSpaceSize());

    a_device_buf.ToDevice(a_gs_ms_ks.mData.data());
    b0_device_buf.ToDevice(b0_gs_ns_ks.mData.data());
    b1_device_buf.ToDevice(b1_gs_os_ns.mData.data());

    auto a_element_op    = AElementOp{};
    auto b0_element_op   = B0ElementOp{};
    auto acc0_element_op = Acc0ElementOp{alpha};
    auto b1_element_op   = B1ElementOp{};
    auto c_element_op    = CElementOp{};

    // do GEMM
    auto gemm    = DeviceGemmInstance{};
    auto invoker = gemm.MakeInvoker();
    auto argument =
        gemm.MakeArgument(static_cast<ADataType*>(a_device_buf.GetDeviceBuffer()),
                          static_cast<B0DataType*>(b0_device_buf.GetDeviceBuffer()),
                          static_cast<B1DataType*>(b1_device_buf.GetDeviceBuffer()),
                          static_cast<CDataType*>(c_device_buf.GetDeviceBuffer()),
                          a_gs_ms_ks_lengths,
                          a_gs_ms_ks_strides,
                          b0_gs_ns_ks_lengths,
                          b0_gs_ns_ks_strides,
                          b1_gs_os_ns_lengths,
                          b1_gs_os_ns_strides,
                          c_gs_ms_os_lengths,
                          c_gs_ms_os_strides,
                          // TODO ANT: add bias
                          // std::array<std::vector<ck::index_t>, 1>{acc0_bias_gs_ms_ns_lengths},
                          // std::array<std::vector<ck::index_t>, 1>{acc0_bias_gs_ms_ns_strides},
                          // std::array<std::vector<ck::index_t>, 1>{acc1_bias_gs_ms_os_lengths},
                          // std::array<std::vector<ck::index_t>, 1>{acc1_bias_gs_ms_os_strides},
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

    ck::index_t BatchCount = G0 * G1;

    float ave_time = invoker.Run(argument, StreamConfig{nullptr, time_kernel});

    std::size_t flop      = (size_t(M) * N * K * 2 + size_t(M) * N * O * 2) * BatchCount;
    std::size_t num_btype = (sizeof(ADataType) * M * K + sizeof(B0DataType) * K * N +
                             sizeof(B1DataType) * N * O + sizeof(CDataType) * M * O) *
                            BatchCount;

    float tflops = static_cast<float>(flop) / 1.E9 / ave_time;

    float gb_per_sec = num_btype / 1.E6 / ave_time;

    std::cout << "Perf: " << ave_time << " ms, " << tflops << " TFlops, " << gb_per_sec << " GB/s, "
              << gemm.GetTypeString() << std::endl;

    if(do_verification)
    {
        c_device_buf.FromDevice(c_gs_ms_os_device_result.mData.data());

        Tensor<ADataType> a_g_m_k({BatchCount, M, K});
        Tensor<B0DataType> b0_g_k_n({BatchCount, K, N});
        Tensor<B1DataType> b1_g_n_o({BatchCount, N, O});
        Tensor<AccDataType> acc0_g_m_n({BatchCount, M, N});        // scratch object after gemm0
        Tensor<ADataType> a1_g_m_n({BatchCount, M, N});            // scratch object after softmax
        Tensor<CDataType> c_g_m_o_host_result({BatchCount, M, O}); // scratch object after gemm1

        std::cout << "a_g_m_k: " << a_g_m_k.mDesc << std::endl;
        std::cout << "b0_g_k_n: " << b0_g_k_n.mDesc << std::endl;
        std::cout << "b1_g_n_o: " << b1_g_n_o.mDesc << std::endl;
        std::cout << "c_g_m_o_host_result: " << c_g_m_o_host_result.mDesc << std::endl;

        // permute
        a_gs_ms_ks.ForEach([&](auto& self, auto idx) {
            a_g_m_k(idx[0] * G1 + idx[1], idx[2], idx[3]) = self(idx);
        });
        b0_gs_ns_ks.ForEach([&](auto& self, auto idx) {
            b0_g_k_n(idx[0] * G1 + idx[1], idx[3], idx[2]) = self(idx);
        });
        b1_gs_os_ns.ForEach([&](auto& self, auto idx) {
            b1_g_n_o(idx[0] * G1 + idx[1], idx[3], idx[2]) = self(idx);
        });

        auto ref_gemm0          = ReferenceGemm0Instance{};
        auto ref_gemm0_invoker  = ref_gemm0.MakeInvoker();
        auto ref_gemm0_argument = ref_gemm0.MakeArgument(
            a_g_m_k, b0_g_k_n, acc0_g_m_n, a_element_op, b0_element_op, acc0_element_op);

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

        // permute
        c_gs_ms_os_host_result.ForEach([&](auto& self, auto idx) {
            const size_t& g0 = idx[0];
            const size_t& g1 = idx[1];

            const size_t g = g0 * G1 + g1;

            self(idx) = c_g_m_o_host_result(g, idx[2], idx[3]);
        });

        return ck::utils::check_err(c_gs_ms_os_device_result.mData, c_gs_ms_os_host_result.mData)
                   ? 0
                   : 1;
    }

    return 0;
}
