// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.
/*
Backprop for Gemm + Softmax + Gemm fused operation, where forward prop is defined as:

  Y_g_m_o = Softmax(alpha * Q_g_m_k * K_g_k_n) * V_g_n_o

Input:

  Q, K, V, Y, dY, and per-row softmax stats computed beforehand during forward prop

Outputs:

  dQ, dK, dV

*/

#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/tensor_specialization.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_batched_gemm_softmax_gemm_permute_xdl_cshuffle.hpp"
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

using PassThrough = ck::tensor_operation::element_wise::PassThrough;
using Scale       = ck::tensor_operation::element_wise::Scale;

using DataType        = F16;
using AccDataType     = F32;
using ShuffleDataType = F32;

static constexpr ck::index_t NumDimG = 2;
static constexpr ck::index_t NumDimM = 1;
static constexpr ck::index_t NumDimN = 1;
static constexpr ck::index_t NumDimK = 1;
static constexpr ck::index_t NumDimO = 1;

#if 0
static constexpr auto GemmSpec = ck::tensor_operation::device::GemmSpecialization::MNKOPadding;
static constexpr auto MaskingSpec =
    ck::tensor_operation::device::MaskingSpecialization::MaskDisabled;

static constexpr auto TensorSpecQ = ck::tensor_operation::device::TensorSpecialization::Default;
static constexpr auto TensorSpecK = ck::tensor_operation::device::TensorSpecialization::Default;
static constexpr auto TensorSpecV = ck::tensor_operation::device::TensorSpecialization::Default;
static constexpr auto TensorSpecY = ck::tensor_operation::device::TensorSpecialization::Default;
#endif

// Ref Gemm0: S = alpha * Q * K^T
// fp16 in, fp32 out
using ReferenceGemm0Instance = ck::tensor_operation::host::ReferenceBatchedGemm<DataType,
                                                                                DataType,
                                                                                AccDataType,
                                                                                AccDataType,
                                                                                PassThrough,
                                                                                PassThrough,
                                                                                Scale>;

// Ref Softmax: P = Softmax(S)
// fp32 in, fp16 out
using ReferenceSoftmaxInstance =
    ck::tensor_operation::host::ReferenceSoftmax<AccDataType, DataType, AccDataType>;

// Ref Gemm1: Y = P * V
// fp16 in, fp16 out
using ReferenceGemm1Instance = ck::tensor_operation::host::ReferenceBatchedGemm<DataType,
                                                                                DataType,
                                                                                DataType,
                                                                                AccDataType,
                                                                                PassThrough,
                                                                                PassThrough,
                                                                                PassThrough>;

// Ref Gemm for backward pass
// fp16 in, fp16 out
using ReferenceGemmGradInstance = ck::tensor_operation::host::ReferenceBatchedGemm<DataType,
                                                                                   DataType,
                                                                                   DataType,
                                                                                   AccDataType,
                                                                                   PassThrough,
                                                                                   PassThrough,
                                                                                   Scale>;
#if 0
// Ref Gemm dP: dP = dY * V^T
// fp16 in, fp16 out
using ReferenceGemmPGradInstance = ck::tensor_operation::host::ReferenceBatchedGemm<DataType,
                                                                                    DataType,
                                                                                    DataType,
                                                                                    AccDataType,
                                                                                    PassThrough,
                                                                                    PassThrough,
                                                                                    PassThrough>;

// Ref Gemm dQ: dQ = alpha * dS * K
// fp16 in, fp16 out
using ReferenceGemmQGradInstance = ck::tensor_operation::host::ReferenceBatchedGemm<DataType,
                                                                                    DataType,
                                                                                    DataType,
                                                                                    AccDataType,
                                                                                    PassThrough,
                                                                                    PassThrough,
                                                                                    Scale>;

// Ref Gemm dK: dK = alpha * dS^T * Q
// fp16 in, fp16 out
using ReferenceGemmKGradInstance = ck::tensor_operation::host::ReferenceBatchedGemm<DataType,
                                                                                    DataType,
                                                                                    DataType,
                                                                                    AccDataType,
                                                                                    PassThrough,
                                                                                    PassThrough,
                                                                                    Scale>;
#endif

int main(int argc, char* argv[]) { return run(argc, argv); }

int run(int argc, char* argv[])
{
    bool do_verification = true;
    int init_method      = 1;
    bool time_kernel     = false;

    // Overall QKV matrices shape
    // Y_g_m_o = Softmax(Q_g_m_k * K_g_k_n) * V_g_n_o
    // Y_g0_g1_m_o = reshape(Y_g_m_o, [G0, G1, M, O])
    // Y_g0_m_g1_o = permute(Y_g0_g1_m_o, [0, 2, 1, 3])
    ck::index_t M  = 128;
    ck::index_t N  = 128;
    ck::index_t K  = 128;
    ck::index_t O  = 128;
    ck::index_t G0 = 1;
    ck::index_t G1 = 1;

    float alpha = 1;

    bool input_permute  = false;
    bool output_permute = true;

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
    else if(argc == 13)
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

        input_permute  = std::stoi(argv[11]);
        output_permute = std::stoi(argv[12]);
    }
    else
    {
        printf("arg1: verification (0=no, 1=yes)\n");
        printf("arg2: initialization (0=no init, 1=integer value, 2=decimal value)\n");
        printf("arg3: time kernel (0=no, 1=yes)\n");
        printf("arg4 to 11: M, N, K, O, G0, G1\n");
        printf("arg10: scale (alpha)\n");
        printf("arg11 to 12: input / output permute\n");
        exit(0);
    }

    std::vector<ck::index_t> q_gs_ms_ks_lengths{G0, G1, M, K};
    std::vector<ck::index_t> q_gs_ms_ks_strides =
        input_permute
            ? std::vector<ck::index_t>{M * G1 * K, K, G1 * K, 1} // Q layout [G0, M, G1, K]
            : std::vector<ck::index_t>{G1 * M * K, M * K, K, 1}; // Q layout [G0, G1, M, K]

    std::vector<ck::index_t> k_gs_ns_ks_lengths{G0, G1, N, K};
    std::vector<ck::index_t> k_gs_ns_ks_strides =
        input_permute
            ? std::vector<ck::index_t>{N * G1 * K, K, G1 * K, 1} // K layout [G0, N, G1, K]
            : std::vector<ck::index_t>{G1 * N * K, N * K, K, 1}; // K layout [G0, G1, N, K]

    std::vector<ck::index_t> v_gs_os_ns_lengths{G0, G1, O, N};
    std::vector<ck::index_t> v_gs_os_ns_strides =
        input_permute
            ? std::vector<ck::index_t>{N * G1 * O, O, 1, G1 * O} // V layout [G0, N, G1, O]
            : std::vector<ck::index_t>{G1 * N * O, N * O, 1, O}; // V layout [G0, G1, N, O]

    std::vector<ck::index_t> y_gs_ms_os_lengths{G0, G1, M, O};
    std::vector<ck::index_t> y_gs_ms_os_strides =
        output_permute
            ? std::vector<ck::index_t>{M * G1 * O, O, G1 * O, 1} // Y layout [G0, M, G1, O]
            : std::vector<ck::index_t>{G1 * M * O, M * O, O, 1}; // Y layout [G0, G1, M, O]

    Tensor<DataType> q_gs_ms_ks(q_gs_ms_ks_lengths, q_gs_ms_ks_strides);
    Tensor<DataType> k_gs_ns_ks(k_gs_ns_ks_lengths, k_gs_ns_ks_strides);
    Tensor<DataType> v_gs_os_ns(v_gs_os_ns_lengths, v_gs_os_ns_strides);
    // Tensor<DataType> y_gs_ms_os_device_result(y_gs_ms_os_lengths, y_gs_ms_os_strides);

    Tensor<DataType> qgrad_gs_ms_ks(q_gs_ms_ks_lengths, q_gs_ms_ks_strides);
    Tensor<DataType> kgrad_gs_ns_ks(k_gs_ns_ks_lengths, k_gs_ns_ks_strides);
    Tensor<DataType> vgrad_gs_os_ns(v_gs_os_ns_lengths, v_gs_os_ns_strides);
    Tensor<DataType> ygrad_gs_ms_os(y_gs_ms_os_lengths, y_gs_ms_os_strides);

    // Tensor<DataType> qgrad_gs_ms_ks_device(q_gs_ms_ks_lengths, q_gs_ms_ks_strides);
    // Tensor<DataType> kgrad_gs_ns_ks_device(k_gs_ns_ks_lengths, k_gs_ns_ks_strides);
    // Tensor<DataType> vgrad_gs_os_ns_device(v_gs_os_ns_lengths, v_gs_os_ns_strides);

    std::cout << "q_gs_ms_ks: " << q_gs_ms_ks.mDesc << std::endl;
    std::cout << "k_gs_ns_ks: " << k_gs_ns_ks.mDesc << std::endl;
    std::cout << "v_gs_os_ns: " << v_gs_os_ns.mDesc << std::endl;

    switch(init_method)
    {
    case 0: break;
    case 1:
        q_gs_ms_ks.GenerateTensorValue(GeneratorTensor_2<DataType>{-2, 2});
        k_gs_ns_ks.GenerateTensorValue(GeneratorTensor_2<DataType>{-2, 2});
        v_gs_os_ns.GenerateTensorValue(GeneratorTensor_2<DataType>{-2, 2});
        ygrad_gs_ms_os.GenerateTensorValue(GeneratorTensor_2<DataType>{-2, 2});
        break;
    case 2:
        q_gs_ms_ks.GenerateTensorValue(GeneratorTensor_3<DataType>{0.0, 1.0});
        k_gs_ns_ks.GenerateTensorValue(GeneratorTensor_3<DataType>{0.0, 1.0});
        v_gs_os_ns.GenerateTensorValue(GeneratorTensor_3<DataType>{-0.5, 0.5});
        ygrad_gs_ms_os.GenerateTensorValue(GeneratorTensor_3<DataType>{-0.5, 0.5});
        break;
    case 3:
        q_gs_ms_ks.GenerateTensorValue(GeneratorTensor_2<DataType>{-2, 2});
        k_gs_ns_ks.GenerateTensorValue(GeneratorTensor_Diagonal<DataType>{});
        v_gs_os_ns.GenerateTensorValue(GeneratorTensor_Diagonal<DataType>{});
        ygrad_gs_ms_os.GenerateTensorValue(GeneratorTensor_Diagonal<DataType>{});
        break;
    default:
        q_gs_ms_ks.GenerateTensorValue(GeneratorTensor_Sequential<2>{});
        k_gs_ns_ks.GenerateTensorValue(GeneratorTensor_Diagonal<DataType>{});
        v_gs_os_ns.GenerateTensorValue(GeneratorTensor_Diagonal<DataType>{});
        ygrad_gs_ms_os.GenerateTensorValue(GeneratorTensor_Diagonal<DataType>{});
    }

#if 0
    DeviceMem q_device_buf(sizeof(DataType) * q_gs_ms_ks.mDesc.GetElementSpaceSize());
    DeviceMem k_device_buf(sizeof(DataType) * k_gs_ns_ks.mDesc.GetElementSpaceSize());
    DeviceMem v_device_buf(sizeof(DataType) * v_gs_os_ns.mDesc.GetElementSpaceSize());

    q_device_buf.ToDevice(q_gs_ms_ks.mData.data());
    k_device_buf.ToDevice(k_gs_ns_ks.mData.data());
    v_device_buf.ToDevice(v_gs_os_ns.mData.data());
#endif

    // TODO ANT: attention backward kernel
#if 0
    auto gemm     = DeviceGemmInstance{};
    auto invoker  = gemm.MakeInvoker();
    auto argument = gemm.MakeArgument(
        static_cast<DataType*>(q_device_buf.GetDeviceBuffer()),
        static_cast<DataType*>(k_device_buf.GetDeviceBuffer()),
        static_cast<DataType*>(v_device_buf.GetDeviceBuffer()),
        static_cast<DataType*>(y_device_buf.GetDeviceBuffer()),
        {}, // std::array<void*, 1> p_acc0_biases;
        {}, // std::array<void*, 1> p_acc1_biases;
        q_gs_ms_ks_lengths,
        q_gs_ms_ks_strides,
        k_gs_ns_ks_lengths,
        k_gs_ns_ks_strides,
        v_gs_os_ns_lengths,
        v_gs_os_ns_strides,
        y_gs_ms_os_lengths,
        y_gs_ms_os_strides,
        {}, // std::array<std::vector<ck::index_t>, 1>{acc0_biases_gs_ms_ns_lengths},
        {}, // std::array<std::vector<ck::index_t>, 1>{acc0_biases_gs_ms_ns_strides},
        {}, // std::array<std::vector<ck::index_t>, 1>{acc1_biases_gs_ms_os_lengths},
        {}, // std::array<std::vector<ck::index_t>, 1>{acc1_biases_gs_ms_os_strides},
        q_element_op,
        k_element_op,
        s_element_op,
        v_element_op,
        y_element_op);

    if(!gemm.IsSupportedArgument(argument))
    {
        std::cout << gemm.GetTypeString() << " does not support this problem" << std::endl;

        return 0;
    }

    ck::index_t BatchCount = G0 * G1;

    float ave_time = invoker.Run(argument, StreamConfig{nullptr, time_kernel});

    std::size_t flop      = (size_t(M) * N * K * 2 + size_t(M) * N * O * 2) * BatchCount;
    std::size_t num_btype = (sizeof(DataType) * M * K + sizeof(DataType) * K * N +
                             sizeof(DataType) * N * O + sizeof(DataType) * M * O) *
                            BatchCount;

    float tflops = static_cast<float>(flop) / 1.E9 / ave_time;

    float gb_per_sec = num_btype / 1.E6 / ave_time;

    std::cout << "Perf: " << ave_time << " ms, " << tflops << " TFlops, " << gb_per_sec << " GB/s, "
              << gemm.GetTypeString() << std::endl;
#endif

    if(do_verification)
    {
        Tensor<DataType> q_g_m_k({BatchCount, M, K});
        Tensor<DataType> k_g_k_n({BatchCount, K, N});
        Tensor<DataType> v_g_n_o({BatchCount, N, O});
        Tensor<AccDataType> s_g_m_n({BatchCount, M, N}); // scratch object after gemm0
        Tensor<DataType> p_g_m_n({BatchCount, M, N});    // scratch object after softmax
        Tensor<DataType> y_g_m_o({BatchCount, M, O});    // scratch object after gemm1

        Tensor<DataType> qgrad_g_m_k({BatchCount, M, K});
        Tensor<DataType> kgrad_g_k_n({BatchCount, K, N});
        Tensor<DataType> vgrad_g_n_o({BatchCount, N, O});
        Tensor<DataType> sgrad_g_m_n({BatchCount, M, N}); // scratch object in bwd pass
        Tensor<DataType> pgrad_g_m_n({BatchCount, M, N}); // scratch object in bwd pass
        Tensor<DataType> ygrad_g_m_n({BatchCount, M, O});

        // permute
        q_gs_ms_ks.ForEach([&](auto& self, auto idx) {
            q_g_m_k(idx[0] * G1 + idx[1], idx[2], idx[3]) = self(idx);
        });
        k_gs_ns_ks.ForEach([&](auto& self, auto idx) {
            k_g_k_n(idx[0] * G1 + idx[1], idx[3], idx[2]) = self(idx);
        });
        v_gs_os_ns.ForEach([&](auto& self, auto idx) {
            v_g_n_o(idx[0] * G1 + idx[1], idx[3], idx[2]) = self(idx);
        });

        // S = alpha * Q * K^T
        auto ref_gemm0          = ReferenceGemm0Instance{};
        auto ref_gemm0_invoker  = ref_gemm0.MakeInvoker();
        auto ref_gemm0_argument = ref_gemm0.MakeArgument(
            q_g_m_k, k_g_k_n, s_g_m_n, PassThrough{}, PassThrough{}, Scale{alpha});

        ref_gemm0_invoker.Run(ref_gemm0_argument);

        // masking
#if 0
        const auto mask = DeviceGemmInstance::C0MatrixMask(N);
        s_g_m_n.ForEach([&](auto& self, auto idx) {
            if(mask.IsMaskedElement(idx[1], idx[2]))
                self(idx) = -ck::NumericLimits<float>::Infinity();
        });
#endif

        // P = Softmax(S)
        auto ref_softmax          = ReferenceSoftmaxInstance{};
        auto ref_softmax_invoker  = ref_softmax.MakeInvoker();
        auto ref_softmax_argument = ref_softmax.MakeArgument(s_g_m_n, p_g_m_n, 1, 0, {2});

        ref_softmax_invoker.Run(ref_softmax_argument);

        // Y = P * V
        auto ref_gemm1          = ReferenceGemm1Instance{};
        auto ref_gemm1_invoker  = ref_gemm1.MakeInvoker();
        auto ref_gemm1_argument = ref_gemm1.MakeArgument(
            p_g_m_n, v_g_n_o, y_g_m_o, PassThrough{}, PassThrough{}, PassThrough{});

        ref_gemm1_invoker.Run(ref_gemm1_argument);

        // Gradients
        auto ref_gemm_grad         = ReferenceGemmGradInstance{};
        auto ref_gemm_grad_invoker = ref_gemm_grad.MakeInvoker();

        {
            auto ref_gemm_grad_argument = ref_gemm_grad.MakeArgument(y_grad, );
        }
        // dP = dY * V^T
        // dS_i = P_i .* (dP_i - P_i^T * dP_i)
        // dV = P^T * dY
        // dQ = alpha * dS * K
        // dK = alpha * dS^T * Q

        // permute
        // y_gs_ms_os.ForEach([&](auto& self, auto idx) {
        //     const size_t& g0 = idx[0];
        //     const size_t& g1 = idx[1];

        //     const size_t g = g0 * G1 + g1;

        //     self(idx) = y_g_m_o(g, idx[2], idx[3]);
        // });

        // return ck::utils::check_err(y_gs_ms_os_device_result.mData, y_gs_ms_os.mData)
        //            ? 0
        //            : 1;
    }

    return 0;
}
