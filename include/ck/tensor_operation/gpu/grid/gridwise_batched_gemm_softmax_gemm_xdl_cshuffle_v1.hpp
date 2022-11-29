// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/multi_index_transform_helper.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_operation/gpu/grid/block_to_ctile_map.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_gemm_pipeline_selector.hpp"
#include "ck/tensor_operation/gpu/block/blockwise_gemm_xdlops.hpp"
#include "ck/tensor_operation/gpu/block/thread_group_tensor_slice_transfer_v4r1.hpp"
#include "ck/tensor_operation/gpu/block/thread_group_tensor_slice_transfer_v6r1.hpp"
#include "ck/tensor_operation/gpu/thread/threadwise_tensor_slice_transfer.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/tensor_operation/gpu/block/blockwise_softmax.hpp"

namespace ck {

template <typename DataType,
          typename AccDataType,
          typename ShuffleDataType,
          typename QTensorElementwiseOperation,
          typename KTensorElementwiseOperation,
          typename STensorElementwiseOperation,
          typename VTensorElementwiseOperation,
          typename YTensorElementwiseOperation,
          InMemoryDataOperationEnum CGlobalMemoryDataOperation, // TODO ANT: remove
          typename QTensorGridDesc_K0_M_K1,
          typename KTensorGridDesc_K0_N_K1,
          typename VTensorGridDesc_N0_O_N1,
          typename YTensorGridDesc_M_O,
          index_t NumGemmKPrefetchStage,
          index_t BlockSize,
          index_t Gemm0MPerBlock,
          index_t Gemm0NPerBlock,
          index_t Gemm0KPerBlock,
          index_t Gemm1OPerBlock,
          index_t Gemm1NPerBlock,
          index_t AK1Value,
          index_t BK1Value,
          index_t BN1Value,
          index_t MPerXdl,
          index_t NPerXdl,
          index_t MXdlPerWave,
          index_t NXdlPerWave,
          index_t OXdlPerWave,
          typename QTensorBlockTransferThreadClusterLengths_AK0_M_AK1,
          typename QTensorBlockTransferThreadClusterArrangeOrder,
          typename QTensorBlockTransferSrcAccessOrder,
          index_t QTensorBlockTransferSrcVectorDim,
          index_t QTensorBlockTransferSrcScalarPerVector,
          index_t QTensorBlockTransferDstScalarPerVector_AK1,
          bool QTensorThreadTransferSrcResetCoordinateAfterRun, // ignored
          index_t ABlockLdsExtraM,
          typename KTensorBlockTransferThreadClusterLengths_BK0_N_BK1,
          typename KTensorBlockTransferThreadClusterArrangeOrder,
          typename KTensorBlockTransferSrcAccessOrder,
          index_t KTensorBlockTransferSrcVectorDim,
          index_t KTensorBlockTransferSrcScalarPerVector,
          index_t KTensorBlockTransferDstScalarPerVector_BK1,
          bool KTensorThreadTransferSrcResetCoordinateAfterRun, // ignored
          index_t KTensorBlockLdsExtraN,
          typename VTensorBlockTransferThreadClusterLengths_BK0_N_BK1,
          typename VTensorBlockTransferThreadClusterArrangeOrder,
          typename VTensorBlockTransferSrcAccessOrder,
          index_t VTensorBlockTransferSrcVectorDim,
          index_t VTensorBlockTransferSrcScalarPerVector,
          index_t VTensorBlockTransferDstScalarPerVector_BK1,
          bool VTensorThreadTransferSrcResetCoordinateAfterRun,
          index_t VTensorBlockLdsExtraN,
          index_t YShuffleMXdlPerWavePerShuffle, // TODO ANT: remove
          index_t YShuffleNXdlPerWavePerShuffle, // TODO ANT: remove
          typename YShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock, // TODO ANT: remove
          index_t YShuffleBlockTransferScalarPerVector_NPerBlock, // TODO ANT: remove
          LoopScheduler LoopSched,
          bool PadN,
          bool MaskOutUpperTriangle,
          PipelineVersion PipelineVer = PipelineVersion::v1>
struct GridwiseBatchedGemmSoftmaxGemm_Xdl_CShuffle // TODO ANT: rename backward attention
{
    static_assert(LoopSched == LoopScheduler::Default,
                  "Non-default loop scheduler is currently not supported");

    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};
    static constexpr auto I3 = Number<3>{};
    static constexpr auto I4 = Number<4>{};
    static constexpr auto I5 = Number<5>{};
    static constexpr auto I6 = Number<6>{};
    static constexpr auto I7 = Number<7>{};

    // K1 should be Number<...>
    // Gemm0
    static constexpr auto AK0 = Number<Gemm0KPerBlock / AK1Value>{};
    static constexpr auto BK0 = Number<Gemm0KPerBlock / BK1Value>{};
    static constexpr auto AK1 = Number<AK1Value>{};
    static constexpr auto BK1 = Number<BK1Value>{};

    static constexpr auto Gemm0MWaves = Gemm0MPerBlock / (MPerXdl * MXdlPerWave);
    static constexpr auto Gemm0NWaves = Gemm0NPerBlock / (NPerXdl * NXdlPerWave);

    // Gemm1
    static constexpr auto B1K0 = Number<Gemm1NPerBlock / BN1Value>{};
    static constexpr auto B1K1 = Number<BN1Value>{};

    // VGrad Gemm
    template <index_t Sum_M_ = MPerXdl * 2>
    struct VGradGemmTile_N_O_M_
    {
        static constexpr index_t Free0_N = Gemm0NPerBlock;
        static constexpr index_t Free1_O = Gemm1OPerBlock;
        static constexpr index_t Sum_M   = Sum_M_;

        static constexpr index_t P_M1     = 8; // P will be row-major
        static constexpr index_t P_M0     = Sum_M / P_M1;
        static constexpr index_t P_LdsPad = 0; // how many multiples of M1 per N * M1 elements

        static constexpr index_t YGrad_M1     = 2; // dY assumed row-major, typically =2 for fp16
        static constexpr index_t YGrad_M0     = Sum_M / YGrad_M1;
        static constexpr index_t YGrad_LdsPad = 0; // how many multiples of M1 per N * M1 elements

        static_assert(Sum_M % MPerXdl == 0, "");

        static constexpr index_t YGrad_SrcVectorDim       = 1; // Free1_O dimension
        static constexpr index_t YGrad_SrcScalarPerVector = 4;

        static constexpr index_t GemmNWave   = 2;
        static constexpr index_t GemmOWave   = BlockSize / get_warp_size() / GemmNWave;
        static constexpr index_t GemmNRepeat = Free0_N / GemmNWave / MPerXdl;
        static constexpr index_t GemmORepeat = Free1_O / GemmOWave / NPerXdl;
        static constexpr index_t GemmMPack =
            math::max(math::lcm(P_M1, YGrad_M1),
                      MfmaSelector<DataType, MPerXdl, NPerXdl>::selected_mfma.k_per_blk);

        using YGrad_BlockSliceLengths = Sequence<YGrad_M0, Free1_O, YGrad_M1>;
        using YGrad_ThreadClusterLengths =
            Sequence<BlockSize / (Free1_O / YGrad_SrcScalarPerVector),
                     Free1_O / YGrad_SrcScalarPerVector,
                     1>;
        using YGrad_ThreadClusterArrangeOrder = Sequence<0, 2, 1>;

        __host__ __device__ static constexpr auto GetPBlockDescriptor_M0_N_M1()
        {
            constexpr index_t P_M0 = Sum_M / P_M1;
            return make_naive_tensor_descriptor(
                make_tuple(Number<P_M0>{}, Number<Free0_N>{}, Number<P_M1>{}),
                make_tuple(Number<Free0_N + P_LdsPad>{} * Number<P_M1>{}, Number<P_M1>{}, I1));
        }
        __host__ __device__ static constexpr auto GetYGradBlockDescriptor_M0_O_M1()
        {
            constexpr index_t YGrad_M0 = Sum_M / YGrad_M1;
            return make_naive_tensor_descriptor(
                make_tuple(Number<YGrad_M0>{}, Number<Free1_O>{}, Number<YGrad_M1>{}),
                make_tuple(
                    Number<Free1_O + YGrad_LdsPad>{} * Number<YGrad_M1>{}, Number<YGrad_M1>{}, I1));
        }

        __host__ __device__ static constexpr auto GetPBlockSliceLengths_M0_N0_M1_N1_M2_N2()
        {
            // perform manual unmerge: m -> m_repeat, m_waves, m_per_xdl
            constexpr index_t m  = Sum_M - 1;
            constexpr index_t m2 = m % MPerXdl;
            constexpr index_t m1 = m / MPerXdl % Gemm0MWaves;
            constexpr index_t m0 = m / MPerXdl / Gemm0MWaves % MXdlPerWave;

            // perform manual unmerge: n -> n_repeat, n_waves, n_per_xdl
            constexpr index_t n  = Free0_N - 1;
            constexpr index_t n2 = n % NPerXdl;
            constexpr index_t n1 = n / NPerXdl % Gemm0NWaves;
            constexpr index_t n0 = n / NPerXdl / Gemm0NWaves % MXdlPerWave;

            // assume 256 decomposed into 2 x 4 x 32
            // 1d idx ( 32 - 1) -> 3d idx 0, 0, 31 -> 3d dim 1 x 1 x 32
            // 1d idx (256 - 1) -> 3d idx 1, 3, 31 -> 3d dim 2 x 4 x 32
            return Sequence<m0, n0, m1, n1, m2, n2>{} + Sequence<1, 1, 1, 1, 1, 1>{};
        }

        // template <typename PBlockDesc_M0_N_M1>
        // __host__ __device__ static constexpr auto
        // MakePMmaTileDescriptor_N0_N1_N2_M(const PBlockDesc_M0_N_M1&)
        // {
        //     constexpr auto lengths = GetPBlockSliceLengths_M0_N0_M1_N1_M2_N2();

        //     return MakeGemmMmaTileDescriptor_MN0_MN1_MN2_K<lengths[I0], lengths[I2],
        //     lengths[I4]>(
        //         PBlockDesc_M0_N_M1{});
        // }

        // template <typename BBlockDesc_BK0_N_BK1>
        // __host__ __device__ static constexpr auto
        // MakeYGradMmaTileDescriptor_N0_N1_N2_K(const BBlockDesc_BK0_N_BK1&)
        // {
        //     constexpr index_t NWaves = Gemm0NPerBlock / (NXdlPerWave * NPerXdl);

        //     return MakeGemmMmaTileDescriptor_MN0_MN1_MN2_K<NXdlPerWave, NWaves, NPerXdl>(
        //         BBlockDesc_BK0_N_BK1{});
        // }
    };

    using VGradGemmTile_N_O_M = VGradGemmTile_N_O_M_<>; // tune later

    // QGrad Gemm
    // KGrad Gemm

    using ThisThreadBlock = ThisThreadBlock<BlockSize>;

    using GridwiseGemmPipe = remove_cvref_t<decltype(
        GridwiseGemmPipeline_Selector<PipelineVer, NumGemmKPrefetchStage>())>;

    template <typename ABlockDesc_AK0_M_AK1>
    __host__ __device__ static constexpr auto
    MakeGemm0AMmaTileDescriptor_M0_M1_M2_K(const ABlockDesc_AK0_M_AK1&)
    {
        constexpr index_t MWaves = Gemm0MPerBlock / (MXdlPerWave * MPerXdl);

        return MakeGemmMmaTileDescriptor_MN0_MN1_MN2_K<MXdlPerWave, MWaves, MPerXdl>(
            ABlockDesc_AK0_M_AK1{});
    }

    template <typename BBlockDesc_BK0_N_BK1>
    __host__ __device__ static constexpr auto
    MakeGemm0BMmaTileDescriptor_N0_N1_N2_K(const BBlockDesc_BK0_N_BK1&)
    {
        constexpr index_t NWaves = Gemm0NPerBlock / (NXdlPerWave * NPerXdl);

        return MakeGemmMmaTileDescriptor_MN0_MN1_MN2_K<NXdlPerWave, NWaves, NPerXdl>(
            BBlockDesc_BK0_N_BK1{});
    }

    template <typename ABlockDesc_AK0_M_AK1>
    __host__ __device__ static constexpr auto
    MakeGemm1AMmaTileDescriptor_M0_M1_M2_K(const ABlockDesc_AK0_M_AK1&)
    {
        return MakeGemmMmaTileDescriptor_MN0_MN1_MN2_K<MXdlPerWave, 1, 1>(ABlockDesc_AK0_M_AK1{});
    }

    template <typename BBlockDesc_BK0_N_BK1>
    __host__ __device__ static constexpr auto
    MakeGemm1BMmaTileDescriptor_N0_N1_N2_K(const BBlockDesc_BK0_N_BK1&)
    {
        constexpr index_t Gemm1NWaves = Gemm1OPerBlock / (OXdlPerWave * NPerXdl);
        return MakeGemmMmaTileDescriptor_MN0_MN1_MN2_K<OXdlPerWave, Gemm1NWaves, NPerXdl>(
            BBlockDesc_BK0_N_BK1{});
    }

    __host__ __device__ static constexpr auto GetABlockDescriptor_AK0PerBlock_MPerBlock_AK1()
    {
        // A matrix in LDS memory, dst of blockwise copy
        return make_naive_tensor_descriptor(
            make_tuple(AK0, Number<Gemm0MPerBlock>{}, AK1),
            make_tuple(Number<Gemm0MPerBlock + ABlockLdsExtraM>{} * AK1, AK1, I1));
    }

    __host__ __device__ static constexpr auto GetBBlockDescriptor_BK0PerBlock_NPerBlock_BK1()
    {
        // B matrix in LDS memory, dst of blockwise copy
        return make_naive_tensor_descriptor(
            make_tuple(BK0, Number<Gemm0NPerBlock>{}, BK1),
            make_tuple(Number<Gemm0NPerBlock + KTensorBlockLdsExtraN>{} * BK1, BK1, I1));
    }

    __host__ __device__ static constexpr auto GetB1BlockDescriptor_BK0PerBlock_NPerBlock_BK1()
    {
        // B1 matrix in LDS memory, dst of blockwise copy
        return make_naive_tensor_descriptor(
            make_tuple(B1K0, Number<Gemm1OPerBlock>{}, B1K1),
            make_tuple(Number<Gemm1OPerBlock + VTensorBlockLdsExtraN>{} * B1K1, B1K1, I1));
    }

    __host__ __device__ static constexpr auto
    GetYShuffleBlockDescriptor_MBlock_MPerBlock_NBlock_NPerBlock()
    {
        constexpr index_t MWave = Gemm0MPerBlock / (MXdlPerWave * MPerXdl);
        constexpr index_t NWave = Gemm1OPerBlock / (OXdlPerWave * NPerXdl);

        constexpr auto c_shuffle_block_desc_mblock_mperblock_nblock_nperblock =
            make_naive_tensor_descriptor_packed(
                make_tuple(I1,
                           Number<YShuffleMXdlPerWavePerShuffle * MWave * MPerXdl>{},
                           I1,
                           Number<YShuffleNXdlPerWavePerShuffle * NWave * NPerXdl>{}));

        return c_shuffle_block_desc_mblock_mperblock_nblock_nperblock;
    }

    __host__ __device__ static constexpr auto
    GetPBlockDescriptor_NBlock_NPerBlock_MBlock_MPerBlock()
    {
        constexpr auto ptrans_block_desc = make_naive_tensor_descriptor_packed(make_tuple(
            I1, Number<VGradGemmTile_N_O_M::Free0_N>{}, I1, Number<VGradGemmTile_N_O_M::Sum_M>{}));

        return ptrans_block_desc;
    }

    __host__ __device__ static constexpr index_t GetSharedMemoryNumberOfByte()
    {
        const index_t gemm0_bytes_end = (SharedMemTrait::a_block_space_size_aligned +
                                         SharedMemTrait::b_block_space_size_aligned) *
                                        sizeof(DataType);
        const index_t gemm1_bytes_end =
            (SharedMemTrait::b1_block_space_offset + SharedMemTrait::b1_block_space_size_aligned) *
            sizeof(DataType);
        const index_t softmax_bytes_end = (SharedMemTrait::reduction_space_offset +
                                           SharedMemTrait::reduction_space_size_aligned) *
                                          sizeof(AccDataType);
        const index_t c_block_bytes_end =
            SharedMemTrait::c_block_space_size * sizeof(ShuffleDataType);

        return math::max(gemm0_bytes_end, gemm1_bytes_end, softmax_bytes_end, c_block_bytes_end);
    }

    // block_id to matrix tile idx (m0, n0) mapping are controlled by {M01, N01}
    template <typename Block2CTileMap>
    __host__ __device__ static constexpr bool
    CheckValidity(const QTensorGridDesc_K0_M_K1& a_grid_desc_ak0_m_ak1,
                  const KTensorGridDesc_K0_N_K1& b_grid_desc_bk0_n_bk1,
                  const VTensorGridDesc_N0_O_N1& b1_grid_desc_bk0_n_bk1,
                  const YTensorGridDesc_M_O& c_grid_desc_m_n,
                  const Block2CTileMap& block_2_ctile_map)
    {
        static_assert((Gemm0MPerBlock % (MPerXdl * MXdlPerWave) == 0) &&
                          (Gemm0NPerBlock % (NXdlPerWave * NPerXdl)) == 0,
                      "Invalid tuning param!");

        const auto M = a_grid_desc_ak0_m_ak1.GetLength(I1);
        const auto N = b_grid_desc_bk0_n_bk1.GetLength(I1);
        const auto K = a_grid_desc_ak0_m_ak1.GetLength(I0) * a_grid_desc_ak0_m_ak1.GetLength(I2);
        const auto Gemm1N = b1_grid_desc_bk0_n_bk1.GetLength(I1);

        if(!(M == c_grid_desc_m_n.GetLength(I0) && Gemm1N == c_grid_desc_m_n.GetLength(I1)))
        {
            return false;
        }

        if(!(M % Gemm0MPerBlock == 0 && N % Gemm0NPerBlock == 0 && K % Gemm0KPerBlock == 0 &&
             Gemm1N % Gemm1OPerBlock == 0))
        {
            return false;
        }

        // check gemm0 gridwise gemm pipeline
        const auto num_gemm0_k_loop = K / Gemm0KPerBlock;
        if(!GridwiseGemmPipe::IsSupported(num_gemm0_k_loop))
        {
            return false;
        }

        // check gemm1 gridwise gemm pipeline
        if(!(Gemm0NPerBlock % Gemm1NPerBlock == 0))
        {
            return false;
        }

        const auto num_gemm1_k_inner_loop = Gemm0NPerBlock / Gemm1NPerBlock;
        if(!GridwiseGemmPipe::IsSupported(num_gemm1_k_inner_loop))
        {
            return false;
        }

        if(!block_2_ctile_map.CheckValidity(c_grid_desc_m_n))
        {
            return false;
        }

        // TODO: also check validity of all components (blockwise-copy, threadwise-copy, etc)
        return true;
    }

    __host__ __device__ static constexpr bool CalculateHasMainKBlockLoop(index_t K)
    {
        const index_t num_loop = K / Gemm0KPerBlock;

        return GridwiseGemmPipe::CalculateHasMainLoop(num_loop);
    }

    __host__ __device__ static constexpr auto
    MakeCGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(const YTensorGridDesc_M_O& c_grid_desc_m_n)
    {
        const auto M = c_grid_desc_m_n.GetLength(I0);
        const auto N = c_grid_desc_m_n.GetLength(I1);

        const auto MBlock = M / Gemm0MPerBlock;
        const auto NBlock = N / Gemm1OPerBlock;

        const auto c_grid_desc_mblock_mperblock_nblock_nperblock = transform_tensor_descriptor(
            c_grid_desc_m_n,
            make_tuple(make_unmerge_transform(make_tuple(MBlock, Number<Gemm0MPerBlock>{})),
                       make_unmerge_transform(make_tuple(NBlock, Number<Gemm1OPerBlock>{}))),
            make_tuple(Sequence<0>{}, Sequence<1>{}),
            make_tuple(Sequence<0, 1>{}, Sequence<2, 3>{}));

        return c_grid_desc_mblock_mperblock_nblock_nperblock;
    }

    // return block_id to C matrix tile idx (m0, n0) mapping
    __host__ __device__ static constexpr auto
    MakeDefaultBlock2CTileMap(const YTensorGridDesc_M_O& c_grid_desc_m_n)
    {
        return BlockToCTileMap_M00_N0_M01Adapt<Gemm0MPerBlock, Gemm1OPerBlock, YTensorGridDesc_M_O>(
            c_grid_desc_m_n);
    }

    using CGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock = remove_cvref_t<decltype(
        MakeCGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(YTensorGridDesc_M_O{}))>;

    using DefaultBlock2CTileMap =
        remove_cvref_t<decltype(MakeDefaultBlock2CTileMap(YTensorGridDesc_M_O{}))>;

    struct SharedMemTrait
    {
        // LDS allocation for A and B: be careful of alignment
        static constexpr auto a_block_desc_ak0_m_ak1 =
            GetABlockDescriptor_AK0PerBlock_MPerBlock_AK1();
        static constexpr auto b_block_desc_bk0_n_bk1 =
            GetBBlockDescriptor_BK0PerBlock_NPerBlock_BK1();
        static constexpr auto b1_block_desc_bk0_n_bk1 =
            GetB1BlockDescriptor_BK0PerBlock_NPerBlock_BK1();

        static constexpr auto max_lds_align = math::lcm(math::lcm(AK1, BK1), B1K1);

        static constexpr auto a_block_space_size_aligned = math::integer_least_multiple(
            a_block_desc_ak0_m_ak1.GetElementSpaceSize(), max_lds_align);
        static constexpr auto b_block_space_size_aligned = math::integer_least_multiple(
            b_block_desc_bk0_n_bk1.GetElementSpaceSize(), max_lds_align);
        static constexpr auto b1_block_space_size_aligned = math::integer_least_multiple(
            b1_block_desc_bk0_n_bk1.GetElementSpaceSize(), max_lds_align);

        static constexpr auto a_block_space_offset  = 0;
        static constexpr auto b_block_space_offset  = a_block_space_size_aligned.value;
        static constexpr auto b1_block_space_offset = 0;

        // LDS allocation for reduction
        static constexpr index_t reduction_space_size_aligned =
            math::integer_least_multiple(BlockSize, max_lds_align);

        static constexpr auto reduction_space_offset = 0;

        // LDS allocation for C shuffle in LDS
        static constexpr auto c_shuffle_block_desc_mblock_mperblock_nblock_nperblock =
            GetYShuffleBlockDescriptor_MBlock_MPerBlock_NBlock_NPerBlock();
        static constexpr auto c_block_space_size =
            c_shuffle_block_desc_mblock_mperblock_nblock_nperblock.GetElementSpaceSize();
    };

    template <bool HasMainKBlockLoop,
              typename Block2CTileMap,
              typename C0MatrixMask,
              typename VGradGridDescriptor_N_O,
              typename YGradTensorGridDesc_M0_O_M1>
    __device__ static void Run(const DataType* __restrict__ p_a_grid,
                               const DataType* __restrict__ p_b_grid,
                               const DataType* __restrict__ p_b1_grid,
                               const DataType* __restrict__ p_c_grid,
                               const DataType* __restrict__ p_ygrad_grid,
                               DataType* __restrict__ p_qgrad_grid,
                               DataType* __restrict__ p_kgrad_grid,
                               DataType* __restrict__ p_vgrad_grid,
                               void* __restrict__ p_shared,
                               const QTensorElementwiseOperation& a_element_op,
                               const KTensorElementwiseOperation& b_element_op,
                               const STensorElementwiseOperation& acc_element_op,
                               const VTensorElementwiseOperation& b1_element_op,
                               const YTensorElementwiseOperation& c_element_op,
                               const QTensorGridDesc_K0_M_K1& a_grid_desc_ak0_m_ak1,
                               const KTensorGridDesc_K0_N_K1& b_grid_desc_bk0_n_bk1,
                               const VTensorGridDesc_N0_O_N1& b1_grid_desc_bk0_n_bk1,
                               const CGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock&
                                   c_grid_desc_mblock_mperblock_nblock_nperblock,
                               const VGradGridDescriptor_N_O& vgrad_grid_desc_n_o,
                               const YGradTensorGridDesc_M0_O_M1& ygrad_grid_desc_m0_o_m1,
                               const Block2CTileMap& block_2_ctile_map,
                               const C0MatrixMask& c0_matrix_mask)
    {
        const auto a_grid_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_a_grid, a_grid_desc_ak0_m_ak1.GetElementSpaceSize());
        const auto b_grid_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_b_grid, b_grid_desc_bk0_n_bk1.GetElementSpaceSize());
        const auto b1_grid_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_b1_grid, b1_grid_desc_bk0_n_bk1.GetElementSpaceSize());
        const auto c_grid_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_c_grid, c_grid_desc_mblock_mperblock_nblock_nperblock.GetElementSpaceSize());

        // divide block work by [M, N]
        const auto block_work_idx =
            block_2_ctile_map.CalculateBottomIndex(make_multi_index(get_block_1d_id()));

        if(!block_2_ctile_map.ValidCTileIndex(
               block_work_idx,
               make_tuple(c_grid_desc_mblock_mperblock_nblock_nperblock.GetLength(I0),
                          c_grid_desc_mblock_mperblock_nblock_nperblock.GetLength(I2))))
        {
            return;
        }

        // HACK: this force m/gemm1_n_block_data_idx_on_grid into SGPR
        const index_t m_block_data_idx_on_grid =
            __builtin_amdgcn_readfirstlane(block_work_idx[I0] * Gemm0MPerBlock);

        const index_t gemm1_n_block_data_idx_on_grid =
            __builtin_amdgcn_readfirstlane(block_work_idx[I1] * Gemm1OPerBlock);

        // A matrix in LDS memory, dst of blockwise copy
        constexpr auto a_block_desc_ak0_m_ak1 = GetABlockDescriptor_AK0PerBlock_MPerBlock_AK1();

        // B matrix in LDS memory, dst of blockwise copy
        constexpr auto b_block_desc_bk0_n_bk1 = GetBBlockDescriptor_BK0PerBlock_NPerBlock_BK1();

        //
        // set up Gemm0
        //

        // A matrix blockwise copy
        auto a_blockwise_copy =
            ThreadGroupTensorSliceTransfer_v4r1<ThisThreadBlock,
                                                QTensorElementwiseOperation,
                                                tensor_operation::element_wise::PassThrough,
                                                InMemoryDataOperationEnum::Set,
                                                Sequence<AK0, Gemm0MPerBlock, AK1>,
                                                QTensorBlockTransferThreadClusterLengths_AK0_M_AK1,
                                                QTensorBlockTransferThreadClusterArrangeOrder,
                                                DataType,
                                                DataType,
                                                decltype(a_grid_desc_ak0_m_ak1),
                                                decltype(a_block_desc_ak0_m_ak1),
                                                QTensorBlockTransferSrcAccessOrder,
                                                Sequence<1, 0, 2>,
                                                QTensorBlockTransferSrcVectorDim,
                                                2,
                                                QTensorBlockTransferSrcScalarPerVector,
                                                QTensorBlockTransferDstScalarPerVector_AK1,
                                                1,
                                                1,
                                                true, // SrcResetCoord
                                                true, // DstResetCoord
                                                NumGemmKPrefetchStage>(
                a_grid_desc_ak0_m_ak1,
                make_multi_index(0, m_block_data_idx_on_grid, 0),
                a_element_op,
                a_block_desc_ak0_m_ak1,
                make_multi_index(0, 0, 0),
                tensor_operation::element_wise::PassThrough{});

        // B matrix blockwise copy
        auto b_blockwise_copy =
            ThreadGroupTensorSliceTransfer_v4r1<ThisThreadBlock,
                                                KTensorElementwiseOperation,
                                                tensor_operation::element_wise::PassThrough,
                                                InMemoryDataOperationEnum::Set,
                                                Sequence<BK0, Gemm0NPerBlock, BK1>,
                                                KTensorBlockTransferThreadClusterLengths_BK0_N_BK1,
                                                KTensorBlockTransferThreadClusterArrangeOrder,
                                                DataType,
                                                DataType,
                                                decltype(b_grid_desc_bk0_n_bk1),
                                                decltype(b_block_desc_bk0_n_bk1),
                                                KTensorBlockTransferSrcAccessOrder,
                                                Sequence<1, 0, 2>,
                                                KTensorBlockTransferSrcVectorDim,
                                                2,
                                                KTensorBlockTransferSrcScalarPerVector,
                                                KTensorBlockTransferDstScalarPerVector_BK1,
                                                1,
                                                1,
                                                true, // SrcResetCoord
                                                true, // DstResetCoord
                                                NumGemmKPrefetchStage>(
                b_grid_desc_bk0_n_bk1,
                make_multi_index(0, 0, 0), // will loop over GemmN dimension
                b_element_op,
                b_block_desc_bk0_n_bk1,
                make_multi_index(0, 0, 0),
                tensor_operation::element_wise::PassThrough{});

        // Fused Gemm+Gemm pipeline
        // for n in N0:
        //   for k in K0:
        //     acc[m][n] += A[m][k] * B0[k][n]
        //   acc1[m][o] += acc[m][n] * B1[n][o]

        // sanity check
        constexpr index_t KPack = math::max(
            math::lcm(AK1, BK1), MfmaSelector<DataType, MPerXdl, NPerXdl>::selected_mfma.k_per_blk);

        auto blockwise_gemm = BlockwiseGemmXdlops_v2<
            BlockSize,
            DataType,
            AccDataType,
            decltype(a_block_desc_ak0_m_ak1),
            decltype(b_block_desc_bk0_n_bk1),
            decltype(MakeGemm0AMmaTileDescriptor_M0_M1_M2_K(a_block_desc_ak0_m_ak1)),
            decltype(MakeGemm0BMmaTileDescriptor_N0_N1_N2_K(b_block_desc_bk0_n_bk1)),
            Gemm0MPerBlock,
            Gemm0NPerBlock,
            Gemm0KPerBlock,
            MPerXdl,
            NPerXdl,
            MXdlPerWave,
            NXdlPerWave,
            KPack,
            true>{}; // TransposeC

        auto acc_thread_buf = blockwise_gemm.GetCThreadBuffer();

        // LDS allocation for A and B: be careful of alignment
        auto a_block_buf = make_dynamic_buffer<AddressSpaceEnum::Lds>(
            static_cast<DataType*>(p_shared) + SharedMemTrait::a_block_space_offset,
            a_block_desc_ak0_m_ak1.GetElementSpaceSize());

        auto b_block_buf = make_dynamic_buffer<AddressSpaceEnum::Lds>(
            static_cast<DataType*>(p_shared) + SharedMemTrait::b_block_space_offset,
            b_block_desc_bk0_n_bk1.GetElementSpaceSize());

        constexpr auto a_block_slice_copy_step = make_multi_index(Gemm0KPerBlock / AK1, 0, 0);
        constexpr auto b_block_slice_copy_step = make_multi_index(Gemm0KPerBlock / BK1, 0, 0);
        const auto a_block_reset_copy_step =
            make_multi_index(-a_grid_desc_ak0_m_ak1.GetLength(I0), 0, 0);
        const auto b_block_reset_copy_step =
            make_multi_index(-b_grid_desc_bk0_n_bk1.GetLength(I0), Gemm0NPerBlock, 0);

        // gridwise GEMM pipeline
        // Only supports LoopScheduler::Default
        const auto gridwise_gemm_pipeline = GridwiseGemmPipeline_Selector<PipelineVer,
                                                                          NumGemmKPrefetchStage,
                                                                          LoopScheduler::Default>();

        const index_t num_k_block_main_loop = __builtin_amdgcn_readfirstlane(
            (a_grid_desc_ak0_m_ak1.GetLength(I0) * a_grid_desc_ak0_m_ak1.GetLength(I2)) /
            Gemm0KPerBlock);

        //
        // set up Gemm1
        //

        // Acc matrix threadwise copy: AccVGPR to VGPR and downcast to XDL input data type
        constexpr auto acc_thread_desc_m0_n0_m1_n1_m2_n2_n3_n4 =
            blockwise_gemm.GetCThreadDescriptor_M0_N0_M1_N1_M2_N2_N3_N4();

        constexpr auto m0 = acc_thread_desc_m0_n0_m1_n1_m2_n2_n3_n4.GetLength(I0);
        constexpr auto n0 = acc_thread_desc_m0_n0_m1_n1_m2_n2_n3_n4.GetLength(I1);
        constexpr auto m1 = acc_thread_desc_m0_n0_m1_n1_m2_n2_n3_n4.GetLength(I2);
        constexpr auto n1 = acc_thread_desc_m0_n0_m1_n1_m2_n2_n3_n4.GetLength(I3);
        constexpr auto m2 = acc_thread_desc_m0_n0_m1_n1_m2_n2_n3_n4.GetLength(I4);
        constexpr auto n2 = acc_thread_desc_m0_n0_m1_n1_m2_n2_n3_n4.GetLength(I5);
        constexpr auto n3 = acc_thread_desc_m0_n0_m1_n1_m2_n2_n3_n4.GetLength(I6);
        constexpr auto n4 = acc_thread_desc_m0_n0_m1_n1_m2_n2_n3_n4.GetLength(I7);

        constexpr auto b1_block_slice_copy_step = make_multi_index(Gemm1NPerBlock / B1K1, 0, 0);

        // acc_thread_desc_m0_n0_m1_n1_m2_n2_n3_n4 to acc_thread_desc_k0_m_k1
        // n0_n1_n2_n3 -> k0
        // m0_m1_m2 -> m
        // n4 -> k1
        // NOTE: had to use merge_v3 or will spit out compilation errors
        constexpr auto acc_thread_desc_k0_m_k1 = transform_tensor_descriptor(
            acc_thread_desc_m0_n0_m1_n1_m2_n2_n3_n4,
            make_tuple(make_merge_transform_v3_division_mod(make_tuple(n0, n1, n2, n3)),
                       make_merge_transform_v3_division_mod(make_tuple(m0, m1, m2)),
                       make_pass_through_transform(n4)),
            make_tuple(Sequence<1, 3, 5, 6>{}, Sequence<0, 2, 4>{}, Sequence<7>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}));

        // A1 matrix in AccVGPR
        // N2 num_groups_per_blk, N3 num_input_blks, N4 group_size
        constexpr auto AccN3 =
            blockwise_gemm.GetCBlockDescriptor_M0_N0_M1_N1_M2_N2_N3_N4().GetLength(I6);

        constexpr auto A1ThreadSlice_K0_M_K1 =
            make_tuple(Number<Gemm1NPerBlock / n4 / AccN3>{}, Number<m0 * m1 * m2>{}, Number<n4>{});

        constexpr auto A1ThreadSliceK0        = A1ThreadSlice_K0_M_K1[I0];
        constexpr auto A1ThreadSliceM         = A1ThreadSlice_K0_M_K1[I1];
        constexpr auto A1ThreadSliceK1        = A1ThreadSlice_K0_M_K1[I2];
        constexpr auto a1_thread_desc_k0_m_k1 = make_naive_tensor_descriptor(
            A1ThreadSlice_K0_M_K1,
            make_tuple(A1ThreadSliceM * A1ThreadSliceK1, A1ThreadSliceK1, I1));

        // B1 matrix in LDS memory, dst of blockwise copy
        constexpr auto b1_block_desc_bk0_n_bk1 = GetB1BlockDescriptor_BK0PerBlock_NPerBlock_BK1();

        // A1 matrix blockwise copy
        auto a1_blockwise_copy = ThreadwiseTensorSliceTransfer_StaticToStatic<
            AccDataType,
            DataType,
            decltype(acc_thread_desc_k0_m_k1),
            decltype(a1_thread_desc_k0_m_k1),
            tensor_operation::element_wise::PassThrough,
            Sequence<A1ThreadSliceK0, A1ThreadSliceM, A1ThreadSliceK1>,
            Sequence<1, 0, 2>,
            2,
            n4>{tensor_operation::element_wise::PassThrough{}};

        // B1 matrix blockwise copy
        auto b1_blockwise_copy =
            ThreadGroupTensorSliceTransfer_v4r1<ThisThreadBlock,
                                                KTensorElementwiseOperation,
                                                tensor_operation::element_wise::PassThrough,
                                                InMemoryDataOperationEnum::Set,
                                                Sequence<B1K0, Gemm1OPerBlock, B1K1>,
                                                VTensorBlockTransferThreadClusterLengths_BK0_N_BK1,
                                                VTensorBlockTransferThreadClusterArrangeOrder,
                                                DataType,
                                                DataType,
                                                decltype(b1_grid_desc_bk0_n_bk1),
                                                decltype(b1_block_desc_bk0_n_bk1),
                                                VTensorBlockTransferSrcAccessOrder,
                                                Sequence<1, 0, 2>,
                                                VTensorBlockTransferSrcVectorDim,
                                                2,
                                                VTensorBlockTransferSrcScalarPerVector,
                                                VTensorBlockTransferDstScalarPerVector_BK1,
                                                1,
                                                1,
                                                VTensorThreadTransferSrcResetCoordinateAfterRun,
                                                true, // DstResetCoord
                                                NumGemmKPrefetchStage>(
                b1_grid_desc_bk0_n_bk1,
                make_multi_index(0, gemm1_n_block_data_idx_on_grid, 0),
                b1_element_op,
                b1_block_desc_bk0_n_bk1,
                make_multi_index(0, 0, 0),
                tensor_operation::element_wise::PassThrough{});

        auto a1_thread_buf = make_static_buffer<AddressSpaceEnum::Vgpr, DataType>(
            a1_thread_desc_k0_m_k1.GetElementSpaceSize());

        // reuse LDS space for gemm0's b_block_buf
        auto b1_block_buf = make_dynamic_buffer<AddressSpaceEnum::Lds>(
            static_cast<DataType*>(p_shared) + SharedMemTrait::b1_block_space_offset,
            b1_block_desc_bk0_n_bk1.GetElementSpaceSize());

        // selected_mfma.group_size or B1K1 <= Gemm1KPack <= selected_mfma.group_size
        // selected_mfma.k_per_blk <= Gemm1KPack
        //
        // Following similar rationale behind Gemm0KPack, let Gemm1KPack be the lowest common
        // multiples of A1K1 (predetermined by selected_mfma.group_size) and B1K1. But in this case
        // Gemm1KPack can't be higher than A1K1 itself because A1 matrix is distributed in VGPRs
        // with 'group_size' amount of contiguous elements. Having Gemm1KPack greater than A1K1 will
        // cause mismatch in summation index for example c[0:7] = a1[[0:3, 8:11]] * b1[0:7].
        // therefore we may just as well assign Gemm1KPack = group_size
        constexpr index_t Gemm1KPack =
            MfmaSelector<DataType, MPerXdl, NPerXdl>::selected_mfma.group_size;

        auto gemm1_blockwise_gemm = BlockwiseGemmXdlops_v2<
            BlockSize,
            DataType,
            AccDataType,
            decltype(a1_thread_desc_k0_m_k1),
            decltype(b1_block_desc_bk0_n_bk1),
            decltype(MakeGemm1AMmaTileDescriptor_M0_M1_M2_K(a1_thread_desc_k0_m_k1)),
            decltype(MakeGemm1BMmaTileDescriptor_N0_N1_N2_K(b1_block_desc_bk0_n_bk1)),
            Gemm0MPerBlock,
            Gemm1OPerBlock,
            Gemm1NPerBlock,
            MPerXdl,
            NPerXdl,
            MXdlPerWave,
            OXdlPerWave,
            Gemm1KPack,
            true,       // TransposeC
            Gemm1KPack, // AMmaKStride
            Gemm1KPack * XdlopsGemm<DataType, MPerXdl, NPerXdl, Gemm1KPack, false>{}.K0PerXdlops>{
            // BMmaKStride
            make_tuple(0, 0, 0, 0)}; // A_origin

        auto acc1_thread_buf = gemm1_blockwise_gemm.GetCThreadBuffer();

        //
        // Blockwise softmax
        //
        auto workspace_buf = make_dynamic_buffer<AddressSpaceEnum::Lds>(
            static_cast<AccDataType*>(p_shared) + SharedMemTrait::reduction_space_offset,
            SharedMemTrait::reduction_space_size_aligned);

        // get acc0 8D thread cluster
        constexpr auto thread_cluster_m0_n0_m1_n1_m2_n2_n3_n4 =
            blockwise_gemm.GetCBlockDescriptor_M0_N0_M1_N1_M2_N2_N3_N4().GetLengths() /
            blockwise_gemm.GetCThreadDescriptor_M0_N0_M1_N1_M2_N2_N3_N4().GetLengths();
        constexpr auto tm0 = thread_cluster_m0_n0_m1_n1_m2_n2_n3_n4.At(I0);
        constexpr auto tn0 = thread_cluster_m0_n0_m1_n1_m2_n2_n3_n4.At(I1);
        constexpr auto tm1 = thread_cluster_m0_n0_m1_n1_m2_n2_n3_n4.At(I2);
        constexpr auto tn1 = thread_cluster_m0_n0_m1_n1_m2_n2_n3_n4.At(I3);
        constexpr auto tm2 = thread_cluster_m0_n0_m1_n1_m2_n2_n3_n4.At(I4);
        constexpr auto tn2 = thread_cluster_m0_n0_m1_n1_m2_n2_n3_n4.At(I5);
        constexpr auto tn3 = thread_cluster_m0_n0_m1_n1_m2_n2_n3_n4.At(I6);
        constexpr auto tn4 = thread_cluster_m0_n0_m1_n1_m2_n2_n3_n4.At(I7);

        // get acc0 thread map
        constexpr auto m0_n_m1_to_m_n_adaptor = make_single_stage_tensor_adaptor(
            make_tuple(make_unmerge_transform(make_tuple(tm0 * tm1, tm2)),
                       make_pass_through_transform(I1)),
            make_tuple(Sequence<0>{}, Sequence<1>{}),
            make_tuple(Sequence<0, 2>{}, Sequence<1>{}));
        constexpr auto threadid_to_m0_n_m1_adaptor = make_single_stage_tensor_adaptor(
            make_tuple(
                make_merge_transform(make_tuple(tm0 * tm1, tn0 * tn1 * tn2 * tn3 * tn4, tm2))),
            make_tuple(Sequence<0, 1, 2>{}),
            make_tuple(Sequence<0>{}));
        const auto threadid_to_m_n_thread_cluster_adaptor =
            chain_tensor_adaptors(m0_n_m1_to_m_n_adaptor, threadid_to_m0_n_m1_adaptor);

        // get acc0 2D thread cluster & 2D thread slice
        constexpr auto thread_cluster_desc_m_n = make_naive_tensor_descriptor_packed(
            make_tuple(tm0 * tm1 * tm2, tn0 * tn1 * tn2 * tn3 * tn4));
        constexpr auto thread_slice_desc_m_n =
            make_naive_tensor_descriptor_packed(make_tuple(m0 * m1 * m2, n0 * n1 * n2 * n3 * n4));

        auto blockwise_softmax = BlockwiseSoftmax<BlockSize,
                                                  AccDataType,
                                                  decltype(threadid_to_m_n_thread_cluster_adaptor),
                                                  decltype(thread_cluster_desc_m_n),
                                                  decltype(thread_slice_desc_m_n)>{};

        const index_t num_gemm1_k_block_outer_loop =
            b_grid_desc_bk0_n_bk1.GetLength(I1) / Gemm0NPerBlock;
        constexpr index_t num_gemm1_k_block_inner_loop = Gemm0NPerBlock / Gemm1NPerBlock;

        // Initialize C
        StaticBuffer<AddressSpaceEnum::Vgpr, AccDataType, acc1_thread_buf.Size(), true>
            c_thread_buf;
        c_thread_buf.Clear();

        // Initialize running sum and max of exponentiating row vectors
        using SoftmaxBuf = typename decltype(blockwise_softmax)::BufferType;
        SoftmaxBuf running_sum, running_sum_new, running_max, running_max_new;
        running_sum     = 0;
        running_sum_new = 0;
        running_max     = NumericLimits<AccDataType>::Lowest();
        running_max_new = NumericLimits<AccDataType>::Lowest();

        //
        // dV
        //

        // P vgpr to lds: writes vgprs of a subgroup to LDS and transform into m0_n_m1
        // m0, n0 are m/n repeat per wave
        // m1, n1 are number of waves
        constexpr auto p_src_thread_desc_m0_n0_m1_n1_m2_n2_n3_n4 =
            blockwise_gemm.GetCThreadDescriptor_M0_N0_M1_N1_M2_N2_N3_N4();

        constexpr auto p_dst_block_desc_m0_n_m1 =
            VGradGemmTile_N_O_M::GetPBlockDescriptor_M0_N_M1();

        constexpr auto p_block_lengths =
            blockwise_gemm.GetCBlockDescriptor_M0_N0_M1_N1_M2_N2_N3_N4().GetLengths();
        constexpr auto P_M0 = p_block_lengths[I0]; // repeats
        constexpr auto P_N0 = p_block_lengths[I1];
        constexpr auto P_M1 = p_block_lengths[I2]; // waves
        constexpr auto P_N1 = p_block_lengths[I3];
        constexpr auto P_M2 = p_block_lengths[I4]; // xdl
        constexpr auto P_N2 = p_block_lengths[I5];
        constexpr auto P_N3 = p_block_lengths[I6];
        constexpr auto P_N4 = p_block_lengths[I7];

        constexpr auto p_dst_block_desc_m0_n0_m1_n1_m2_n2_n3_n4 = [&]() constexpr
        {
            constexpr auto p_dst_block_desc_m_n = transform_tensor_descriptor(
                p_dst_block_desc_m0_n_m1,
                make_tuple(make_merge_transform_v3_division_mod(
                               make_tuple(VGradGemmTile_N_O_M::P_M0, VGradGemmTile_N_O_M::P_M1)),
                           make_pass_through_transform(VGradGemmTile_N_O_M::Free0_N)),
                make_tuple(Sequence<0, 2>{}, Sequence<1>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}));

            return transform_tensor_descriptor(
                p_dst_block_desc_m_n,
                make_tuple(make_unmerge_transform(make_tuple(P_M0, P_M1, P_M2)),
                           make_unmerge_transform(make_tuple(P_N0, P_N1, P_N2, P_N3, P_N4))),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0, 2, 4>{}, Sequence<1, 3, 5, 6, 7>{}));
        }
        ();

        // TODO ANT: check lds offset
        auto p_block_buf = make_dynamic_buffer<AddressSpaceEnum::Lds>(
            static_cast<DataType*>(p_shared), p_dst_block_desc_m0_n_m1.GetElementSpaceSize());

        const auto p_dst_thread_origin = [&]() {
            const auto c_thread_mtx_on_block =
                blockwise_gemm.CalculateCThreadOriginDataIndex(I0, I0, I0, I0);

            const index_t m_thread_data_on_block = c_thread_mtx_on_block[I0];
            const index_t n_thread_data_on_block = c_thread_mtx_on_block[I1];

            const auto m_thread_data_on_block_to_m0_m1_m2_adaptor =
                make_single_stage_tensor_adaptor(
                    make_tuple(make_merge_transform(make_tuple(P_M0, P_M1, P_M2))),
                    make_tuple(Sequence<0, 1, 2>{}),
                    make_tuple(Sequence<0>{}));

            const auto m_thread_data_on_block_idx =
                m_thread_data_on_block_to_m0_m1_m2_adaptor.CalculateBottomIndex(
                    make_multi_index(m_thread_data_on_block));

            const auto n_thread_data_on_block_to_n0_n1_n2_n3_n4_adaptor =
                make_single_stage_tensor_adaptor(
                    make_tuple(make_merge_transform(make_tuple(P_N0, P_N1, P_N2, P_N3, P_N4))),
                    make_tuple(Sequence<0, 1, 2, 3, 4>{}),
                    make_tuple(Sequence<0>{}));

            const auto n_thread_data_on_block_idx =
                n_thread_data_on_block_to_n0_n1_n2_n3_n4_adaptor.CalculateBottomIndex(
                    make_multi_index(n_thread_data_on_block));

            return make_tuple(0,                              // mrepeat
                              0,                              // nrepeat
                              m_thread_data_on_block_idx[I1], // mwave
                              n_thread_data_on_block_idx[I1], // nwave
                              m_thread_data_on_block_idx[I2], // xdlops
                              n_thread_data_on_block_idx[I2],
                              n_thread_data_on_block_idx[I3],
                              n_thread_data_on_block_idx[I4]);
        }();

        constexpr auto p_block_slice_lengths_m0_n0_m1_n1_m2_n2 = // mrepeat, nrepeat, mwaves,
                                                                 // nwaves, mperxdl, nperxdl
            VGradGemmTile_N_O_M::GetPBlockSliceLengths_M0_N0_M1_N1_M2_N2();

        // how to properly perform copy for a sub-workgroup?
        auto p_thread_copy_vgpr_to_lds = ThreadwiseTensorSliceTransfer_v1r3<
            AccDataType,
            DataType,
            decltype(p_src_thread_desc_m0_n0_m1_n1_m2_n2_n3_n4),
            decltype(p_dst_block_desc_m0_n0_m1_n1_m2_n2_n3_n4),
            tensor_operation::element_wise::PassThrough,
            Sequence<p_block_slice_lengths_m0_n0_m1_n1_m2_n2[I0], // ThreadSliceLengths
                     p_block_slice_lengths_m0_n0_m1_n1_m2_n2[I1],
                     I1,
                     I1,
                     I1,
                     P_N2,
                     I1,
                     P_N4>,
            Sequence<0, 1, 2, 3, 4, 5, 6, 7>,
            7, // DstVectorDim
            1, // DstScalarPerVector
            InMemoryDataOperationEnum::Set,
            1, // DstScalarStrideInVector
            true>{
            p_dst_block_desc_m0_n0_m1_n1_m2_n2_n3_n4,
            make_multi_index(p_dst_thread_origin[I0],
                             p_dst_thread_origin[I1],
                             p_dst_thread_origin[I2] % p_block_slice_lengths_m0_n0_m1_n1_m2_n2[I2],
                             p_dst_thread_origin[I3] % p_block_slice_lengths_m0_n0_m1_n1_m2_n2[I3],
                             p_dst_thread_origin[I4],
                             p_dst_thread_origin[I5],
                             p_dst_thread_origin[I6],
                             p_dst_thread_origin[I7]),
            tensor_operation::element_wise::PassThrough{}};

        // construct space filling curve
        // p_thread_copy_vgpr_to_lds.Run();

        constexpr auto ygrad_dst_block_desc_m0_o_m1 =
            VGradGemmTile_N_O_M::GetYGradBlockDescriptor_M0_O_M1();

        auto ygrad_blockwise_copy = ThreadGroupTensorSliceTransfer_v4r1<
            ThisThreadBlock,
            tensor_operation::element_wise::PassThrough,
            tensor_operation::element_wise::PassThrough,
            InMemoryDataOperationEnum::Set,
            typename VGradGemmTile_N_O_M::YGrad_BlockSliceLengths,
            typename VGradGemmTile_N_O_M::YGrad_ThreadClusterLengths,
            typename VGradGemmTile_N_O_M::YGrad_ThreadClusterArrangeOrder,
            DataType,
            DataType,
            decltype(ygrad_grid_desc_m0_o_m1),
            decltype(ygrad_dst_block_desc_m0_o_m1),
            typename VGradGemmTile_N_O_M::YGrad_ThreadClusterArrangeOrder, // access order == thread
                                                                           // order
            Sequence<1, 0, 2>,
            VGradGemmTile_N_O_M::YGrad_SrcVectorDim,
            2, // DstVectorDim
            VGradGemmTile_N_O_M::YGrad_SrcScalarPerVector,
            VGradGemmTile_N_O_M::YGrad_M1,
            1,
            1,
            true,
            true,
            1>(ygrad_grid_desc_m0_o_m1,
               make_multi_index(0, gemm1_n_block_data_idx_on_grid, 0),
               tensor_operation::element_wise::PassThrough{},
               ygrad_dst_block_desc_m0_o_m1,
               make_multi_index(0, 0, 0),
               tensor_operation::element_wise::PassThrough{});

        auto vgrad_blockwise_gemm = BlockwiseGemmXdlops_k0mk1_k0nk1_m0n0m1n1m2m3m4n2_v1<
            BlockSize,
            DataType,
            AccDataType,
            decltype(p_dst_block_desc_m0_n_m1),
            decltype(ygrad_dst_block_desc_m0_o_m1),
            MPerXdl,
            NPerXdl,
            VGradGemmTile_N_O_M::GemmNRepeat, // NRepeat
            VGradGemmTile_N_O_M::GemmORepeat, // ORepeat
            VGradGemmTile_N_O_M::GemmMPack>{};

        constexpr auto vgrad_block_lengths =
            vgrad_blockwise_gemm.GetCBlockDescriptor_M0_N0_M1_N1_M2_M3_M4_N2().GetLengths();

        const auto vgrad_grid_desc_n0_o0_n1_o1_n2_o2 = transform_tensor_descriptor(
            vgrad_grid_desc_n_o,
            make_tuple(
                make_unmerge_transform(make_tuple(I1, // may place a dummy variable
                                                  p_block_slice_lengths_m0_n0_m1_n1_m2_n2[I2],
                                                  p_block_slice_lengths_m0_n0_m1_n1_m2_n2[I4])),
                make_unmerge_transform(make_tuple(I1,
                                                  p_block_slice_lengths_m0_n0_m1_n1_m2_n2[I3],
                                                  p_block_slice_lengths_m0_n0_m1_n1_m2_n2[I5]))),
            make_tuple(Sequence<0>{}, Sequence<1>{}),
            make_tuple(Sequence<0, 2, 4>{}, Sequence<1, 3, 5>{}));

        constexpr auto vgrad_thread_desc_n0_o0_n1_o1_n2_n3_n4_o2 =
            vgrad_blockwise_gemm.GetCThreadDescriptor_M0_N0_M1_N1_M2_M3_M4_N2();

        const auto vgrad_grid_desc_n0_o0_n1_o1_n2_n3_n4_o2 =
            vgrad_blockwise_gemm.xdlops_gemm.MakeCDescriptor_M0_N0_M1_N1_M2_M3_M4_N2(
                vgrad_grid_desc_n0_o0_n1_o1_n2_o2);

        const auto vgrad_thread_mtx_on_block_n_o =
            vgrad_blockwise_gemm.CalculateCThreadOriginDataIndex(I0, I0, I0, I0);

        constexpr auto vgrad_block_desc_n0_o0_n1_o1_n2_n3_n4_o2 =
            decltype(vgrad_blockwise_gemm)::GetCBlockDescriptor_M0_N0_M1_N1_M2_M3_M4_N2();
        constexpr auto VGrad_N0 = vgrad_block_desc_n0_o0_n1_o1_n2_n3_n4_o2.GetLength(I0);
        constexpr auto VGrad_O0 = vgrad_block_desc_n0_o0_n1_o1_n2_n3_n4_o2.GetLength(I1);
        constexpr auto VGrad_N1 = vgrad_block_desc_n0_o0_n1_o1_n2_n3_n4_o2.GetLength(I2);
        constexpr auto VGrad_O1 = vgrad_block_desc_n0_o0_n1_o1_n2_n3_n4_o2.GetLength(I3);
        constexpr auto VGrad_N2 = vgrad_block_desc_n0_o0_n1_o1_n2_n3_n4_o2.GetLength(I4);
        constexpr auto VGrad_N3 = vgrad_block_desc_n0_o0_n1_o1_n2_n3_n4_o2.GetLength(I5);
        constexpr auto VGrad_N4 = vgrad_block_desc_n0_o0_n1_o1_n2_n3_n4_o2.GetLength(I6);
        constexpr auto VGrad_O2 = vgrad_block_desc_n0_o0_n1_o1_n2_n3_n4_o2.GetLength(I7);

        const index_t n_thread_data_idx_on_grid =
            vgrad_thread_mtx_on_block_n_o[I0]; // TODO ANT: step n after each Gemm1 outer loop

        const index_t o_thread_data_idx_on_grid =
            vgrad_thread_mtx_on_block_n_o[I1] + gemm1_n_block_data_idx_on_grid;

        const auto n_thread_data_on_grid_to_n0_n1_n2_n3_n4_adaptor =
            make_single_stage_tensor_adaptor(
                make_tuple(make_merge_transform(
                    make_tuple(VGrad_N0, VGrad_N1, VGrad_N2, VGrad_N3, VGrad_N4))),
                make_tuple(Sequence<0, 1, 2, 3, 4>{}),
                make_tuple(Sequence<0>{}));

        const auto n_thread_data_nd_idx_on_grid =
            n_thread_data_on_grid_to_n0_n1_n2_n3_n4_adaptor.CalculateBottomIndex(
                make_multi_index(n_thread_data_idx_on_grid));

        const auto o_thread_data_on_grid_to_o0_o1_o2_adaptor = make_single_stage_tensor_adaptor(
            make_tuple(make_merge_transform(make_tuple(VGrad_O0, VGrad_O1, VGrad_O2))),
            make_tuple(Sequence<0, 1, 2>{}),
            make_tuple(Sequence<0>{}));

        const auto o_thread_data_nd_idx_on_grid =
            o_thread_data_on_grid_to_o0_o1_o2_adaptor.CalculateBottomIndex(
                make_multi_index(o_thread_data_idx_on_grid));

        auto vgrad_thread_copy_vgpr_to_global = ThreadwiseTensorSliceTransfer_v1r3<
            AccDataType,
            DataType,
            decltype(vgrad_thread_desc_n0_o0_n1_o1_n2_n3_n4_o2),
            decltype(vgrad_grid_desc_n0_o0_n1_o1_n2_n3_n4_o2),
            tensor_operation::element_wise::PassThrough, // YTensorElementwiseOperation
            decltype(vgrad_blockwise_gemm.GetCThreadDescriptor_M0_N0_M1_N1_M2_M3_M4_N2()
                         .GetLengths()),          // SliceLengths
            Sequence<0, 1, 2, 3, 4, 5, 6, 7>,     // AccessOrder
            7,                                    // VectorDim
            1,                                    // ScalarPerVector
            InMemoryDataOperationEnum::AtomicAdd, // GlobalMemoryDataOperation
            1,
            true>(vgrad_grid_desc_n0_o0_n1_o1_n2_n3_n4_o2,
                  make_multi_index(n_thread_data_nd_idx_on_grid[I0],
                                   o_thread_data_nd_idx_on_grid[I0],
                                   n_thread_data_nd_idx_on_grid[I1],
                                   o_thread_data_nd_idx_on_grid[I1],
                                   n_thread_data_nd_idx_on_grid[I2],
                                   n_thread_data_nd_idx_on_grid[I3],
                                   n_thread_data_nd_idx_on_grid[I4],
                                   o_thread_data_nd_idx_on_grid[I2]),
                  tensor_operation::element_wise::PassThrough{});

        // TODO ANT: ygrad slice window step size

#if 0
        // gemm1 K loop
        index_t gemm1_k_block_outer_index = 0;
        do
        {
            auto n_block_data_idx_on_grid =
                __builtin_amdgcn_readfirstlane(gemm1_k_block_outer_index * Gemm0NPerBlock);
            if(c0_matrix_mask.IsTileSkippable(
                   m_block_data_idx_on_grid, n_block_data_idx_on_grid, Gemm0MPerBlock, Gemm0NPerBlock))
            {
                continue;
            }
            // gemm0
            gridwise_gemm_pipeline.template Run<HasMainKBlockLoop>(a_grid_desc_ak0_m_ak1,
                                                                   a_block_desc_ak0_m_ak1,
                                                                   a_blockwise_copy,
                                                                   a_grid_buf,
                                                                   a_block_buf,
                                                                   a_block_slice_copy_step,
                                                                   b_grid_desc_bk0_n_bk1,
                                                                   b_block_desc_bk0_n_bk1,
                                                                   b_blockwise_copy,
                                                                   b_grid_buf,
                                                                   b_block_buf,
                                                                   b_block_slice_copy_step,
                                                                   blockwise_gemm,
                                                                   acc_thread_buf,
                                                                   num_k_block_main_loop);

            // do MNK padding or upper triangular masking
            if constexpr(MaskOutUpperTriangle || PadN)
            {
                // 8d thread_desc in thread scope
                constexpr auto c_thread_lengths =
                    blockwise_gemm.GetCThreadDescriptor_M0_N0_M1_N1_M2_N2_N3_N4().GetLengths();

                // 8d block_desc in block scope
                constexpr auto c_block_lengths =
                    blockwise_gemm.GetCBlockDescriptor_M0_N0_M1_N1_M2_N2_N3_N4().GetLengths();

                constexpr auto M0 = c_block_lengths[I0];
                constexpr auto N0 = c_block_lengths[I1];
                constexpr auto M1 = c_block_lengths[I2];
                constexpr auto N1 = c_block_lengths[I3];
                constexpr auto M2 = c_block_lengths[I4];
                constexpr auto N2 = c_block_lengths[I5];
                constexpr auto N3 = c_block_lengths[I6];
                constexpr auto N4 = c_block_lengths[I7];

                // works like multi-dimension static_for (static_ford), but provides both the linear
                // index as well as n-d index
                using Acc0TileIterator = SpaceFillingCurve<
                    decltype(c_thread_lengths),
                    typename arithmetic_sequence_gen<0, c_thread_lengths.Size(), 1>::type,
                    typename uniform_sequence_gen<c_thread_lengths.Size(), 1>::type,
                    false>; // SnakeCurved

                auto acc0_thread_origin = blockwise_gemm.CalculateCThreadOriginDataIndex8D(
                    Number<0>{}, Number<0>{}, Number<0>{}, Number<0>{});

                constexpr auto block_idx_to_m_n_adaptor = make_single_stage_tensor_adaptor(
                    make_tuple(make_unmerge_transform(make_tuple(M0, M1, M2)),
                               make_unmerge_transform(make_tuple(N0, N1, N2, N3, N4))),
                    make_tuple(Sequence<0>{}, Sequence<1>{}),
                    make_tuple(Sequence<0, 2, 4>{}, Sequence<1, 3, 5, 6, 7>{}));

                static_for<0, Acc0TileIterator::GetNumOfAccess(), 1>{}([&](auto i) {
                    auto acc0_thread_idx = Acc0TileIterator::GetIndex(i) + acc0_thread_origin;
                    auto m_local =
                        block_idx_to_m_n_adaptor.CalculateBottomIndex(acc0_thread_idx)[I0];
                    auto n_local =
                        block_idx_to_m_n_adaptor.CalculateBottomIndex(acc0_thread_idx)[I1];
                    auto m_global = m_local + m_block_data_idx_on_grid;
                    auto n_global = n_local + n_block_data_idx_on_grid;
                    if(c0_matrix_mask.IsMaskedElement(m_global, n_global))
                    {
                        acc_thread_buf(i) = -ck::NumericLimits<float>::Infinity();
                    }
                    else
                    {
                        acc_element_op(acc_thread_buf(i), acc_thread_buf[i]);
                    }
                });
            }

            block_sync_lds(); // wait for lds read in gemm0 blockwise gemm

            // softmax
            SoftmaxBuf& max = blockwise_softmax.max_value_buf;
            SoftmaxBuf& sum = blockwise_softmax.sum_value_buf;

            blockwise_softmax.Run(acc_thread_buf, workspace_buf);

            // TODO: may convert to log domain
            running_max_new = mathext::max(max, running_max);
            running_sum_new = mathext::exp(running_max - running_max_new) * running_sum +
                              mathext::exp(max - running_max_new) * sum;

            // gemm1
            {
                // TODO: explore using dynamic buffer for a1 thread buffer
                // For a1_blockwise_copy, the goal is to satisfy pipeline requirements RunRead(),
                // RunWrite(), and MoveSliceWindow(). But it is impossible to implement given that
                // the A1 source buffer is static buffer holding the output of first GEMM and
                // requires constexpr offset by design. Therefore, we pass tensor coordinate offset
                // explicitly in Run() below.

                // Initialize acc1
                acc1_thread_buf.Clear();

                // preload data into LDS
                b1_blockwise_copy.RunRead(b1_grid_desc_bk0_n_bk1, b1_grid_buf);

                b1_blockwise_copy.MoveSrcSliceWindow(b1_grid_desc_bk0_n_bk1,
                                                     b1_block_slice_copy_step);

                block_sync_lds(); // wait for reduction LDS read

                b1_blockwise_copy.RunWrite(b1_block_desc_bk0_n_bk1, b1_block_buf);

                // main body
                if constexpr(num_gemm1_k_block_inner_loop > 1)
                {
                    static_for<0, num_gemm1_k_block_inner_loop - 1, 1>{}([&](auto i) {
                        a1_blockwise_copy.Run(acc_thread_desc_k0_m_k1,
                                              make_tuple(Number<i * A1ThreadSliceK0>{}, I0, I0),
                                              acc_thread_buf,
                                              a1_thread_desc_k0_m_k1,
                                              make_tuple(I0, I0, I0),
                                              a1_thread_buf);

                        b1_blockwise_copy.RunRead(b1_grid_desc_bk0_n_bk1, b1_grid_buf);

                        block_sync_lds();

                        gemm1_blockwise_gemm.Run(a1_thread_buf, b1_block_buf, acc1_thread_buf);

                        block_sync_lds();

                        b1_blockwise_copy.MoveSrcSliceWindow(b1_grid_desc_bk0_n_bk1,
                                                             b1_block_slice_copy_step);

                        b1_blockwise_copy.RunWrite(b1_block_desc_bk0_n_bk1, b1_block_buf);
                    });
                }
                // tail
                {
                    a1_blockwise_copy.Run(
                        acc_thread_desc_k0_m_k1,
                        make_tuple(
                            Number<(num_gemm1_k_block_inner_loop - 1) * A1ThreadSliceK0>{}, I0, I0),
                        acc_thread_buf,
                        a1_thread_desc_k0_m_k1,
                        make_tuple(I0, I0, I0),
                        a1_thread_buf);

                    block_sync_lds();

                    gemm1_blockwise_gemm.Run(a1_thread_buf, b1_block_buf, acc1_thread_buf);
                }
            } // end gemm1

            constexpr auto c_thread_desc_m0_n0_m1_n1_m2_n2_n3_n4 =
                gemm1_blockwise_gemm.GetCThreadDescriptor_M0_N0_M1_N1_M2_N2_N3_N4();
            constexpr auto cm0 = c_thread_desc_m0_n0_m1_n1_m2_n2_n3_n4.GetLength(I0);
            constexpr auto cn0 = c_thread_desc_m0_n0_m1_n1_m2_n2_n3_n4.GetLength(I1);
            constexpr auto cm1 = c_thread_desc_m0_n0_m1_n1_m2_n2_n3_n4.GetLength(I2);
            constexpr auto cn1 = c_thread_desc_m0_n0_m1_n1_m2_n2_n3_n4.GetLength(I3);
            constexpr auto cm2 = c_thread_desc_m0_n0_m1_n1_m2_n2_n3_n4.GetLength(I4);
            constexpr auto cn2 = c_thread_desc_m0_n0_m1_n1_m2_n2_n3_n4.GetLength(I5);
            constexpr auto cn3 = c_thread_desc_m0_n0_m1_n1_m2_n2_n3_n4.GetLength(I6);
            constexpr auto cn4 = c_thread_desc_m0_n0_m1_n1_m2_n2_n3_n4.GetLength(I7);
            constexpr auto c_thread_slice_desc_m_n = make_naive_tensor_descriptor_packed(
                make_tuple(cm0 * cm1 * cm2, cn0 * cn1 * cn2 * cn3 * cn4));
            constexpr auto c_thread_buf_slice_m = c_thread_slice_desc_m_n.GetLength(I0);
            constexpr auto c_thread_buf_slice_n = c_thread_slice_desc_m_n.GetLength(I1);

            static_for<0, c_thread_buf_slice_m, 1>{}([&](auto iM) {
                static_for<0, c_thread_buf_slice_n, 1>{}([&](auto iN) {
                    auto I = Number<c_thread_slice_desc_m_n.CalculateOffset(make_tuple(iM, iN))>{};
                    AccDataType acc1 = acc1_thread_buf[I]; // P*V
                    AccDataType c    = c_thread_buf[I];    // O
                    AccDataType c_new =
                        (running_sum[iM] * math::exp(running_max[iM] - running_max_new[iM]) * c +
                         math::exp(max[iM] - running_max_new[iM]) * acc1) /
                        running_sum_new[iM]; // Formula by Dao et al.,
                                             // https://arxiv.org/pdf/2205.14135v2.pdf section 3.1

                    c_thread_buf(I) = c_new; // O_new
                });
            });

            a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc_ak0_m_ak1,
                                                a_block_reset_copy_step); // rewind K
            b_blockwise_copy.MoveSrcSliceWindow(b_grid_desc_bk0_n_bk1,
                                                b_block_reset_copy_step); // rewind K and step N

            // update before next j iteration
            running_max = running_max_new;
            running_sum = running_sum_new;

            block_sync_lds(); // wait for gemm1 LDS read
        } while(++gemm1_k_block_outer_index < num_gemm1_k_block_outer_loop); // end j loop
#endif

        // TODO ANT:
        // shuffle dQ and write
        if constexpr(false)
        {
            static_assert(MXdlPerWave % YShuffleMXdlPerWavePerShuffle == 0 &&
                              OXdlPerWave % YShuffleNXdlPerWavePerShuffle == 0,
                          "wrong!");

            constexpr index_t MWave = Gemm0MPerBlock / (MXdlPerWave * MPerXdl);
            constexpr index_t NWave = Gemm1OPerBlock / (OXdlPerWave * NPerXdl);

            // TODO: hacky, fix it!
            constexpr auto c_thread_desc_m0_n0_m1_n1_m2_n2_n3_n4 =
                gemm1_blockwise_gemm.GetCThreadDescriptor_M0_N0_M1_N1_M2_N2_N3_N4();

            // TODO: hacky, fix it!
            // c_block_desc_m0_n0_m1_n1_m2_n2_n3_n4_tmp is only used to get lengths
            constexpr auto c_block_desc_m0_n0_m1_n1_m2_n2_n3_n4_tmp =
                gemm1_blockwise_gemm.GetCBlockDescriptor_M0_N0_M1_N1_M2_N2_N3_N4();

            constexpr auto M0 = c_block_desc_m0_n0_m1_n1_m2_n2_n3_n4_tmp.GetLength(I0);
            constexpr auto N0 = c_block_desc_m0_n0_m1_n1_m2_n2_n3_n4_tmp.GetLength(I1);
            constexpr auto M1 = c_block_desc_m0_n0_m1_n1_m2_n2_n3_n4_tmp.GetLength(I2);
            constexpr auto N1 = c_block_desc_m0_n0_m1_n1_m2_n2_n3_n4_tmp.GetLength(I3);
            constexpr auto M2 = c_block_desc_m0_n0_m1_n1_m2_n2_n3_n4_tmp.GetLength(I4);
            constexpr auto N2 = c_block_desc_m0_n0_m1_n1_m2_n2_n3_n4_tmp.GetLength(I5);
            constexpr auto N3 = c_block_desc_m0_n0_m1_n1_m2_n2_n3_n4_tmp.GetLength(I6);
            constexpr auto N4 = c_block_desc_m0_n0_m1_n1_m2_n2_n3_n4_tmp.GetLength(I7);

            constexpr auto c_shuffle_block_desc_mblock_mperblock_nblock_nperblock =
                GetYShuffleBlockDescriptor_MBlock_MPerBlock_NBlock_NPerBlock();

            auto c_shuffle_block_buf = make_dynamic_buffer<AddressSpaceEnum::Lds>(
                static_cast<ShuffleDataType*>(p_shared),
                c_shuffle_block_desc_mblock_mperblock_nblock_nperblock.GetElementSpaceSize());

            constexpr auto c_block_desc_m0_n0_m1_n1_m2_n2_n3_n4 = transform_tensor_descriptor(
                c_shuffle_block_desc_mblock_mperblock_nblock_nperblock,
                make_tuple(
                    make_freeze_transform(I0),
                    make_unmerge_transform(make_tuple(
                        Number<YShuffleMXdlPerWavePerShuffle>{}, // M0 (MXdlPerWave) per shuffle
                        M1,                                      // M1 = MWave
                        M2)),                                    // M2 = MPerXdl
                    make_freeze_transform(I0),
                    make_unmerge_transform(make_tuple(
                        Number<YShuffleNXdlPerWavePerShuffle>{}, // N0 (NXdlPerWave) per shuffle
                        N1,                                      // N1 = NWave
                        N2,                                      // N2 * N3 * N4 = NPerXdl
                        N3,
                        N4))),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
                make_tuple(
                    Sequence<>{}, Sequence<0, 2, 4>{}, Sequence<>{}, Sequence<1, 3, 5, 6, 7>{}));

            // calculate origin of thread output tensor on global memory
            //     blockwise GEMM c matrix starting index
            const auto c_thread_mtx_on_block =
                gemm1_blockwise_gemm.CalculateCThreadOriginDataIndex(I0, I0, I0, I0);

            const index_t m_thread_data_on_block = c_thread_mtx_on_block[I0];
            const index_t n_thread_data_on_block = c_thread_mtx_on_block[I1];

            const auto m_thread_data_on_block_to_m0_m1_m2_adaptor =
                make_single_stage_tensor_adaptor(
                    make_tuple(make_merge_transform(make_tuple(M0, M1, M2))),
                    make_tuple(Sequence<0, 1, 2>{}),
                    make_tuple(Sequence<0>{}));

            const auto m_thread_data_on_block_idx =
                m_thread_data_on_block_to_m0_m1_m2_adaptor.CalculateBottomIndex(
                    make_multi_index(m_thread_data_on_block));

            const auto n_thread_data_on_block_to_n0_n1_n2_n3_n4_adaptor =
                make_single_stage_tensor_adaptor(
                    make_tuple(make_merge_transform(make_tuple(N0, N1, N2, N3, N4))),
                    make_tuple(Sequence<0, 1, 2, 3, 4>{}),
                    make_tuple(Sequence<0>{}));

            const auto n_thread_data_on_block_idx =
                n_thread_data_on_block_to_n0_n1_n2_n3_n4_adaptor.CalculateBottomIndex(
                    make_multi_index(n_thread_data_on_block));

            // shuffle: threadwise copy C from VGPR to LDS
            auto c_thread_copy_vgpr_to_lds =
                ThreadwiseTensorSliceTransfer_v1r3<AccDataType,
                                                   ShuffleDataType,
                                                   decltype(c_thread_desc_m0_n0_m1_n1_m2_n2_n3_n4),
                                                   decltype(c_block_desc_m0_n0_m1_n1_m2_n2_n3_n4),
                                                   tensor_operation::element_wise::PassThrough,
                                                   Sequence<YShuffleMXdlPerWavePerShuffle,
                                                            YShuffleNXdlPerWavePerShuffle,
                                                            I1,
                                                            I1,
                                                            I1,
                                                            N2,
                                                            I1,
                                                            N4>,
                                                   Sequence<0, 1, 2, 3, 4, 5, 6, 7>,
                                                   7,
                                                   1,
                                                   InMemoryDataOperationEnum::Set,
                                                   1,
                                                   true>{
                    c_block_desc_m0_n0_m1_n1_m2_n2_n3_n4,
                    make_multi_index(0,
                                     0,
                                     m_thread_data_on_block_idx[I1],
                                     n_thread_data_on_block_idx[I1],
                                     m_thread_data_on_block_idx[I2],
                                     n_thread_data_on_block_idx[I2],
                                     n_thread_data_on_block_idx[I3],
                                     n_thread_data_on_block_idx[I4]),
                    tensor_operation::element_wise::PassThrough{}};

            // shuffle: blockwise copy C from LDS to global
            auto c_shuffle_block_copy_lds_to_global = ThreadGroupTensorSliceTransfer_v6r1<
                ThisThreadBlock,            // ThreadGroup
                YTensorElementwiseOperation,      // TensorElementwiseOperation,
                CGlobalMemoryDataOperation, // DstInMemOp,
                Sequence<1,
                         YShuffleMXdlPerWavePerShuffle * MWave * MPerXdl,
                         1,
                         YShuffleNXdlPerWavePerShuffle * NWave * NPerXdl>, // BlockSliceLengths,
                YShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
                Sequence<0, 1, 2, 3>, // typename ThreadClusterArrangeOrder,
                ShuffleDataType,        // typename SrcData,
                DataType,             // typename DstData,
                decltype(c_shuffle_block_desc_mblock_mperblock_nblock_nperblock),
                decltype(c_grid_desc_mblock_mperblock_nblock_nperblock),
                Sequence<0, 1, 2, 3>,                           // typename DimAccessOrder,
                3,                                              // index_t VectorDim,
                YShuffleBlockTransferScalarPerVector_NPerBlock, // index_t ScalarPerVector,
                true,  // bool ThreadTransferSrcResetCoordinateAfterRun,
                false> // bool ThreadTransferDstResetCoordinateAfterRun>
                {c_shuffle_block_desc_mblock_mperblock_nblock_nperblock,
                 make_multi_index(0, 0, 0, 0),
                 c_grid_desc_mblock_mperblock_nblock_nperblock,
                 make_multi_index(block_work_idx[I0], 0, block_work_idx[I1], 0),
                 c_element_op};

            // space filling curve for threadwise C in VGPR
            constexpr auto sfc_c_vgpr =
                SpaceFillingCurve<Sequence<MXdlPerWave, OXdlPerWave, 1, 1, 1, N2, 1, N4>,
                                  Sequence<0, 1, 2, 3, 4, 5, 6, 7>,
                                  Sequence<YShuffleMXdlPerWavePerShuffle,
                                           YShuffleNXdlPerWavePerShuffle,
                                           1,
                                           1,
                                           1,
                                           N2,
                                           1,
                                           N4>>{};

            // space filling curve for shuffled blockwise C in global mem
            constexpr auto sfc_c_global =
                SpaceFillingCurve<Sequence<1, Gemm0MPerBlock, 1, Gemm1OPerBlock>,
                                  Sequence<0, 2, 1, 3>,
                                  Sequence<1,
                                           YShuffleMXdlPerWavePerShuffle * MWave * MPerXdl,
                                           1,
                                           YShuffleNXdlPerWavePerShuffle * NWave * NPerXdl>>{};

            constexpr index_t num_access = sfc_c_vgpr.GetNumOfAccess();

            static_assert(num_access == sfc_c_global.GetNumOfAccess(), "wrong!");

            static_for<0, num_access, 1>{}([&](auto access_id) {
                // make sure it's safe to write to LDS
                block_sync_lds();

                // each thread write its data from VGPR to LDS
                c_thread_copy_vgpr_to_lds.Run(c_thread_desc_m0_n0_m1_n1_m2_n2_n3_n4,
                                              sfc_c_vgpr.GetIndexTupleOfNumber(access_id),
                                              c_thread_buf,
                                              c_block_desc_m0_n0_m1_n1_m2_n2_n3_n4,
                                              c_shuffle_block_buf);

                // make sure it's safe to read from LDS
                block_sync_lds();

                // each block copy its data from LDS to global
                c_shuffle_block_copy_lds_to_global.Run(
                    c_shuffle_block_desc_mblock_mperblock_nblock_nperblock,
                    c_shuffle_block_buf,
                    c_grid_desc_mblock_mperblock_nblock_nperblock,
                    c_grid_buf);

                if constexpr(access_id < num_access - 1)
                {
                    constexpr auto c_global_step = sfc_c_global.GetForwardStep(access_id);

                    // move on C
                    c_shuffle_block_copy_lds_to_global.MoveDstSliceWindow(
                        c_grid_desc_mblock_mperblock_nblock_nperblock, c_global_step);
                }
            });
        }
    }
};

} // namespace ck
