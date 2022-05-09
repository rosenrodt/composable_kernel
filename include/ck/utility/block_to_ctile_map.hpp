#ifndef UTILITY_BLOCK_TO_CTILE_MAP
#define UTILITY_BLOCK_TO_CTILE_MAP

#include "utility/math.hpp"
#include "utility/number.hpp"
#include "tensor_description/tensor_adaptor.hpp"
#include "tensor_description/multi_index_transform_helper.hpp"

namespace ck {

// Blocks of row-vectors
template <index_t MPerBlock, index_t NPerBlock, typename CGridDesc_M_N>
struct BlockToCTileMap_M00_N00_M01_N01
{
    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};

    __host__ BlockToCTileMap_M00_N00_M01_N01(const CGridDesc_M_N& c_grid_desc_m_n,
                                             index_t M01 = 1,
                                             index_t N01 = 1)
        : M01_(M01), N01_(N01), underlying_map_(GetBlockToCTileMap(c_grid_desc_m_n, M01, N01))
    {
    }

    __host__ constexpr index_t CalculateGridSize(const CGridDesc_M_N& c_grid_desc_m_n) const
    {
        const auto M = c_grid_desc_m_n.GetLength(I0);
        const auto N = c_grid_desc_m_n.GetLength(I1);

        const auto M0 = math::integer_divide_ceil(M, MPerBlock);
        const auto N0 = math::integer_divide_ceil(N, NPerBlock);

        const auto M00 = math::integer_divide_ceil(M0, M01_);
        const auto N00 = math::integer_divide_ceil(N0, N01_);

        const index_t grid_size = M00 * M01_ * N00 * N01_;

        return grid_size;
    }

    template <typename TopIdx>
    __host__ __device__ constexpr auto CalculateBottomIndex(const TopIdx& idx_top) const
    {
        return underlying_map_.CalculateBottomIndex(idx_top);
    }

    template <typename CTileIdx, typename CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock>
    __host__ __device__ bool ValidCTileIndex(
        const CTileIdx& c_tile_idx,
        const CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock& c_grid_desc_m0_m1_n0_n1) const
    {
        return DefaultValidCTileIndex(c_tile_idx, c_grid_desc_m0_m1_n0_n1);
    }

    private:
    template <typename CGridDesc_M_N_>
    __host__ static constexpr auto
    GetBlockToCTileMap(const CGridDesc_M_N_& c_grid_desc_m_n, index_t M01, index_t N01)
    {
        const auto M = c_grid_desc_m_n.GetLength(I0);
        const auto N = c_grid_desc_m_n.GetLength(I1);

        constexpr auto M1 = Number<MPerBlock>{};
        constexpr auto N1 = Number<NPerBlock>{};

        const auto M0 = M / M1;
        const auto N0 = N / N1;

        const auto M00 = M0 / M01;
        const auto N00 = N0 / N01;

        const auto m00_m01_n00_n01_to_m0_n0_block_cluster_adaptor =
            make_single_stage_tensor_adaptor(
                make_tuple(make_unmerge_transform(make_tuple(M00, M01)),
                           make_unmerge_transform(make_tuple(N00, N01))),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0, 2>{}, Sequence<1, 3>{}));

        const auto cblockid_to_m00_m01_n00_n01_block_cluster_adaptor =
            make_single_stage_tensor_adaptor(
                make_tuple(make_merge_transform(make_tuple(M00, N00, M01, N01))),
                make_tuple(Sequence<0, 1, 2, 3>{}),
                make_tuple(Sequence<0>{}));

        const auto cblockid_to_m0_n0_block_cluster_adaptor =
            chain_tensor_adaptors(m00_m01_n00_n01_to_m0_n0_block_cluster_adaptor,
                                  cblockid_to_m00_m01_n00_n01_block_cluster_adaptor);

        return cblockid_to_m0_n0_block_cluster_adaptor;
    }

    index_t M01_, N01_;
    using UnderlyingMap = decltype(GetBlockToCTileMap(CGridDesc_M_N{}, 1, 1));
    UnderlyingMap underlying_map_;
};

template <typename CTileIdx, typename CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock>
__host__ __device__ bool
DefaultValidCTileIndex(const CTileIdx& c_tile_idx,
                       const CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock& c_grid_desc_m0_m1_n0_n1)
{
    bool is_valid = false;

    const auto m_block =
        __builtin_amdgcn_readfirstlane(c_grid_desc_m0_m1_n0_n1.GetLength(Number<0>{}));
    const auto n_block =
        __builtin_amdgcn_readfirstlane(c_grid_desc_m0_m1_n0_n1.GetLength(Number<2>{}));

    const index_t m_block_idx = __builtin_amdgcn_readfirstlane(c_tile_idx[Number<0>{}]);
    const index_t n_block_idx = __builtin_amdgcn_readfirstlane(c_tile_idx[Number<1>{}]);

    if(0 <= m_block_idx && m_block_idx < m_block && 0 <= n_block_idx && n_block_idx < n_block)
        is_valid = true;

    return is_valid;
}

} // namespace ck

#endif // UTILITY_BLOCK_TO_CTILE_MAP
