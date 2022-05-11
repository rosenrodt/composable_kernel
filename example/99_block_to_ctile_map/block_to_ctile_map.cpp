#include <ck/config.hpp>
#include <ck/utility/block_to_ctile_map.hpp>

#include <iostream>

using namespace ck;

static auto I0 = Number<0>{};
static auto I1 = Number<1>{};
static auto I2 = Number<2>{};
static auto I3 = Number<3>{};

int main()
{
    const index_t M         = 768;
    const index_t N         = 768;
    const index_t MPerBlock = 128;
    const index_t NPerBlock = 128;
    const index_t MBlock    = M / MPerBlock;
    const index_t NBlock    = N / NPerBlock;

    auto c_grid_desc_m0_m1_n0_n1 = make_naive_tensor_descriptor(
        make_tuple(MBlock, MPerBlock, NBlock, NPerBlock), make_tuple(I1, I1, I1, I1));

    std::cout << c_grid_desc_m0_m1_n0_n1.GetLength(I0) << ", "
              << c_grid_desc_m0_m1_n0_n1.GetLength(I1) << ", "
              << c_grid_desc_m0_m1_n0_n1.GetLength(I2) << ", "
              << c_grid_desc_m0_m1_n0_n1.GetLength(I3) << std::endl;

    // clang-format off
    // BlockToCTileMap_M00_N00_M01_N01<decltype(c_grid_desc_m0_m1_n0_n1)> tile_map(c_grid_desc_m0_m1_n0_n1, 4, 4);
    BlockToCTileMap_KSplit_M00_N00_M01_N01<decltype(c_grid_desc_m0_m1_n0_n1)> tile_map(c_grid_desc_m0_m1_n0_n1, 4, 4, 2);
    // BlockToCTileMap_N00_M0_N01Adapt<decltype(c_grid_desc_m0_m1_n0_n1)> tile_map(c_grid_desc_m0_m1_n0_n1, 4);
    // BlockToCTileMap_M00_N0_M01Adapt<decltype(c_grid_desc_m0_m1_n0_n1)> tile_map(c_grid_desc_m0_m1_n0_n1, 4);
    // clang-format on
    for(index_t i = 0; i < tile_map.CalculateGridSize(c_grid_desc_m0_m1_n0_n1); i++)
    {
        auto m0n0_idx = tile_map.CalculateBottomIndex(make_multi_index(i));
        // std::cout << "i = " << i << ", m0, n0 = " << m0n0_idx[I0] << ", " << m0n0_idx[I1];
        std::cout << "i = " << i << ", k, m0, n0 = " << m0n0_idx[I0] << ", " << m0n0_idx[I1] << ", "
                  << m0n0_idx[I2];
        std::cout << ", valid = " << tile_map.ValidCTileIndex(m0n0_idx, c_grid_desc_m0_m1_n0_n1)
                  << std::endl;
    }
    return 0;
}
