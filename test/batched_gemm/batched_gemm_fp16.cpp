#include "profile_batched_gemm_impl.hpp"

#include <iostream>

namespace {
using ADataType = ck::half_t;
using BDataType = ck::half_t;
using CDataType = ck::half_t;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;
} // namespace

int main()
{
    int M          = 512;
    int N          = 256;
    int K          = 128;
    int BatchCount = 3;

    bool pass = true;

    pass = pass &&
           ck::profiler::profile_batched_gemm_impl<ADataType, BDataType, CDataType, Row, Row, Row>(
               true, 1, false, 1, M, N, K, K, N, N, BatchCount);

    pass = pass &&
           ck::profiler::profile_batched_gemm_impl<ADataType, BDataType, CDataType, Row, Col, Row>(
               true, 1, false, 1, M, N, K, K, K, N, BatchCount);

    pass = pass &&
           ck::profiler::profile_batched_gemm_impl<ADataType, BDataType, CDataType, Col, Row, Row>(
               true, 1, false, 1, M, N, K, M, N, N, BatchCount);

    pass = pass &&
           ck::profiler::profile_batched_gemm_impl<ADataType, BDataType, CDataType, Col, Col, Row>(
               true, 1, false, 1, M, N, K, M, K, N, BatchCount);

    std::cout << "test BatchedGEMM fp16: " << (pass ? "Pass" : "Fail") << std::endl;
    return pass ? 0 : 1;
}
