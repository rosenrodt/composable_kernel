#ifndef DEVICE_CONV_BWD_DATA_HPP
#define DEVICE_CONV_BWD_DATA_HPP

#include <iostream>
#include "device_base.hpp"
#include "element_wise_operation.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

template <typename InElementwiseOperation,
          typename WeiElementwiseOperation,
          typename OutElementwiseOperation>
struct DeviceConvBwdData : public BaseOperator
{
    virtual std::unique_ptr<BaseArgument>
    MakeArgumentPointer(void* p_in,
                        const void* p_wei,
                        const void* p_out,
                        ck::index_t N,
                        ck::index_t K,
                        ck::index_t C,
                        std::vector<ck::index_t> input_spatial_lengths,
                        std::vector<ck::index_t> filter_spatial_lengths,
                        std::vector<ck::index_t> output_spatial_lengths,
                        std::vector<ck::index_t> conv_filter_strides,
                        std::vector<ck::index_t> conv_filter_dilations,
                        std::vector<ck::index_t> input_left_pads,
                        std::vector<ck::index_t> input_right_pads,
                        InElementwiseOperation in_element_op,
                        WeiElementwiseOperation wei_element_op,
                        OutElementwiseOperation out_element_op) = 0;

    virtual std::unique_ptr<BaseInvoker> MakeInvokerPointer() = 0;
};

template <typename InElementwiseOperation,
          typename WeiElementwiseOperation,
          typename OutElementwiseOperation>
using DeviceConvBwdDataPtr = std::unique_ptr<
    DeviceConvBwdData<InElementwiseOperation, WeiElementwiseOperation, OutElementwiseOperation>>;

} // namespace device
} // namespace tensor_operation
} // namespace ck
#endif
