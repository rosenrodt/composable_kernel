#ifndef CK_BLOCKWISE_GENERIC_TENSOR_SLICE_COPY_HPP
#define CK_BLOCKWISE_GENERIC_TENSOR_SLICE_COPY_HPP

#include "common_header.hpp"
#include "ConstantTensorDescriptor.hpp"
#include "ConstantMergedTensorDescriptor.hpp"
#include "tensor_coordinate.hpp"
#include "tensor_view.hpp"
#include "threadwise_generic_tensor_slice_copy.hpp"
#include "tensor_descriptor.hpp"
#include "tensor_coordinate_v2.hpp"

#ifndef CK_EXPERIMENTAL_USE_MORE_COMPILE_STATIC_BLOCKWISE_GENERIC_SLICE_COPY_V1
#define CK_EXPERIMENTAL_USE_MORE_COMPILE_STATIC_BLOCKWISE_GENERIC_SLICE_COPY_V1 1
#endif

namespace ck {

// Slice a (normal or merged) tensor, and copy it into another (normal or merged) tensor
// memory layout (ordering of dimensions) can be different between src and dst.
// This functions assume each thread is reading and writing a normal (not merged) tensor,
// to simplify index calculations. To satisfy this assumption, the user need to make sure
// that, on a merged dimension that constains multiple original dimensions, the length of
// the last original dimension need to be evenly dividable by its sub-lengths. Also, the
// repeat-length on the merged dimension need to be 1. These sanity checks are performed
// in constructor of BlockwiseGenericTensorSliceCopy_v1
template <index_t BlockSize,
          typename SrcDesc,
          typename DstDesc,
          typename SliceLengths,
          typename SubLengths,
          typename ThreadClusterLengths,
          typename ThreadClusterArrangeOrder,
          typename SrcDimAccessOrder,
          typename DstDimAccessOrder,
          index_t SrcVectorAccessDim,
          index_t DstVectorAccessDim,
          index_t SrcDataPerAccess,
          index_t DstDataPerAccess>
struct BlockwiseGenericTensorSliceCopy_v1
{
    static constexpr index_t nDim = SrcDesc::GetNumOfDimension();

    static constexpr index_t nOriginalDimSrc =
        SrcDesc::GetOriginalTensorDescriptor().GetNumOfDimension();
    static constexpr index_t nOriginalDimDst =
        DstDesc::GetOriginalTensorDescriptor().GetNumOfDimension();

    // per-thread offset
    index_t mThreadSrcOffset;
    index_t mThreadDstOffset;

    // "mThreadSrcOriginalMultiId", "mThreadSrcPartialOffsets, "mThreadDstOriginalMultiId",
    // "mThreadDstPartialOffsets" are always calculated inside constructor, and would be
    // updated if slicing-window is moved. However, they will not be used if you always move
    // the slicing-window along a non-merged dimension. In that case, compiler should be
    // able to remove these calculation.
    // TODO: make sure compiler would actually remove them in that case

    // partial offset in each (merged) dimension
    Array<index_t, nDim> mThreadSrcPartialOffsets;
    Array<index_t, nDim> mThreadDstPartialOffsets;

    // multi-id of original tensor
    Array<index_t, nOriginalDimSrc> mThreadSrcOriginalMultiId;
    Array<index_t, nOriginalDimDst> mThreadDstOriginalMultiId;

    __device__ BlockwiseGenericTensorSliceCopy_v1(Array<index_t, nDim> src_block_data_id_begin,
                                                  Array<index_t, nDim> dst_block_data_id_begin)
    {
        // check NDim consistency
        static_assert(
            nDim == SrcDesc::GetNumOfDimension() && nDim == DstDesc::GetNumOfDimension() &&
                nDim == SliceLengths::GetSize() && nDim == SubLengths::GetSize() &&
                nDim == ThreadClusterLengths::GetSize() &&
                nDim == ThreadClusterArrangeOrder::GetSize() &&
                nDim == SrcDimAccessOrder::GetSize() && nDim == DstDimAccessOrder::GetSize(),
            "wrong");

        // check thread arrange order and read/write access order are valid
        static_assert(is_valid_sequence_map<ThreadClusterArrangeOrder>::value &&
                          is_valid_sequence_map<SrcDimAccessOrder>::value &&
                          is_valid_sequence_map<DstDimAccessOrder>::value,
                      "wrong!");

        // thread cluster
        constexpr auto thread_cluster_desc = make_ConstantTensorDescriptor_packed(
            ThreadClusterLengths::ReorderGivenNew2Old(ThreadClusterArrangeOrder{}));

        // BlockSize
        static_assert(BlockSize == thread_cluster_desc.GetElementSize(), "wrong! BlockSize");

        // divide work
        constexpr auto data_per_cluster_per_dims = SubLengths{} * ThreadClusterLengths{};

        static_for<0, nDim, 1>{}([&](auto IDim) {
            static_assert(SliceLengths::Get(IDim) % data_per_cluster_per_dims.Get(IDim) == 0,
                          "wrong! cannot evenly divide sliced tensor into cluster");
        });

        constexpr auto repeat_lengths = SliceLengths{} / data_per_cluster_per_dims;

        // additional check for merged dimension
        static_for<0, nDim, 1>{}([&](auto IDim_) {
            // src
            static_if<SrcDesc::ContainMultipleOriginalDimensions(IDim_)>{}([&](auto) {
                constexpr auto IDim = decltype(IDim_){};

                // on a merged dimension that constains multiple original dimensions,
                // the length of the last original dimension need to evenly dividable by its
                // sub-length,
                // so each thread is effectively reading a normal (not merged) tensor
                constexpr auto idim_last_original_src =
                    SrcDesc::GetContainedOriginalDimensions(IDim).Back();
                static_assert(
                    SrcDesc::GetOriginalTensorDescriptor().GetLength(idim_last_original_src) %
                            SubLengths::Get(IDim) ==
                        0,
                    "wrong!");

                // merged dimension should have repeat_lengths = 1
                static_assert(repeat_lengths[IDim] == 1,
                              "wrong! repeat_lengths shoud be 1 on merged dimension");
            });

            // dst
            static_if<DstDesc::ContainMultipleOriginalDimensions(IDim_)>{}([&](auto) {
                constexpr auto IDim = decltype(IDim_){};

                // on a merged dimension that constains multiple original dimensions,
                // the length of the last original dimension need to evenly dividable by its
                // sub-length,
                // so each thread is effectively reading a normal (not merged) tensor
                constexpr auto idim_last_original_dst =
                    DstDesc::GetContainedOriginalDimensions(IDim).Back();
                static_assert(
                    DstDesc::GetOriginalTensorDescriptor().GetLength(idim_last_original_dst) %
                            SubLengths::Get(IDim) ==
                        0,
                    "wrong!");

                // merged dimension should have repeat_lengths = 1
                static_assert(repeat_lengths[IDim] == 1,
                              "wrong! repeat_lengths shoud be 1 on merged dimension");
            });
        });

        // calculate mThreadSrcOffset, mThreadDstOffset
        const auto thread_cluster_id =
            thread_cluster_desc.GetMultiIndexFrom1dIndex(get_thread_local_1d_id());

        const auto data_cluster_id =
            reorder_array_given_old2new(thread_cluster_id, ThreadClusterArrangeOrder{});

        const auto thread_data_id_begin = data_cluster_id * SubLengths{};

        // original multi-id
        mThreadSrcOriginalMultiId = SrcDesc::GetOriginalMultiIndexFromMultiIndex(
            src_block_data_id_begin + thread_data_id_begin);

        mThreadDstOriginalMultiId = DstDesc::GetOriginalMultiIndexFromMultiIndex(
            dst_block_data_id_begin + thread_data_id_begin);

        // partial offset on each dimension
        static_for<0, nDim, 1>{}([&](auto IDim) {
            constexpr auto src_partial_original_dims =
                SrcDesc::GetContainedOriginalDimensions(IDim);

            constexpr auto src_partial_original_desc =
                SrcDesc::GetOriginalTensorDescriptor().Extract(src_partial_original_dims);

            mThreadSrcPartialOffsets(IDim) = src_partial_original_desc.GetOffsetFromMultiIndex(
                extract_array(mThreadSrcOriginalMultiId, src_partial_original_dims));
        });

        static_for<0, nDim, 1>{}([&](auto IDim) {
            constexpr auto dst_partial_original_dims =
                DstDesc::GetContainedOriginalDimensions(IDim);

            constexpr auto dst_partial_original_desc =
                DstDesc::GetOriginalTensorDescriptor().Extract(dst_partial_original_dims);

            mThreadDstPartialOffsets(IDim) = dst_partial_original_desc.GetOffsetFromMultiIndex(
                extract_array(mThreadDstOriginalMultiId, dst_partial_original_dims));
        });

        // complete offset
        mThreadSrcOffset = accumulate_on_array(
            mThreadSrcPartialOffsets, math::plus<index_t>{}, static_cast<index_t>(0));

        mThreadDstOffset = accumulate_on_array(
            mThreadDstPartialOffsets, math::plus<index_t>{}, static_cast<index_t>(0));
    }

    __device__ static constexpr auto GetRegisterBufferDescriptor()
    {
        constexpr auto repeat_lengths = SliceLengths{} / (SubLengths{} * ThreadClusterLengths{});

        return make_ConstantTensorDescriptor_packed(SubLengths{} * repeat_lengths);
    }

    __device__ static constexpr index_t GetRegisterBufferSize()
    {
        return GetRegisterBufferDescriptor().GetElementSpace();
    }

    template <typename TData>
    __device__ void RunLoadRegisterBuffer(const TData* __restrict__ p_src,
                                          TData* __restrict__ p_buffer) const
    {
        constexpr auto thread_sub_tensor_lengths = SubLengths{};

        constexpr auto data_per_cluster_per_dims =
            thread_sub_tensor_lengths * ThreadClusterLengths{};

        constexpr auto repeat_lengths = SliceLengths{} / (SubLengths{} * ThreadClusterLengths{});

        constexpr auto thread_buffer_desc = GetRegisterBufferDescriptor();

#if CK_EXPERIMENTAL_USE_MORE_COMPILE_STATIC_BLOCKWISE_GENERIC_SLICE_COPY_V1
        static_ford<decltype(repeat_lengths)>{}([&](auto repeat_id) {
            constexpr auto src_thread_data_id_begin = repeat_id * data_per_cluster_per_dims;

            constexpr auto buffer_data_id_begin = repeat_id * thread_sub_tensor_lengths;

            constexpr index_t src_offset =
                SrcDesc::GetOffsetFromMultiIndex(src_thread_data_id_begin);

            constexpr index_t buffer_offset =
                thread_buffer_desc.GetOffsetFromMultiIndex(buffer_data_id_begin);
#else
        ford<decltype(repeat_lengths)>{}([&](auto repeat_id) {
            const auto src_thread_data_id_begin = repeat_id * data_per_cluster_per_dims;

            const auto buffer_data_id_begin = repeat_id * thread_sub_tensor_lengths;

            const index_t src_offset = SrcDesc::GetOffsetFromMultiIndex(src_thread_data_id_begin);

            const index_t buffer_offset =
                thread_buffer_desc.GetOffsetFromMultiIndex(buffer_data_id_begin);
#endif

            // By position the origin of the per-thread window at the point, where multi-index
            // of the SrcDesc (might be a merged tensor) is all-zero. This threadwise slice copy
            // is assuming each thread is copy a noraml (not merged) tensor.
            // To satisfy this assumption, the user need to make sure that, on a merged dimension
            // that constains multiple original dimensions, the length of the last original
            // dimension need to be evenly dividable by its sub-lengths. Also, the repeat-length on
            // the merged dimension need to be 1. These sanity checks are performed in constructor
            // of BlockwiseGenericTensorSliceCopy_v1
            ThreadwiseGenericTensorSliceCopy_v1r2<SrcDesc,
                                                  decltype(thread_buffer_desc),
                                                  SubLengths,
                                                  SrcDimAccessOrder,
                                                  SrcVectorAccessDim,
                                                  SrcDataPerAccess,
                                                  1>(make_zero_array<index_t, nDim>(),
                                                     make_zero_array<index_t, nDim>())
                .Run(p_src + src_offset + mThreadSrcOffset, p_buffer + buffer_offset);
        });
    }

    template <typename TData>
    __device__ void RunStoreRegisterBuffer(const TData* __restrict__ p_buffer,
                                           TData* __restrict__ p_dst) const
    {
        constexpr auto thread_sub_tensor_lengths = SubLengths{};

        constexpr auto data_per_cluster_per_dims =
            thread_sub_tensor_lengths * ThreadClusterLengths{};

        constexpr auto repeat_lengths = SliceLengths{} / (SubLengths{} * ThreadClusterLengths{});

        constexpr auto thread_buffer_desc = GetRegisterBufferDescriptor();

#if CK_EXPERIMENTAL_USE_MORE_COMPILE_STATIC_BLOCKWISE_GENERIC_SLICE_COPY_V1
        static_ford<decltype(repeat_lengths)>{}([&](auto repeat_id) {
            constexpr auto buffer_data_id_begin = repeat_id * thread_sub_tensor_lengths;

            constexpr auto dst_data_id_begin = repeat_id * data_per_cluster_per_dims;

            constexpr index_t buffer_offset =
                thread_buffer_desc.GetOffsetFromMultiIndex(buffer_data_id_begin);

            constexpr index_t dst_offset = DstDesc::GetOffsetFromMultiIndex(dst_data_id_begin);
#else
        ford<decltype(repeat_lengths)>{}([&](auto repeat_id) {
            const auto buffer_data_id_begin = repeat_id * thread_sub_tensor_lengths;

            const auto dst_data_id_begin = repeat_id * data_per_cluster_per_dims;

            const index_t buffer_offset =
                thread_buffer_desc.GetOffsetFromMultiIndex(buffer_data_id_begin);

            const index_t dst_offset = DstDesc::GetOffsetFromMultiIndex(dst_data_id_begin);
#endif

            // By position the origin of the per-thread window at the point, where multi-index
            // of the SrcDesc (might be a merged tensor) is all-zero. This threadwise slice copy
            // is assuming each thread is copy a noraml (not merged) tensor.
            // To satisfy this assumption, the user need to make sure that, on a merged dimension
            // that constains multiple original dimensions, the length of the last original
            // dimension need to be evenly dividable by its sub-lengths. Also, the repeat-length on
            // the merged dimension need to be 1. These sanity checks are performed in constructor
            // of BlockwiseGenericTensorSliceCopy_v1
            ThreadwiseGenericTensorSliceCopy_v1r2<decltype(thread_buffer_desc),
                                                  DstDesc,
                                                  SubLengths,
                                                  DstDimAccessOrder,
                                                  DstVectorAccessDim,
                                                  1,
                                                  DstDataPerAccess>(
                make_zero_array<index_t, nDim>(), make_zero_array<index_t, nDim>())
                .Run(p_buffer + buffer_offset, p_dst + dst_offset + mThreadDstOffset);
        });
    }

    template <typename TData>
    __device__ void Run(const TData* __restrict__ p_src, TData* __restrict__ p_dst) const
    {
        TData p_buffer[GetRegisterBufferSize()];

        RunLoadRegisterBuffer(p_src, p_buffer);
        RunStoreRegisterBuffer(p_buffer, p_dst);
    }

    // When moving the slicing windows along a merged dimension, if the strides of the
    // contained (by the merged dimension) original dimensions are not in descending order,
    // then there is no guarantee that the new offset will be larger than the old offset
    // for movement in positive direction (vice versue for movement in negative direction).
    // As a result, there is the possiblity that the offset calculation may result in
    // unsigned integer underflow (due to "-" operation). However, this hazard should not
    // happen, as long as the users make sure the slicing window would not be moved out of
    // the boundary of the tensor being sliced. This functions doesn't do runtime sanity
    // check on out-of-bound slicing window, for performance reason
    template <index_t IDim_, index_t StepSize, bool PositiveDirection>
    __device__ void MoveSlicingWindowOnSourceTensor(
        Number<IDim_>, Number<StepSize>, integral_constant<bool, PositiveDirection> direction)
    {
        constexpr auto IDim = Number<IDim_>{};

        static_if<SrcDesc::ContainMultipleOriginalDimensions(IDim)>{}([&](auto) {
            // logic for a merged dimension, also works for non-merged dimension, but its logic may
            // be unncessarily complicated for compiler to remove calculations that are useless for
            // a non-merged dimension

            // extract partial original dimensions
            constexpr auto src_partial_original_dims =
                SrcDesc::GetContainedOriginalDimensions(IDim);

            constexpr auto src_partial_original_desc =
                SrcDesc::GetOriginalTensorDescriptor().Extract(src_partial_original_dims);

            // calculate new partial original multi-id
            auto old_src_partial_original_id =
                extract_array(mThreadSrcOriginalMultiId, src_partial_original_dims);

            auto new_src_partial_original_id =
                src_partial_original_desc.UpdateMultiIndexGivenStepSizeOf1dIndex(
                    old_src_partial_original_id, StepSize, direction);

            // update "mThreadSrcOriginalMultiId"
            static_for<0, decltype(src_partial_original_dims)::GetSize(), 1>{}([&](auto I) {
                constexpr auto IDimOriginal = src_partial_original_dims[I];

                mThreadSrcOriginalMultiId(IDimOriginal) = new_src_partial_original_id[I];
            });

            // calculate new partial offset on this merged dimension
            const index_t old_src_partial_offset = mThreadSrcPartialOffsets[IDim];

            const index_t new_src_partial_offset =
                src_partial_original_desc.GetOffsetFromMultiIndex(new_src_partial_original_id);

            // update "mThreadSrcPartialOffsets"
            mThreadSrcPartialOffsets(IDim) = new_src_partial_offset;

            // update "mThreadSrcOffset", do "+" before "-" to avoid underflow
            mThreadSrcOffset = (mThreadSrcOffset + new_src_partial_offset) - old_src_partial_offset;
        }).Else([&](auto) {
            // Logic for non-merged dimension. If you are never going to move the slicing window on
            // a merged dimension, then "mThreadSrcOriginalMultiId" and "mThreadSrcPartialOffsets",
            // which are being calculated here, will never be used later. In this case, compiler
            // should be able to remove these calculations.
            // TODO: make sure compiler would actually remove them in this case.

            // It is the user's responsiblity to make sure the slicing window will not be moved out
            // of the boundary of the tensor being sliced. Otherwise, there might be hazard like
            // unsigned integer underflow. That is NO runtime sanity check to prevent the hazard

            constexpr auto IDimOriginal = SrcDesc::GetContainedOriginalDimensions(IDim).Front();

            static_if<PositiveDirection>{}([&](auto fwd) {
                mThreadSrcOffset += StepSize * fwd(SrcDesc{}).GetStride(IDim);

                mThreadSrcOriginalMultiId(IDimOriginal) += StepSize;

                mThreadSrcPartialOffsets(IDim) += StepSize * fwd(SrcDesc{}).GetStride(IDim);
            }).Else([&](auto fwd) {
                mThreadSrcOffset -= StepSize * fwd(SrcDesc{}).GetStride(IDim);

                mThreadSrcOriginalMultiId(IDimOriginal) -= StepSize;

                mThreadSrcPartialOffsets(IDim) -= StepSize * fwd(SrcDesc{}).GetStride(IDim);
            });
        });
    }

    template <typename T, bool PositiveDirection>
    __device__ void
    MoveSrcSliceWindow(T step_sizes, integral_constant<bool, PositiveDirection> positive_direction)
    {
        static_for<0, nDim, 1>{}([&](auto idim) {
            if(step_sizes[idim] != 0)
            {
                MoveSlicingWindowOnSourceTensor(idim, step_sizes[idim], positive_direction);
            }
        });
    }
};

// This version use TensorCoordiante
// Slice a (normal or merged) tensor, and copy it into another (normal or merged) tensor
// memory layout (ordering of dimensions) can be different between src and dst.
template <index_t BlockSize,
          typename SrcDesc,
          typename DstDesc,
          typename SliceLengths,
          typename SubLengths,
          typename ThreadClusterLengths,
          typename ThreadClusterArrangeOrder,
          typename SrcDimAccessOrder,
          typename DstDimAccessOrder,
          index_t SrcVectorAccessDim,
          index_t DstVectorAccessDim,
          index_t SrcDataPerAccess,
          index_t DstDataPerAccess>
struct BlockwiseGenericTensorSliceCopy_v2
{
    static constexpr index_t nDim = SrcDesc::GetNumOfDimension();

    using Index = MultiIndex<nDim>;

    __device__ constexpr BlockwiseGenericTensorSliceCopy_v2(const Index& src_block_slice_origin,
                                                            const Index& dst_block_slice_origin)
    {
        static_assert(
            nDim == SrcDesc::GetNumOfDimension() && nDim == DstDesc::GetNumOfDimension() &&
                nDim == SliceLengths::GetSize() && nDim == SubLengths::GetSize() &&
                nDim == ThreadClusterLengths::GetSize() &&
                nDim == ThreadClusterArrangeOrder::GetSize() &&
                nDim == SrcDimAccessOrder::GetSize() && nDim == DstDimAccessOrder::GetSize(),
            "wrong! nDim not consistent");

        static_assert(is_same<SliceLengths, decltype(SubLengths{} * ThreadClusterLengths{})>{},
                      "wrong! threads should be mapped to cover entire slicing window");

        constexpr auto thread_cluster_desc = make_ConstantTensorDescriptor_packed(
            ThreadClusterLengths::ReorderGivenNew2Old(ThreadClusterArrangeOrder{}));

        static_assert(BlockSize == thread_cluster_desc.GetElementSize(),
                      "wrong! BlockSize not consistent with ThreadClusterLengths");

        const auto thread_cluster_id =
            thread_cluster_desc.GetMultiIndexFrom1dIndex(get_thread_local_1d_id());

        const auto data_cluster_id =
            reorder_array_given_old2new(thread_cluster_id, ThreadClusterArrangeOrder{});

        const auto thread_data_id_begin = data_cluster_id * SubLengths{};

        mThreadwiseLoad.SetSrcSliceOrigin(src_block_slice_origin + thread_data_id_begin);
        mThreadwiseLoad.SetDstSliceOrigin(make_zero_array<index_t, nDim>());

        mThreadwiseStore.SetSrcSliceOrigin(make_zero_array<index_t, nDim>());
        mThreadwiseStore.SetDstSliceOrigin(dst_block_slice_origin + thread_data_id_begin);
    }

    __device__ static constexpr index_t GetRegisterBufferSize()
    {
        return RegisterBufferDesc::GetElementSpace();
    }

    template <typename TData>
    __device__ void RunLoadRegisterBuffer(const TData* p_src, TData* p_buffer) const
    {
#if 0
        mThreadwiseLoad.Run(p_src, p_buffer);
#else
        // hardcoded: global to register
        mThreadwiseLoad.template Run_amd_experiment<TData, 2, 0>(p_src, p_buffer);
#endif
    }

    template <typename TData>
    __device__ void RunStoreRegisterBuffer(const TData* p_buffer, TData* p_dst) const
    {
#if 0
        mThreadwiseStore.Run(p_buffer, p_dst);
#else
        // hardcoded: register to LDS
        mThreadwiseStore.template Run_amd_experiment<TData, 0, 1>(p_buffer, p_dst);
#endif
    }

    template <typename TData>
    __device__ void Run(const TData* p_src, TData* p_dst) const
    {
        TData p_buffer[GetRegisterBufferSize()];

        RunLoadRegisterBuffer(p_src, p_buffer);
        RunStoreRegisterBuffer(p_buffer, p_dst);
    }

    template <typename T, bool PositiveDirection>
    __device__ void
    MoveSrcSliceWindow(T step_sizes, integral_constant<bool, PositiveDirection> positive_direction)
    {
        mThreadwiseLoad.MoveSrcSliceWindow(step_sizes, positive_direction);
    }

    template <typename T, bool PositiveDirection>
    __device__ void
    MoveDstSliceWindow(T step_sizes, integral_constant<bool, PositiveDirection> positive_direction)
    {
        mThreadwiseStore.MoveDstSliceWindow(step_sizes, positive_direction);
    }

    private:
    using RegisterBufferDesc = decltype(make_ConstantTensorDescriptor_packed(SubLengths{}));

    using ThreadwiseLoad = ThreadwiseGenericTensorSliceCopy_v2r1<SrcDesc,
                                                                 RegisterBufferDesc,
                                                                 SubLengths,
                                                                 SrcDimAccessOrder,
                                                                 SrcDimAccessOrder,
                                                                 SrcVectorAccessDim,
                                                                 SrcVectorAccessDim,
                                                                 SrcDataPerAccess,
                                                                 1>;

    using ThreadwiseStore = ThreadwiseGenericTensorSliceCopy_v2r1<RegisterBufferDesc,
                                                                  DstDesc,
                                                                  SubLengths,
                                                                  DstDimAccessOrder,
                                                                  DstDimAccessOrder,
                                                                  DstVectorAccessDim,
                                                                  DstVectorAccessDim,
                                                                  1,
                                                                  DstDataPerAccess>;

    ThreadwiseLoad mThreadwiseLoad;
    ThreadwiseStore mThreadwiseStore;
};

// this version use TensorView and TensorCoordinate
template <index_t BlockSize,
          typename SrcTensor,
          typename DstTensor,
          typename SliceLengths,
          typename SubLengths,
          typename ThreadClusterLengths,
          typename ThreadClusterArrangeOrder,
          typename SrcDimAccessOrder,
          typename DstDimAccessOrder,
          index_t SrcVectorAccessDim,
          index_t DstVectorAccessDim,
          index_t SrcDataPerAccess,
          index_t DstDataPerAccess>
struct BlockwiseGenericTensorSliceCopy_v3
{
    static constexpr index_t nDim = SrcTensor::GetNumOfDimension();
    using data_type               = remove_cv_t<typename SrcTensor::data_type>;

    using SrcCoordinate = typename SrcTensor::coordinate_type;
    using DstCoordinate = typename DstTensor::coordinate_type;

    __device__ constexpr BlockwiseGenericTensorSliceCopy_v3(SrcTensor src_block,
                                                            SrcCoordinate src_block_slice_origin,
                                                            DstTensor dst_block,
                                                            DstCoordinate dst_block_slice_origin)
        : mThreadBuffer{make_TensorView(ThreadBufferDesc{}, mpBuffer)}
    {
        static_assert(
            nDim == SrcTensor::GetNumOfDimension() && nDim == DstTensor::GetNumOfDimension() &&
                nDim == SliceLengths::GetSize() && nDim == SubLengths::GetSize() &&
                nDim == ThreadClusterLengths::GetSize() &&
                nDim == ThreadClusterArrangeOrder::GetSize() &&
                nDim == SrcDimAccessOrder::GetSize() && nDim == DstDimAccessOrder::GetSize(),
            "wrong! nDim not consistent");

        static_assert(is_same<SliceLengths, decltype(SubLengths{} * ThreadClusterLengths{})>{},
                      "wrong! threads should be mapped to cover entire slicing window");

        static_assert(is_same<remove_cv_t<typename SrcTensor::data_type>,
                              remove_cv_t<typename DstTensor::data_type>>{},
                      "wrong! type conversion not supported yet");

        constexpr auto thread_cluster_desc = make_ConstantTensorDescriptor_packed(
            ThreadClusterLengths::ReorderGivenNew2Old(ThreadClusterArrangeOrder{}));

        static_assert(BlockSize == thread_cluster_desc.GetElementSize(),
                      "wrong! BlockSize not consistent with ThreadClusterLengths");

        const auto thread_cluster_id =
            thread_cluster_desc.GetMultiIndexFrom1dIndex(get_thread_local_1d_id());

        const auto data_cluster_id =
            reorder_array_given_old2new(thread_cluster_id, ThreadClusterArrangeOrder{});

        const auto thread_data_id_begin = data_cluster_id * SubLengths{};

        mThreadwiseLoad = ThreadwiseLoad(src_block,
                                         src_block_slice_origin + thread_data_id_begin,
                                         mThreadBuffer,
                                         make_zero_array<index_t, nDim>());

        mThreadwiseStore = ThreadwiseStore(mThreadBuffer,
                                           make_zero_array<index_t, nDim>(),
                                           dst_block,
                                           dst_block_slice_origin + thread_data_id_begin);
    }

    __device__ void RunLoadRegisterBuffer() { mThreadwiseLoad.Run(); }

    __device__ void RunStoreRegisterBuffer() const { mThreadwiseStore.Run(); }

    __device__ void Run()
    {
        mThreadwiseLoad.Run();
        mThreadwiseStore.Run();
    }

    template <typename T, bool PositiveDirection>
    __device__ void
    MoveSrcSliceWindow(T step_sizes, integral_constant<bool, PositiveDirection> positive_direction)
    {
        mThreadwiseLoad.MoveSrcSliceWindow(step_sizes, positive_direction);
    }

    template <typename T, bool PositiveDirection>
    __device__ void
    MoveDstSliceWindow(T step_sizes, integral_constant<bool, PositiveDirection> positive_direction)
    {
        mThreadwiseStore.MoveDstSliceWindow(step_sizes, positive_direction);
    }

    private:
    using ThreadBufferDesc   = decltype(make_ConstantTensorDescriptor_packed(SubLengths{}));
    using ThreadBufferTensor = NormalTensorView<ThreadBufferDesc, data_type>;

    using ThreadwiseLoad = ThreadwiseGenericTensorSliceCopy_v3r1<SrcTensor,
                                                                 ThreadBufferTensor,
                                                                 SubLengths,
                                                                 SrcDimAccessOrder,
                                                                 SrcDimAccessOrder,
                                                                 SrcVectorAccessDim,
                                                                 SrcVectorAccessDim,
                                                                 SrcDataPerAccess,
                                                                 1>;

    using ThreadwiseStore = ThreadwiseGenericTensorSliceCopy_v3r1<ThreadBufferTensor,
                                                                  DstTensor,
                                                                  SubLengths,
                                                                  DstDimAccessOrder,
                                                                  DstDimAccessOrder,
                                                                  DstVectorAccessDim,
                                                                  DstVectorAccessDim,
                                                                  1,
                                                                  DstDataPerAccess>;

    data_type mpBuffer[ThreadBufferDesc::GetElementSpace()];

    ThreadBufferTensor mThreadBuffer;

    ThreadwiseLoad mThreadwiseLoad;
    ThreadwiseStore mThreadwiseStore;
};

template <index_t BlockSize,
          typename SrcDesc,
          typename DstDesc,
          typename SliceLengths,
          typename SubLengths,
          typename ThreadClusterLengths,
          typename ThreadClusterArrangeOrder,
          typename SrcDimAccessOrder,
          typename DstDimAccessOrder,
          index_t SrcVectorAccessDim,
          index_t DstVectorAccessDim,
          index_t SrcDataPerAccess,
          index_t DstDataPerAccess>
struct BlockwiseGenericTensorSliceCopy_v4
{
    static constexpr index_t nDim = SrcDesc::GetNumOfDimension();
    using Index                   = MultiIndex<nDim>;

    __device__ constexpr BlockwiseGenericTensorSliceCopy_v4(const Index& src_block_slice_origin,
                                                            const Index& dst_block_slice_origin)
    {
        static_assert(nDim == SrcDesc::GetNumOfDimension() &&
                          nDim == DstDesc::GetNumOfDimension() && nDim == SliceLengths::Size() &&
                          nDim == SubLengths::Size() && nDim == ThreadClusterLengths::Size() &&
                          nDim == ThreadClusterArrangeOrder::Size() &&
                          nDim == SrcDimAccessOrder::Size() && nDim == DstDimAccessOrder::Size(),
                      "wrong! nDim not consistent");

        static_assert(is_same<SliceLengths, decltype(SubLengths{} * ThreadClusterLengths{})>{},
                      "wrong! threads should be mapped to cover entire slicing window");

        constexpr auto thread_cluster_desc = make_ConstantTensorDescriptor_packed(
            ThreadClusterLengths::ReorderGivenNew2Old(ThreadClusterArrangeOrder{}));

        static_assert(BlockSize == thread_cluster_desc.GetElementSize(),
                      "wrong! BlockSize not consistent with ThreadClusterLengths");

        const auto thread_cluster_id =
            thread_cluster_desc.GetMultiIndexFrom1dIndex(get_thread_local_1d_id());

        const auto data_cluster_id =
            reorder_array_given_old2new(thread_cluster_id, ThreadClusterArrangeOrder{});

        const auto thread_data_id_begin = data_cluster_id * SubLengths{};

        mThreadwiseLoad.SetSrcSliceOrigin(src_block_slice_origin + thread_data_id_begin);
        mThreadwiseLoad.SetDstSliceOrigin(make_zero_array<index_t, nDim>());

        mThreadwiseStore.SetSrcSliceOrigin(make_zero_array<index_t, nDim>());
        mThreadwiseStore.SetDstSliceOrigin(dst_block_slice_origin + thread_data_id_begin);
    }

    __device__ static constexpr index_t GetRegisterBufferSize()
    {
        return RegisterBufferDesc::GetElementSpace();
    }

    template <typename SrcData, typename BufferData, address_space_t SrcAddressSpace = address_space_t::generic>
    __device__ void RunLoadRegisterBuffer(const SrcData* p_src, BufferData* p_buffer) const
    {
#if 1
        mThreadwiseLoad.template Run_generic<SrcData, BufferData, SrcAddressSpace, address_space_t::generic>(
            p_src, p_buffer);
#else
        mThreadwiseLoad.template Run_optimized_src_address_calculation<SrcData,
                                                                       BufferData,
                                                                       SrcAddressSpace,
                                                                       address_space_t::generic>(
            p_src, p_buffer);
#endif
    }

    template <typename BufferData, typename DstData, address_space_t DstAddressSpace = address_space_t::generic>
    __device__ void RunStoreRegisterBuffer(const BufferData* p_buffer, DstData* p_dst) const
    {
#if 1
        mThreadwiseStore.template Run_generic<BufferData, DstData, address_space_t::generic, DstAddressSpace>(
            p_buffer, p_dst);
#else
        mThreadwiseStore.template Run_optimized_dst_address_calculation<BufferData,
                                                                        DstData,
                                                                        address_space_t::generic,
                                                                        DstAddressSpace>(p_buffer,
                                                                                         p_dst);
#endif
    }

    template <typename SrcData,
              typename DstData,
              address_space_t SrcAddressSpace = address_space_t::generic,
              address_space_t DstAddressSpace = address_space_t::generic>
    __device__ void Run(const SrcData* p_src, DstData* p_dst) const
    {
        SrcData p_src_buffer[GetRegisterBufferSize()];

        RunLoadRegisterBuffer<SrcData, SrcData, SrcAddressSpace>(p_src, p_buffer);
        RunStoreRegisterBuffer<SrcData, DstData, DstAddressSpace>(p_buffer, p_dst);
    }

    template <typename T, bool PositiveDirection>
    __device__ void
    MoveSrcSliceWindow(const T& step_sizes,
                       integral_constant<bool, PositiveDirection> positive_direction)
    {
        mThreadwiseLoad.MoveSrcSliceWindow(step_sizes, positive_direction);
    }

    template <typename T, bool PositiveDirection>
    __device__ void
    MoveDstSliceWindow(const T& step_sizes,
                       integral_constant<bool, PositiveDirection> positive_direction)
    {
        mThreadwiseStore.MoveDstSliceWindow(step_sizes, positive_direction);
    }

    private:
    using RegisterBufferDesc = decltype(make_native_tensor_descriptor_packed(SubLengths{}));

    using ThreadwiseLoad = ThreadwiseGenericTensorSliceCopy_v4r2<SrcDesc,
                                                                 RegisterBufferDesc,
                                                                 SubLengths,
                                                                 SrcDimAccessOrder,
                                                                 SrcVectorAccessDim,
                                                                 SrcDataPerAccess,
                                                                 1>;

    using ThreadwiseStore = ThreadwiseGenericTensorSliceCopy_v4r2<RegisterBufferDesc,
                                                                  DstDesc,
                                                                  SubLengths,
                                                                  DstDimAccessOrder,
                                                                  DstVectorAccessDim,
                                                                  1,
                                                                  DstDataPerAccess>;

    ThreadwiseLoad mThreadwiseLoad;
    ThreadwiseStore mThreadwiseStore;
};

} // namespace ck

#endif
