// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include <cassert>

#include "ck/library/utility/host_tensor.hpp"

std::vector<size_t> CalculateStrides(std::vector<size_t> lens)
{
    std::vector<size_t> strides;
    strides.clear();
    strides.resize(lens.size(), 0);
    strides.back() = 1;
    std::partial_sum(
        lens.rbegin(), lens.rend() - 1, strides.rbegin() + 1, std::multiplies<std::size_t>());

    return strides;
}

std::size_t GetElementSize(std::vector<std::size_t> lens, std::vector<std::size_t> strides)
{
    if (strides.empty()) strides = CalculateStrides(lens);
    assert(lens.size() == strides.size());
    return std::accumulate(
        lens.begin(), lens.end(), std::size_t{1}, std::multiplies<std::size_t>());
}

std::size_t GetElementSpaceSize(std::vector<std::size_t> lens, std::vector<std::size_t> strides)
{
    if (strides.empty()) strides = CalculateStrides(lens);
    std::size_t space = 1;
    for(std::size_t i = 0; i < lens.size(); ++i)
    {
        space += (lens[i] - 1) * strides[i];
    }
    return space;
}

void HostTensorDescriptor::CalculateStrides()
{
    mStrides = ::CalculateStrides(mLens);
}

std::size_t HostTensorDescriptor::GetNumOfDimension() const { return mLens.size(); }

std::size_t HostTensorDescriptor::GetElementSize() const
{
    return ::GetElementSize(mLens, mStrides);
}

std::size_t HostTensorDescriptor::GetElementSpaceSize() const
{
    return ::GetElementSpaceSize(mLens, mStrides);
}

const std::vector<std::size_t>& HostTensorDescriptor::GetLengths() const { return mLens; }

const std::vector<std::size_t>& HostTensorDescriptor::GetStrides() const { return mStrides; }

std::ostream& operator<<(std::ostream& os, const HostTensorDescriptor& desc)
{
    os << "dim " << desc.GetNumOfDimension() << ", ";

    os << "lengths {";
    LogRange(os, desc.GetLengths(), ", ");
    os << "}, ";

    os << "strides {";
    LogRange(os, desc.GetStrides(), ", ");
    os << "}";

    return os;
}
