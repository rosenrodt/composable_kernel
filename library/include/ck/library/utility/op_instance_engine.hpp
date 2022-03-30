#pragma once

#include <memory>
#include <tuple>

#include "check_err.hpp"
#include "device_base.hpp"
#include "functional2.hpp"

namespace ck {
namespace utils {

struct ProfileBestConfig
{
    std::string best_op_name;
    float best_avg_time   = std::numeric_limits<float>::max();
    float best_tflops     = std::numeric_limits<float>::max();
    float best_gb_per_sec = std::numeric_limits<float>::max();
};

/**
 * @brief      This class describes an operation instance(s).
 *
 *             Op instance defines a particular specializations of operator
 *             template. Thanks to this specific input/output data types, data
 *             layouts and modifying elementwise operations it is able to create
 *             it's input/output tensors, provide pointers to instances which
 *             can execute it and all operation specific parameters.
 */
template <typename OutDataType, typename... InArgTypes>
class OpInstance
{
    public:
    template <typename T>
    using TensorPtr      = std::shared_ptr<Tensor<T>>;
    using InTensorsTuple = std::tuple<TensorPtr<InArgTypes>...>;
    using DeviceMemPtr   = std::shared_ptr<DeviceMem>;
    using DeviceBuffers  = std::vector<DeviceMemPtr>;

    OpInstance()                  = default;
    OpInstance(const OpInstance&) = default;
    OpInstance& operator=(const OpInstance&) = default;
    virtual ~OpInstance();

    virtual InTensorsTuple getInputTensors() = 0;
    virtual TensorPtr<OutDataType> getOutputTensor() const      = 0;
    virtual std::unique_ptr<tensor_operation::device::BaseInvoker>
        makeInvokerPointer(std::unique_ptr<tensor_operation::device::BaseOperator>&) const = 0;
    virtual std::unique_ptr<tensor_operation::device::BaseArgument>
    makeArgumentPointer(std::unique_ptr<tensor_operation::device::BaseOperator>&,
                        const DeviceBuffers&,
                        const DeviceMemPtr&) const = 0;
    virtual std::size_t getFlops() const = 0;
    virtual std::size_t getBtype() const = 0;

};

/**
 * @brief      A generic operation instance run engine.
 */
template <typename OutDataType, typename... InArgTypes>
class OpInstanceRunEngine
{
    public:
    using OpInstanceT = OpInstance<InArgTypes..., OutDataType>;
    template <typename T>
    using TensorPtr      = std::shared_ptr<Tensor<T>>;
    using DeviceMemPtr   = std::unique_ptr<DeviceMem>;
    using InTensorsTuple = std::tuple<TensorPtr<InArgTypes>...>;
    using DeviceBuffers  = std::vector<DeviceMemPtr>;

    OpInstanceRunEngine() = delete;

    // TODO instead of ref_output, pass functor which will calculate ref_output and take tensors 
    // as input??
    OpInstanceRunEngine(const OpInstanceT& op_instance, const std::vector<OutDataType>& ref_output) 
        : m_op_instance{op_instance}
        , m_ref_output{ref_output}
    {
        m_in_tensors = m_op_instance.getInputTensors();
        m_out_tensor = m_op_instance.getOutputTensor();

        ck::static_for<0, std::tuple_size_v<InTensorsTuple>, 1>{}(
            [this](const auto& ts) {
                // This is ugly...
                using vec_type = decltype(ts->mData);
                this->m_in_device_buffers
                    .emplace_back(std::make_unique<DeviceMem>(sizeof(vec_type::value_type) *
                                                              ts->mDesc.GetElementSpace()))
                    .ToDevice(ts->mData.data());
            });

        using vec_type = decltype(m_out_tensor.mData);
        m_out_device_buffer =
            std::make_unique(sizeof(vec_type::value_type) * m_out_tensor.mDesc.GetElementSpace());
        m_out_device_buffer->SetZero();
    }

    virtual ~OpInstanceRunEngine(){};

    template <typename OpInstancePtr>
    bool test(const std::vector<OpInstancePtr>& op_ptrs)
    {
        bool res{true};
        for(auto& op_ptr : op_ptrs)
        {
            auto invoker = m_op_instance.makeInvokerPointer(op_ptr);
            auto argument =
                m_op_instance.makeArgumentPointer(op_ptr, m_in_device_buffers, m_out_device_buffer);
            if(op_ptr->IsSupportedArgument(argument.get()))
            {
                invoker->Run(argument.get());
                m_out_device_buffer->FromDevice(m_out_tensor.mData.data());
                // TODO: enable flexible use of custom check_error functions
                res = res && check_err(m_out_tensor.mData, m_ref_output.mData);
                m_out_device_buffer->SetZero();
            }
        }
        return res;
    }

    template <typename OpInstancePtr>
    ProfileBestConfig profile(const std::vector<OpInstancePtr>& op_ptrs,
                              int nrepeat = 100,
                              bool do_verification = false)
    {
        bool res{true};
        ProfileBestConfig best_config;

        for(auto& op_ptr : op_ptrs)
        {
            auto invoker = m_op_instance.makeInvokerPointer(op_ptr);
            auto argument =
                m_op_instance.makeArgumentPointer(op_ptr, m_in_device_buffers, m_out_device_buffer);
            if(op_ptr->IsSupportedArgument(argument.get()))
            {
                std::string op_name = op_ptr->GetTypeString();
                float avg_time = invoker->Run(argument.get(), nrepeat);

                std::size_t flops = m_op_instance.getFlops();
                std::size_t num_btype = m_op_instance.getBtype();
                float tflops = static_cast<float>(flops) / 1.E9 / avg_time;
                float gb_per_sec = num_btype / 1.E6 / avg_time;

                std::cout << "Perf: " << avg_time << " ms, " << tflops << " TFlops, " << gb_per_sec
                      << " GB/s, " << op_name << std::endl;

                if (tflops < best_config.best_tflops)
                {
                    best_config.best_op_name = op_name;
                    best_config.best_tflops = tflops;
                    best_config.best_gb_per_sec = gb_per_sec;
                    best_config.best_avg_time = avg_time;
                }

                if (do_verification)
                {
                    m_out_device_buffer->FromDevice(m_out_tensor.mData.data());
                    // TODO: enable flexible use of custom check_error functions
                    res = res && check_err(m_out_tensor.mData, m_ref_output.mData);

                    // if (do_log)
                    // {

                    // }
                }
                m_out_device_buffer->SetZero();
            }
        }
        return best_config;
    }

    void setAtol(double a) { m_atol = a; }
    void setRtol(double r) { m_rtol = r; }

    private:
    const OpInstanceT& m_op_instance;
    const std::vector<OutDataType>& m_ref_output;
    double m_rtol{1e-5};
    double m_atol{1e-8};

    InTensorsTuple m_in_tensors;
    TensorPtr<OutDataType> m_out_tensor;

    DeviceBuffers m_in_device_buffers;
    DeviceMemPtr m_out_device_buffer;

    template <typename T>
    bool check_err(const std::vector<T>& dev_out, const std::vector<T>& ref_out) const
    {
        return ck::utils::check_err(dev_out, ref_out, "Error: incorrect results!", m_atol, m_rtol);
    }
};

} // namespace utils
} // namespace ck
