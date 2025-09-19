// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "pytensor.hpp"

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <vector>

#include <fmt/format.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "ttnn/tensor/tensor_impl_wrapper.hpp"
#include "tools/profiler/op_profiler.hpp"
#include "ttnn-pybind/small_vector_caster.hpp"  // NOLINT - for pybind11 SmallVector binding support.
#include "ttnn/common/queue_id.hpp"
#include "ttnn/core.hpp"
#include "ttnn/distributed/api.hpp"
#include "ttnn/distributed/distributed_tensor.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/run_operation.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/tensor_impl.hpp"
#include "ttnn/tensor/tensor_utils.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/tensor/storage.hpp"
#include <tt-metalium/graph_tracking.hpp>
#include <tt-metalium/host_buffer.hpp>
#include <tt_stl/overloaded.hpp>
#include <tt_stl/span.hpp>

#include <tracy/Tracy.hpp>

#include <ttnn/tensor/tensor_conversion.hpp>

using namespace tt::tt_metal;

namespace ttnn::tensor {

template <typename... Args>
void log_from_cpp(const char* file, int line, const char* func, Args&&... message) {
    auto logging = pybind11::module_::import("logging");
    auto logger = logging.attr("getLogger")("py_log_cxx");

    // Convert arguments to Python objects and join with spaces
    auto builtins = pybind11::module_::import("builtins");
    auto str_func = builtins.attr("str");

    std::vector<pybind11::object> py_args;
    auto convert_arg = [&](auto&& arg) { py_args.push_back(str_func(std::forward<decltype(arg)>(arg))); };

    (convert_arg(std::forward<Args>(message)), ...);

    auto join_str = pybind11::str(" ");
    auto formatted_message = join_str.attr("join")(py_args);

    // Create a LogRecord manually
    auto log_record = logging.attr("LogRecord")(
        "py_log_cxx",                          // name
        logging.attr("DEBUG"),                 // level
        std::filesystem::path(file).string(),  // pathname
        line,                                  // lineno
        formatted_message,                     // msg
        pybind11::tuple(),                     // args
        pybind11::none(),                      // exc_info
        func,                                  // func
        pybind11::none()                       // stack_info
    );

    // Handle the record
    logger.attr("handle")(log_record);
}

void pytensor_logger_redirect(const char* file, int line, const char* func, const std::string& message) {
    log_from_cpp(file, line, func, message);
}

#define py_log(...) log_from_cpp(__FILE__, __LINE__, __func__, "[" #__VA_ARGS__ "] =" __VA_OPT__(, ) __VA_ARGS__);

// Virtual base class for list wrapper
class ListWrapper {
public:
    virtual ~ListWrapper() = default;

    // Core interface
    virtual bool is_leaf() const = 0;
    virtual size_t size() const = 0;
    virtual bool empty() const = 0;

    // Access elements
    virtual std::shared_ptr<ListWrapper> get_element(size_t index) const = 0;
    virtual double get_scalar_value() const = 0;

    // For formatting
    virtual std::vector<double> get_all_values() const = 0;
};

// Python list wrapper implementation
class PythonListWrapper : public ListWrapper {
private:
    py::object obj_;

public:
    explicit PythonListWrapper(py::object obj) : obj_(std::move(obj)) {}

    bool is_leaf() const override { return !py::isinstance<py::list>(obj_); }

    size_t size() const override {
        if (is_leaf()) {
            return 1;
        }
        return obj_.cast<py::list>().size();
    }

    bool empty() const override {
        if (is_leaf()) {
            return false;
        }
        return obj_.cast<py::list>().size() == 0;
    }

    std::shared_ptr<ListWrapper> get_element(size_t index) const override {
        if (is_leaf()) {
            throw std::runtime_error("Cannot get element from leaf node");
        }
        py::list list_obj = obj_.cast<py::list>();
        return std::make_shared<PythonListWrapper>(py::object(list_obj[index]));
    }

    double get_scalar_value() const override {
        if (!is_leaf()) {
            throw std::runtime_error("Cannot get scalar value from non-leaf node");
        }
        try {
            return obj_.cast<double>();
        } catch (...) {
            return 0.0;
        }
    }

    std::vector<double> get_all_values() const override {
        std::vector<double> values;
        if (is_leaf()) {
            values.push_back(get_scalar_value());
        } else {
            py::list list_obj = obj_.cast<py::list>();
            for (auto item : list_obj) {
                auto wrapper = std::make_shared<PythonListWrapper>(item.cast<py::object>());
                auto sub_values = wrapper->get_all_values();
                values.insert(values.end(), sub_values.begin(), sub_values.end());
            }
        }
        return values;
    }
};

// Tensor data wrapper implementation
template <typename T>
class TensorListWrapper : public ListWrapper {
private:
    std::span<const T> data_;
    Shape shape_;
    std::vector<size_t> current_indices_;

    size_t calculate_flat_index(const std::vector<size_t>& indices) const {
        size_t flat_index = 0;
        size_t stride = 1;

        for (int i = shape_.size() - 1; i >= 0; --i) {
            if (i < indices.size()) {
                flat_index += indices[i] * stride;
            }
            stride *= shape_[i];
        }
        return flat_index;
    }

    bool is_current_leaf() const { return current_indices_.size() == shape_.size() - 1; }

public:
    TensorListWrapper(std::span<const T> data, const Shape& shape) : data_(data), shape_(shape) {}

    TensorListWrapper(std::span<const T> data, const Shape& shape, std::vector<size_t> indices) :
        data_(data), shape_(shape), current_indices_(std::move(indices)) {}

    bool is_leaf() const override { return current_indices_.size() == shape_.size(); }

    size_t size() const override {
        if (is_leaf()) {
            return 1;
        }

        size_t dim_index = current_indices_.size();
        return shape_[dim_index];
    }

    bool empty() const override { return shape_.empty() || data_.empty(); }

    std::shared_ptr<ListWrapper> get_element(size_t index) const override {
        if (is_leaf()) {
            throw std::runtime_error("Cannot get element from leaf node");
        }

        std::vector<size_t> new_indices = current_indices_;
        new_indices.push_back(index);

        return std::make_shared<TensorListWrapper<T>>(data_, shape_, std::move(new_indices));
    }

    double get_scalar_value() const override {
        if (!is_leaf()) {
            throw std::runtime_error("Cannot get scalar value from non-leaf node");
        }

        size_t flat_index = calculate_flat_index(current_indices_);
        if (flat_index >= data_.size()) {
            throw std::runtime_error("Index out of bounds");
        }

        return static_cast<double>(data_[flat_index]);
    }

    std::vector<double> get_all_values() const override {
        std::vector<double> values;

        if (is_leaf()) {
            values.push_back(get_scalar_value());
        } else {
            for (size_t i = 0; i < size(); ++i) {
                auto sub_wrapper = get_element(i);
                auto sub_values = sub_wrapper->get_all_values();
                values.insert(values.end(), sub_values.begin(), sub_values.end());
            }
        }

        return values;
    }
};

// Implementation function that works with ListWrapper
std::string format_tensor_as_string_impl(const ListWrapper& wrapper, int precision = 4) {
    auto calculate_col_width = [&]() -> int {
        auto all_values = wrapper.get_all_values();
        if (all_values.empty()) {
            return precision + 4;
        }

        int max_len = 0;
        for (double val : all_values) {
            std::ostringstream oss;
            if (std::abs(val) < 1e-10) {
                oss << "0.0";
            } else {
                oss << std::fixed << std::setprecision(precision) << val;
            }
            max_len = std::max(max_len, static_cast<int>(oss.str().length()));
        }
        return std::max(max_len + 2, precision + 4);
    };

    auto format_number = [&](double val, int width) -> std::string {
        std::ostringstream oss;
        if (std::abs(val) < 1e-10) {
            oss << "0.0";
        } else {
            oss << std::fixed << std::setprecision(precision) << val;
        }

        std::string formatted = oss.str();
        if (formatted.length() < width) {
            return std::string(width - formatted.length(), ' ') + formatted;
        }
        return formatted;
    };

    std::function<std::string(const ListWrapper&, int, int)> format_recursive;
    format_recursive = [&](const ListWrapper& list_wrapper, int depth, int col_width) -> std::string {
        if (list_wrapper.is_leaf()) {
            return format_number(list_wrapper.get_scalar_value(), col_width);
        }

        if (list_wrapper.empty()) {
            return "[]";
        }

        // Check if all children are leaves (1D array case)
        bool all_children_leaves = true;
        for (size_t i = 0; i < list_wrapper.size(); ++i) {
            auto child = list_wrapper.get_element(i);
            if (!child->is_leaf()) {
                all_children_leaves = false;
                break;
            }
        }

        if (all_children_leaves) {
            std::ostringstream oss;
            oss << "[ ";
            for (size_t i = 0; i < list_wrapper.size(); ++i) {
                if (i > 0) {
                    oss << "   ";
                }
                auto child = list_wrapper.get_element(i);
                oss << format_number(child->get_scalar_value(), col_width);
            }
            oss << " ]";
            return oss.str();
        }

        // Multi-dimensional case
        std::vector<std::string> lines;
        std::string indent(depth, ' ');

        for (size_t i = 0; i < list_wrapper.size(); ++i) {
            auto child = list_wrapper.get_element(i);
            std::string formatted_item = format_recursive(*child, depth + 1, col_width);

            if (i == 0) {
                lines.push_back("[" + formatted_item);
            } else {
                lines.push_back(indent + " " + formatted_item);
            }
        }

        if (!lines.empty()) {
            lines.back() += "]";
        }

        std::ostringstream result;
        for (size_t i = 0; i < lines.size(); ++i) {
            if (i > 0) {
                result << "\n";
            }
            result << lines[i];
        }
        return result.str();
    };

    if (wrapper.empty()) {
        return "[]";
    }

    int col_width = calculate_col_width();
    return format_recursive(wrapper, 0, col_width);
}

// Entry point function with pybind11 tensor checking
std::string format_tensor_as_string(pybind11::object tensor, int precision = 4) {
    py::object np = py::module_::import("numpy");
    py::object torch = py::module_::import("torch");

    pybind11::object tensor_list;

    if (pybind11::hasattr(tensor, "tolist")) {
        tensor_list = tensor.attr("tolist")();
    } else if (pybind11::hasattr(tensor, "to_list")) {
        tensor_list = tensor.attr("to_list")();
    } else {
        return "Unsupported tensor type";
    }

    // Create appropriate wrapper
    std::shared_ptr<ListWrapper> wrapper = std::make_shared<PythonListWrapper>(tensor_list);

    // Generate header
    std::string header;
    auto builtins = pybind11::module_::import("builtins");
    auto str_func = builtins.attr("str");

    if (py::isinstance(tensor, np.attr("ndarray"))) {
        header = fmt::format(
            "torch.Tensor: shape: {} dtype: {} size: {}",
            str_func(tensor.attr("shape")).cast<std::string>(),
            str_func(tensor.attr("dtype")).cast<std::string>(),
            str_func(tensor.attr("size")).cast<std::string>());
    } else if (py::isinstance(tensor, torch.attr("Tensor"))) {
        header = fmt::format(
            "numpy.ndarray: shape: {} dtype: {} numel: {}",
            str_func(tensor.attr("shape")).cast<std::string>(),
            str_func(tensor.attr("dtype")).cast<std::string>(),
            str_func(tensor.attr("numel")()).cast<std::string>());
    } else if (pybind11::hasattr(tensor, "to_list")) {
        header = fmt::format(
            "ttnn.tensor: layout: {} padded_shape: {} shape: {} dtype: {} volume: {}",
            str_func(tensor.attr("layout")).cast<std::string>(),
            str_func(tensor.attr("padded_shape")).cast<std::string>(),
            str_func(tensor.attr("shape")).cast<std::string>(),
            str_func(tensor.attr("dtype")).cast<std::string>(),
            str_func(tensor.attr("volume")()).cast<std::string>());
    }

    // Check if tensor is too large
    std::string body;
    const int big_size = 1024;
    if (py::isinstance(tensor, np.attr("ndarray")) && big_size < tensor.attr("size").cast<int>()) {
        body = "1024 < volume";
    } else if (py::isinstance(tensor, torch.attr("Tensor")) && big_size < tensor.attr("numel")().cast<int>()) {
        body = "1024 < volume";
    } else if (py::hasattr(tensor, "volume") && big_size < tensor.attr("volume")().cast<int>()) {
        body = "1024 < volume";
    } else {
        body = format_tensor_as_string_impl(*wrapper, precision);
    }

    return header + "\n" + body;
}

// Helper function to create TensorListWrapper from HostBuffer based on type_info
std::shared_ptr<ListWrapper> create_tensor_wrapper_from_hostbuffer(const HostBuffer& buffer, const Shape& shape) {
    const std::type_info& type_info = *buffer.type_info();

    if (type_info == typeid(float)) {
        auto span = buffer.view_as<const float>();
        return std::make_shared<TensorListWrapper<float>>(span, shape);
    } else if (type_info == typeid(double)) {
        auto span = buffer.view_as<const double>();
        return std::make_shared<TensorListWrapper<double>>(span, shape);
    } else if (type_info == typeid(int8_t)) {
        auto span = buffer.view_as<const int8_t>();
        return std::make_shared<TensorListWrapper<int8_t>>(span, shape);
    } else if (type_info == typeid(int16_t)) {
        auto span = buffer.view_as<const int16_t>();
        return std::make_shared<TensorListWrapper<int16_t>>(span, shape);
    } else if (type_info == typeid(int32_t)) {
        auto span = buffer.view_as<const int32_t>();
        return std::make_shared<TensorListWrapper<int32_t>>(span, shape);
    } else if (type_info == typeid(int64_t)) {
        auto span = buffer.view_as<const int64_t>();
        return std::make_shared<TensorListWrapper<int64_t>>(span, shape);
    } else if (type_info == typeid(uint8_t)) {
        auto span = buffer.view_as<const uint8_t>();
        return std::make_shared<TensorListWrapper<uint8_t>>(span, shape);
    } else if (type_info == typeid(uint16_t)) {
        auto span = buffer.view_as<const uint16_t>();
        return std::make_shared<TensorListWrapper<uint16_t>>(span, shape);
    } else if (type_info == typeid(uint32_t)) {
        auto span = buffer.view_as<const uint32_t>();
        return std::make_shared<TensorListWrapper<uint32_t>>(span, shape);
    } else if (type_info == typeid(uint64_t)) {
        auto span = buffer.view_as<const uint64_t>();
        return std::make_shared<TensorListWrapper<uint64_t>>(span, shape);
    } else if (type_info == typeid(bfloat16)) {
        auto span = buffer.view_as<const bfloat16>();
        return std::make_shared<TensorListWrapper<bfloat16>>(span, shape);
    } else {
        TT_FATAL(false, "Unsupported type in HostBuffer: {}", type_info.name());
    }
}

// Format tensor as string from HostBuffer
std::string format_tensor_as_string(const HostBuffer& buffer, const Shape& shape, int precision = 4) {
    // Check if tensor is too large
    const int big_size = 1024;
    if (shape.volume() > big_size) {
        // Format shape manually
        std::ostringstream shape_stream;
        auto shape_view = shape.view();
        for (size_t i = 0; i < shape_view.size(); ++i) {
            if (i > 0) {
                shape_stream << "x";
            }
            shape_stream << shape_view[i];
        }

        return fmt::format("shape: {} volume: {} (too large to display)", shape_stream.str(), shape.volume());
    }

    // Create appropriate wrapper based on type_info
    auto wrapper = create_tensor_wrapper_from_hostbuffer(buffer, shape);

    // Generate header with buffer information
    std::ostringstream shape_stream;
    auto shape_view = shape.view();
    for (size_t i = 0; i < shape_view.size(); ++i) {
        if (i > 0) {
            shape_stream << "x";
        }
        shape_stream << shape_view[i];
    }

    std::string header =
        fmt::format("shape: {} dtype: {} volume: {}", shape_stream.str(), buffer.type_info()->name(), shape.volume());

    // Format the actual data
    std::string body = format_tensor_as_string_impl(*wrapper, precision);

    return header + "\n" + body;
}

std::size_t get_element_count(const HostBuffer& buffer) {
    const std::type_info& type_info = *buffer.type_info();
    auto byte_span = buffer.view_bytes();

    if (type_info == typeid(float)) {
        return byte_span.size() / sizeof(float);
    } else if (type_info == typeid(double)) {
        return byte_span.size() / sizeof(double);
    } else if (type_info == typeid(int8_t)) {
        return byte_span.size() / sizeof(int8_t);
    } else if (type_info == typeid(int16_t)) {
        return byte_span.size() / sizeof(int16_t);
    } else if (type_info == typeid(int32_t)) {
        return byte_span.size() / sizeof(int32_t);
    } else if (type_info == typeid(int64_t)) {
        return byte_span.size() / sizeof(int64_t);
    } else if (type_info == typeid(uint8_t)) {
        return byte_span.size() / sizeof(uint8_t);
    } else if (type_info == typeid(uint16_t)) {
        return byte_span.size() / sizeof(uint16_t);
    } else if (type_info == typeid(uint32_t)) {
        return byte_span.size() / sizeof(uint32_t);
    } else if (type_info == typeid(uint64_t)) {
        return byte_span.size() / sizeof(uint64_t);
    } else if (type_info == typeid(bfloat16)) {
        return byte_span.size() / sizeof(bfloat16);
    } else {
        TT_FATAL(false, "Unsupported type in HostBuffer: {}", type_info.name());
    }
}

// Overload without explicit shape (assumes 1D)
std::string format_tensor_as_string(const HostBuffer& buffer, int precision = 4) {
    // Calculate number of elements based on buffer size and type
    size_t element_count = get_element_count(buffer);
    Shape shape({static_cast<uint32_t>(element_count)});

    return format_tensor_as_string(buffer, shape, precision);
}

namespace CMAKE_UNIQUE_NAMESPACE {
namespace {

#ifdef DEBUG

void log_external_operation(const operation::ExternalOperation& operation, const std::vector<Tensor>& input_tensors) {
    log_debug(tt::LogOp, "Launching External Operation: \"{}\"", operation.get_type_name());

    auto attributes = operation.attributes();
    if (not attributes.empty()) {
        log_debug(tt::LogOp, "Attributes:");
        for (auto&& [name, value] : attributes) {
            log_debug(tt::LogOp, "\t{} = {}", name, value);
        }
    }

    log_debug(tt::LogOp, "Input std::vector<Tensor>:");
    for (auto index = 0; index < input_tensors.size(); index++) {
        const auto& tensor = input_tensors[index];
        log_debug(tt::LogOp, "\t{}: {}", index, tensor);
    }

    log_debug(tt::LogOp, "");
}
#else

void log_external_operation(const operation::ExternalOperation& operation, const std::vector<Tensor>& input_tensors) {}

#endif

// Wrapper around HostBuffer that provides a row-major view of the data, handles padding / logical view, and provides
// `shape` and `data_type` information.
struct RowMajorHostBuffer {
    static RowMajorHostBuffer create_padded(HostBuffer buffer, const ttnn::TensorSpec& tensor_spec) {
        tt::stl::Span<const uint32_t> shape_view = tensor_spec.padded_shape().view();
        return RowMajorHostBuffer{
            .buffer = std::move(buffer),
            .shape = std::vector<uint32_t>(shape_view.begin(), shape_view.end()),
            .data_type = tensor_spec.data_type(),
        };
    }

    static RowMajorHostBuffer create_logical(HostBuffer buffer, const ttnn::TensorSpec& tensor_spec) {
        tt::stl::Span<const uint32_t> shape_view = tensor_spec.logical_shape().view();
        return RowMajorHostBuffer{
            .buffer = std::move(buffer),
            .shape = std::vector<uint32_t>(shape_view.begin(), shape_view.end()),
            .data_type = tensor_spec.data_type(),
        };
    }

    HostBuffer buffer;
    std::vector<uint32_t> shape;
    ttnn::DataType data_type = ttnn::DataType::INVALID;
};

// Converts a TT tensor to a RowMajorHostBuffer.
//
// If `padded_output` is true, the returned buffer will be padded to the tile size.
// If `padded_output` is false, the returned buffer will be in logical view.
RowMajorHostBuffer convert_to_row_major_host_buffer(const Tensor& tt_tensor, const bool padded_output) {
    const auto& tensor_spec = tt_tensor.tensor_spec();

    // Performs logical data conversion on the concrete data type.
    auto dispatch_to_concrete = [&tensor_spec, padded_output]<typename T>(HostBuffer host_buffer) {
        if (padded_output) {
            if (tensor_spec.layout() == Layout::TILE) {
                auto row_major_data = tensor_impl::convert_layout_tile_to_row_major(
                    tensor_spec.physical_shape(), tensor_spec.tile(), host_buffer.view_as<const T>());
                return RowMajorHostBuffer::create_padded(HostBuffer(std::move(row_major_data)), tensor_spec);
            }
            return RowMajorHostBuffer::create_padded(std::move(host_buffer), tensor_spec);
        }

        // No modifications needed; direclty return buffer
        if (tensor_impl::logical_matches_physical(tensor_spec)) {
            return RowMajorHostBuffer::create_logical(std::move(host_buffer), tensor_spec);
        }

        auto logical_data = tensor_impl::decode_tensor_data(host_buffer.view_as<const T>(), tensor_spec);
        return RowMajorHostBuffer::create_logical(HostBuffer(std::move(logical_data)), tensor_spec);
    };

    auto convert_to_logical = [&tensor_spec, &dispatch_to_concrete](const HostBuffer& buffer) {
        const auto tt_dtype = tensor_spec.data_type();
        switch (tt_dtype) {
            case DataType::UINT8: return dispatch_to_concrete.template operator()<uint8_t>(buffer);
            case DataType::UINT16: return dispatch_to_concrete.template operator()<uint16_t>(buffer);
            case DataType::INT32: return dispatch_to_concrete.template operator()<int32_t>(buffer);
            case DataType::UINT32: return dispatch_to_concrete.template operator()<uint32_t>(buffer);
            case DataType::FLOAT32: return dispatch_to_concrete.template operator()<float>(buffer);
            case DataType::BFLOAT16: return dispatch_to_concrete.template operator()<bfloat16>(buffer);
            case DataType::BFLOAT8_B:
            case DataType::BFLOAT4_B: {
                const auto& tile = tensor_spec.tile();
                tt::stl::Span<const std::uint32_t> uint32_data = host_buffer::get_as<std::uint32_t>(buffer);
                auto float_unpacked_data = tt_dtype == DataType::BFLOAT8_B
                                               ? unpack_bfp8_tiles_into_float_vec(
                                                     uint32_data, /*row_major_output=*/false, /*is_exp_a=*/false, tile)
                                               : unpack_bfp4_tiles_into_float_vec(
                                                     uint32_data, /*row_major_output=*/false, /*is_exp_a=*/false, tile);
                auto input_float_buffer = tt::tt_metal::HostBuffer(std::move(float_unpacked_data));
                return dispatch_to_concrete.template operator()<float>(input_float_buffer);
            }
            case DataType::INVALID: TT_THROW("Unsupported DataType: {}", tt_dtype);
        }
        TT_THROW("Unreachable");
    };

    return convert_to_logical(std::visit(
        tt::stl::overloaded{
            [](const HostStorage& storage) {
                std::vector<HostBuffer> buffers;
                storage.buffer().apply([&buffers](const HostBuffer& shard) { buffers.push_back(shard); });
                TT_FATAL(
                    buffers.size() == 1,
                    "Can't convert a tensor distributed on {} mesh to row-major logical tensor. Supply a mesh composer "
                    "to concatenate multi-device shards.",
                    storage.buffer().shape());
                return buffers.front();
            },
            [&tt_tensor](auto&&) -> HostBuffer {
                TT_THROW(
                    "Tensor with {} cannot be converted to torch",
                    tt::stl::get_active_type_name_in_variant(tt_tensor.storage()));
            },
        },
        tt_tensor.storage()));
}

// Overload that converts a distributed tensor to a RowMajorHostBuffer.
//
// The returned buffer will be in logical view.
RowMajorHostBuffer convert_to_row_major_host_buffer(
    const Tensor& tt_tensor, const ttnn::distributed::MeshToTensor& mesh_composer) {
    auto dispatch_to_concrete = [&mesh_composer]<typename T>(const Tensor& tt_tensor) {
        auto [data, shape] = mesh_composer.compose<T>(tt_tensor);
        tt::stl::Span<const uint32_t> shape_view = shape.view();
        return RowMajorHostBuffer{
            .buffer = HostBuffer(std::move(data)),
            .shape = std::vector<uint32_t>(shape_view.begin(), shape_view.end()),
            .data_type = tt_tensor.dtype(),
        };
    };

    switch (tt_tensor.dtype()) {
        case DataType::UINT8: return dispatch_to_concrete.template operator()<uint8_t>(tt_tensor);
        case DataType::UINT16: return dispatch_to_concrete.template operator()<uint16_t>(tt_tensor);
        case DataType::INT32: return dispatch_to_concrete.template operator()<int32_t>(tt_tensor);
        case DataType::UINT32: return dispatch_to_concrete.template operator()<uint32_t>(tt_tensor);
        case DataType::BFLOAT16: return dispatch_to_concrete.template operator()<bfloat16>(tt_tensor);
        case DataType::BFLOAT8_B:
        case DataType::BFLOAT4_B:
        case DataType::FLOAT32: return dispatch_to_concrete.template operator()<float>(tt_tensor);
        case DataType::INVALID: TT_THROW("Unsupported DataType: {}", tt_tensor.dtype());
    }
    TT_THROW("Unreachable");
}

py::object convert_tt_tensor_to_torch_tensor(const RowMajorHostBuffer& row_major_host_buffer) {
    GraphTracker::instance().track_function_start(
        "tt::tt_metal::detail::convert_tt_tensor_to_torch_tensor", row_major_host_buffer);

    py::object torch = py::module_::import("torch");
    auto frombuffer = torch.attr("frombuffer");

    py::object torch_dtype = [&]() {
        switch (row_major_host_buffer.data_type) {
            case DataType::UINT8: return torch.attr("uint8");
            case DataType::UINT16: return torch.attr("int16");
            case DataType::INT32:
            case DataType::UINT32: return torch.attr("int32");
            case DataType::BFLOAT16: return torch.attr("bfloat16");
            case DataType::BFLOAT8_B:
            case DataType::BFLOAT4_B:
            case DataType::FLOAT32: return torch.attr("float32");
            case DataType::INVALID: TT_THROW("Invalid data type");
        }
        TT_THROW("Unreachable");
    }();

    auto tensor = [&]() {
        if (row_major_host_buffer.buffer.view_bytes().empty()) {
            auto pytorch_empty = torch.attr("empty");
            return pytorch_empty(row_major_host_buffer.shape, py::arg("dtype") = torch_dtype);
        }
        return frombuffer(row_major_host_buffer.buffer, py::arg("dtype") = torch_dtype);
    }();

    tensor = tensor.attr("reshape")(row_major_host_buffer.shape);
    tensor = tensor.attr("contiguous")();

    GraphTracker::instance().track_function_end(tensor);
    return tensor;
}

py::object convert_tt_tensor_to_numpy_tensor(const RowMajorHostBuffer& row_major_host_buffer) {
    GraphTracker::instance().track_function_start(
        "tt::tt_metal::detail::convert_tt_tensor_to_numpy_tensor", row_major_host_buffer);

    py::object np = py::module_::import("numpy");
    auto frombuffer = np.attr("frombuffer");

    py::object np_dtype = [&]() {
        switch (row_major_host_buffer.data_type) {
            case DataType::UINT8: return np.attr("ubyte");
            case DataType::UINT16: return np.attr("int16");
            case DataType::INT32:
            case DataType::UINT32: return np.attr("int32");
            case DataType::BFLOAT16: TT_THROW("Bfloat16 is not supported for numpy!");
            case DataType::BFLOAT8_B:
            case DataType::BFLOAT4_B:
            case DataType::FLOAT32: return np.attr("float32");
            case DataType::INVALID: TT_THROW("Invalid data type");
        }
        TT_THROW("Unreachable");
    }();

    auto tensor = frombuffer(row_major_host_buffer.buffer, py::arg("dtype") = np_dtype);
    tensor = tensor.attr("reshape")(row_major_host_buffer.shape);
    tensor = np.attr("ascontiguousarray")(tensor);
    GraphTracker::instance().track_function_end(tensor);
    return tensor;
}

auto parse_external_operation(
    const py::function& external_operation,
    const py::args& args,
    const py::kwargs& kwargs,
    std::optional<std::string> function_name_override = std::nullopt) {
    std::string function_name;
    if (function_name_override.has_value()) {
        function_name = function_name_override.value();
    } else {
        function_name = py::cast<std::string>(external_operation.attr("__qualname__"));
    }

    std::vector<Tensor> input_tensors;
    tt::stl::reflection::Attributes attributes;

    auto process_name_and_value = [&function_name, &input_tensors, &attributes](const auto& name, const auto& value) {
        py::object torch = py::module_::import("torch");
        py::object ttnn = py::module_::import("ttnn");
        if (py::isinstance<Tensor>(value)) {
            // TODO(arakhmati): figure out how to handle this without causing extra memory usage
            // auto tensor = py::cast<Tensor>(value);
            // input_tensors.push_back(tensor);
        } else if (py::isinstance(value, ttnn.attr("Tensor"))) {
            // TODO(arakhmati): figure out how to handle this without causing extra memory usage
            // auto tensor = py::cast<Tensor>(value.attr("value"));
            // input_tensors.push_back(tensor);
        } else if (py::isinstance(value, torch.attr("nn").attr("Module"))) {
            // do nothing
        } else if (py::isinstance(value, torch.attr("Tensor"))) {
            // TODO(arakhmati): figure out how to handle this without causing extra memory usage
            // auto tensor = detail::convert_torch_tensor_to_tt_tensor(value);
            // input_tensors.push_back(tensor);
        } else {
            // TODO(MO): Exclude tensor data as it is not an attribute
            // attributes.push_back({name, fmt::format("{}", value)});
        }
    };

    auto arg_index = 0;
    for (const auto& value : args) {
        auto name = fmt::format("arg_{}", arg_index++);
        process_name_and_value(name, value);
    }

    for (const auto& [name, value] : kwargs) {
        process_name_and_value(py::cast<std::string>(name), value);
    }

    auto operation = tt::tt_metal::operation::ExternalOperation{function_name, attributes};
    return std::make_tuple(operation, input_tensors);
}

host_buffer_data_type get_py_tensor_type_info(const py::handle& py_tensor) {
    if (py::object torch = py::module_::import("torch"); py::isinstance(py_tensor, torch.attr("Tensor"))) {
        const auto py_dtype = py_tensor.attr("dtype");
        if (py_dtype.equal(torch.attr("float32"))) {
            return host_buffer_data_type::FLOAT32;
        } else if (py_dtype.equal(torch.attr("float64"))) {
            return host_buffer_data_type::FLOAT64;
        } else if (py_dtype.equal(torch.attr("float16"))) {
            return host_buffer_data_type::FLOAT16;
        } else if (py_dtype.equal(torch.attr("bfloat16"))) {
            return host_buffer_data_type::BFLOAT16;
        } else if (py_dtype.equal(torch.attr("int8"))) {
            return host_buffer_data_type::INT8;
        } else if (py_dtype.equal(torch.attr("int16"))) {
            return host_buffer_data_type::INT16;
        } else if (py_dtype.equal(torch.attr("int32"))) {
            return host_buffer_data_type::INT32;
        } else if (py_dtype.equal(torch.attr("int64"))) {
            return host_buffer_data_type::INT64;
        } else if (py_dtype.equal(torch.attr("uint8"))) {
            return host_buffer_data_type::UINT8;
        } else if (py_dtype.equal(torch.attr("bool"))) {
            return host_buffer_data_type::BOOL;
        } else {
            TT_THROW("Unsupported torch tensor dtype!");
        }
    } else if (py::object np = py::module_::import("numpy"); py::isinstance(py_tensor, np.attr("ndarray"))) {
        const auto py_dtype = py_tensor.attr("dtype");
        if (py_dtype.equal(np.attr("float32"))) {
            return host_buffer_data_type::FLOAT32;
        } else if (py_dtype.equal(np.attr("float64"))) {
            return host_buffer_data_type::FLOAT64;
        } else if (py_dtype.equal(np.attr("float16"))) {
            return host_buffer_data_type::FLOAT16;
        } else if (py_dtype.equal(np.attr("int8"))) {
            return host_buffer_data_type::INT8;
        } else if (py_dtype.equal(np.attr("int16"))) {
            return host_buffer_data_type::INT16;
        } else if (py_dtype.equal(np.attr("int32"))) {
            return host_buffer_data_type::INT32;
        } else if (py_dtype.equal(np.attr("int64"))) {
            return host_buffer_data_type::INT64;
        } else if (py_dtype.equal(np.attr("uint8"))) {
            return host_buffer_data_type::UINT8;
        } else if (py_dtype.equal(np.attr("uint16"))) {
            return host_buffer_data_type::UINT16;
        } else if (py_dtype.equal(np.attr("uint32"))) {
            return host_buffer_data_type::UINT32;
        } else if (py_dtype.equal(np.attr("uint64"))) {
            return host_buffer_data_type::UINT64;
        } else if (py_dtype.equal(np.attr("bool_"))) {
            return host_buffer_data_type::BOOL;
        } else {
            TT_THROW("Unsupported numpy array dtype!");
        }
    } else {
        TT_THROW("The argument must be of type torch.Tensor or numpy.ndarray!");
    }
}

HostBuffer convert_py_tensor_to_host_buffer(const py::handle& py_tensor, DataType target_dtype) {
    auto to_buffer =
        []<typename T>(
            const void* py_data_ptr, std::size_t num_elements, const py::object& contiguous_py_tensor) -> HostBuffer {
        // Important: `py::object` copying and destruction must be done while holding GIL, which pybind ensures for a
        // thread that calls the C++ APIs. We wrap `py::object` in `MemoryPin` so that multi-threaded C++ code only
        // increments / decrements the reference count on the memory pin; the last decrement to the pin should be
        // triggered from the pybind caller thread, which will correctly decrement the `py::object` reference count
        // while hodling GIL.
        tt::tt_metal::MemoryPin pydata_pin(std::make_shared<py::object>(contiguous_py_tensor));
        T* typed_py_ptr = const_cast<T*>(static_cast<const T*>(py_data_ptr));
        return HostBuffer(tt::stl::Span<T>(typed_py_ptr, typed_py_ptr + num_elements), pydata_pin);
    };

    py_log("converting py tensor to host buffer", target_dtype);

    if (target_dtype == DataType::BFLOAT4_B || target_dtype == DataType::BFLOAT8_B) {
        TT_THROW("BFLOAT4_B and BFLOAT8_B data types are not supported for tensor conversion!");
    }

    if (target_dtype == DataType::INVALID) {
        TT_THROW("DataType::INVALID data type specified!");
    }

    if (py::object torch = py::module_::import("torch"); py::isinstance(py_tensor, torch.attr("Tensor"))) {
        py::object cont_tensor = py_tensor.attr("contiguous")();
        const auto py_dtype = cont_tensor.attr("dtype");

        auto maybe_convert = [&cont_tensor, &py_dtype, &torch](const char* target_py_dtype) {
            if (not py_dtype.equal(torch.attr(target_py_dtype))) {
                cont_tensor = cont_tensor.attr("to")(torch.attr(target_py_dtype));
            }
        };

        // Convert to target dtype
        switch (target_dtype) {
            case DataType::BFLOAT16: maybe_convert("bfloat16"); break;
            case DataType::FLOAT32: maybe_convert("float32"); break;
            case DataType::UINT32: maybe_convert("int32"); break;
            case DataType::UINT8: maybe_convert("uint8"); break;
            case DataType::UINT16: maybe_convert("int16"); break;
            case DataType::INT32: maybe_convert("int32"); break;
            default: TT_THROW("Unsupported target DataType!");
        }

        auto numel = py::cast<std::size_t>(cont_tensor.attr("numel")());
        auto ptr = reinterpret_cast<const void*>(py::cast<uintptr_t>(cont_tensor.attr("data_ptr")()));

        switch (target_dtype) {
            case DataType::BFLOAT16: return to_buffer.operator()<bfloat16>(ptr, numel, cont_tensor);
            case DataType::FLOAT32: return to_buffer.operator()<float>(ptr, numel, cont_tensor);
            case DataType::UINT32: return to_buffer.operator()<uint32_t>(ptr, numel, cont_tensor);
            case DataType::UINT8: return to_buffer.operator()<uint8_t>(ptr, numel, cont_tensor);
            case DataType::UINT16: return to_buffer.operator()<uint16_t>(ptr, numel, cont_tensor);
            case DataType::INT32: return to_buffer.operator()<int32_t>(ptr, numel, cont_tensor);
            default: TT_THROW("Unsupported target DataType!");
        }
    } else if (py::object np = py::module_::import("numpy"); py::isinstance(py_tensor, np.attr("ndarray"))) {
        if (target_dtype == DataType::BFLOAT16) {
            return convert_py_tensor_to_host_buffer(torch.attr("from_numpy")(py_tensor), target_dtype);
        }

        py::object cont_tensor = np.attr("ascontiguousarray")(py_tensor);
        const auto py_dtype = cont_tensor.attr("dtype");

        auto maybe_convert = [&cont_tensor, &py_dtype, &np](const char* target_py_dtype) {
            if (not py_dtype.equal(np.attr(target_py_dtype))) {
                cont_tensor = cont_tensor.attr("astype")(np.attr(target_py_dtype));
            }
        };

        // Convert to target dtype
        switch (target_dtype) {
            case DataType::FLOAT32: maybe_convert("float32"); break;
            case DataType::UINT32: maybe_convert("uint32"); break;
            case DataType::UINT8: maybe_convert("uint8"); break;
            case DataType::UINT16: maybe_convert("uint16"); break;
            case DataType::INT32: maybe_convert("int32"); break;
            default: TT_THROW("Unsupported target DataType!");
        }

        auto size = py::cast<std::size_t>(cont_tensor.attr("size"));
        auto ptr = reinterpret_cast<const void*>(py::cast<uintptr_t>(
            py::cast<py::tuple>(py::cast<py::dict>(cont_tensor.attr("__array_interface__"))[py::str("data")])[0]));

        switch (target_dtype) {
            case DataType::BFLOAT16: return to_buffer.operator()<bfloat16>(ptr, size, cont_tensor);
            case DataType::FLOAT32: return to_buffer.operator()<float>(ptr, size, cont_tensor);
            case DataType::UINT32: return to_buffer.operator()<uint32_t>(ptr, size, cont_tensor);
            case DataType::UINT8: return to_buffer.operator()<uint8_t>(ptr, size, cont_tensor);
            case DataType::UINT16: return to_buffer.operator()<uint16_t>(ptr, size, cont_tensor);
            case DataType::INT32: return to_buffer.operator()<int32_t>(ptr, size, cont_tensor);
            default: TT_THROW("Unsupported target DataType!");
        }
    } else {
        TT_THROW("The argument must be of type torch.Tensor or numpy.ndarray!");
    }
}

std::optional<DataType> map_torch_data_type_to_ttnn(const py::object& py_dtype, const py::object& torch) {
    if (py_dtype.equal(torch.attr("float32"))) {
        return DataType::FLOAT32;
    } else if (py_dtype.equal(torch.attr("float16"))) {
        return DataType::BFLOAT16;
    } else if (py_dtype.equal(torch.attr("bfloat16"))) {
        return DataType::BFLOAT16;
    } else if (py_dtype.equal(torch.attr("int64"))) {
        return DataType::UINT32;
    } else if (py_dtype.equal(torch.attr("int32"))) {
        return DataType::INT32;
    } else if (py_dtype.equal(torch.attr("int16"))) {
        return DataType::UINT16;
    } else if (py_dtype.equal(torch.attr("uint8"))) {
        return DataType::UINT8;
    } else {
        return std::nullopt;
    }
}

std::optional<DataType> map_numpy_data_type_to_ttnn(const py::object& py_dtype, const py::object& np) {
    if (py_dtype.equal(np.attr("float32"))) {
        return DataType::FLOAT32;
    } else if (py_dtype.equal(np.attr("float16"))) {
        return DataType::BFLOAT16;
    } else if (py_dtype.equal(np.attr("int64"))) {
        return DataType::UINT32;
    } else if (py_dtype.equal(np.attr("int32"))) {
        return DataType::INT32;
    } else if (py_dtype.equal(np.attr("int16"))) {
        return DataType::UINT16;
    } else if (py_dtype.equal(np.attr("uint8"))) {
        return DataType::UINT8;
    } else {
        return std::nullopt;
    }
}

DataType get_target_type(std::optional<DataType> optional_data_type, const py::handle& py_tensor) {
    if (optional_data_type.has_value()) {
        return optional_data_type.value();
    } else if (py::object torch = py::module_::import("torch"); py::isinstance(py_tensor, torch.attr("Tensor"))) {
        auto result = map_torch_data_type_to_ttnn(py_tensor.attr("dtype"), torch);
        TT_FATAL(result.has_value(), "Could not map torch type to TTNN: {}", py::cast<std::string>(py_tensor("dtype")));
        return result.value();
    } else if (py::object np = py::module_::import("numpy"); py::isinstance(py_tensor, np.attr("ndarray"))) {
        auto result = map_numpy_data_type_to_ttnn(py_tensor.attr("dtype"), np);
        TT_FATAL(result.has_value(), "Could not map numpy type to TTNN: {}", py::cast<std::string>(py_tensor("dtype")));
        return result.value();
    } else {
        TT_THROW(
            "Could not get target type: the dtype was not explicitly specified and the input tensor object is not a "
            "torch or numpy tensor: {}",
            py::cast<std::string>(py::str(py_tensor.get_type())));
    }
}

Tensor convert_python_tensor_to_tt_tensor(
    const py::handle& py_tensor,
    std::optional<DataType> optional_data_type,
    std::optional<Layout> optional_layout,
    const std::optional<Tile>& optional_tile,
    const MemoryConfig& memory_config,
    ttnn::distributed::MeshDevice* device,
    ttnn::QueueId cq_id,
    float pad_value,
    const ttnn::distributed::TensorToMesh* mesh_mapper) {
    ZoneScoped;

    py_log(optional_data_type);
    py_log(optional_layout);
    py_log("input tensor", format_tensor_as_string(py::reinterpret_borrow<py::object>(py_tensor)));

    GraphTracker::instance().track_function_start(
        "tt::tt_metal::detail::convert_python_tensor_to_tt_tensor",
        py_tensor,
        optional_data_type,
        optional_layout,
        optional_tile,
        memory_config,
        device,
        cq_id,
        pad_value,
        mesh_mapper);

    const auto shape = ttnn::Shape(py::cast<ttnn::SmallVector<uint32_t>>(py_tensor.attr("shape")));

    Tensor output = create_device_tensor_from_host_data(
        TensorSpec(
            shape,
            TensorLayout(
                get_target_type(optional_data_type, py_tensor),
                PageConfig(optional_layout.value_or(Layout::ROW_MAJOR), optional_tile),
                memory_config)),
        get_py_tensor_type_info(py_tensor),
        [&](const DataType& dtype) -> HostBuffer {
            py::object tensor = py::reinterpret_borrow<pybind11::object>(py_tensor);
            HostBuffer buffer = convert_py_tensor_to_host_buffer(py_tensor, dtype);
            py_log("Created host buffer with type ID", buffer.type_info()->name());
            py_log("shapeless buffer", format_tensor_as_string(buffer));
            py_log("shaped buffer", format_tensor_as_string(buffer, shape));
            return buffer;
        },
        device,
        cq_id,
        pad_value,
        mesh_mapper);

    py_log("output tensor", format_tensor_as_string(py::cast(output)));

    GraphTracker::instance().track_function_end(output);
    return output;
}

}  // namespace
}  // namespace CMAKE_UNIQUE_NAMESPACE

void pytensor_module_types(py::module& m_tensor) {
    // Tensor constructors that accept device and .to_device() function use keep alive call policy to communicate that
    // Device needs to outlive Tensor. This is because when tensors on device are destroyed they need to deallocate
    // their buffers via device. keep_alive increases the ref count of the Device object being passed into the
    // constructor and .to_device() function. For additional info see:
    // https://pybind11.readthedocs.io/en/stable/advanced/functions.html#keep-alive
    auto pyTensor = py::class_<Tensor>(m_tensor, "Tensor", R"doc(

        Class constructor supports tensors of rank 4.
        The constructor takes following arguments:

        +------------+--------------------------------------------------------+---------------------------+------------------------------------+----------+
        |  Argument  |                 Description                            |       Data type           |           Valid range              | Required |
        +============+========================================================+===========================+====================================+==========+
        | data       | Data to store in TT tensor                             | List[float/int]           |                                    | Yes      |
        +------------+--------------------------------------------------------+---------------------------+------------------------------------+----------+
        | shape      | Shape of TT tensor                                     | List[int[4]]              |                                    | Yes      |
        +------------+--------------------------------------------------------+---------------------------+------------------------------------+----------+
        | data_type  | Data type of numbers in TT tensor                      | ttnn.DataType             | ttnn.DataType.BFLOAT16             | Yes      |
        |            |                                                        |                           |                                    |          |
        |            |                                                        |                           | ttnn.DataType.FLOAT32              |          |
        |            |                                                        |                           |                                    |          |
        |            |                                                        |                           | ttnn.DataType.UINT32               |          |
        |            |                                                        |                           |                                    |          |
        |            |                                                        |                           | ttnn.DataType.BFLOAT8_B            |          |
        |            |                                                        |                           |                                    |          |
        |            |                                                        |                           | ttnn.DataType.BFLOAT4_B            |          |
        +------------+--------------------------------------------------------+---------------------------+------------------------------------+----------+
        | layout     | Layout of tensor data in memory                        | ttnn.Layout               | ttnn.Layout.ROW_MAJOR              | Yes      |
        |            |                                                        |                           |                                    |          |
        |            |                                                        |                           | ttnn.Layout.TILE                   |          |
        +------------+--------------------------------------------------------+---------------------------+------------------------------------+----------+
        | device     | Device on which tensor will be created                 | ttnn.Device               | Host or TT accelerator device      | No       |
        +------------+--------------------------------------------------------+---------------------------+------------------------------------+----------+
        | mem_config | Layout of tensor in TT Accelerator device memory banks | ttnn.MemoryConfig         |                                    | No       |
        +------------+--------------------------------------------------------+---------------------------+------------------------------------+----------+

    )doc");
}

void pytensor_module(py::module& m_tensor) {
    m_tensor.def(
        "decorate_external_operation",
        [](const py::function& function, const std::optional<std::string>& function_name) -> py::function {
            return py::cpp_function(
                std::function([function, function_name](const py::args& args, const py::kwargs& kwargs) {
                    ZoneScopedN("TT_DNN_FALLBACK_OP");
                    auto [operation, input_tensors] =
                        CMAKE_UNIQUE_NAMESPACE::parse_external_operation(function, args, kwargs, function_name);
                    GraphTracker::instance().track_function_start(operation.get_type_name(), args, kwargs);
                    CMAKE_UNIQUE_NAMESPACE::log_external_operation(operation, input_tensors);
                    auto output = function(*args, **kwargs);
                    TracyOpTTNNExternal(
                        operation, input_tensors, ttnn::CoreIDs::instance().fetch_and_increment_device_operation_id());
                    GraphTracker::instance().track_function_end(output);
                    return output;
                }));
        },
        py::arg("function").noconvert(),
        py::arg("function_name").noconvert() = std::nullopt,
        R"doc(
        Decorate external operation for purposes of reporting and profiling.

            +----------+----------------------+-----------+-------------+----------+
            | Argument | Description          | Data type | Valid range | Required |
            +==========+======================+===========+=============+==========+
            | function | Fallback Operation   | Function  |             | Yes      |
            +----------+----------------------+-----------+-------------+----------+
            | args     | Packed args          | tuple     |             | No       |
            +----------+----------------------+-----------+-------------+----------+
            | kwargs   | Packed kwargs        | dict      |             | No       |
            +----------+----------------------+-----------+-------------+----------+
    )doc");

    auto pyTensor = static_cast<py::class_<Tensor>>(m_tensor.attr("Tensor"));
    pyTensor.def(py::init<ttnn::Tensor&>())
        .def(
            py::init<>([](std::vector<float>&& data,
                          const std::array<uint32_t, 4>& shape,
                          DataType data_type,
                          Layout layout,
                          const std::optional<Tile>& tile,
                          float pad_value) {
                return Tensor::from_vector(
                    std::move(data),
                    TensorSpec(ttnn::Shape(shape), TensorLayout(data_type, PageConfig(layout, tile), MemoryConfig{})),
                    /*device=*/nullptr,
                    ttnn::DefaultQueueId,
                    pad_value);
            }),
            py::arg("data"),
            py::arg("shape"),
            py::arg("data_type"),
            py::arg("layout"),
            py::arg("tile") = std::nullopt,
            py::arg("pad_value") = 0.0f,
            py::return_value_policy::move,
            R"doc(
                +---------------+----------------------+
                | Argument      | Name                 |
                +===============+======================+
                | arg0          | data                 |
                +---------------+----------------------+
                | arg1          | shape                |
                +---------------+----------------------+
                | arg2          | data_type            |
                +---------------+----------------------+
                | arg3          | layout               |
                +---------------+----------------------+
                | arg4          | tile (optional)      |
                +---------------+----------------------+
                | arg5          | pad_value (optional) |
                +---------------+----------------------+

                Example of creating a TT Tensor on host:

                .. code-block:: python

                    py_tensor = torch.randn((1, 1, 32, 32))
                    ttnn.Tensor(
                        py_tensor.reshape(-1).tolist(),
                        py_tensor.size(),
                        ttnn.DataType.BFLOAT16,
                        ttnn.Layout.ROW_MAJOR,
                    )
            )doc")
        .def(
            py::init<>([](std::vector<float>&& data,
                          const std::array<uint32_t, 4>& shape,
                          DataType data_type,
                          Layout layout,
                          std::optional<MeshDevice*> device,
                          const std::optional<Tile>& tile,
                          float pad_value) {
                return Tensor::from_vector(
                    std::move(data),
                    TensorSpec(ttnn::Shape(shape), TensorLayout(data_type, PageConfig(layout, tile), MemoryConfig{})),
                    device.value_or(nullptr),
                    ttnn::DefaultQueueId,
                    pad_value);
            }),
            py::keep_alive<1, 6>(),
            py::arg("data"),
            py::arg("shape"),
            py::arg("data_type"),
            py::arg("layout"),
            py::arg("device") = std::nullopt,
            py::arg("tile") = std::nullopt,
            py::arg("pad_value") = 0.0f,
            py::return_value_policy::move,
            R"doc(
                +---------------+----------------------+
                | Argument      | Name                 |
                +===============+======================+
                | arg0          | data                 |
                +---------------+----------------------+
                | arg1          | shape                |
                +---------------+----------------------+
                | arg2          | data_type            |
                +---------------+----------------------+
                | arg3          | layout               |
                +---------------+----------------------+
                | arg4          | device (optional)    |
                +---------------+----------------------+
                | arg5          | tile (optional)      |
                +---------------+----------------------+
                | arg6          | pad_value (optional) |
                +---------------+----------------------+

                Only BFLOAT16 (in ROW_MAJOR or TILE layout) and BFLOAT8_B, BFLOAT4_B (in TILE layout) are supported on device.

                Note that TT Tensor in ROW_MAJOR layout on TT Accelerator device must have size of last dimension divisble by 2.

                Example of creating a TT Tensor on TT accelerator device:

                .. code-block:: python

                    py_tensor = torch.randn((1, 1, 32, 32))
                    tt_device = ttnn.CreateDevice(0)
                    // ...
                    ttnn.Tensor(
                        py_tensor.reshape(-1).tolist(),
                        py_tensor.size(),
                        ttnn.DataType.BFLOAT16,
                        ttnn.Layout.ROW_MAJOR,
                        tt_device
                    )
            )doc")
        .def(
            py::init<>([](std::vector<float>&& data,
                          const std::array<uint32_t, 4>& shape,
                          DataType data_type,
                          Layout layout,
                          std::optional<MeshDevice*> device,
                          const MemoryConfig& memory_config,
                          const std::optional<Tile>& tile,
                          float pad_value) {
                return Tensor::from_vector(
                    std::move(data),
                    TensorSpec(ttnn::Shape(shape), TensorLayout(data_type, PageConfig(layout, tile), memory_config)),
                    device.value_or(nullptr),
                    ttnn::DefaultQueueId,
                    pad_value);
            }),
            py::keep_alive<1, 7>(),
            py::arg("data"),
            py::arg("shape"),
            py::arg("data_type"),
            py::arg("layout"),
            py::arg("device") = std::nullopt,
            py::arg("memory_config"),
            py::arg("tile") = std::nullopt,
            py::arg("pad_value") = 0.0f,
            py::return_value_policy::move,
            R"doc(
                +---------------+----------------------+
                | Argument      | Name                 |
                +===============+======================+
                | arg0          | data                 |
                +---------------+----------------------+
                | arg1          | shape                |
                +---------------+----------------------+
                | arg2          | data_type            |
                +---------------+----------------------+
                | arg3          | layout               |
                +---------------+----------------------+
                | arg4          | device               |
                +---------------+----------------------+
                | arg5          | mem_config           |
                +---------------+----------------------+
                | arg6          | tile (optional)      |
                +---------------+----------------------+
                | arg7          | pad_value (optional) |
                +---------------+----------------------+

                Only BFLOAT16 (in ROW_MAJOR or TILE layout) and BFLOAT8_B, BFLOAT4_B (in TILE layout) are supported on device.

                Note that TT Tensor in ROW_MAJOR layout on TT Accelerator device must have size of last dimension divisble by 2.

                Example of creating a TT Tensor on TT accelerator device with specified mem_config:

                .. code-block:: python

                    py_tensor = torch.randn((1, 1, 32, 32))
                    tt_device = ttnn.CreateDevice(0)
                    mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED)
                    // ...
                    ttnn.Tensor(
                        py_tensor.reshape(-1).tolist(),
                        py_tensor.size(),
                        ttnn.DataType.BFLOAT16,
                        ttnn.Layout.ROW_MAJOR,
                        tt_device,
                        mem_config
                    )
            )doc")
        .def(
            py::init<>([](const py::object& python_tensor,
                          std::optional<DataType> data_type,
                          std::optional<MeshDevice*> device,
                          std::optional<Layout> layout,
                          const std::optional<MemoryConfig>& mem_config,
                          const std::optional<Tile>& tile,
                          ttnn::QueueId cq_id,
                          std::optional<float> pad_value,
                          const distributed::TensorToMesh* mesh_mapper) {
                return CMAKE_UNIQUE_NAMESPACE::convert_python_tensor_to_tt_tensor(
                    python_tensor,
                    data_type,
                    layout,
                    tile,
                    mem_config.value_or(MemoryConfig{}),
                    device.value_or(nullptr),
                    cq_id,
                    pad_value.value_or(0.0f),
                    mesh_mapper);
            }),
            py::arg("tensor"),
            py::arg("data_type") = std::nullopt,
            py::arg("device") = std::nullopt,
            py::arg("layout").noconvert() = std::nullopt,
            py::arg("mem_config").noconvert() = std::nullopt,
            py::arg("tile").noconvert() = std::nullopt,
            py::arg("cq_id") = ttnn::DefaultQueueId,
            py::arg("pad_value") = std::nullopt,
            py::arg("mesh_mapper") = nullptr,
            py::return_value_policy::move,
            R"doc(
                +--------------+--------------------------------+
                | Argument     | Description                    |
                +==============+================================+
                | tensor       | Pytorch or Numpy Tensor        |
                +--------------+--------------------------------+
                | data_type    | TT Tensor data type (optional) |
                +--------------+--------------------------------+
                | device       | TT device ptr (optional)       |
                +--------------+--------------------------------+
                | layout       | TT layout (optional)           |
                +--------------+--------------------------------+
                | mem_config   | TT memory_config (optional)    |
                +--------------+--------------------------------+
                | tile         | TT Tile Spec (optional)        |
                +--------------+--------------------------------+
                | cq_id        | TT Command Queue ID (optional) |
                +--------------+--------------------------------+
                | pad_value    | Padding value (optional)       |
                +--------------+--------------------------------+
                | mesh_mapper  | TT-NN Mesh Mapper (optional)    |
                +--------------+--------------------------------+

                Example of creating a TT Tensor from numpy tensor:

                .. code-block:: python

                    device = ttnn.open_device(device_id=0)
                    py_tensor = np.zeros((1, 1, 32, 32))
                    ttnn.Tensor(py_tensor, ttnn.bfloat16, device, ttnn.TILE_LAYOUT)
            )doc")
        .def_property_readonly("spec", [](const Tensor& self) { return self.tensor_spec(); })
        .def_property_readonly("shape", [](const Tensor& self) { return self.logical_shape(); })
        .def_property_readonly("padded_shape", [](const Tensor& self) { return self.padded_shape(); })
        .def_property_readonly("dtype", [](const Tensor& self) { return self.dtype(); })
        .def_property_readonly("layout", [](const Tensor& self) { return self.layout(); })
        .def_property_readonly("tile", [](const Tensor& self) { return self.tensor_spec().tile(); })
        .def(
            "deallocate",
            [](Tensor& self, bool force) { return self.deallocate(force); },
            py::arg("force") = false,
            R"doc(
                Dellocates all data of a tensor. This either deletes all host data or deallocates tensor data from device memory.
            )doc")
        .def(
            "to",
            [](const Tensor& self, MeshDevice* device, std::optional<const MemoryConfig>& mem_config, QueueId cq_id) {
                return self.to_device(device, mem_config, cq_id);
            },
            py::arg("device").noconvert(),
            py::arg("mem_config").noconvert() = std::nullopt,
            py::arg("cq_id") = ttnn::DefaultQueueId,
            py::keep_alive<0, 2>(),
            R"doc(
            Move TT Tensor from host device to TT accelerator device.

            Only BFLOAT16 (in ROW_MAJOR or TILE layout) and BFLOAT8_B, BFLOAT4_B (in TILE layout) are supported on device.

            If ``arg1`` is not supplied, memory config from the source tensor will be used.

            +-----------+-------------------------------------------------+----------------------------+-----------------------+----------+
            | Argument  | Description                                     | Data type                  | Valid range           | Required |
            +===========+=================================================+============================+=======================+==========+
            | arg0      | MeshDevice to which tensor will be moved        | ttnn.MeshDevice            | TT accelerator device | Yes      |
            +-----------+-------------------------------------------------+----------------------------+-----------------------+----------+
            | arg1      | MemoryConfig of tensor of TT accelerator device | ttnn.MemoryConfig          |                       | No       |
            +-----------+-------------------------------------------------+----------------------------+-----------------------+----------+
            | arg2      | CQ ID of TT accelerator device to use           | uint8_t                    |                       | No       |
            +-----------+-------------------------------------------------+----------------------------+-----------------------+----------+

            .. code-block:: python

                tt_tensor = tt_tensor.to(tt_device)
        )doc")
        .def(
            "extract_shard",
            [](const Tensor& self, CoreCoord core) { return self.extract_shard(core); },
            py::arg("core").noconvert(),
            py::keep_alive<0, 2>(),
            R"doc(
            Move TT Tensor from host device to TT accelerator device.

            Only BFLOAT16 (in ROW_MAJOR or TILE layout) and BFLOAT8_B, BFLOAT4_B (in TILE layout) are supported on device.

            If ``arg1`` is not supplied, default ``MemoryConfig`` with ``interleaved`` set to ``True``.

            +-----------+-------------------------------------------------+----------------------------+-----------------------+----------+
            | Argument  | Description                                     | Data type                  | Valid range           | Required |
            +===========+=================================================+============================+=======================+==========+
            | arg0      | Core who's shard we want                        | ttnn.CoreCoord             | TT accelerator device | Yes      |
            +-----------+-------------------------------------------------+----------------------------+-----------------------+----------+


            .. code-block:: python

                tt_tensor = tt_tensor.to(tt_device)
        )doc")
        .def(
            "extract_shard",
            [](const Tensor& self, const uint32_t& core_id) { return self.extract_shard(core_id); },
            py::arg("core_id").noconvert(),
            py::keep_alive<0, 2>(),
            R"doc(
            Move TT Tensor from host device to TT accelerator device.

            Only BFLOAT16 (in ROW_MAJOR or TILE layout) and BFLOAT8_B, BFLOAT4_B (in TILE layout) are supported on device.

            If ``arg1`` is not supplied, default ``MemoryConfig`` with ``interleaved`` set to ``True``.

            +-----------+-------------------------------------------------+----------------------------+-----------------------+----------+
            | Argument  | Description                                     | Data type                  | Valid range           | Required |
            +===========+=================================================+============================+=======================+==========+
            | arg0      | Core who's shard we want                        | uint32_t                   | TT accelerator device | Yes      |
            +-----------+-------------------------------------------------+----------------------------+-----------------------+----------+


            .. code-block:: python

                tt_tensor = tt_tensor.to(tt_device)
        )doc")
        .def(
            "cpu",
            [](const Tensor& self, bool blocking, QueueId cq_id) { return self.cpu(blocking, cq_id); },
            py::arg("blocking") = true,
            py::arg("cq_id") = ttnn::DefaultQueueId,
            R"doc(
            Move TT Tensor from TT accelerator device to host device.

            .. code-block:: python

                tt_tensor = tt_tensor.cpu()
        )doc")
        .def(
            "item",
            [](const Tensor& self) -> py::object {
                switch (self.dtype()) {
                    case DataType::FLOAT32: return py::cast(self.item<float>());
                    case DataType::BFLOAT16: return py::cast(static_cast<float>(self.item<bfloat16>()));
                    case DataType::BFLOAT8_B:
                    case DataType::BFLOAT4_B: return py::cast(self.item<float>());
                    case DataType::INT32: return py::cast(self.item<int32_t>());
                    case DataType::UINT32: return py::cast(self.item<uint32_t>());
                    case DataType::UINT16: return py::cast(self.item<uint16_t>());
                    case DataType::UINT8: return py::cast(self.item<uint8_t>());
                    case DataType::INVALID: TT_THROW("Unsupported DataType");
                }
                TT_THROW("Unreachable");
            },
            R"doc(
                 Extract the scalar value from a tensor containing exactly one element.

                 Similar to PyTorch's tensor.item(), this method returns the value of this tensor as a standard Python number.
                 This only works for tensors with one element.

                 Returns:
                     Python scalar: The scalar value contained in the tensor.

                 Raises:
                     RuntimeError: If the tensor doesn't contain exactly one element.

                 .. code-block:: python

                     # Create a tensor with one element
                     scalar_tensor = ttnn.from_torch(torch.tensor([3.14]), device=device)
                     value = scalar_tensor.item()  # Returns 3.14
             )doc")
        .def(
            "to",
            py::overload_cast<Layout>(&Tensor::to_layout, py::const_),
            py::arg("target_layout").noconvert(),
            R"doc(
            Convert TT Tensor to provided memory layout. Available layouts conversions are:

            * ROW_MAJOR to TILE
            * TILE to ROW_MAJOR

            +-----------+-------------------------------------------------+----------------------------+--------------------------------+----------+
            | Argument  | Description                                     | Data type                  | Valid range                    | Required |
            +===========+=================================================+============================+================================+==========+
            | arg0      | Target memory layout                            | ttnn.Layout                | ROW_MAJOR, TILE                | Yes      |
            +-----------+-------------------------------------------------+----------------------------+--------------------------------+----------+

            .. code-block:: python

                tt_tensor = tt_tensor.to(ttnn.Layout.TILE)
        )doc")
        .def(
            "pad",
            [](const Tensor& self,
               const std::array<uint32_t, 4>& output_tensor_shape,
               const std::array<uint32_t, 4>& input_tensor_start,
               float pad_value) {
                return self.pad(ttnn::Shape(output_tensor_shape), ttnn::Shape(input_tensor_start), pad_value);
            },
            R"doc(
            Pad TT Tensor with given pad value ``arg2``.

            The input tensor must be on host and in ROW_MAJOR layout.

            Returns an output tensor that contains the input tensor at the given input tensor start indices ``arg1`` and the padded value everywhere else.

            +---------------------+------------------------------------------------------+--------------+-----------------------------------------------------+----------+
            | Argument            | Description                                          | Data type    | Valid range                                         | Required |
            +=====================+======================================================+==============+=====================================================+==========+
            | arg0                | Shape of output tensor                               | List[int[4]] |                                                     | Yes      |
            +---------------------+------------------------------------------------------+--------------+-----------------------------------------------------+----------+
            | arg1                | Start indices to place input tensor in output tensor | List[int[4]] | Values along each dim must be                       | Yes      |
            |                     |                                                      |              |                                                     |          |
            |                     |                                                      |              | <= (output_tensor_shape[i] - input_tensor_shape[i]) |          |
            +---------------------+------------------------------------------------------+--------------+-----------------------------------------------------+----------+
            | arg2                | Value to pad input tensor                            | float        |                                                     | Yes      |
            +---------------------+------------------------------------------------------+--------------+-----------------------------------------------------+----------+

            .. code-block:: python

                input_tensor_shape = [1, 1, 3, 3]
                output_tensor_shape = [1, 2, 5, 5]
                input_tensor_start = [0, 1, 1, 1]
                pad_value = 0

                inp = torch.Tensor(
                    [ 1, 2, 3,
                    4, 5, 6,
                    7, 8, 9 ]
                )
                tt_tensor = ttnn.Tensor(
                    inp.tolist(),
                    input_tensor_shape,
                    ttnn.DataType.BFLOAT16,
                    ttnn.Layout.ROW_MAJOR,
                )
                tt_tensor_padded = tt_tensor.pad(output_tensor_shape, input_tensor_start, pad_value)

                print("Input tensor:")
                print(tt_tensor)
                print("\nPadded tensor:")
                print(tt_tensor_padded)

            Example output:

            .. code-block::

                Input tensor:
                [ [[[1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9]]] dtype=bfloat16 ]

                Padded tensor:
                [ [[[0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0]],

                    [[0, 0, 0, 0, 0],
                    [0, 1, 2, 3, 0],
                    [0, 4, 5, 6, 0],
                    [0, 7, 8, 9, 0],
                    [0, 0, 0, 0, 0]]] dtype=bfloat16 ]
        )doc")
        .def(
            "unpad",
            [](const Tensor& self,
               const ttnn::SmallVector<uint32_t>& output_tensor_start,
               const ttnn::SmallVector<uint32_t>& output_tensor_end) {
                return self.unpad(ttnn::Shape(output_tensor_start), ttnn::Shape(output_tensor_end));
            },
            R"doc(
            Unpad this TT Tensor.

            This tensor must be on host and in ROW_MAJOR layout.

            Returns an output tensor from output tensor start indices ``arg0`` to output tensor end indices ``arg1`` (inclusive) of the input tensor.

            +---------------------+----------------------------------------------+--------------+-----------------------------------------------------+----------+
            | Argument            | Description                                  | Data type    | Valid range                                         | Required |
            +=====================+==============================================+==============+=====================================================+==========+
            | arg0                | Start indices of input tensor                | List[int]    | Values along each dim must be                       | Yes      |
            |                     |                                              |              |                                                     |          |
            |                     |                                              |              | < input_tensor_shape[i] and <= output_tensor_end[i] |          |
            +---------------------+----------------------------------------------+--------------+-----------------------------------------------------+----------+
            | arg1                | End indices of input tensor in output tensor | List[int]    | Values along each dim must be                       | Yes      |
            |                     |                                              |              |                                                     |          |
            |                     |                                              |              | < input_tensor_shape[i]                             |          |
            +---------------------+----------------------------------------------+--------------+-----------------------------------------------------+----------+

            .. code-block:: python

                input_tensor_shape = [1, 1, 5, 5]
                output_tensor_start = [0, 0, 1, 1]
                output_tensor_end = [0, 0, 3, 3]

                inp = torch.Tensor(
                    [ 0, 0, 0, 0, 0,
                    0, 1, 2, 3, 0,
                    0, 4, 5, 6, 0,
                    0, 7, 8, 9, 0,
                    0, 0, 0, 0, 0 ]
                )
                tt_tensor = ttnn.Tensor(
                    inp.tolist(),
                    input_tensor_shape,
                    ttnn.DataType.BFLOAT16,
                    ttnn.Layout.ROW_MAJOR,
                )
                tt_tensor_unpadded = tt_tensor.unpad(output_tensor_start, output_tensor_end)

                print("Input tensor:")
                print(tt_tensor)
                print("\nUnpadded tensor:")
                print(tt_tensor_unpadded)

            Example output:

            .. code-block::

                Input tensor:
                [ [[[0, 0, 0, 0, 0],
                    [0, 1, 2, 3, 0],
                    [0, 4, 5, 6, 0],
                    [0, 7, 8, 9, 0],
                    [0, 0, 0, 0, 0]]] dtype=bfloat16 ]

                Unpadded tensor:
                [ [[[1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9]]] dtype=bfloat16 ]
        )doc")
        .def(
            "pad_to_tile", [](const Tensor& self, float pad_value) { return self.pad_to_tile(pad_value); }, R"doc(
            Pads TT Tensor with given pad value ``arg0``.

            The input tensor must be on host and in ROW_MAJOR layout.

            Returns an output tensor that contains the input tensor padded with the padded value in the last two dims to multiples of 32.

            Padding will be added to the right and bottom of the tensor.

            +---------------------+------------------------------------------------------+--------------+-----------------------------------------------------+----------+
            | Argument            | Description                                          | Data type    | Valid range                                         | Required |
            +=====================+======================================================+==============+=====================================================+==========+
            | arg0                | Value to pad input tensor                            | float        |                                                     | Yes      |
            +---------------------+------------------------------------------------------+--------------+-----------------------------------------------------+----------+

            .. code-block:: python

                input_tensor_shape = [1, 1, 3, 3]
                pad_value = 0

                inp = torch.Tensor(
                    [ 1, 2, 3,
                    4, 5, 6,
                    7, 8, 9 ]
                )
                tt_tensor = ttnn.Tensor(
                    inp.tolist(),
                    input_tensor_shape,
                    ttnn.DataType.BFLOAT16,
                    ttnn.Layout.ROW_MAJOR,
                )
                tt_tensor_padded = tt_tensor.pad_to_tile(pad_value)

                print("Input tensor:")
                print(tt_tensor)
                print("\nPadded tensor:")
                print(tt_tensor_padded)

            Example output:

            .. code-block::

                Input tensor:
                [ [[[1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9]]] dtype=bfloat16 ]

                Padded tensor:
                [ [[[1, 2, 3, 0, ..., 0],
                    [4, 5, 6, 0, ..., 0],
                    [7, 8, 9, 0, ..., 0],
                    [0, 0, 0, 0, ..., 0],
                    ...,
                    [0, 0, 0, 0, ..., 0]]] dtype=bfloat16 ]
        )doc")
        .def(
            "unpad_from_tile",
            [](const Tensor& self, const ttnn::SmallVector<uint32_t>& output_tensor_shape) {
                return self.unpad_from_tile(ttnn::Shape(output_tensor_shape));
            },
            R"doc(
            Unpads TT Tensor from given input tensor ``arg0``.

            The input tensor must be on host and in ROW_MAJOR layout.

            This function expects the real data to aligned on the top left of the tensor.

            Returns an output tensor with padding removed from the right and bottom of the input tensor.

            +---------------------+----------------------------------------------+--------------+------------------------------------------------------------------------------+----------+
            | Argument            | Description                                  | Data type    | Valid range                                                                  | Required |
            +=====================+==============================================+==============+==============================================================================+==========+
            | arg0                | Shape of output tensor                       | List[int[4]] | All dims must match the input tensor dims apart from the last two dims.      | Yes      |
            |                     |                                              |              |                                                                              |          |
            |                     |                                              |              | Last two dims have the following restrictions:                               |          |
            |                     |                                              |              |                                                                              |          |
            |                     |                                              |              | input_tensor_shape[i] must be a multiple of 32                               |          |
            |                     |                                              |              |                                                                              |          |
            |                     |                                              |              | input_tensor_shape[i] - 32 < output_tensor_shape[i] <= input_tensor_shape[i] |          |
            +---------------------+----------------------------------------------+--------------+------------------------------------------------------------------------------+----------+


            .. code-block:: python

                input_tensor_shape = [1, 1, 32, 32]
                output_tensor_shape = [1, 1, 3, 3]

                inp = torch.arange(start=1.0, end=10.0).reshape(1, 1, 3, 3)
                inp = torch.nn.functional.pad(inp, [0, input_tensor_shape[3] - inp.shape[3], 0, input_tensor_shape[2] - inp.shape[2]]).reshape(-1)
                tt_tensor = ttnn.Tensor(
                    inp.tolist(),
                    input_tensor_shape,
                    ttnn.DataType.BFLOAT16,
                    ttnn.Layout.ROW_MAJOR,
                )
                tt_tensor_unpadded = tt_tensor.unpad_from_tile(output_tensor_shape)

                print("Input tensor:")
                print(tt_tensor)
                print("\nUnpadded tensor:")
                print(tt_tensor_unpadded)

            Example output:

            .. code-block::

                Input tensor:
                [ [[[1, 2, 3, 0, ..., 0],
                    [4, 5, 6, 0, ..., 0],
                    [7, 8, 9, 0, ..., 0],
                    [0, 0, 0, 0, ..., 0],
                    ...,
                    [0, 0, 0, 0, ..., 0]]] dtype=bfloat16 ]

                Unpadded tensor:
                [ [[[1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9]]] dtype=bfloat16 ]
        )doc")
        .def(
            "__repr__", [](const Tensor& self) { return self.write_to_string(); }, R"doc(
            Prints the tensor as list of nested lists. Number of levels of nesting is equal to tensor rank.

            .. code-block:: python

                print(tt_tensor)

            Example output for a rank 4 TT Tensor with shape (1, 1, 32, 32):

            .. code-block::

                [ [[[0.220703, 0.839844, 0.960938, ..., 0.378906, 0.507812],
                [0.03125, 0.511719, 0.0407715, ..., 0.945312, 0.671875],
                ...
                [0.433594, 0.165039, 0.980469, ..., , 0.349609]]] dtype=bfloat16 ]

        )doc")
        .def(
            // TODO: Rename to physical_volume
            "volume",
            [](const Tensor& self) { return self.physical_volume(); },
            R"doc(
            Get the volume of the tensor.

            .. code-block:: python

                volume = tt_tensor.physical_volume()

        )doc")
        .def(
            "logical_volume",
            [](const Tensor& self) { return self.logical_volume(); },
            R"doc(
            Get the logical volume of the tensor.

            .. code-block:: python

                volume = tt_tensor.logical_volume()

        )doc")
        .def(
            "storage_type", [](const Tensor& self) { return self.storage_type(); }, R"doc(
            Check if the tensor is on host

            .. code-block:: python

                storage_type = tt_tensor.storage_type()

        )doc")
        .def(
            "device",
            [](const Tensor& self) { return dynamic_cast<MeshDevice*>(self.device()); },
            R"doc(
            Get the device of the tensor.

            .. code-block:: python

                device = tt_tensor.device()

        )doc",
            py::return_value_policy::reference)
        .def(
            "devices",
            [](const Tensor& self) { return std::vector<MeshDevice*>{dynamic_cast<MeshDevice*>(self.device())}; },
            R"doc(
            Get devices tensor is mapped on to.

            .. code-block:: python

                devices = tt_tensor.devices()

        )doc",
            py::return_value_policy::reference)
        .def(
            "to_torch_with_padded_shape",
            [](const Tensor& self) -> py::object {
                using namespace CMAKE_UNIQUE_NAMESPACE;

                auto buffer = convert_to_row_major_host_buffer(self, /*padded_output=*/true);
                return convert_tt_tensor_to_torch_tensor(buffer);
            },
            R"doc(
            Convert tensor to torch tensor using legacy padded shape.
            WARNING: Will be deprecated soon!

            The tensor must be on host when calling this function.

            .. code-block:: python

                data = tt_tensor.cpu().to_torch_with_padded_shape() # move TT Tensor to host and convert it to torch tensor

        )doc")
        .def(
            "to_torch",
            [](const Tensor& self, const ttnn::distributed::MeshToTensor* mesh_composer) -> py::object {
                using namespace CMAKE_UNIQUE_NAMESPACE;

                auto buffer = mesh_composer ? convert_to_row_major_host_buffer(self, *mesh_composer)
                                            : convert_to_row_major_host_buffer(self, /*padded_output=*/false);
                return convert_tt_tensor_to_torch_tensor(buffer);
            },
            py::arg("mesh_composer") = nullptr,
            R"doc(
            Convert tensor to torch tensor.

            The tensor must be on host when calling this function.

            .. code-block:: python

                data = tt_tensor.cpu().to_torch() # move TT Tensor to host and convert it to torch tensor

        )doc")
        .def(
            "to_numpy",
            [](const Tensor& self, const ttnn::distributed::MeshToTensor* mesh_composer) -> py::object {
                using namespace CMAKE_UNIQUE_NAMESPACE;

                auto buffer = mesh_composer ? convert_to_row_major_host_buffer(self, *mesh_composer)
                                            : convert_to_row_major_host_buffer(self, /*padded_output=*/false);
                return convert_tt_tensor_to_numpy_tensor(buffer);
            },
            py::arg("mesh_composer") = nullptr,
            R"doc(
            Convert tensor to numpy tensor.

            The tensor must be on host when calling this function.

            .. code-block:: python

                data = tt_tensor.cpu().to_numpy() # move TT Tensor to host and convert it to numpy tensor

        )doc")
        .def(
            "host_buffer",
            [](Tensor& self) -> DistributedHostBuffer {
                TT_FATAL(self.storage_type() == StorageType::HOST, "Tensor must be on host to access host_buffer");
                return self.host_storage().buffer();
            },
            R"doc(
            Get the underlying host buffer.

            The tensor must be on the cpu when calling this function.

            .. code-block:: python

                buffer = tt_tensor.cpu().host_buffer() # move TT Tensor to host and get the buffer

        )doc")
        .def(
            "buffer_address",
            [](const Tensor& self) -> uint32_t {
                return std::visit(
                    tt::stl::overloaded{
                        [](const DeviceStorage& s) -> uint32_t {
                            TT_FATAL(s.mesh_buffer != nullptr, "Tensor is not allocated.");
                            return s.mesh_buffer->address();
                        },
                        [&](auto&&) -> uint32_t {
                            TT_THROW(
                                "{} doesn't support buffer_address method",
                                tt::stl::get_active_type_name_in_variant(self.storage()));
                        },
                    },
                    self.storage());
            },
            R"doc(
            Get the address of the underlying buffer.

            The tensor must be on the single device when calling this function.

            .. code-block:: python

                address = tt_tensor.buffer_address()

        )doc")
        .def(
            "get_layout", [](const Tensor& self) { return self.layout(); }, R"doc(
            Get memory layout of TT Tensor.

            .. code-block:: python

                layout = tt_tensor.layout()

        )doc")
        .def(
            "get_tile", [](const Tensor& self) { return self.tensor_spec().tile(); }, R"doc(
            Get tile dims of TT Tensor.

            .. code-block:: python

                tile = tt_tensor.get_tile()

        )doc")
        .def(
            "memory_config", [](const Tensor& self) { return self.memory_config(); }, R"doc(
            Get buffer type of TT Tensor.

            .. code-block:: python

                memory_config = tt_tensor.memory_config()

        )doc")
        .def(
            "is_allocated", [](const Tensor& self) { return self.is_allocated(); }, R"doc(
            Check if TT Tensor is allocated.

            .. code-block:: python

                is_sharded = tt_tensor.is_sharded()

        )doc")
        .def(
            "is_sharded", [](const Tensor& self) { return self.is_sharded(); }, R"doc(
            Check if TT Tensor is sharded.

            .. code-block:: python

                is_sharded = tt_tensor.is_sharded()

        )doc")
        .def(
            "get_dtype", [](const Tensor& self) { return self.dtype(); }, R"doc(
            Get dtype of TT Tensor.

            .. code-block:: python

                dtype = tt_tensor.dtype()
        )doc")
        .def(
            "reshape",
            [](Tensor& self, int N, int C, int H, int W) {
                return ttnn::reshape(self, infer_dims_for_reshape(self, ttnn::SmallVector<int>{N, C, H, W}));
            },
            R"doc(
                Reshapes TT tensor

                .. code-block:: python

                    reshaped_tensor = tt_tensor.reshape(N, C, H, W)
            )doc")
        .def(
            "reshape",
            [](Tensor& self, const ttnn::Shape& shape) -> Tensor { return ttnn::reshape(self, shape); },
            R"doc(
                Reshapes TT tensor

                .. code-block:: python

                    reshaped_tensor = tt_tensor.reshape((4, 3, 32))
            )doc")
        .def(
            "reshape",
            [](Tensor& self, const ttnn::SmallVector<int32_t>& shape) -> Tensor {
                return ttnn::reshape(self, infer_dims_for_reshape(self, shape));
            },
            R"doc(
                Reshapes TT tensor

                .. code-block:: python

                    reshaped_tensor = tt_tensor.reshape((4, -1, 32))
            )doc")
        .def(
            "to_list",
            [](Tensor& self) {
                using namespace tt::tt_metal::tensor_impl;
                return dispatch(self.dtype(), [&]<typename T>() -> py::list {
                    const auto& logical_shape = self.logical_shape();
                    std::vector<uint32_t> shape{logical_shape.cbegin(), logical_shape.cend()};

                    if constexpr (
                        std::is_same_v<T, bfloat8_b> || std::is_same_v<T, bfloat4_b> || std::is_same_v<T, bfloat16>) {
                        return py::array(shape, self.to_vector<float>().data()).attr("tolist")();
                    } else {
                        return py::array(shape, self.to_vector<T>().data()).attr("tolist")();
                    }
                });
            },
            R"doc(
                Return TT tensor values as python list

                .. code-block:: python

                    py_list = tt_tensor.to_list()
            )doc")
        .def(
            "tensor_topology",
            [](const Tensor& self) { return self.tensor_topology(); },
            R"doc(
                Get the topology of the tensor.

                .. code-block:: python

                    topology = tt_tensor.tensor_topology()
            )doc")
        .def_property(
            "tensor_id",
            [](const Tensor& self) { return self.tensor_id; },
            [](Tensor& self, std::size_t tensor_id) { self.tensor_id = tensor_id; });
}

}  // namespace ttnn::tensor
