#pragma once

#include "ttnn/tensor/shape/shape.hpp"
#include "ttnn/tensor/tensor_spec.hpp"
#include "ttnn/tensor/layout/tensor_layout.hpp"
#include "ttnn/tensor/layout/page_config.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/conv/conv2d/conv2d.hpp"

#include <array>
#include <cstdint>

namespace tt::tt_metal {
// Structure to hold all parameters for a test instance
struct QueryOpConstraintsParams {
    // void* device;
    ::ttnn::TensorSpec input_spec;
    std::array<uint32_t, 4> original_weights_shape;
    ::ttnn::TensorSpec weight_spec;
    uint32_t in_channels;
    uint32_t out_channels;
    uint32_t batch_size;
    uint32_t input_height;
    uint32_t input_width;
    std::array<uint32_t, 2> kernel_size;
    std::array<uint32_t, 2> stride;
    std::array<uint32_t, 4> padding;
    std::array<uint32_t, 2> dilation;
    uint32_t groups;
    std::optional<::ttnn::TensorSpec> bias_spec;
    ::ttnn::operations::conv::conv2d::Conv2dConfig conv2d_config;
    MemoryConfig memory_config;
};

const std::vector<QueryOpConstraintsParams> params = {};

inline ::tt::tt_metal::Shape MakeShape(const std::vector<int64_t>& dims) {
    std::vector<uint32_t> dims_u32;
    dims_u32.reserve(dims.size());
    for (auto d : dims) {
        dims_u32.push_back(static_cast<uint32_t>(d));
    }
    // Use initializer_list constructor
    return ::tt::tt_metal::Shape(dims_u32);
}
// Helper function to construct TensorSpec with memory config
inline ::ttnn::TensorSpec MakeTensorSpec(
    const std::vector<int64_t>& logical_shape,
    ::ttnn::DataType dtype,
    Layout layout_type,
    const std::array<uint32_t, 2>& tile_shape,
    const std::array<uint32_t, 2>& face_shape,
    uint32_t num_faces,
    const MemoryConfig& memory_config) {
    auto shape = MakeShape(logical_shape);
    auto tile = ::tt::tt_metal::Tile(tile_shape);
    auto page_config = ::tt::tt_metal::PageConfig(layout_type, tile);
    auto layout = ::tt::tt_metal::TensorLayout(dtype, page_config, memory_config);
    return ::ttnn::TensorSpec(shape, layout);
}

// Helper for ShardSpec (adapt as needed for your API)
inline ::tt::tt_metal::ShardSpec MakeShardSpec(
    const std::vector<std::pair<CoreCoord, CoreCoord>>& grid_ranges,
    const std::array<uint32_t, 2>& shape,
    ::tt::tt_metal::ShardOrientation orientation) {
    std::set<CoreRange> ranges;
    for (const std::pair<CoreCoord, CoreCoord>& range : grid_ranges) {
        ranges.insert(CoreRange(range.first, range.second));
    }
    CoreRangeSet core_range_set(ranges);
    return ::tt::tt_metal::ShardSpec(core_range_set, shape, orientation);
}

// Helper for MemoryConfig
inline MemoryConfig MakeMemoryConfig(
    TensorMemoryLayout layout, BufferType buffer_type, const std::optional<ShardSpec>& shard_spec) {
    return MemoryConfig(layout, buffer_type, shard_spec);
}

}  // namespace tt::tt_metal
