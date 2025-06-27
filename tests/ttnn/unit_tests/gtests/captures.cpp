#include <gtest/gtest.h>
#include <iostream>
#include <optional>
#include <array>
#include <vector>
#include <fmt/base.h>
#include <gtest/gtest.h>
#include <stdint.h>
#include <array>
#include <memory>
#include <optional>

#include "device.hpp"
#include "ttnn/cpp/ttnn/operations/conv/conv2d/device/conv2d_op.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/conv/conv2d/prepare_conv2d_weights.hpp"
#include "device/device_impl.hpp"
#include "gtest/gtest.h"
#include "ttnn/operations/creation.hpp"
#include "ttnn/operations/conv/conv2d/conv2d.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_types.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/operations/functions.hpp"
#include "ttnn/operations/matmul/device/matmul_op.hpp"
#include "ttnn/operations/matmul/matmul.hpp"
#include "ttnn/tensor/shape/shape.hpp"

#include <tt-metalium/assert.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/shape.hpp>
#include "ttnn/common/queue_id.hpp"
#include "ttnn/decorators.hpp"
#include "ttnn/device.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/functions.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/types.hpp"
#include "ttnn_suite_device_fixture.hpp"
#include "ttnn_test_fixtures.hpp"

#include "ttnn/graph/graph_processor.hpp"
#include "ttnn/graph/graph_query_op_constraints.hpp"
#include "ttnn/tensor/tensor_spec.hpp"
#include "ttnn/tensor/layout/tensor_layout.hpp"
#include "ttnn/tensor/layout/page_config.hpp"
#include "ttnn/operations/conv/conv2d/device/conv2d_op.hpp"
#include "ttnn/tensor/tensor.hpp"

#include <gtest/gtest.h>
#include <optional>
#include <array>
#include <vector>

#include "captured_api_calls_small.h"

namespace tt::tt_metal {

class QueryOpConstraintsTest : public TTNNSuiteDeviceFixture {};

extern std::vector<QueryOpConstraintsParams> kAllQueryOpConstraintsParams;

static ::tt::tt_metal::HostBuffer createHostBuffer(uint32_t numElements, ::tt::tt_metal::DataType dataType) {
    switch (dataType) {
        case ::tt::tt_metal::DataType::FLOAT32: {
            std::vector<float> data(numElements);
            return ::tt::tt_metal::HostBuffer(std::move(data));
        }
        case ::tt::tt_metal::DataType::BFLOAT16: {
            std::vector<bfloat16> data(numElements);
            return ::tt::tt_metal::HostBuffer(std::move(data));
        }
        default: throw std::runtime_error("Unsupported data type");
    }
}

static ::tt::tt_metal::Tensor createMetalHostTensor(
    const std::array<uint32_t, 4>& shape, ::tt::tt_metal::DataType dataType) {
    // Calculate total volume of the tensor
    uint32_t volume = 1;
    for (size_t i = 0; i < shape.size(); i++) {
        volume *= shape[i];
    }

    auto hostBuffer = createHostBuffer(volume, dataType);
    ::tt::tt_metal::PageConfig pageconfig(::tt::tt_metal::Layout::ROW_MAJOR);
    ::tt::tt_metal::TensorLayout layout(dataType, pageconfig, ::tt::tt_metal::MemoryConfig{});
    ::tt::tt_metal::TensorSpec tensorSpec(::tt::tt_metal::Shape(shape), layout);

    return ::tt::tt_metal::Tensor(std::move(hostBuffer), tensorSpec);
}

TEST_F(QueryOpConstraintsTest, InstantiatesAndRunsBatch) {
    IDevice* device = device_;

    int query_count = 0;
    for (int loop = 0; loop < 100; loop++) {
        for (const auto& p : kAllQueryOpConstraintsParams) {
            std::cout << "Loop " << loop << " Query #" << query_count << std::endl;
            std::optional<Tensor> bias_tensor = std::nullopt;
            if (p.bias_spec) {
                std::cout << "Creating bias tensor" << std::endl;
                bias_tensor = create_device_tensor(p.bias_spec.value(), device);
            }

            auto weightTensor = createMetalHostTensor(p.original_weights_shape, p.weight_spec.data_type());
            const std::optional<const ttnn::operations::conv::conv2d::Conv2dConfig> conv2localcfg = p.conv2d_config;
            auto q_prep_conv2d_weights = ::ttnn::graph::query_op_constraints(
                &::ttnn::operations::conv::conv2d::prepare_conv_weights<IDevice>,
                device,
                weightTensor,
                p.input_spec.memory_config(),
                p.input_spec.layout(),
                "OIHW",
                p.in_channels,
                p.out_channels,
                p.batch_size,
                p.input_height,
                p.input_width,
                p.kernel_size,
                p.stride,
                p.padding,
                p.dilation,
                p.bias_spec ? true : false,
                p.groups,
                device,
                // p.conv2d_config,
                conv2localcfg,
                std::optional<
                    const std::variant<ttnn::GrayskullComputeKernelConfig, ttnn::WormholeComputeKernelConfig>>{},
                std::optional<const ttnn::operations::conv::Conv2dSliceConfig>{});

            auto result = ::ttnn::graph::query_op_constraints(
                ::ttnn::conv2d,
                device,
                p.input_spec,
                p.weight_spec,
                device,
                p.in_channels,
                p.out_channels,
                p.batch_size,
                p.input_height,
                p.input_width,
                p.kernel_size,
                p.stride,
                p.padding,
                p.dilation,
                p.groups,
                bias_tensor,
                p.conv2d_config,
                std::nullopt,  // compute config
                p.memory_config);
            query_count++;
            if (query_count >= 10000000) {
                break;
            }
        }
    }

    device->close();

    std::map<chip_id_t, IDevice*> devices = detail::CreateDevices(
        std::vector<chip_id_t>{0},
        1,  // num_hw_cqs
        TTNNSuiteDeviceFixture::L1_SMALL_SIZE,
        TTNNSuiteDeviceFixture::TRACE_REGION_SIZE);

    std::cout << "Opened devices: " << std::endl;
    for (auto& [device_id, device] : devices) {
        std::cout << "Device " << device_id << std::endl;
    }

    query_count = 0;

    device = devices.begin()->second;
    device->enable_program_cache();

    // call graph::query_op_constraints again
    for (const auto& p : kAllQueryOpConstraintsParams) {
        std::optional<Tensor> bias_tensor = std::nullopt;
        if (p.bias_spec) {
            bias_tensor = create_device_tensor(p.bias_spec.value(), device);
        }
        auto result = ::ttnn::graph::query_op_constraints(
            ::ttnn::conv2d,
            device,
            p.input_spec,
            p.weight_spec,
            device,
            p.in_channels,
            p.out_channels,
            p.batch_size,
            p.input_height,
            p.input_width,
            p.kernel_size,
            p.stride,
            p.padding,
            p.dilation,
            p.groups,
            bias_tensor,
            p.conv2d_config,
            std::nullopt,  // compute config
            p.memory_config);

        if (++query_count > 0) {
            // lets break immediately
            break;
        }
    }

    ::ttnn::Tensor inputTensorA =
        ttnn::ones(::ttnn::Shape({8, 1, 49, 2048}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE);

    ::ttnn::Tensor inputTensorB =
        ttnn::ones(::ttnn::Shape({8, 1, 49, 2048}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE);

    auto inputTensorA_on_device = ttnn::to_device(
        inputTensorA, device, ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM});
    auto inputTensorB_on_device = ttnn::to_device(
        inputTensorB, device, ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM});

    auto a = ttnn::add(inputTensorA_on_device, inputTensorB_on_device, ::ttnn::DataType::BFLOAT16);

    for (auto& [device_id, device] : devices) {
        // Close the device
        std::cout << "Closing device " << device_id << std::endl;
        device->close();
    }
}

// Example instantiation for one log line (fill with real values from your logs)
// INSTANTIATE_TEST_SUITE_P(
//     FromLogs,
//     QueryOpConstraintsTest,
//     ::testing::Values(QueryOpConstraintsParams{
//         /* input_spec */
//         MakeTensorSpec(
//             {1, 1, 401408, 3},
//             ::ttnn::DataType::BFLOAT16,
//             Layout::TILE,
//             {32, 32},
//             {16, 16},
//             4,
//             MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM, std::nullopt}),
//         /* weight_spec */
//         MakeTensorSpec(
//             {1, 1, 147, 64},
//             ::ttnn::DataType::BFLOAT16,
//             Layout::TILE,
//             {32, 32},
//             {16, 16},
//             4,
//             MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM, std::nullopt}),
//         /* in_channels */ 3,
//         /* out_channels */ 64,
//         /* batch_size */ 8,
//         /* input_height */ 224,
//         /* input_width */ 224,
//         /* kernel_size */ {7, 7},
//         /* stride */ {2, 2},
//         /* padding */ {3, 3, 3, 3},
//         /* dilation */ {1, 1},
//         /* groups */ 1,
//         /* bias_spec */ std::nullopt,
//         /* conv2d_config */
//         [] {
//             ::ttnn::operations::conv::conv2d::Conv2dConfig cfg;
//             cfg.dtype = ::ttnn::DataType::BFLOAT16;
//             cfg.weights_dtype = ::ttnn::DataType::BFLOAT16;
//             cfg.activation = "";
//             cfg.deallocate_activation = false;
//             cfg.reallocate_halo_output = true;
//             cfg.act_block_h_override = 0;
//             cfg.act_block_w_div = 1;
//             cfg.reshard_if_not_optimal = false;
//             cfg.override_sharding_config = false;
//             cfg.shard_layout = std::nullopt;
//             cfg.core_grid = std::nullopt;
//             cfg.transpose_shards = true;
//             cfg.output_layout = ::ttnn::Layout::TILE;
//             cfg.preprocess_weights_on_device = false;
//             cfg.enable_act_double_buffer = false;
//             cfg.enable_weights_double_buffer = false;
//             cfg.enable_split_reader = false;
//             cfg.enable_subblock_padding = false;
//             cfg.in_place = false;
//             return cfg;
//         }(),
//         /* memory_config */
//         [] {
//             auto shard_spec =
//                 MakeShardSpec({{{0, 0}, {7, 7}}}, {1568, 64}, ShardOrientation::ROW_MAJOR);
//             return MakeMemoryConfig(TensorMemoryLayout::HEIGHT_SHARDED, BufferType::L1, shard_spec);
//         }()}));

// INSTANTIATE_TEST_SUITE_P(
//     );

// INSTANTIATE_TEST_SUITE_P(
//     Debug,
//     QueryOpConstraintsTest,
//     ::testing::Values(
//         QueryOpConstraintsParams{
//             /* input_spec */
//             MakeTensorSpec(
//                 {1, 1, 392, 512},
//                 ::ttnn::DataType::BFLOAT16,
//                 Layout::TILE,
//                 {32, 32},
//                 {16, 16},
//                 4,
//                 MakeMemoryConfig(TensorMemoryLayout::BLOCK_SHARDED, BufferType::L1, MakeShardSpec({{{0, 0}, {7, 6}}},
//                 {64, 64}, ShardOrientation::ROW_MAJOR))),
//             /* weight_spec */
//             MakeTensorSpec(
//                 {1, 1, 512, 2048},
//                 ::ttnn::DataType::BFLOAT16,
//                 Layout::TILE,
//                 {32, 32},
//                 {16, 16},
//                 4,
//                 MakeMemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::DRAM, std::nullopt)),
//             /* in_channels */ 512,
//             /* out_channels */ 2048,
//             /* batch_size */ 8,
//             /* input_height */ 7,
//             /* input_width */ 7,
//             /* kernel_size */ {1, 1},
//             /* stride */ {1, 1},
//             /* padding */ {0, 0, 0, 0},
//             /* dilation */ {1, 1},
//             /* groups */ 1,
//             /* bias_spec */ std::nullopt,
//             /* conv2d_config */
//             [] {
//                 ::ttnn::operations::conv::conv2d::Conv2dConfig cfg;
//                 cfg.dtype = ::ttnn::DataType::BFLOAT16;
//                 cfg.weights_dtype = ::ttnn::DataType::BFLOAT16;
//                 cfg.activation = "";
//                 cfg.deallocate_activation = false;
//                 cfg.reallocate_halo_output = true;
//                 cfg.act_block_h_override = 0;
//                 cfg.act_block_w_div = 1;
//                 cfg.reshard_if_not_optimal = false;
//                 cfg.override_sharding_config = false;
//                 cfg.shard_layout = std::nullopt;
//                 cfg.core_grid = std::nullopt;
//                 cfg.transpose_shards = true;
//                 cfg.output_layout = ::ttnn::Layout::TILE;
//                 cfg.preprocess_weights_on_device = false;
//                 cfg.enable_act_double_buffer = false;
//                 cfg.enable_weights_double_buffer = false;
//                 cfg.enable_split_reader = false;
//                 cfg.enable_subblock_padding = false;
//                 cfg.in_place = false;
//                 return cfg;
//             }(),
//             /* memory_config */
//             MakeMemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::DRAM, std::nullopt)
//         },
//         QueryOpConstraintsParams{
//             /* input_spec */
//             MakeTensorSpec(
//                 {1, 1, 392, 512},
//                 ::ttnn::DataType::BFLOAT16,
//                 Layout::TILE,
//                 {32, 32},
//                 {16, 16},
//                 4,
//                 MakeMemoryConfig(TensorMemoryLayout::BLOCK_SHARDED, BufferType::L1, MakeShardSpec({{{0, 0}, {7, 6}}},
//                 {64, 64}, ShardOrientation::ROW_MAJOR))),
//             /* weight_spec */
//             MakeTensorSpec(
//                 {1, 1, 512, 2048},
//                 ::ttnn::DataType::BFLOAT16,
//                 Layout::TILE,
//                 {32, 32},
//                 {16, 16},
//                 4,
//                 MakeMemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::DRAM, std::nullopt)),
//             /* in_channels */ 512,
//             /* out_channels */ 2048,
//             /* batch_size */ 8,
//             /* input_height */ 7,
//             /* input_width */ 7,
//             /* kernel_size */ {1, 1},
//             /* stride */ {1, 1},
//             /* padding */ {0, 0, 0, 0},
//             /* dilation */ {1, 1},
//             /* groups */ 1,
//             /* bias_spec */ std::nullopt,
//             /* conv2d_config */
//             [] {
//                 ::ttnn::operations::conv::conv2d::Conv2dConfig cfg;
//                 cfg.dtype = ::ttnn::DataType::BFLOAT16;
//                 cfg.weights_dtype = ::ttnn::DataType::BFLOAT16;
//                 cfg.activation = "";
//                 cfg.deallocate_activation = false;
//                 cfg.reallocate_halo_output = true;
//                 cfg.act_block_h_override = 0;
//                 cfg.act_block_w_div = 1;
//                 cfg.reshard_if_not_optimal = false;
//                 cfg.override_sharding_config = false;
//                 cfg.shard_layout = std::nullopt;
//                 cfg.core_grid = std::nullopt;
//                 cfg.transpose_shards = true;
//                 cfg.output_layout = ::ttnn::Layout::TILE;
//                 cfg.preprocess_weights_on_device = false;
//                 cfg.enable_act_double_buffer = false;
//                 cfg.enable_weights_double_buffer = false;
//                 cfg.enable_split_reader = false;
//                 cfg.enable_subblock_padding = false;
//                 cfg.in_place = false;
//                 return cfg;
//             }(),
//             /* memory_config */
//             MakeMemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::DRAM, std::nullopt)
//         },
//         QueryOpConstraintsParams{
//             /* input_spec */
//             MakeTensorSpec(
//                 {1, 1, 392, 512},
//                 ::ttnn::DataType::BFLOAT16,
//                 Layout::TILE,
//                 {32, 32},
//                 {16, 16},
//                 4,
//                 MakeMemoryConfig(TensorMemoryLayout::BLOCK_SHARDED, BufferType::L1, MakeShardSpec({{{0, 0}, {7, 6}}},
//                 {64, 64}, ShardOrientation::ROW_MAJOR))),
//             /* weight_spec */
//             MakeTensorSpec(
//                 {1, 1, 512, 2048},
//                 ::ttnn::DataType::BFLOAT16,
//                 Layout::TILE,
//                 {32, 32},
//                 {16, 16},
//                 4,
//                 MakeMemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::DRAM, std::nullopt)),
//             /* in_channels */ 512,
//             /* out_channels */ 2048,
//             /* batch_size */ 8,
//             /* input_height */ 7,
//             /* input_width */ 7,
//             /* kernel_size */ {1, 1},
//             /* stride */ {1, 1},
//             /* padding */ {0, 0, 0, 0},
//             /* dilation */ {1, 1},
//             /* groups */ 1,
//             /* bias_spec */ std::nullopt,
//             /* conv2d_config */
//             [] {
//                 ::ttnn::operations::conv::conv2d::Conv2dConfig cfg;
//                 cfg.dtype = ::ttnn::DataType::BFLOAT16;
//                 cfg.weights_dtype = ::ttnn::DataType::BFLOAT16;
//                 cfg.activation = "";
//                 cfg.deallocate_activation = false;
//                 cfg.reallocate_halo_output = true;
//                 cfg.act_block_h_override = 0;
//                 cfg.act_block_w_div = 1;
//                 cfg.reshard_if_not_optimal = false;
//                 cfg.override_sharding_config = false;
//                 cfg.shard_layout = std::nullopt;
//                 cfg.core_grid = std::nullopt;
//                 cfg.transpose_shards = true;
//                 cfg.output_layout = ::ttnn::Layout::TILE;
//                 cfg.preprocess_weights_on_device = false;
//                 cfg.enable_act_double_buffer = false;
//                 cfg.enable_weights_double_buffer = false;
//                 cfg.enable_split_reader = false;
//                 cfg.enable_subblock_padding = false;
//                 cfg.in_place = false;
//                 return cfg;
//             }(),
//             /* memory_config */
//             MakeMemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::DRAM, std::nullopt)
//         },
//         QueryOpConstraintsParams{
//             /* input_spec */
//             MakeTensorSpec(
//                 {1, 1, 392, 512},
//                 ::ttnn::DataType::BFLOAT16,
//                 Layout::TILE,
//                 {32, 32},
//                 {16, 16},
//                 4,
//                 MakeMemoryConfig(TensorMemoryLayout::BLOCK_SHARDED, BufferType::L1, MakeShardSpec({{{0, 0}, {7, 6}}},
//                 {64, 64}, ShardOrientation::ROW_MAJOR))),
//             /* weight_spec */
//             MakeTensorSpec(
//                 {1, 1, 512, 2048},
//                 ::ttnn::DataType::BFLOAT16,
//                 Layout::TILE,
//                 {32, 32},
//                 {16, 16},
//                 4,
//                 MakeMemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::DRAM, std::nullopt)),
//             /* in_channels */ 512,
//             /* out_channels */ 2048,
//             /* batch_size */ 8,
//             /* input_height */ 7,
//             /* input_width */ 7,
//             /* kernel_size */ {1, 1},
//             /* stride */ {1, 1},
//             /* padding */ {0, 0, 0, 0},
//             /* dilation */ {1, 1},
//             /* groups */ 1,
//             /* bias_spec */ std::nullopt,
//             /* conv2d_config */
//             [] {
//                 ::ttnn::operations::conv::conv2d::Conv2dConfig cfg;
//                 cfg.dtype = ::ttnn::DataType::BFLOAT16;
//                 cfg.weights_dtype = ::ttnn::DataType::BFLOAT16;
//                 cfg.activation = "";
//                 cfg.deallocate_activation = false;
//                 cfg.reallocate_halo_output = true;
//                 cfg.act_block_h_override = 0;
//                 cfg.act_block_w_div = 1;
//                 cfg.reshard_if_not_optimal = false;
//                 cfg.override_sharding_config = false;
//                 cfg.shard_layout = std::nullopt;
//                 cfg.core_grid = std::nullopt;
//                 cfg.transpose_shards = true;
//                 cfg.output_layout = ::ttnn::Layout::TILE;
//                 cfg.preprocess_weights_on_device = false;
//                 cfg.enable_act_double_buffer = false;
//                 cfg.enable_weights_double_buffer = false;
//                 cfg.enable_split_reader = false;
//                 cfg.enable_subblock_padding = false;
//                 cfg.in_place = false;
//                 return cfg;
//             }(),
//             /* memory_config */
//             MakeMemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::DRAM, std::nullopt)
//         }
//     ));
}  // namespace tt::tt_metal
