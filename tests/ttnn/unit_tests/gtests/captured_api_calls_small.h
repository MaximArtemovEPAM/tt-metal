#pragma once

#include "captures.h"

using namespace tt::tt_metal;
using namespace std;

namespace tt {
namespace tt_metal {

std::vector<QueryOpConstraintsParams> kAllQueryOpConstraintsParams = {
    QueryOpConstraintsParams{/* input_spec */
                             MakeTensorSpec(
                                 {1, 1, 401408, 3},
                                 ::ttnn::DataType::BFLOAT16,
                                 Layout::TILE,
                                 {32, 32},
                                 {16, 16},
                                 4,
                                 MakeMemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::DRAM, std::nullopt)),
                             /* original_weights_shape */
                             {64, 3, 7, 7},
                             /* weight_spec */
                             MakeTensorSpec(
                                 {1, 1, 147, 64},
                                 ::ttnn::DataType::BFLOAT16,
                                 Layout::TILE,
                                 {32, 32},
                                 {16, 16},
                                 4,
                                 MakeMemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::DRAM, std::nullopt)),
                             /* in_channels */ 3,
                             /* out_channels */ 64,
                             /* batch_size */ 8,
                             /* input_height */ 224,
                             /* input_width */ 224,
                             /* kernel_size */ {7, 7},
                             /* stride */ {2, 2},
                             /* padding */ {3, 3, 3, 3},
                             /* dilation */ {1, 1},
                             /* groups */ 1,
                             /* bias_spec */
                             MakeTensorSpec(
                                 {1, 1, 1, 64},
                                 ::ttnn::DataType::BFLOAT16,
                                 Layout::TILE,
                                 {32, 32},
                                 {16, 16},
                                 4,
                                 MakeMemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::DRAM, std::nullopt)),

                             /* conv2d_config */
                             [] {
                                 ::ttnn::operations::conv::conv2d::Conv2dConfig cfg;
                                 cfg.dtype = ::ttnn::DataType::BFLOAT16;
                                 cfg.weights_dtype = ::ttnn::DataType::BFLOAT16;
                                 cfg.activation = "";
                                 cfg.deallocate_activation = false;
                                 cfg.reallocate_halo_output = true;
                                 cfg.act_block_h_override = 0;
                                 cfg.act_block_w_div = 1;
                                 cfg.reshard_if_not_optimal = false;
                                 cfg.override_sharding_config = false;
                                 cfg.shard_layout = std::nullopt;
                                 cfg.core_grid = std::nullopt;
                                 cfg.transpose_shards = true;
                                 cfg.output_layout = ::ttnn::Layout::TILE;
                                 cfg.preprocess_weights_on_device = false;
                                 cfg.enable_act_double_buffer = false;
                                 cfg.enable_weights_double_buffer = false;
                                 cfg.enable_split_reader = false;
                                 cfg.enable_subblock_padding = false;
                                 cfg.in_place = false;
                                 return cfg;
                             }(),
                             /* memory_config */
                             MakeMemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::DRAM, std::nullopt)},
    QueryOpConstraintsParams{/* input_spec */
                             MakeTensorSpec(
                                 {1, 1, 401408, 3},
                                 ::ttnn::DataType::BFLOAT16,
                                 Layout::TILE,
                                 {32, 32},
                                 {16, 16},
                                 4,
                                 MakeMemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::DRAM, std::nullopt)),
                             /* original_weights_shape */
                             {64, 3, 7, 7},
                             /* weight_spec */
                             MakeTensorSpec(
                                 {1, 1, 147, 64},
                                 ::ttnn::DataType::BFLOAT16,
                                 Layout::TILE,
                                 {32, 32},
                                 {16, 16},
                                 4,
                                 MakeMemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::DRAM, std::nullopt)),
                             /* in_channels */ 3,
                             /* out_channels */ 64,
                             /* batch_size */ 8,
                             /* input_height */ 224,
                             /* input_width */ 224,
                             /* kernel_size */ {7, 7},
                             /* stride */ {2, 2},
                             /* padding */ {3, 3, 3, 3},
                             /* dilation */ {1, 1},
                             /* groups */ 1,
                             /* bias_spec */
                             MakeTensorSpec(
                                 {1, 1, 1, 64},
                                 ::ttnn::DataType::BFLOAT16,
                                 Layout::TILE,
                                 {32, 32},
                                 {16, 16},
                                 4,
                                 MakeMemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::DRAM, std::nullopt)),

                             /* conv2d_config */
                             [] {
                                 ::ttnn::operations::conv::conv2d::Conv2dConfig cfg;
                                 cfg.dtype = ::ttnn::DataType::BFLOAT16;
                                 cfg.weights_dtype = ::ttnn::DataType::BFLOAT16;
                                 cfg.activation = "";
                                 cfg.deallocate_activation = false;
                                 cfg.reallocate_halo_output = true;
                                 cfg.act_block_h_override = 0;
                                 cfg.act_block_w_div = 1;
                                 cfg.reshard_if_not_optimal = false;
                                 cfg.override_sharding_config = false;
                                 cfg.shard_layout = std::nullopt;
                                 cfg.core_grid = std::nullopt;
                                 cfg.transpose_shards = true;
                                 cfg.output_layout = ::ttnn::Layout::TILE;
                                 cfg.preprocess_weights_on_device = false;
                                 cfg.enable_act_double_buffer = false;
                                 cfg.enable_weights_double_buffer = false;
                                 cfg.enable_split_reader = false;
                                 cfg.enable_subblock_padding = false;
                                 cfg.in_place = false;
                                 return cfg;
                             }(),
                             /* memory_config */
                             MakeMemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::DRAM, std::nullopt)},
    QueryOpConstraintsParams{/* input_spec */
                             MakeTensorSpec(
                                 {1, 1, 401408, 3},
                                 ::ttnn::DataType::BFLOAT16,
                                 Layout::TILE,
                                 {32, 32},
                                 {16, 16},
                                 4,
                                 MakeMemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::DRAM, std::nullopt)),
                             /* original_weights_shape */
                             {64, 3, 7, 7},
                             /* weight_spec */
                             MakeTensorSpec(
                                 {1, 1, 147, 64},
                                 ::ttnn::DataType::BFLOAT16,
                                 Layout::TILE,
                                 {32, 32},
                                 {16, 16},
                                 4,
                                 MakeMemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::DRAM, std::nullopt)),
                             /* in_channels */ 3,
                             /* out_channels */ 64,
                             /* batch_size */ 8,
                             /* input_height */ 224,
                             /* input_width */ 224,
                             /* kernel_size */ {7, 7},
                             /* stride */ {2, 2},
                             /* padding */ {3, 3, 3, 3},
                             /* dilation */ {1, 1},
                             /* groups */ 1,
                             /* bias_spec */
                             MakeTensorSpec(
                                 {1, 1, 1, 64},
                                 ::ttnn::DataType::BFLOAT16,
                                 Layout::TILE,
                                 {32, 32},
                                 {16, 16},
                                 4,
                                 MakeMemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::DRAM, std::nullopt)),

                             /* conv2d_config */
                             [] {
                                 ::ttnn::operations::conv::conv2d::Conv2dConfig cfg;
                                 cfg.dtype = ::ttnn::DataType::BFLOAT16;
                                 cfg.weights_dtype = ::ttnn::DataType::BFLOAT16;
                                 cfg.activation = "";
                                 cfg.deallocate_activation = false;
                                 cfg.reallocate_halo_output = true;
                                 cfg.act_block_h_override = 0;
                                 cfg.act_block_w_div = 1;
                                 cfg.reshard_if_not_optimal = false;
                                 cfg.override_sharding_config = false;
                                 cfg.shard_layout = std::nullopt;
                                 cfg.core_grid = std::nullopt;
                                 cfg.transpose_shards = true;
                                 cfg.output_layout = ::ttnn::Layout::TILE;
                                 cfg.preprocess_weights_on_device = false;
                                 cfg.enable_act_double_buffer = false;
                                 cfg.enable_weights_double_buffer = false;
                                 cfg.enable_split_reader = false;
                                 cfg.enable_subblock_padding = false;
                                 cfg.in_place = false;
                                 return cfg;
                             }(),
                             /* memory_config */
                             MakeMemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::DRAM, std::nullopt)}};
}  // namespace tt_metal

}  // namespace tt
