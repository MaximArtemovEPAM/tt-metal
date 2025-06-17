// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/decorators.hpp"
#include "ttnn/operations/transformer/sdpa_config.hpp"

namespace ttnn {
namespace operations::transformer {

struct ExecuteFlashMLADecode {
    static ttnn::Tensor invoke(
        QueueId queue_id,
        const ttnn::Tensor& input_tensor_q,
        const ttnn::Tensor& input_tensor_k,
        const uint32_t head_dim_v,
        const bool is_causal = true,
        const std::optional<const Tensor>& attn_mask = std::nullopt,
        const std::vector<uint32_t>& cur_pos = std::vector<uint32_t>(),
        const std::optional<const Tensor>& cur_pos_tensor = std::nullopt,
        std::optional<float> scale = std::nullopt,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        std::optional<SDPAProgramConfig> program_config = std::nullopt,
        std::optional<DeviceComputeKernelConfig> compute_kernel_config = std::nullopt);

    static ttnn::Tensor invoke(
        const ttnn::Tensor& input_tensor_q,
        const ttnn::Tensor& input_tensor_k,
        const uint32_t head_dim_v,
        const bool is_causal = true,
        const std::optional<const Tensor>& attn_mask = std::nullopt,
        const std::vector<uint32_t>& cur_pos = std::vector<uint32_t>(),
        const std::optional<const Tensor>& cur_pos_tensor = std::nullopt,
        std::optional<float> scale = std::nullopt,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        std::optional<SDPAProgramConfig> program_config = std::nullopt,
        std::optional<DeviceComputeKernelConfig> compute_kernel_config = std::nullopt);
};

struct ExecutePagedFlashMLADecode {
    static ttnn::Tensor invoke(
        QueueId queue_id,
        const ttnn::Tensor& input_tensor_q,
        const ttnn::Tensor& input_tensor_k,
        const uint32_t head_dim_v,
        const ttnn::Tensor& page_table_tensor,
        const bool is_causal = true,
        const std::optional<const Tensor>& attn_mask = std::nullopt,
        const std::optional<const Tensor>& cur_pos_tensor = std::nullopt,
        std::optional<float> scale = std::nullopt,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        std::optional<SDPAProgramConfig> program_config = std::nullopt,
        std::optional<DeviceComputeKernelConfig> compute_kernel_config = std::nullopt);

    static ttnn::Tensor invoke(
        const ttnn::Tensor& input_tensor_q,
        const ttnn::Tensor& input_tensor_k,
        const uint32_t head_dim_v,
        const ttnn::Tensor& page_table_tensor,
        const bool is_causal = true,
        const std::optional<const Tensor>& attn_mask = std::nullopt,
        const std::optional<const Tensor>& cur_pos_tensor = std::nullopt,
        std::optional<float> scale = std::nullopt,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        std::optional<SDPAProgramConfig> program_config = std::nullopt,
        std::optional<DeviceComputeKernelConfig> compute_kernel_config = std::nullopt);
};

}  // namespace operations::transformer

namespace transformer {

constexpr auto flash_mla_decode = ttnn::
    register_operation<"ttnn::transformer::flash_mla_decode", ttnn::operations::transformer::ExecuteFlashMLADecode>();

constexpr auto paged_flash_mla_decode = ttnn::register_operation<
    "ttnn::transformer::paged_flash_mla_decode",
    ttnn::operations::transformer::ExecutePagedFlashMLADecode>();

}  // namespace transformer

}  // namespace ttnn
