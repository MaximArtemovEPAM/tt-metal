// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/decorators.hpp"
#include "ttnn/operations/transformer/sdpa_config.hpp"

namespace ttnn {
namespace operations::transformer {

struct ExecuteFlashMLAPrefill {
    static ttnn::Tensor invoke(
        QueueId queue_id,
        const ttnn::Tensor& input_tensor_q,
        const ttnn::Tensor& input_tensor_k,
        const std::optional<ttnn::Tensor>& attn_mask = std::nullopt,
        bool is_causal = true,
        std::optional<float> scale = std::nullopt,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        std::optional<SDPAProgramConfig> program_config = std::nullopt,
        std::optional<DeviceComputeKernelConfig> compute_kernel_config = std::nullopt);

    static ttnn::Tensor invoke(
        const ttnn::Tensor& input_tensor_q,
        const ttnn::Tensor& input_tensor_k,
        const std::optional<ttnn::Tensor>& attn_mask = std::nullopt,
        bool is_causal = true,
        std::optional<float> scale = std::nullopt,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        std::optional<SDPAProgramConfig> program_config = std::nullopt,
        std::optional<DeviceComputeKernelConfig> compute_kernel_config = std::nullopt);
};

struct ExecuteChunkedFlashMLAPrefill {
    static ttnn::Tensor invoke(
        QueueId queue_id,
        const ttnn::Tensor& input_tensor_q,
        const ttnn::Tensor& input_tensor_k,
        const ttnn::Tensor& page_table_tensor,
        int64_t chunk_start_idx,
        std::optional<float> scale = std::nullopt,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        std::optional<SDPAProgramConfig> program_config = std::nullopt,
        std::optional<DeviceComputeKernelConfig> compute_kernel_config = std::nullopt);

    static ttnn::Tensor invoke(
        const ttnn::Tensor& input_tensor_q,
        const ttnn::Tensor& input_tensor_k,
        const ttnn::Tensor& page_table_tensor,
        int64_t chunk_start_idx,
        std::optional<float> scale = std::nullopt,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        std::optional<SDPAProgramConfig> program_config = std::nullopt,
        std::optional<DeviceComputeKernelConfig> compute_kernel_config = std::nullopt);
};

struct ExecuteJointFlashMLAPrefill {
    static std::tuple<ttnn::Tensor, ttnn::Tensor> invoke(
        QueueId queue_id,
        const ttnn::Tensor& input_tensor_q,
        const ttnn::Tensor& input_tensor_k,
        const ttnn::Tensor& joint_tensor_q,
        const ttnn::Tensor& joint_tensor_k,
        const std::string& joint_strategy,
        SDPAProgramConfig program_config,
        std::optional<float> scale = std::nullopt,
        std::optional<DeviceComputeKernelConfig> compute_kernel_config = std::nullopt);

    static std::tuple<ttnn::Tensor, ttnn::Tensor> invoke(
        const ttnn::Tensor& input_tensor_q,
        const ttnn::Tensor& input_tensor_k,
        const ttnn::Tensor& joint_tensor_q,
        const ttnn::Tensor& joint_tensor_k,
        const std::string& joint_strategy,
        SDPAProgramConfig program_config,
        std::optional<float> scale = std::nullopt,
        std::optional<DeviceComputeKernelConfig> compute_kernel_config = std::nullopt);
};

}  // namespace operations::transformer

namespace transformer {

constexpr auto flash_mla_prefill = ttnn::
    register_operation<"ttnn::transformer::flash_mla_prefill", ttnn::operations::transformer::ExecuteFlashMLAPrefill>();

constexpr auto chunked_flash_mla_prefill = ttnn::register_operation<
    "ttnn::transformer::chunked_flash_mla_prefill",
    ttnn::operations::transformer::ExecuteChunkedFlashMLAPrefill>();

constexpr auto joint_flash_mla_prefill = ttnn::register_operation<
    "ttnn::transformer::joint_flash_mla_prefill",
    ttnn::operations::transformer::ExecuteJointFlashMLAPrefill>();

}  // namespace transformer

}  // namespace ttnn
