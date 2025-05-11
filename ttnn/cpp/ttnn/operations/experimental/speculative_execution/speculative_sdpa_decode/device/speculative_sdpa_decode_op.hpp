// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/operations/transformer/sdpa_config.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "cpp/ttnn/operations/ccl/ccl_host_types.hpp"

namespace ttnn::operations::experimental::speculative_execution {

struct SpeculativeScaledDotProductAttentionDecode {
    const std::optional<float> lambda_;
    const bool is_causal;
    std::vector<uint32_t> cur_pos;
    const std::optional<float> scale;
    const MemoryConfig output_mem_config;
    const std::optional<SDPAProgramConfig> program_config;
    const DeviceComputeKernelConfig compute_kernel_config;
    const uint32_t k_chunk_size;
    const bool paged_attention;
    const bool ccl_enabled;
    const uint32_t num_devices;
    const uint32_t device_index;
    const ttnn::ccl::Topology topology;
    std::optional<GlobalSemaphore> semaphore;
    std::optional<IDevice*> forward_device;
    std::optional<IDevice*> backward_device;
    const std::optional<bool> share_cache;

    void validate(
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const Tensor>>& optional_input_tensors) const;

    std::vector<TensorSpec> compute_output_specs(const std::vector<Tensor>& input_tensors) const;

    operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const Tensor>>& optional_input_tensors,
        std::vector<Tensor>& output_tensors) const;

    operation::Hash compute_program_hash(
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const Tensor>>& optional_input_tensors) const;
};

}  // namespace ttnn::operations::experimental::speculative_execution
