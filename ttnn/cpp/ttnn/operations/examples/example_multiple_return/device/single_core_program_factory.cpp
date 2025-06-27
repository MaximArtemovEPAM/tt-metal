// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "example_multiple_return_device_operation.hpp"
#include <tt-metalium/work_split.hpp>
#include <iostream>

// #define DEB

namespace ttnn::operations::examples {
ExampleMultipleReturnDeviceOperation::SingleCore::cached_program_t
ExampleMultipleReturnDeviceOperation::SingleCore::create(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
#ifdef DEB
    std::cout << "*********************" << std::endl;
    std::cout << "ttnn/cpp/ttnn/operations/examples/example_multiple_return/device/single_core_program_factory.cpp"
              << std::endl;
#endif

    using namespace tt;
    using namespace tt::tt_metal;

    const auto& input = tensor_args.input;
    const auto& other = tensor_args.other;

    auto output = tensor_return_value;

    auto src1_buffer = input.buffer();
    auto src2_buffer = other.buffer();

    tt::tt_metal::Program program{};

    tt::DataFormat cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input.get_dtype());
    uint32_t single_tile_size = tt::tt_metal::detail::TileSize(cb_data_format);

    auto output_dtype = output.get_dtype();
    tt::DataFormat cb_data_format_output = tt::tt_metal::datatype_to_dataformat_converter(output_dtype);
    uint32_t single_tile_size_output = tt::tt_metal::detail::TileSize(cb_data_format_output);

    uint32_t num_tiles = input.volume() / tt::constants::TILE_HW;

    tt::tt_metal::IDevice* device = input.device();

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
        split_work_to_cores(compute_with_storage_grid_size, num_tiles);

#ifdef DEB
    std::cout << "Num_cores " << num_cores << " num_cores_x " << num_cores_x << " num_cores_y " << num_cores_y
              << std::endl;
#endif

    uint32_t src0_cb_index = tt::CBIndex::c_0;
    uint32_t num_input_tiles = 2;
    tt::tt_metal::CircularBufferConfig cb_src0_config =
        tt::tt_metal::CircularBufferConfig(num_input_tiles * single_tile_size, {{src0_cb_index, cb_data_format}})
            .set_page_size(src0_cb_index, single_tile_size);
    auto cb_src0 = tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_src0_config);

    uint32_t src1_cb_index = tt::CBIndex::c_1;
    tt::tt_metal::CircularBufferConfig cb_src1_config =
        tt::tt_metal::CircularBufferConfig(num_input_tiles * single_tile_size, {{src1_cb_index, cb_data_format}})
            .set_page_size(src1_cb_index, single_tile_size);
    auto cb_src1 = tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_src1_config);

    uint32_t output_cb_index = tt::CBIndex::c_2;
    uint32_t num_output_tiles = 2;
    tt::tt_metal::CircularBufferConfig cb_output_config =
        tt::tt_metal::CircularBufferConfig(
            num_output_tiles * single_tile_size_output, {{output_cb_index, cb_data_format_output}})
            .set_page_size(output_cb_index, single_tile_size_output);
    auto cb_output1 = tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_output_config);

    tt::tt_metal::KernelHandle unary_reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/examples/example_multiple_return/device/kernels/dataflow/reader_unary.cpp",
        all_cores,
        tt::tt_metal::ReaderDataMovementConfig());

    tt::tt_metal::KernelHandle unary_writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/examples/example_multiple_return/device/kernels/dataflow/writer_unary.cpp",
        all_cores,
        tt::tt_metal::WriterDataMovementConfig());

    std::map<string, string> compute_defines;
    compute_defines["UNOPS"] = std::to_string(std::getenv("UNOPS") ? std::stoi(std::getenv("UNOPS")) : 0);
    compute_defines["MNOPS"] = std::to_string(std::getenv("MNOPS") ? std::stoi(std::getenv("MNOPS")) : 0);
    compute_defines["PNOPS"] = std::to_string(std::getenv("PNOPS") ? std::stoi(std::getenv("PNOPS")) : 0);
    bool math_approx_mode = false;
    auto eltwise_unary_kernel_group_1_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/examples/example_multiple_return/device/kernels/compute/eltwise_sfpu.cpp",
        core_group_1,
        tt::tt_metal::ComputeConfig{.math_fidelity = MathFidelity::HiFi4, .math_approx_mode = math_approx_mode, .defines = compute_defines});

    for (uint32_t i = 0, num_tiles_written = 0; i < num_cores; i++) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};
        uint32_t num_tiles_per_core = 0;
        if (core_group_1.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_2;
        } else {
            TT_ASSERT(false, "Core not in specified core ranges");
        }

        tt::tt_metal::SetRuntimeArgs(
            program, unary_reader_kernel_id, core, {src1_buffer->address(), src2_buffer->address(), 1});

        auto dst_buffer_address = output.buffer()->address();
        tt::tt_metal::SetRuntimeArgs(program, unary_writer_kernel_id, core, {dst_buffer_address, i});
    }

    return {
        std::move(program),
        {.unary_reader_kernel_id = unary_reader_kernel_id, .unary_writer_kernel_id = unary_writer_kernel_id}};
}

void ExampleMultipleReturnDeviceOperation::SingleCore::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    auto& program = cached_program.program;
    auto& unary_reader_kernel_id = cached_program.shared_variables.unary_reader_kernel_id;
    auto& unary_writer_kernel_id = cached_program.shared_variables.unary_writer_kernel_id;

    const auto& input = tensor_args.input;
    const auto& other = tensor_args.other;
    auto output = tensor_return_value;

    auto src_buffer1 = input.buffer();
    auto src_buffer2 = other.buffer();
    auto dst_buffer = output.buffer();

    {
        auto& runtime_args = tt::tt_metal::GetRuntimeArgs(program, unary_reader_kernel_id, CoreCoord{0, 0});
        runtime_args[0] = src_buffer1->address();
        runtime_args[1] = src_buffer2->address();
    }

    {
        auto& runtime_args = tt::tt_metal::GetRuntimeArgs(program, unary_writer_kernel_id, CoreCoord{0, 0});
        runtime_args[0] = dst_buffer->address();
    }
}

}  // namespace ttnn::operations::examples
