// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <memory>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/command_queue.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/data_types.hpp>

using namespace tt;
using namespace tt::tt_metal;

void unicast_swap(IDevice* device, const std::vector<float>& input0, const std::vector<float>& input1) {
    using namespace tt;
    using namespace tt::tt_metal;

    std::cout << "-- unicast_swap --" << std::endl;

    CommandQueue& cq = device->command_queue();
    Program program = CreateProgram();

    constexpr CoreCoord core0 = {0, 0};
    constexpr CoreCoord core1 = {1, 0};

    const auto core0_physical_coord = device->worker_core_from_logical_core(core0);
    const auto core1_physical_coord = device->worker_core_from_logical_core(core1);

    constexpr uint32_t tile_size = tt::constants::TILE_HW;
    std::cout << "tile size = " << tile_size << std::endl;

    // TODO: We also want core1 to have a semaphore
    CoreRangeSet core_set({CoreRange(core0, core1)});
    const uint32_t sem_id = CreateSemaphore(program, core_set, 0);

    InterleavedBufferConfig dram_config{
        .device = device, .size = tile_size, .page_size = tile_size, .buffer_type = BufferType::DRAM};
    InterleavedBufferConfig sram_config{
        .device = device, .size = tile_size, .page_size = tile_size, .buffer_type = BufferType::L1};

    std::shared_ptr<Buffer> input0_dram_buffer = CreateBuffer(dram_config);
    std::shared_ptr<Buffer> input1_dram_buffer = CreateBuffer(dram_config);

    std::shared_ptr<Buffer> input0_sram_buffer = CreateBuffer(sram_config);
    std::shared_ptr<Buffer> input1_sram_buffer = CreateBuffer(sram_config);

    EnqueueWriteBuffer(cq, input0_sram_buffer, input0.data(), false);
    EnqueueWriteBuffer(cq, input1_sram_buffer, input1.data(), false);

    // Set up 2 input circular buffers + 1 output circular buffer
    const DataFormat input_cb_dataformat = tt::DataFormat::Float32;
    const DataFormat input_other_cb_dataformat = input_cb_dataformat;
    const DataFormat output_cb_dataformat = input_cb_dataformat;

    constexpr uint32_t input_cb_idx = tt::CBIndex::c_0;
    constexpr uint32_t input_other_cb_idx = tt::CBIndex::c_1;
    constexpr uint32_t output_cb_idx = tt::CBIndex::c_2;

    constexpr uint32_t input_size = tile_size * 1;

    const CircularBufferConfig input_cb_config =
        CircularBufferConfig(input_size, {{input_cb_idx, input_cb_dataformat}}).set_page_size(input_cb_idx, tile_size);

    const CircularBufferConfig input_other_cb_config =
        CircularBufferConfig(input_size, {{input_other_cb_idx, input_other_cb_dataformat}})
            .set_page_size(input_other_cb_idx, tile_size);

    const CircularBufferConfig output_cb_config = CircularBufferConfig(
                                                      input_size, {{output_cb_idx, output_cb_dataformat}}

                                                      )
                                                      .set_page_size(output_cb_idx, tile_size);

    CBHandle input_cb = CreateCircularBuffer(program, core_set, input_cb_config);
    CBHandle input_other_cb = CreateCircularBuffer(program, core_set, input_other_cb_config);
    CBHandle output_cb = CreateCircularBuffer(program, core_set, output_cb_config);

    // Define compile time arguments
    std::vector<uint32_t> reader_compile_time_args = {input_cb_idx, input_other_cb_idx};
    std::vector<uint32_t> writer_compile_time_args = {output_cb_idx};
    std::vector<uint32_t> compute_compile_time_args = {input_cb_idx, input_other_cb_idx, output_cb_idx};

    // Load kernels
    const std::string reader_kernel_path =
        "tt_metal/programming_examples/unicast_swap_2_tensors/kernels/dataflow/unicast_swap_reader.cpp";
    const std::string writer_kernel_path =
        "tt_metal/programming_examples/unicast_swap_2_tensors/kernels/dataflow/unicast_swap_writer.cpp";
    const std::string compute_kernel_path =
        "tt_metal/programming_examples/unicast_swap_2_tensors/kernels/compute/unicast_swap_compute.cpp";

    KernelHandle reader_kernel_id =
        CreateKernel(program, reader_kernel_path, core_set, ReaderDataMovementConfig{reader_compile_time_args});

    KernelHandle writer_kernel_id =
        CreateKernel(program, writer_kernel_path, core_set, WriterDataMovementConfig{writer_compile_time_args});

    KernelHandle compute_kernel_id =
        CreateKernel(program, compute_kernel_path, core_set, ComputeConfig{.compile_args = compute_compile_time_args});

    // Set runtime arguments
    {  // core 0
        SetRuntimeArgs(
            program,
            reader_kernel_id,
            core0,
            {input0_sram_buffer->address(),
             sem_id,
             core0_physical_coord.x,
             core0_physical_coord.y,
             core1_physical_coord.x,
             core1_physical_coord.y});
        SetRuntimeArgs(program, writer_kernel_id, core0, {input0_sram_buffer->address()});

        SetRuntimeArgs(program, compute_kernel_id, core0, {});
    }
    {  // core 1
        SetRuntimeArgs(
            program,
            reader_kernel_id,
            core1,
            {input1_sram_buffer->address(),
             sem_id,
             core1_physical_coord.x,
             core1_physical_coord.y,
             core0_physical_coord.x,
             core0_physical_coord.y});
        SetRuntimeArgs(program, writer_kernel_id, core1, {input1_sram_buffer->address()});

        SetRuntimeArgs(program, compute_kernel_id, core1, {});
    }

    EnqueueProgram(cq, program, false);
    Finish(cq);

    std::vector<float> results0;
    std::vector<float> results1;

    results0.resize(input0.size());
    results1.resize(input1.size());

    EnqueueReadBuffer(cq, input0_sram_buffer, results0.data(), true);
    EnqueueReadBuffer(cq, input1_sram_buffer, results1.data(), true);

    std::cout << "results0 = [";
    for (const auto& val : results0) {
        std::cout << ", " << val;
    }
    std::cout << "]\n";

    std::cout << "results1 = [";
    for (const auto& val : results1) {
        std::cout << ", " << val;
    }
    std::cout << "]\n";
}

int main(int argc, char** argv) {
    using namespace tt::tt_metal;

    int device_id = 0;
    IDevice* device = CreateDevice(device_id);

    std::vector<float> input0 = {1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<float> input1 = {-1, -2, -3, -4, -5, -6, -7, -8};

    input0.resize(256, 0);
    input1.resize(256, 0);

    unicast_swap(device, input0, input1);

    CloseDevice(device);

    return 0;
}
