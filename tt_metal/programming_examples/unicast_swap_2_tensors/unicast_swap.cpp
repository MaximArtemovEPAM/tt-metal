// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <memory>
#include <random>
#include <thread>
#include <cassert>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/command_queue.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/data_types.hpp>

using namespace tt;
using namespace tt::tt_metal;

// Round up to a multiple of a know power of up
// n: number to be rounded up
// m: must be a power of two
constexpr size_t roundup_kpow2(size_t n, int m) { return (n + (m - 1)) & -m; }

void unicast_swap(
    IDevice* device,
    const std::vector<float>& input0,
    const std::vector<float>& input1,
    std::vector<float>& output0,
    std::vector<float>& output1) {
    using namespace tt;
    using namespace tt::tt_metal;

    using DataType = float;

    constexpr uint32_t tile_size = tt::constants::TILE_HW * sizeof(DataType);

    CommandQueue& cq = device->command_queue();
    Program program = CreateProgram();

    // === CORE SETUP ===
    constexpr CoreCoord core0 = {0, 0};
    constexpr CoreCoord core1 = {1, 0};

    const auto grid = device->compute_with_storage_grid_size();

    const auto core0_physical_coord = device->worker_core_from_logical_core(core0);
    const auto core1_physical_coord = device->worker_core_from_logical_core(core1);

    CoreRangeSet core_set({CoreRange(core0, core1)});

    // == SEMAPHORE SETUP ===
    const uint32_t sem_id = CreateSemaphore(program, core_set, 0);

    // === BUFFER SETUP ===

    const uint32_t dram_buffer_size = roundup_kpow2(input0.size() * sizeof(DataType), tile_size);

    InterleavedBufferConfig dram_config{
        .device = device, .size = dram_buffer_size, .page_size = tile_size, .buffer_type = BufferType::DRAM};

    InterleavedBufferConfig sram_config{
        .device = device, .size = tile_size, .page_size = tile_size, .buffer_type = BufferType::L1};

    std::shared_ptr<Buffer> input0_dram_buffer = CreateBuffer(dram_config);
    std::shared_ptr<Buffer> input1_dram_buffer = CreateBuffer(dram_config);

    std::shared_ptr<Buffer> intermed0_sram_buffer = CreateBuffer(sram_config);
    std::shared_ptr<Buffer> intermed1_sram_buffer = CreateBuffer(sram_config);

    // Write to buffers
    EnqueueWriteBuffer(cq, input0_dram_buffer, input0.data(), false);
    EnqueueWriteBuffer(cq, input1_dram_buffer, input1.data(), false);

    // Set up 2 input circular buffers + 1 output circular buffer
    const DataFormat input_cb_dataformat = tt::DataFormat::Float32;
    const DataFormat input_other_cb_dataformat = input_cb_dataformat;
    const DataFormat output_cb_dataformat = input_cb_dataformat;

    constexpr uint32_t input_cb_idx = tt::CBIndex::c_0;
    constexpr uint32_t input_other_cb_idx = tt::CBIndex::c_1;
    constexpr uint32_t output_cb_idx = tt::CBIndex::c_2;

    constexpr uint32_t tiles_per_cb = 4;
    constexpr uint32_t cb_size = tile_size * tiles_per_cb;

    const CircularBufferConfig input_cb_config =
        CircularBufferConfig(cb_size, {{input_cb_idx, input_cb_dataformat}}).set_page_size(input_cb_idx, tile_size);

    const CircularBufferConfig input_other_cb_config =
        CircularBufferConfig(cb_size, {{input_other_cb_idx, input_other_cb_dataformat}})
            .set_page_size(input_other_cb_idx, tile_size);

    const CircularBufferConfig output_cb_config = CircularBufferConfig(
                                                      cb_size, {{output_cb_idx, output_cb_dataformat}}

                                                      )
                                                      .set_page_size(output_cb_idx, tile_size);

    CBHandle input_cb = CreateCircularBuffer(program, core_set, input_cb_config);
    CBHandle input_other_cb = CreateCircularBuffer(program, core_set, input_other_cb_config);
    CBHandle output_cb = CreateCircularBuffer(program, core_set, output_cb_config);

    const uint32_t num_tiles = input0.size() / tt::constants::TILE_HW;

    // === KERNEL SETUP ===
    std::vector<uint32_t> reader_compile_time_args = {input_cb_idx, input_other_cb_idx, num_tiles, grid.x, grid.y};
    std::vector<uint32_t> writer_compile_time_args = {output_cb_idx, num_tiles};

    constexpr uint32_t SELECT_MIN = 1;
    constexpr uint32_t SELECT_MAX = 0;
    std::vector<uint32_t> compute_min_compile_time_args = {
        input_cb_idx, input_other_cb_idx, output_cb_idx, num_tiles, SELECT_MIN};
    std::vector<uint32_t> compute_max_compile_time_args = {
        input_cb_idx, input_other_cb_idx, output_cb_idx, num_tiles, SELECT_MAX};

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

    KernelHandle compute_min_kernel_id =
        CreateKernel(program, compute_kernel_path, core0, ComputeConfig{.compile_args = compute_min_compile_time_args});

    KernelHandle compute_max_kernel_id =
        CreateKernel(program, compute_kernel_path, core1, ComputeConfig{.compile_args = compute_max_compile_time_args});

    // Set runtime arguments
    {  // core 0
        SetRuntimeArgs(
            program,
            reader_kernel_id,
            core0,
            {input0_dram_buffer->address(),
             intermed0_sram_buffer->address(),
             sem_id,
             core0_physical_coord.x,
             core0_physical_coord.y,
             core1_physical_coord.x,
             core1_physical_coord.y});
        SetRuntimeArgs(program, writer_kernel_id, core0, {input0_dram_buffer->address()});

        SetRuntimeArgs(program, compute_min_kernel_id, core0, {});
    }
    {  // core 1
        SetRuntimeArgs(
            program,
            reader_kernel_id,
            core1,
            {input1_dram_buffer->address(),
             intermed0_sram_buffer->address(),
             sem_id,
             core1_physical_coord.x,
             core1_physical_coord.y,
             core0_physical_coord.x,
             core0_physical_coord.y});
        SetRuntimeArgs(program, writer_kernel_id, core1, {input1_dram_buffer->address()});

        SetRuntimeArgs(program, compute_max_kernel_id, core1, {});
    }

    // === LAUNCH KERNEL ===
    std::cout << "Launching kernel" << std::endl;
    EnqueueProgram(cq, program, false);
    Finish(cq);

    // === READ RESULTS ===

    output0.resize(input0.size());
    output1.resize(input1.size());

    EnqueueReadBuffer(cq, input0_dram_buffer, output0.data(), true);
    EnqueueReadBuffer(cq, input1_dram_buffer, output1.data(), true);

    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    std::cout << "results0 = [";
    for (const auto& val : output0) {
        std::cout << ", " << val;
    }
    std::cout << "]\n\n";

    std::cout << "results1 = [";
    for (const auto& val : output1) {
        std::cout << ", " << val;
    }
    std::cout << "]\n";
}

void generate_randint(std::vector<float>& output, uint64_t seed) {
    std::mt19937 mt(seed);
    std::uniform_int_distribution<> distr(-9, 10);

    for (auto& entry : output) {
        entry = distr(mt);
    }
}

void verify_output(
    const std::vector<float>& input0,
    const std::vector<float>& input1,
    const std::vector<float>& output0,
    const std::vector<float>& output1) {
    bool is_ok = false;
    for (size_t i = 0; i < input0.size(); i++) {
        float input_val0 = input0[i];
        float input_val1 = input1[i];
        float output_val0 = output0[i];
        float output_val1 = output1[i];

        assert(output_val0 == std::min(input_val0, input_val1));
        assert(output_val1 == std::max(input_val0, input_val1));
    }
}

int main(int argc, char** argv) {
    using namespace tt::tt_metal;

    int device_id = 0;
    IDevice* device = CreateDevice(device_id);

    constexpr size_t len = 4096;

    std::vector<float> input0(len);
    std::vector<float> input1(len);

    std::vector<float> output0;
    std::vector<float> output1;

    generate_randint(input0, 0);
    generate_randint(input1, 1);

    unicast_swap(device, input0, input1, output0, output1);

    verify_output(input0, input1, output0, output1);

    CloseDevice(device);

    return 0;
}
