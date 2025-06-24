// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "device_fixture.hpp"
#include "tt_metal/test_utils/comparison.hpp"
#include "tt_metal/test_utils/stimulus.hpp"
#include "tt_metal/test_utils/print_helpers.hpp"
#include "dm_common.hpp"

namespace tt::tt_metal {

using namespace std;
using namespace tt;
using namespace tt::test_utils;

namespace unit_tests::dm::dram {
// Test config, i.e. test parameters
struct I2SConfig {
    uint32_t test_id = 0;
    uint32_t num_of_transactions = 0;
    uint32_t transaction_size_pages = 0;
    uint32_t page_size_bytes = 0;
    DataFormat l1_data_format = DataFormat::Invalid;
    CoreRangeSet cores = CoreRangeSet();
    array<uint32_t, 2> tensor_shape_in_pages = {0, 0};
    array<uint32_t, 2> num_dram_banks = {0, 0};
};

/// @brief Does Dram --> Reader --> L1 CB --> Writer --> Dram.
/// @param device
/// @param test_config - Configuration of the test -- see struct
/// @return
bool run_dm(IDevice* device, const I2SConfig& test_config) {
    // Program
    Program program = CreateProgram();

    // DRAM Buffers
    const size_t total_size_bytes =
        test_config.num_of_transactions * test_config.transaction_size_pages * test_config.page_size_bytes;

    InterleavedBufferConfig interleaved_dram_config{
        .device = device, .size = 4096 * 2, .page_size = 32 * 32 * 2, .buffer_type = BufferType::DRAM};
    // InterleavedBufferConfig interleaved_dram_config{
    //     .device = device, .size = total_size_bytes, .page_size = total_size_bytes, .buffer_type = BufferType::DRAM};
    std::shared_ptr<Buffer> input_dram_buffer;
    // if (!test_config.num_dram_banks[0]) {
    input_dram_buffer = CreateBuffer(interleaved_dram_config);

    auto output_spec = ShardSpecBuffer(test_config.cores, {64, 64}, ShardOrientation::ROW_MAJOR, {32, 32}, {2, 2});
    auto output_L1_buffer = CreateBuffer(ShardedBufferConfig{
        .device = device,
        .size = 4096 * 2,
        .page_size = test_config.page_size_bytes,
        .buffer_type = BufferType::L1,
        .buffer_layout = TensorMemoryLayout::WIDTH_SHARDED,
        .shard_parameters = std::move(output_spec),
    });

    uint32_t input_dram_byte_address = input_dram_buffer->address();
    // auto output_dram_buffer = CreateBuffer(interleaved_dram_config);
    // uint32_t output_dram_byte_address = output_dram_buffer->address();
    uint32_t output_L1_byte_address = output_L1_buffer->address();

    // Input
    vector<uint32_t> packed_input = generate_packed_uniform_random_vector<uint32_t, bfloat16>(
        -100.0f, 100.0f, total_size_bytes / bfloat16::SIZEOF, chrono::system_clock::now().time_since_epoch().count());

    vector<uint32_t> sequential_input =
        create_arange_vector_of_bfloat16(4096 * 2, false);  // 4096 elements starting at 0 incrementing 1 each time
                                                            // add the include for hpp and use tilize func

    // Golden output
    vector<uint32_t> packed_golden = packed_input;

    uint8_t l1_cb_index = CBIndex::c_0;

    // Compile-time arguments for kernels
    vector<uint32_t> reader_compile_args = {
        (uint32_t)input_dram_byte_address,
        (uint32_t)0,
        (uint32_t)test_config.num_of_transactions,
        (uint32_t)test_config.transaction_size_pages,
        (uint32_t)test_config.page_size_bytes,
        (uint32_t)l1_cb_index,
        (uint32_t)test_config.test_id};

    vector<uint32_t> writer_compile_args = {
        (uint32_t)output_dram_byte_address,
        (uint32_t)0,
        (uint32_t)test_config.num_of_transactions,
        (uint32_t)test_config.transaction_size_pages,
        (uint32_t)test_config.page_size_bytes,
        (uint32_t)l1_cb_index,
        (uint32_t)test_config.test_id};

    // Create circular buffers
    CircularBufferConfig l1_cb_config = CircularBufferConfig(4096 * 2, {{l1_cb_index, tt::DataFormat::Float16_b}})
                                            .set_page_size(l1_cb_index, 32 * 32 * 2);
    auto l1_cb = CreateCircularBuffer(program, test_config.cores, l1_cb_config);

    // Kernels
    auto reader_kernel = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/data_movement/interleaved_to_sharded/kernels/kernel1.cpp",
        test_config.cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = reader_compile_args});

    auto writer_kernel = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/data_movement/dram_unary/kernels/writer_unary.cpp",
        test_config.cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = writer_compile_args});

    // Assign unique id
    log_info(tt::LogTest, "Running Test ID: {}, Run ID: {}", test_config.test_id, unit_tests::dm::runtime_host_id);
    program.set_runtime_id(unit_tests::dm::runtime_host_id++);

    // Launch program and record outputs
    vector<uint32_t> packed_output;
    detail::WriteToBuffer(input_dram_buffer, packed_input);
    MetalContext::instance().get_cluster().dram_barrier(device->id());
    detail::LaunchProgram(device, program);
    detail::ReadFromBuffer(output_L1_buffer, packed_output);

    // Results comparison
    bool pcc = is_close_packed_vectors<bfloat16, uint32_t>(
        packed_output, packed_golden, [&](const bfloat16& a, const bfloat16& b) { return is_close(a, b); });

    if (!pcc) {
        log_error(tt::LogTest, "PCC Check failed");
        log_info(tt::LogTest, "Golden vector");
        print_vector<uint32_t>(packed_golden);
        log_info(tt::LogTest, "Output vector");
        print_vector<uint32_t>(packed_output);
    }

    return pcc;
}
}  // namespace unit_tests::dm::dram

/* ========== Directed ideal test case; Test id = 3 ========== */
TEST_F(DeviceFixture, TensixDataMovementInterleavedToSharded) {
    // Parameters
    uint32_t num_of_transactions = 180;
    uint32_t transaction_size_pages = 4 * 32;
    uint32_t page_size_bytes = arch_ == tt::ARCH::BLACKHOLE ? 64 : 32;  // =Flit size: 32 bytes for WH, 64 for BH
    // Max transaction size = 4 * 32 pages = 128 * 32 bytes = 4096 bytes for WH; 8192 bytes for BH
    // Max total transaction size = 180 * 8192 bytes = 1474560 bytes = 1.4 MB = L1 capacity

    // Cores
    CoreRange core_range({0, 0}, {0, 0});
    CoreRangeSet core_range_set({core_range});

    // Test config
    unit_tests::dm::dram::I2SConfig test_config = {
        .test_id = 3,
        .num_of_transactions = num_of_transactions,
        .transaction_size_pages = transaction_size_pages,
        .page_size_bytes = page_size_bytes,
        .l1_data_format = DataFormat::Float16_b,
        .cores = core_range_set};

    // Run
    for (unsigned int id = 0; id < num_devices_; id++) {
        EXPECT_TRUE(run_dm(devices_.at(id), test_config));
    }
}

}  // namespace tt::tt_metal
