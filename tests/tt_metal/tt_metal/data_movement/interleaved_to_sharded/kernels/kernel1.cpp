// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

// NAME TBD + gotta figure out if need 1 or 2 kernels

void kernel_main() {
    // start with base case: interleaved dram has 4 tiles of data (but how do we make sure its formatted right with
    // input tensor 64x64?) sharded to 1 core

    /*
    get compiletime and/or runtime args
    */

    constexpr uint32_t tile_bytes = get_tile_size(cb_id_in0);  // cb_id_in0 = compile time arg, cb index
    constexpr DataFormat data_format = get_dataformat(cb_id_in0);

    const InterleavedAddrGenFast<true> s = {
        .bank_base_address = src_addr,
        .page_size = tile_bytes,
        .data_format = data_format};  // src_addr = run time arg, addr of interleaved dram buffer(?)

    cb_reserve_back(cb_id_in0, 4);                      // may not need
    uint32_t l1_write_addr = get_write_ptr(cb_id_in0);  // replace with compile time arg of output buffer address
    for (uint32_t i = 0; i < 4; i++) {
        noc_async_read_tile(i, s, l1_write_addr);
        l1_write_addr += tile_bytes;
    }
    noc_async_read_barrier();
    cb_push_back(cb_id_in0, 4);
}

// from reader_unary of test unary dram
//  DRAM to L1 read
void kernel_main1() {
    uint32_t src_addr = get_compile_time_arg_val(0);
    constexpr uint32_t bank_id = get_compile_time_arg_val(1);
    constexpr uint32_t num_of_transactions = get_compile_time_arg_val(2);
    constexpr uint32_t transaction_num_pages = get_compile_time_arg_val(3);
    constexpr uint32_t page_size_bytes = get_compile_time_arg_val(4);
    constexpr uint32_t cb_id_in0 = get_compile_time_arg_val(5);
    constexpr uint32_t test_id = get_compile_time_arg_val(6);

    constexpr uint32_t transaction_size_bytes = transaction_num_pages * page_size_bytes;
    constexpr uint32_t total_num_pages = num_of_transactions * transaction_num_pages;

    DeviceTimestampedData("Number of transactions", num_of_transactions);
    DeviceTimestampedData("Transaction size in bytes", transaction_size_bytes);
    DeviceTimestampedData("Test id", test_id);

    cb_reserve_back(cb_id_in0, 1);
    uint32_t l1_write_addr = get_write_ptr(cb_id_in0);
    {
        DeviceZoneScopedN("RISCV1");
        uint64_t src_base_noc_addr = get_noc_addr_from_bank_id<true>(bank_id, 0);
        for (uint32_t i = 0; i < num_of_transactions; i++) {
            /* The 64-bit NOC addresses consists of a 32-bit local address and a NOC XY coordinate. The local address
             * occupies the lower 32 bits and the NOC XY coordinate occupies the next 12 (unicast) to 24 (multicast)
             * bits. In the get_noc_addr call, we set the local address to 0 to get the base address. Then, we OR it
             * with the local address (src_addr) in each iteration to get the full NOC address. */
            uint64_t src_noc_addr = src_base_noc_addr | src_addr;

            noc_async_read(src_noc_addr, l1_write_addr, transaction_size_bytes);

            src_addr += transaction_size_bytes;
            l1_write_addr += transaction_size_bytes;
        }
        noc_async_read_barrier();
    }
    cb_push_back(cb_id_in0, 1);
}

// from reader unary sharded blocks interleaved start id kernel
#include <stdint.h>
#include <cstdint>

#include "tensix_types.h"

// #include "debug/dprint.h"

// Target 8KB of data before a single barrier for 8x8 grid of readers
template <uint32_t tile_bytes, uint32_t num_readers>
constexpr uint32_t get_barrier_read_threshold() {
    return ((512 / num_readers) * (1024 + 128)) / tile_bytes;
}

void kernel_main2() {
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t block_height_tiles = get_arg_val<uint32_t>(1);
    const uint32_t block_width_tiles = get_arg_val<uint32_t>(2);
    const uint32_t padded_offset_bytes = get_arg_val<uint32_t>(3);       // input width in tiles - block width in tiles
    const uint32_t input_width_offset_tiles = get_arg_val<uint32_t>(4);  // input width in tiles - block width in tiles
    const uint32_t block_num_tiles = get_arg_val<uint32_t>(5);           // block_height_tiles * block_width_tiles
    const uint32_t start_id_offset = get_arg_val<uint32_t>(6);
    const uint32_t start_id_base = get_arg_val<uint32_t>(7);
    const uint32_t start_id = start_id_base + start_id_offset;

    constexpr uint32_t cb_id_in0 = get_compile_time_arg_val(0);
    constexpr bool src_is_dram = get_compile_time_arg_val(1) == 1;
    constexpr uint32_t num_readers = get_compile_time_arg_val(2);

    constexpr uint32_t tile_bytes = get_tile_size(cb_id_in0);
    constexpr DataFormat data_format = get_dataformat(cb_id_in0);

    const InterleavedAddrGenFast<src_is_dram> s = {
        .bank_base_address = src_addr, .page_size = tile_bytes, .data_format = data_format};

    constexpr uint32_t barrier_threshold = get_barrier_read_threshold<tile_bytes, num_readers>();
    uint32_t barrier_count = 0;
    uint32_t curr_tile_id = start_id;
    cb_reserve_back(cb_id_in0, block_num_tiles);
    uint32_t l1_write_addr = get_write_ptr(cb_id_in0);
    for (uint32_t h = 0; h < block_height_tiles; h++) {
        uint32_t tile_id = curr_tile_id;
        for (uint32_t w = 0; w < block_width_tiles; w++) {
            noc_async_read_tile(tile_id, s, l1_write_addr);
            tile_id++;
            l1_write_addr += tile_bytes;
            if (++barrier_count == barrier_threshold) {
                noc_async_read_barrier();
                barrier_count = 0;
            }
        }
        l1_write_addr += padded_offset_bytes;
        curr_tile_id += input_width_offset_tiles;
    }
    noc_async_read_barrier();
    cb_push_back(cb_id_in0, block_num_tiles);
}
