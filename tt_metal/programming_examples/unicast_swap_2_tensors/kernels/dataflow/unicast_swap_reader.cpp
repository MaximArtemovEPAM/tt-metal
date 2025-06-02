// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"

#include "debug/dprint.h"

constexpr uint32_t compute_core_id(uint32_t core_x, uint32_t core_y, uint32_t grid_size_x, uint32_t grid_size_y) {
    return core_x + core_y * grid_size_x;
}

void print_tile_row0(uint32_t tile_addr) {
    DPRINT << "tile = [";
    const float* ptr = (float*)tile_addr;
    for (int i = 0; i < 32; i++) {
        DPRINT << ptr[i] << " ";
    }
    DPRINT << "]";
}

void kernel_main() {
    DPRINT << "[Reader] starting" << ENDL();

    const uint32_t input_dram_addr = get_arg_val<uint32_t>(0);
    const uint32_t intermed_sram_addr = get_arg_val<uint32_t>(1);
    const uint32_t sem_input = get_semaphore(get_arg_val<uint32_t>(2));
    const uint32_t this_core_x = get_arg_val<uint32_t>(3);
    const uint32_t this_core_y = get_arg_val<uint32_t>(4);
    const uint32_t other_core_x = get_arg_val<uint32_t>(5);
    const uint32_t other_core_y = get_arg_val<uint32_t>(6);

    constexpr uint32_t input_cb_index = get_compile_time_arg_val(0);
    constexpr uint32_t input_other_cb_index = get_compile_time_arg_val(1);
    constexpr uint32_t num_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t compute_with_storage_grid_size_x = get_compile_time_arg_val(3);
    constexpr uint32_t compute_with_storage_grid_size_y = get_compile_time_arg_val(4);

    constexpr uint32_t one_tile = 1;

    DPRINT << "[Reader] arguments parsed" << ENDL();

    const uint32_t this_core_id =
        compute_core_id(this_core_x, this_core_y, compute_with_storage_grid_size_x, compute_with_storage_grid_size_y);
    const uint32_t other_core_id =
        compute_core_id(other_core_x, other_core_y, compute_with_storage_grid_size_x, compute_with_storage_grid_size_y);

    // Read 1 tile from sharded buffer
    // Push and read tile from other buffer

    // 1. Send data to other core
    // 2. Receive data from other core
    // Warning: beware of deadlock => use total order on physical core id
    // 3. write data to compute kernel

    const uint32_t input_tile_size = get_tile_size(input_cb_index);
    const auto input_dataformat = get_dataformat(input_cb_index);

    const InterleavedAddrGenFast<true> dram_input_addrg = {
        .bank_base_address = input_dram_addr, .page_size = input_tile_size, .data_format = input_dataformat};

    const InterleavedAddrGenFast<false> sram_input_addrg = {
        .bank_base_address = intermed_sram_addr, .page_size = input_tile_size, .data_format = input_dataformat};

    // Setup semaphore
    uint64_t sem_input_other_noc_addr = get_noc_addr(other_core_x, other_core_y, sem_input);
    auto sem_self_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_input);

    // Read input value data
    const uint64_t intermed_other_noc_addr = get_noc_addr(other_core_x, other_core_y, intermed_sram_addr);
    const uint64_t intermed_this_noc_addr = get_noc_addr(this_core_x, this_core_y, intermed_sram_addr);

    for (uint32_t i = 0; i < num_tiles; i++) {
        // Copy tile from input buffer into circular buffer
        DPRINT << "[Reader " << this_core_id << "] #" << i << "/" << num_tiles << ENDL();

        // 1) Read tile from DRAM and store into temp SRAM buffer
        // 2) Copy tile into circular buffer

        cb_reserve_back(input_cb_index, one_tile);
        const uint32_t input_cb_write_addr = get_write_ptr(input_cb_index);

        // DRAM => Circular Buffer
        noc_async_read_tile(i, dram_input_addrg, input_cb_write_addr);
        noc_async_read_barrier();

        cb_reserve_back(input_other_cb_index, one_tile);
        uint32_t input_other_cb_write_addr = get_write_ptr(input_other_cb_index);
        uint64_t input_other_this_noc_addr = get_noc_addr(this_core_x, this_core_y, input_other_cb_write_addr);
        uint64_t input_other_noc_addr = get_noc_addr(other_core_x, other_core_y, input_other_cb_write_addr);

        DPRINT << "[Reader " << this_core_id << "] input_other addr = " << input_other_cb_write_addr
               << ", this noc addr = " << input_other_this_noc_addr << ", noc addr = " << input_other_noc_addr
               << ENDL();

        if (this_core_id < other_core_id) {  // total order => avoid deadlocks
            // Write remote first / read then

            // Do not write until peer is ready => wait for him
            // We can't proceed before this because we don't know if it is safe to write into buffer
            noc_semaphore_wait(sem_self_ptr, 1);
            noc_semaphore_set(sem_self_ptr, 0);  // reset semaphore

            // Peer is ready => send data
            noc_async_write(input_cb_write_addr, input_other_noc_addr, input_tile_size);
            noc_async_write_barrier();

            // Data has been sent and written => we are ready to receive
            // Before this, we wake up other thread
            noc_semaphore_inc(sem_input_other_noc_addr, 1);

            // We need another semaphore to know when said data has been written
            // It will copy data into desired circular buffer => no need to to more
            noc_semaphore_wait(sem_self_ptr, 1);
            noc_semaphore_set(sem_self_ptr, 0);  // reset semaphore

        } else {
            // This section is similar to previous one, except that
            // 1) waiting and writing are reversed
            // 2)

            noc_semaphore_inc(sem_input_other_noc_addr, 1);

            // Note: other core does not need to notify this core that data has been written.
            // Because if next `noc_semaphore_wait()` passes, then that means other is ready to read
            // i.e. has written its data (write barrier ensures this behavior)

            // Wait for 1 reader
            noc_semaphore_wait(sem_self_ptr, 1);
            noc_semaphore_set(sem_self_ptr, 0);  // reset semaphore

            // Send data
            noc_async_write(input_cb_write_addr, input_other_noc_addr, input_tile_size);
            noc_async_write_barrier();

            // notify other that data has been written
            noc_semaphore_inc(sem_input_other_noc_addr, 1);
        }
        cb_push_back(input_cb_index, one_tile);

        // Temp Buffer => Circular buffer
        // cb_reserve_back(input_other_cb_index, one_tile);

        // Put temporary data into circular buffer
        // Technically, we could also have copied data directly to circular buffer but that would mean making sure
        // we are not having any data-race with both the reader and the unpacker
        // uint32_t input_other_cb_write_addr = get_write_ptr(input_other_cb_index);

        // noc_async_write(intermed_sram_addr, input_other_cb_write_addr, input_tile_size);
        // noc_async_write_barrier();

        print_tile_row0(input_other_cb_write_addr);
        DPRINT << ENDL();

        cb_push_back(input_other_cb_index, one_tile);

        DPRINT << "[Reader " << this_core_id << "] finished with loop #" << i << "/" << num_tiles << ENDL();
    }
}
