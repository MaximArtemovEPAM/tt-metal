// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"

#include "debug/dprint.h"

void kernel_main() {
    // Parse run-time arguments
    const uint32_t input_addr = get_arg_val<uint32_t>(0);
    const uint32_t sem_input = get_semaphore(get_arg_val<uint32_t>(1));
    const uint32_t this_core_x = get_arg_val<uint32_t>(2);
    const uint32_t this_core_y = get_arg_val<uint32_t>(3);
    const uint32_t other_core_x = get_arg_val<uint32_t>(4);
    const uint32_t other_core_y = get_arg_val<uint32_t>(5);

    constexpr uint32_t input_cb_index = get_compile_time_arg_val(0);
    constexpr uint32_t input_other_cb_index = get_compile_time_arg_val(1);

    constexpr uint32_t one_tile = 1;

    // uint32_t this_core_x = get_absolute_logical_x();
    // uint32_t this_core_y = get_absolute_logical_y();

    DPRINT << "[Reader " << this_core_x << "] start" << ENDL();
    DPRINT << "[Reader " << this_core_x << "] other core = " << other_core_x << ", sem input " << sem_input << ENDL();

    // Read 1 tile from sharded buffer
    // Push and read tile from other buffer

    // 1. Send data to other core
    // 2. Receive data from other core
    // Warning: beware of deadlock => use total order on physical core id
    // 3. write data to compute kernel
    // TODO: Does Tensor memory config matter here ?

    const uint32_t input_tile_size = get_tile_size(input_cb_index);
    const auto input_dataformat = get_dataformat(input_cb_index);

    const InterleavedAddrGenFast<false> l1_input_addrg = {
        .bank_base_address = input_addr, .page_size = input_tile_size, .data_format = input_dataformat};

    // Setup semaphore
    uint64_t sem_input_remote_addr = get_noc_addr(other_core_x, other_core_y, sem_input);
    auto sem_self_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_input);

    uint64_t sem_self_global_addr = get_noc_addr(this_core_x, this_core_y, sem_input);

    uint32_t tile_offset = 0;
    // Read input value data

    DPRINT << "input_cb_index = " << input_cb_index << ", one_tile = " << one_tile << ENDL();

    cb_reserve_back(input_cb_index, one_tile);

    DPRINT << "[Reader " << this_core_x << "] reading tile from SRAM" << ENDL();

    const uint32_t l1_write_addr = get_write_ptr(input_cb_index);
    noc_async_read_tile(tile_offset, l1_input_addrg, l1_write_addr);
    noc_async_read_barrier();

    cb_reserve_back(input_other_cb_index, one_tile);

    const uint32_t input_other_addr = get_write_ptr(input_other_cb_index);  // Local address

    const uint32_t input_cb_addr = get_read_ptr(input_cb_index);
    const uint64_t input_remote_addr = get_noc_addr(other_core_x, other_core_y, input_other_addr);

    // We could put unicast code in a separate loop, but we'd need another circular buffer
    // TODO: Move this to a function
    if (this_core_x < other_core_x) {  // total order => avoid deadlocks
        // Write remote first / read then

        DPRINT << __LINE__ << " [Reader " << this_core_x << "] waiting on " << ENDL();
        noc_semaphore_wait(sem_self_ptr, 1);
        noc_semaphore_set(sem_self_ptr, 0);  // reset semaphore

        DPRINT << __LINE__ << " [Reader " << this_core_x << "] writing 1 tile to " << input_remote_addr << " from "
               << l1_write_addr << ENDL();
        noc_async_write(l1_write_addr, input_remote_addr, input_tile_size);
        noc_async_write_barrier();

        // Notify writer that ready to read
        noc_semaphore_inc(sem_input_remote_addr, 1);

        // We need another semaphore to know when said data has been written
        // It will copy data into desired circular buffer => no need to to more
        noc_semaphore_wait(sem_self_ptr, 1);
        noc_semaphore_set(sem_self_ptr, 0);  // reset semaphore

    } else {
        // Notify writer
        DPRINT << __LINE__ << "[Reader " << this_core_x << "] incrementing sem " << sem_input_remote_addr << ENDL();

        noc_semaphore_inc(sem_input_remote_addr, 1);

        // Note: other core does not need to notify this core that data has been written.
        // Because if next `noc_semaphore_wait()` passes, then that means other is ready to read
        // i.e. has written its data (write barrier ensures this behavior)

        // Wait for 1 reader
        DPRINT << __LINE__ << "[Reader " << this_core_x << "] waiting on " << ENDL();
        noc_semaphore_wait(sem_self_ptr, 1);
        noc_semaphore_set(sem_self_ptr, 0);  // reset semaphore

        DPRINT << __LINE__ << "[Reader " << this_core_x << "] writing 1 tile to " << input_remote_addr << " from "
               << l1_write_addr << ENDL();

        noc_async_write(l1_write_addr, input_remote_addr, input_tile_size);  // send data
        noc_async_write_barrier();

        // notify other that data has been written
        DPRINT << __LINE__ << "[Reader " << this_core_x << "] incrementing sem " << sem_input_remote_addr << ENDL();
        noc_semaphore_inc(sem_input_remote_addr, 1);
    }

    cb_push_back(input_other_cb_index, one_tile);
    cb_push_back(input_cb_index, one_tile);

    DPRINT << "[Reader " << this_core_x << "] finshed " << ENDL();
}
