// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"

#include "debug/dprint.h"

/*
To improve performance of both reader and writer kernels the work has been split so that they both prepare input and
save output data.

Reader:
    * Reads input value data from DRAM and writes it to L1 circular buffer.
    * Write processed index data from L1 to DRAM.

Writer:
    * Generates index input data and writes it to L1 circular buffer.
    * Write output values from L1 to DRAM.
*/
void kernel_main() {
    // Runtime args
    const uint32_t input_tensor_buffer_addr = get_arg_val<uint32_t>(0);
    const uint32_t index_tensor_buffer_addr = get_arg_val<uint32_t>(1);
    const uint32_t core_loop_count = get_arg_val<uint32_t>(2);

    // Compile time args
    constexpr uint32_t input_tensor_cb_index = get_compile_time_arg_val(0);
    constexpr uint32_t index_tensor_output_cb_index = get_compile_time_arg_val(1);
    constexpr bool input_tensor_is_dram = get_compile_time_arg_val(2) == 1;
    constexpr bool index_tensor_is_dram = get_compile_time_arg_val(3) == 1;
    constexpr uint32_t Wt = get_compile_time_arg_val(4);
    constexpr uint32_t Ht = get_compile_time_arg_val(5);
    constexpr uint32_t total_number_of_cores = get_compile_time_arg_val(6);
    constexpr uint32_t compute_with_storage_grid_size_x = get_compile_time_arg_val(7);
    constexpr uint32_t compute_with_storage_grid_size_y = get_compile_time_arg_val(8);

    // Input tensor config
    constexpr uint32_t one_tile = 1;
    constexpr uint32_t tile_size_bytes = get_tile_size(input_tensor_cb_index);
    constexpr DataFormat input_tensor_data_format = get_dataformat(input_tensor_cb_index);
    const InterleavedAddrGenFast<input_tensor_is_dram> interleaved_accessor0 = {
        .bank_base_address = input_tensor_buffer_addr,
        .page_size = tile_size_bytes,
        .data_format = input_tensor_data_format};

    // Index tensor config
    const uint32_t index_tensor_output_tile_size_bytes = get_tile_size(index_tensor_output_cb_index);
    const DataFormat index_tensor_output_data_format = get_dataformat(index_tensor_output_cb_index);
    const InterleavedAddrGenFast<index_tensor_is_dram> interleaved_accessor1 = {
        .bank_base_address = index_tensor_buffer_addr,
        .page_size = index_tensor_output_tile_size_bytes,
        .data_format = index_tensor_output_data_format};

    for (uint32_t core_loop = 0; core_loop < core_loop_count; core_loop++) {
        // Calculate tile h coordinate
        // TODO: Adapt with new logic where more than 1 core can process row
        // Two cores must be able to process the same row, but
        const uint32_t h = core_loop * total_number_of_cores +
                           get_absolute_logical_y() * compute_with_storage_grid_size_x + get_absolute_logical_x();

        // Read input value data
        for (uint32_t w = 0; w < Wt; w++) {
            cb_reserve_back(input_tensor_cb_index, one_tile);
            const uint32_t l1_write_addr = get_write_ptr(input_tensor_cb_index);
            noc_async_read_tile(h * Wt + w, interleaved_accessor0, l1_write_addr);
            noc_async_read_barrier();
            cb_push_back(input_tensor_cb_index, one_tile);
        }  // Wt loop

        // Write output index data
        for (uint32_t w = 0; w < Wt; w++) {
            cb_wait_front(index_tensor_output_cb_index, one_tile);
            const uint32_t l1_write_addr_index = get_read_ptr(index_tensor_output_cb_index);
            noc_async_write_tile(h * Wt + w, interleaved_accessor1, l1_write_addr_index);
            noc_async_write_barrier();
            cb_pop_front(index_tensor_output_cb_index, one_tile);
        }  // Wt loop
    }  // core_loop_count loop

    // =======
    // FOR SHARDING ACROSS SAME AXIS

    // We need Reader/Writer kernel to synchronize here
    // Otherwise, reader (and writer) could be reader and forwarding invalid data
    // Idea: We can do that with 2 'dummy' CB

    // Notify writer that reader is done (i.e. local sort is complete)
    cb_reserve_back(cb_reader_done, one_tile);
    cb_push_back(cb_reader_done, one_tile);
    // Wait until writer is done by reaing from 'dummy' buffer
    cb_wait_front(cb_writer_done, one_tile);
    cb_pop_front(cb_writer_done, one_tile);

    // 1. Send data to other core
    // 2. Receive data from other core
    // Warning: beware of deadlock => use total order on physical core id
    // 3. write data to compute kernel
    // TODO: Does Tensor memory config matter here ?

    // TODO: Beware of deadlock if CB < total input size
    // Solution: Interleave copies
    const uint32_t sem_input = get_semaphore(get_arg_val<uint32_t>(0));  // TODO: Pass argument
    auto sem_self_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_input);

    for (uint32_t core_loop = 0; core_loop < core_loop_count; core_loop++) {
        const uint32_t h = core_loop * total_number_of_cores +
                           get_absolute_logical_y() * compute_with_storage_grid_size_x + get_absolute_logical_x();

        const uint32_t tile_offset = h * Wt + w;
        // Read input value data
        for (uint32_t w = 0; w < Wt; w++) {
            cb_reserve_back(input_tensor_cb_index, one_tile);
            const uint32_t l1_write_addr = get_write_ptr(input_tensor_cb_index);
            noc_async_read_tile(tile_offset, interleaved_accessor0, l1_write_addr);
            noc_async_read_barrier();

            cb_reserve_back(input_other_cb_index, one_tile);
            const uint32_t input_other_addr = get_write_ptr(input_other_cb_index);  // Local address

            const uint64_t input_remote_noc_addr = get_noc_addr(other, core_y, OTHER_DATA_PTR);  // TODO

            // We could put unicast code in a separate loop, but we'd need another circular buffer
            // TODO: Move this to a function
            if (this_core < other_core) {  // total order => avoid deadlocks
                // Write remote first / read then

                noc_semaphore_wait(input_self_sem_addr, 1);

                noc_async_write(l1_write_addr, input_remote_noc_addr, one_tile);
                noc_async_write_barrier();

                // Notify writer that ready to read
                noc_semaphore_inc(input_other_sem_addr, 1);
                noc_semaphore_set(input_other_sem_addr, 0);  // reset semaphore

                // We need another semaphore to know when said data has been written
                // It will copy data into desired circular buffer => no need to to more
                noc_semaphore_wait(sem_other_addr, 1);
            } else {
                // Notify writer
                noc_semaphore_inc(input_other_sem_addr, 1);

                // Note: other core does not need to notify this core that data has been written.
                // Because if next `noc_semaphore_wait()` passes, then that means other is ready to read
                // i.e. has written its data (write barrier ensures this behavior)

                // Wait for 1 reader
                noc_semaphore_wait(input_self_mem_addr, 1);
                noc_async_write(l1_write_addr, input_remote_noc_addr, one_tile);  // send data
                noc_async_write_barrier();

                // notify other that data has been written
                noc_semaphore_inc(sem_other_addr, 1);
                noc_semaphore_set(sem_other_addr, 0);  // reset semaphore
            }

            cb_push_back(input_other_cb_index, one_tile);
            cb_push_back(input_tensor_cb_index, one_tile);
        }  // Wt loop

        // Write output index data
        for (uint32_t w = 0; w < Wt; w++) {
            cb_wait_front(index_tensor_output_cb_index, one_tile);
            uint32_t l1_write_addr_index = get_write_ptr(index_tensor_cb_index);
            noc_async_write(h * Wt + w, interleaved_accessor1, l1_write_addr_index);
            noc_async_write_barrier();

            if (this_core < other_core) {  // total order => avoid deadlocks

            } else {
            }

            cb_pop_front(index_tensor_output_cb_index, one_tile);
        }  // Wt loop
    }
}
