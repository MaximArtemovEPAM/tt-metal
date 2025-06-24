// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"

#include "debug/dprint.h"
#include "sort_distributed_common.hpp"
#include "../sort_debug_common.hpp"

FORCE_INLINE void print_row_bf16(uint16_t* ptr, uint32_t len) {
    DPRINT << TERM_READER;
    DPRINT << "[";
    for (uint32_t i = 0; i < len; i++) {
        uint16_t val = ptr[i];
        DPRINT << " " << BF16(val) << " (" << HEX{} << val << DEC{} << ")";
    }
    DPRINT << "]" << TERM_RESET << ENDL();
}

FORCE_INLINE void print_tile_bf16(uint16_t* tile_ptr) {
    constexpr uint32_t FACE_LEN = 16;
    constexpr uint32_t TILE_LEN = 32 * 32;
    constexpr uint32_t TILE_HEIGHT = TILE_LEN / FACE_LEN;

    uint32_t offset = 0;
    for (uint32_t i = 0; i < TILE_HEIGHT; i++) {
        DPRINT << TERM_READER << i << " | ";
        for (uint32_t j = 0; j < FACE_LEN; j++, offset++) {
            uint32_t val = tile_ptr[offset];

            DPRINT << SETW(6) << BF16(val) << " ";
        }
        DPRINT << TERM_RESET << ENDL();
    }
}

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
    const uint32_t this_core_logical_id = get_arg_val<uint32_t>(3);
    const uint32_t this_core_x = get_arg_val<uint32_t>(4);
    const uint32_t this_core_y = get_arg_val<uint32_t>(5);

    const uint32_t other_core0_x = get_arg_val<uint32_t>(6);
    const uint32_t other_core0_y = get_arg_val<uint32_t>(7);

    const uint32_t other_core1_x = get_arg_val<uint32_t>(8);
    const uint32_t other_core1_y = get_arg_val<uint32_t>(9);

    // Compile time args
    constexpr uint32_t input_tensor_cb_index = get_compile_time_arg_val(0);
    constexpr uint32_t index_tensor_output_cb_index = get_compile_time_arg_val(1);
    constexpr uint32_t index_tensor_transposed_cb_index = get_compile_time_arg_val(2);
    constexpr uint32_t index_tensor_other_cb_index = get_compile_time_arg_val(3);
    constexpr uint32_t sync_with_reader_cb_index = get_compile_time_arg_val(4);

    constexpr bool input_tensor_is_dram = get_compile_time_arg_val(5) == 1;
    constexpr bool index_tensor_is_dram = get_compile_time_arg_val(6) == 1;
    constexpr uint32_t Wt = get_compile_time_arg_val(7);
    constexpr uint32_t Wt_per_core = get_compile_time_arg_val(8);
    constexpr uint32_t Ht = get_compile_time_arg_val(9);
    constexpr uint32_t total_number_of_cores = get_compile_time_arg_val(10);
    constexpr uint32_t num_cores_y = get_compile_time_arg_val(11);
    constexpr uint32_t intercore_stages = get_compile_time_arg_val(12);
    constexpr uint32_t compute_with_storage_grid_size_x = get_compile_time_arg_val(13);
    constexpr uint32_t compute_with_storage_grid_size_y = get_compile_time_arg_val(14);
    const uint32_t sem_index_addr = get_semaphore(get_compile_time_arg_val(15));

    const uint32_t this_core_id =
        compute_core_id(this_core_x, this_core_y, compute_with_storage_grid_size_x, compute_with_storage_grid_size_y);

    const uint32_t other_core0_id = compute_core_id(
        other_core0_x, other_core0_y, compute_with_storage_grid_size_x, compute_with_storage_grid_size_y);
    const uint32_t other_core1_id = compute_core_id(
        other_core1_x, other_core1_y, compute_with_storage_grid_size_x, compute_with_storage_grid_size_y);

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

    const uint32_t w_start = get_absolute_logical_x() * Wt_per_core;

    // DPRINT << TERM_READER << "[Reader] starting..." << TERM_RESET << ENDL();
    DPRINT << TERM_READER << "[Reader] Wt = " << Wt << ", Wt_per_core = " << Wt_per_core << ", w_start = " << w_start
           << ", intercore stages = " << intercore_stages << TERM_RESET << ENDL();

    for (uint32_t core_loop = 0; core_loop < core_loop_count; core_loop++) {
        // Calculate tile h coordinate
        // TODO: Adapt with new logic where more than 1 core can process row
        // Two cores must be able to process the same row, but
        // const uint32_t h = core_loop * total_number_of_cores +
        //                    get_absolute_logical_y() * compute_with_storage_grid_size_x + get_absolute_logical_x();
        const uint32_t h = core_loop * num_cores_y + get_absolute_logical_y();
        // DPRINT << TERM_READER << "[Reader] core loop = " << core_loop << " Ht = " << h
        //        << ", num_cores_y = " << num_cores_y << TERM_RESET << ENDL();

        // Read input value data
        // DPRINT << TERM_READER << "[Reader] writing value tiles" << TERM_RESET << ENDL();
        for (uint32_t w = w_start; w < w_start + Wt_per_core; w++) {
            cb_reserve_back(input_tensor_cb_index, one_tile);
            const uint32_t l1_write_addr = get_write_ptr(input_tensor_cb_index);
            noc_async_read_tile(h * Wt + w, interleaved_accessor0, l1_write_addr);
            // DPRINT << TERM_READER << "[Reader] tile [" << w << "] = " << TERM_RESET << ENDL();
            // print_row_bf16(reinterpret_cast<uint16_t*>(l1_write_addr), 8);

            noc_async_read_barrier();
            cb_push_back(input_tensor_cb_index, one_tile);
        }  // Wt loop

        sem_ptr_t sem_self_index_other_ptr = reinterpret_cast<sem_ptr_t>(sem_index_addr);
        const uint32_t index_tensor_other_tile_size_bytes = get_tile_size(index_tensor_other_cb_index);

        for (uint32_t core_stage = 1; core_stage <= intercore_stages; core_stage++) {
            for (uint32_t sub = core_stage; sub > 0; sub--) {
                // DPRINT << TERM_READER << "[Reader] stage #" << core_stage << "/" << intercore_stages << ", core sub =
                // " << sub << TERM_RESET << ENDL();

                uint32_t sub_dist = 1 << (sub - 1);

                uint32_t other_core_logical_id = 0 ^ sub_dist;  // TODO: Replace 0 with this_core_logical_id and use LUT

                uint32_t other_core_x = other_core0_x;
                uint32_t other_core_y = other_core0_y;
                uint32_t other_core_id = other_core0_id;
                // Non modular version: assumes only 2 inter-core stages (i.e. 4 cores per row) and hardcoded values
                // TODO: Make it generic
                if (other_core_logical_id == 2) {
                    other_core_x = other_core1_x;
                    other_core_y = other_core1_y;
                    other_core_id = other_core1_id;
                }

                // Get other core coords
                uint64_t sem_index_other_noc_addr = get_noc_addr(other_core_x, other_core_y, sem_index_addr);
                uint64_t sem_index_noc_addr = get_noc_addr(this_core_x, this_core_y, sem_index_addr);  // only debug

                // Wait for Compute for complete
                // Use sync_with_writer_cb as barrier
                // DPRINT << TERM_READER << "[Reader] synchronizing with compute" << TERM_RESET << ENDL();
                cb_wait_front(sync_with_reader_cb_index, one_tile);
                cb_pop_front(sync_with_reader_cb_index, one_tile);
                // DPRINT << TERM_READER << "[Reader] synchronized with compute" << TERM_RESET << ENDL();

                // Exchange Index tile with peer
                DPRINT << TERM_READER << "[Reader] stage #" << core_stage << ", sub = " << sub_dist << ", "
                       << this_core_id << " -> " << other_core_id << TERM_RESET << ENDL();
                for (uint32_t w = w_start; w < w_start + Wt_per_core; w++) {
                    cb_wait_front(index_tensor_transposed_cb_index, one_tile);
                    const uint32_t l1_read_ptr = get_read_ptr(index_tensor_transposed_cb_index);

                    cb_reserve_back(index_tensor_other_cb_index, one_tile);
                    uint32_t index_other_cb_write_addr = get_write_ptr(index_tensor_other_cb_index);
                    uint64_t index_other_noc_addr = get_noc_addr(other_core_x, other_core_y, index_other_cb_write_addr);

                    // DPRINT << TERM_READER << "[Reader] exchanging tile #" << (w - w_start) << "/" << Wt_per_core << "
                    // with "
                    //        << other_core_id << " (self = " << this_core_id << ")"
                    //        << ", sem_self = " << sem_index_noc_addr << " (" << sem_index_addr
                    //        << "), sem_other_noc = " << sem_index_other_noc_addr << TERM_RESET << ENDL();
                    sort_noc_exchange_tiles(
                        this_core_id,
                        other_core_id,
                        sem_self_index_other_ptr,
                        sem_index_other_noc_addr,
                        l1_read_ptr,
                        index_other_noc_addr,
                        index_tensor_other_tile_size_bytes);

                    constexpr uint32_t DEBUG_PRINT_LEN = 8;  // only print first 8 elements

                    // DPRINT << TERM_READER
                    //        << "[Reader] sending other tile back to compute, other_cb = " <<
                    //        index_tensor_other_cb_index
                    //        << TERM_RESET << ENDL();
                    cb_push_back(index_tensor_other_cb_index, one_tile);

                    cb_pop_front(index_tensor_transposed_cb_index, one_tile);
                }  // Wt
            }  // sub
        }  // core_stage

        // TODO: Move it back down and handle inter-core handshakes
        //       Right now, if we read input tiles before index then we have a deadlock
        //       Indeed, index_tensor_output_cb_index gets filled, which blocks compute and writer
        //       But since they (currently) produce value tiles at the end (after index tiles),
        //       there is a deadlock
        // Write output index data
        // DPRINT << TERM_READER << "[Reader] writing index tiles" << TERM_RESET << ENDL();
        for (uint32_t w = w_start; w < w_start + Wt_per_core; w++) {
            cb_wait_front(index_tensor_output_cb_index, one_tile);
            const uint32_t l1_write_addr_index = get_read_ptr(index_tensor_output_cb_index);
            noc_async_write_tile(h * Wt + w, interleaved_accessor1, l1_write_addr_index);
            noc_async_write_barrier();
            cb_pop_front(index_tensor_output_cb_index, one_tile);
        }  // Wt loop

    }  // core_loop_count loop

    DPRINT << TERM_READER << "[Reader] completing" << TERM_RESET << ENDL();
}
