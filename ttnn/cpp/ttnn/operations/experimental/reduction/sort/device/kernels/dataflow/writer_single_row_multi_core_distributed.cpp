// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

#include "debug/dprint.h"
#include "sort_distributed_common.hpp"
#include "../sort_debug_common.hpp"
#include "tt-metalium/bfloat16.hpp"

FORCE_INLINE void generate_index_tile(const uint32_t cb_id, const uint32_t wt) {
    constexpr uint32_t one_tile = 1;
    // Reserve space
    cb_reserve_back(cb_id, one_tile);

    // Writer config
    const uint32_t writer_addr = get_write_ptr(cb_id);
    volatile tt_l1_ptr uint16_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(writer_addr);
    const uint16_t wt_offset = wt << 5;  // wt * 2^(5)

    // Writer loop
    uint32_t count = 0;
    /*
    The 32x32 tile is subdivided into four 16x16 quadrants(faces): top-left, top-right, bottom-left, and bottom-right.
    These quadrants are stored contiguously in memory. Therefore, indices must be written in memory according
    to their respective quadrant, rather than sequentially from left to right across the entire tile.
    */
    constexpr uint32_t tile_faces = 2;
    constexpr uint32_t face_size = 16;
    for (uint32_t i = 0; i < tile_faces; ++i) {
        for (uint32_t j = 0; j < tile_faces; ++j) {
            for (uint32_t k = 0; k < face_size; ++k) {
                for (uint32_t l = 0; l < face_size; l++) {
                    const uint16_t value = l + face_size * j + wt_offset;
                    ptr[count] = value;
                    count++;
                }  // l loop
            }  // k loop
        }  // j loop
    }  // i loop

    // Push the tile
    cb_push_back(cb_id, one_tile);
}

FORCE_INLINE void print_row_bf16(bfloat16* ptr, uint32_t len) {
    DPRINT << TERM_WRITER;
    for (uint32_t i = 0; i < len; i++) {
        DPRINT <<
    }
    DPRINT << TERM_RESET << ENDL();
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
    const uint32_t value_tensor_buffer_addr = get_arg_val<uint32_t>(0);
    const uint32_t core_loop_count = get_arg_val<uint32_t>(1);
    const uint32_t this_core_x = get_arg_val<uint32_t>(2);
    const uint32_t this_core_y = get_arg_val<uint32_t>(3);
    const uint32_t other_core_x = get_arg_val<uint32_t>(4);
    const uint32_t other_core_y = get_arg_val<uint32_t>(5);

    // Compile time args
    constexpr uint32_t value_tensor_cb_index = get_compile_time_arg_val(0);
    constexpr uint32_t index_tensor_cb_index = get_compile_time_arg_val(1);
    constexpr uint32_t input_tensor_transposed_cb_index = get_compile_time_arg_val(2);
    constexpr uint32_t value_tensor_other_cb_index = get_compile_time_arg_val(3);
    constexpr uint32_t sync_cb_index = get_compile_time_arg_val(4);  // TO-ADD

    constexpr bool value_tensor_is_dram = get_compile_time_arg_val(5);
    constexpr uint32_t Wt = get_compile_time_arg_val(6);
    constexpr uint32_t Wt_per_core = get_compile_time_arg_val(7);
    constexpr uint32_t Ht = get_compile_time_arg_val(8);
    constexpr uint32_t total_number_of_cores = get_compile_time_arg_val(9);
    constexpr uint32_t num_cores_y = get_compile_time_arg_val(10);
    constexpr uint32_t compute_with_storage_grid_size_x = get_compile_time_arg_val(11);
    constexpr uint32_t compute_with_storage_grid_size_y = get_compile_time_arg_val(12);
    const uint32_t sem_value_addr = get_semaphore(get_compile_time_arg_val(13));

    const uint32_t this_core_id =
        compute_core_id(this_core_x, this_core_y, compute_with_storage_grid_size_x, compute_with_storage_grid_size_y);

    const uint32_t other_core_id =
        compute_core_id(other_core_x, other_core_y, compute_with_storage_grid_size_x, compute_with_storage_grid_size_y);

    // Output tensor config
    constexpr uint32_t one_tile = 1;
    const uint32_t value_tensor_tile_size_bytes = get_tile_size(value_tensor_cb_index);
    const DataFormat value_tensor_data_format = get_dataformat(value_tensor_cb_index);
    const InterleavedAddrGenFast<value_tensor_is_dram> interleaved_accessor0 = {
        .bank_base_address = value_tensor_buffer_addr,
        .page_size = value_tensor_tile_size_bytes,
        .data_format = value_tensor_data_format};

    const uint32_t w_start = get_absolute_logical_x() * Wt_per_core;

    DPRINT << TERM_WRITER << "[Writer] Wt = " << Wt << ", Wt_per_core = " << Wt_per_core << ", w_start = " << w_start
           << ", this core = {" << this_core_x << ", " << this_core_y << "} (id =" << this_core_id << ")" << TERM_RESET
           << ENDL();

    DPRINT << TERM_WRITER << "[Writer] other core = {" << other_core_x << ", " << other_core_y
           << "} (id = " << other_core_id << ")" << TERM_RESET << ENDL();

    // Move data from L1 to DRAMs
    for (uint32_t core_loop = 0; core_loop < core_loop_count; core_loop++) {
        // Calculate tile h coordinate
        // const uint32_t h = core_loop * total_number_of_cores +
        //                    get_absolute_logical_y() * compute_with_storage_grid_size_x + get_absolute_logical_x();
        const uint32_t h = core_loop * num_cores_y + get_absolute_logical_y();

        // Generate index tiles
        for (uint32_t w = w_start; w < w_start + Wt; w++) {
            generate_index_tile(index_tensor_cb_index, w);
        }  // Wt loop

        // This will synchronize with compute kernel, as well as with peer core
        // At this point, compute kernel has completed its sorting, we read its data and write
        // it to the remote peer core
        // Conversely, the peer core will write its data to a tile (in input_other_cb_index)
        // Once we have received it (after noc_exchange_tiles()), we send the data to the compute
        // core
        // In this case, we only use input_tensor_transposed_cb_index to synchronize with compute kernel,
        // we do not overwrite its data
        uint64_t sem_value_other_noc_addr = get_noc_addr(other_core_x, other_core_y, sem_value_addr);
        uint64_t sem_value_noc_addr = get_noc_addr(this_core_x, this_core_y, sem_value_addr);

        sem_ptr_t sem_self_value_other_ptr = reinterpret_cast<sem_ptr_t>(sem_value_addr);
        const uint32_t value_tensor_other_tile_size_bytes = get_tile_size(value_tensor_other_cb_index);

        // TODO: Add syncrhonisation barrier with compute kernel
        // Wait for Compute for complete
        // Use sync_cb as barrier
        cb_wait_front(sync_cb_index, one_tile);
        cb_pop_front(sync_cb_index, one_tile);

        DPRINT << TERM_WRITER << "[Writer] waiting for compute..." << TERM_RESET << ENDL();

        // TODO: Beware of synchronization between Reader and Compute regarding input_tensor_transposed_cb_index
        uint32_t dbg_w = 0;
        for (uint32_t w = w_start; w < w_start + Wt_per_core; w++, dbg_w++) {
            cb_wait_front(input_tensor_transposed_cb_index, one_tile);
            const uint32_t l1_read_ptr = get_read_ptr(input_tensor_transposed_cb_index);

            cb_reserve_back(value_tensor_other_cb_index, one_tile);
            uint32_t input_other_cb_write_addr = get_write_ptr(value_tensor_other_cb_index);
            uint64_t input_other_noc_addr = get_noc_addr(other_core_x, other_core_y, input_other_cb_write_addr);

            DPRINT << TERM_WRITER << "[Writer] exchanging tile #" << dbg_w << "/" << Wt_per_core << " with "
                   << other_core_id << " (self = " << this_core_id << ")"
                   << ", sem_self = " << sem_value_noc_addr << " (" << sem_value_addr
                   << "), sem_other_noc = " << sem_value_other_noc_addr << TERM_RESET << ENDL();
            sort_noc_exchange_tiles(
                this_core_id,
                other_core_id,
                sem_self_value_other_ptr,
                sem_value_other_noc_addr,
                l1_read_ptr,
                input_other_noc_addr,
                value_tensor_other_tile_size_bytes);

            DPRINT << TERM_WRITER
                   << "[Writer] sending other tile back to compute, other_cb = " << value_tensor_other_cb_index
                   << TERM_RESET << ENDL();
            cb_push_back(value_tensor_other_cb_index, one_tile);

            cb_pop_front(input_tensor_transposed_cb_index, one_tile);
        }  // Wt loop

        DPRINT << TERM_WRITER << "[Writer] exchange complete" << TERM_RESET << ENDL();
        // Sort is compolete, we read final results from value_tensor_cb_index
        // Write value tensor to DRAM
        for (uint32_t w = w_start; w < w_start + Wt_per_core; w++) {
            cb_wait_front(value_tensor_cb_index, one_tile);
            const uint32_t l1_write_addr_val = get_read_ptr(value_tensor_cb_index);
            noc_async_write_tile(h * Wt + w, interleaved_accessor0, l1_write_addr_val);
            noc_async_write_barrier();
            cb_pop_front(value_tensor_cb_index, one_tile);
        }  // Wt loop

        DPRINT << TERM_WRITER << "[Writer] complete" << TERM_RESET << ENDL();

    }  // core_loop_count loop

    //
}
