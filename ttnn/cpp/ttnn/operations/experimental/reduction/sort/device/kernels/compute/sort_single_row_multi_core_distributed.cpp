// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "compute_kernel_api.h"
#include "compute_kernel_api/transpose_wh.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/reconfig_data_format.h"
#include "compute_kernel_api/pack.h"
#include "compute_kernel_api/binary_max_min.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"

#include "sort_common.hpp"
#include "../sort_debug_common.hpp"

#include "debug/dprint.h"

namespace NAMESPACE {
/*
This sorting algorithm is based on Bitonic Merge Sort and operates on input data arranged in tiles.

The algorithm processes the data such that the dimension to be sorted becomes the last dimension of the tensor.
From the perspective of tile arrangement, sorting is performed row by row in a matrix-like structure.

### Overview:
1. **Tile Initialization**:
    - A full row of tiles (size `Wt`) is read from DRAM into L1 memory.
    - Corresponding tiles containing the initial data indices are also generated.

2. **Sorting Mechanism**:
    - The core of the sorting is performed using `ckernel::topk_local_sort`, which:
      - Sorts two input tiles in-place.
      - Updates the indices of the data to reflect the new order.
    - Since `ckernel::topk_local_sort` operates on columns, an additional transposition step is required.
    - The number of tiles in the `Wt` dimension must be a multiple of 64 (2 * Tile_Width (32)) to ensure compatibility.

3. **Bitonic Sequence Formation**:
    - The function `sort_Wt_tiles_row_to_bitonic_sequence`:
      - Sorts pairs of tiles alternately in ascending and descending order.
      - Produces a set of sorted tile pairs with alternating sorting directions.

4. **Bitonic Merge Sort**:
    - The tiles are further sorted in stages to ensure the entire row is sorted.
    - At each stage, tile indices are calculated, and tiles are sorted pairwise.
    - This process continues until all tiles in the row are sorted.

5. **Multicore Calculation**:
    - Multicore parallelism is enabled by assigning each row of tiles (`Wt`) to a separate core.
    - If the number of rows (`Ht`) exceeds the number of available cores, the workload is distributed such that some
cores process multiple rows.
    - This ensures efficient utilization of all cores and minimizes idle time during computation.

6. **Final Steps**:
    - Once sorted, the tiles are transposed back to the desired dimension.
    - The sorted data is then written back to DRAM.

### Example:
- Input: A 64x128 matrix, represented as 2x4 tiles: T0, T1, T2, T3
                                                    T4, T5, T6, T7
- Sorting (ascending order):
0. Distributing workload across cores:
   - Core 0 processes T0, T1, T2, T3
   - Core 1 processes T4, T5, T6, T7
Calculation of each row:
  1. **Pairwise Sorting**:
      - T0 and T1 are sorted as a pair in ascending order.
      - T2 and T3 are sorted as a pair in descending order.
  2. **Sorting Across Pairs**:
      - **Stage 1**: T0 and T2 are sorted in ascending order, and T1 and T3 are sorted in ascending order.
      - **Stage 2**: T0 and T1 are sorted in ascending order, and T2 and T3 are sorted in ascending order.
  3. **Data Saving**:
      - The tiles are now fully sorted along the desired dimension and ready to be saved.
 */
void MAIN {
    // Runtime args
    const uint32_t core_loop_count = get_arg_val<uint32_t>(0);
    const uint32_t select_min = get_arg_val<uint32_t>(1);

    DPRINT << TERM_COMPUTE << "[Compute] starting..." << TERM_RESET << ENDL();

    // Compile time args
    constexpr uint32_t input_tensor_cb_index = get_compile_time_arg_val(0);
    constexpr uint32_t index_tensor_cb_index = get_compile_time_arg_val(1);
    constexpr uint32_t input_tensor_transposed_cb_index = get_compile_time_arg_val(2);
    constexpr uint32_t index_tensor_transposed_cb_index = get_compile_time_arg_val(3);
    constexpr uint32_t value_tensor_cb_index = get_compile_time_arg_val(4);
    constexpr uint32_t index_tensor_output_cb_index = get_compile_time_arg_val(5);
    constexpr uint32_t value_tensor_other_cb_index = get_compile_time_arg_val(6);
    constexpr uint32_t sync_cb_index = get_compile_time_arg_val(7);

    constexpr uint32_t Wt = get_compile_time_arg_val(8);
    constexpr uint32_t Wt_per_core = get_compile_time_arg_val(9);
    constexpr bool descending = get_compile_time_arg_val(10);
    constexpr bool stable =
        get_compile_time_arg_val(11);  // TODO: In the future change LLK to have the option or add additional step with
                                       // checking values and indexes after the sorting
                                       // Issue: https://github.com/tenstorrent/tt-metal/issues/20625
    constexpr uint32_t compute_with_storage_grid_size_x = get_compile_time_arg_val(12);
    constexpr uint32_t compute_with_storage_grid_size_y = get_compile_time_arg_val(13);

    constexpr uint32_t one_tile = 1;

    constexpr uint32_t input_dest_start = 0;
    constexpr uint32_t index_dest_start = 2;
    constexpr uint32_t input_dest_end = 1;
    constexpr uint32_t index_dest_end = 3;

    DPRINT_MATH(
        DPRINT << TERM_COMPUTE << "[Compute] Wt = " << Wt << ", Wt_per_core = " << Wt_per_core
               << ", select_min = " << select_min << TERM_RESET << ENDL());

    ckernel::topk_tile_init();
    transpose_wh_init(input_tensor_cb_index, input_tensor_transposed_cb_index);

    for (uint32_t core_loop = 0; core_loop < core_loop_count; core_loop++) {
        const bool ascending = (!descending);

        DPRINT << TERM_COMPUTE << "[Compute] building bitonic sequence..." << TERM_RESET << ENDL();
        sort_Wt_tiles_row_to_bitonic_sequence(
            input_tensor_cb_index,
            index_tensor_cb_index,
            input_tensor_transposed_cb_index,
            index_tensor_transposed_cb_index,
            Wt_per_core,
            /*switch_dir=*/true,
            ascending,
            /*end_phase(log2(K))=*/5);

        DPRINT << TERM_COMPUTE << "[Compute] built bitonic sequence" << TERM_RESET << ENDL();
        // Wait for bitonic sequence of Wt tiles
        cb_wait_front(input_tensor_transposed_cb_index, Wt_per_core);
        // DPRINT << TERM_COMPUTE << "[Compute] cb_wait_front(input_tensor_cb)" << TERM_RESET << ENDL();
        cb_wait_front(index_tensor_transposed_cb_index, Wt_per_core);
        // DPRINT << TERM_COMPUTE << "[Compute] cb_wait_frotn(index_tensor_cb)" << TERM_RESET << ENDL();

        // Sort and merge step of bitonic merge sort
        uint32_t stages = 0;
        for (uint32_t i = Wt_per_core; i > 1; i >>= 1) {
            stages++;
        }

        for (uint32_t stage = 2; stage <= stages; stage++) {
            for (uint32_t sub = stage; sub > 0; sub--) {
                uint32_t sub_dist = 1 << (sub - 1);
                for (uint32_t i = 0; i < Wt_per_core; i++) {
                    uint32_t j = i ^ sub_dist;
                    if (j > i) {
                        // Determine direction for this comparison block
                        const bool ascending_block = ((i >> stage) & 1) == 0;
                        const bool dir = ascending_block == ascending;

                        // Get indexes of tiles to compare
                        const uint32_t left_tile_id = i;
                        const uint32_t right_tile_id = j;
                        /**
                         * Compute kernel for performing bitonic sort on tiles with synchronization caveats.
                         *
                         * Potential Bug: Unpacker and Packer Threads Synchronization Issue
                         *
                         * After migrating to the blackhole architecture, undefined behavior was observed, resulting in
                         * incorrect results. The core of the issue lies in the synchronization between the unpacker
                         * (reading tiles from CB to registers) and packer (writing tiles from registers back to CB)
                         * threads.
                         *
                         * In the this loop, two tiles are read from a circular buffer (CB) into LLK registers, an LLK
                         * operation is performed, and then the results are written back to the same CB. If there is
                         * insufficient synchronization between the packer and unpacker threads, it is possible that the
                         * packer thread does not have enough time to fully pack the tiles from the registers back to
                         * the CB before the next loop iteration begins. As a result, the unpacker thread in the next
                         * iteration may read tiles into registers before the previous packing operation is complete,
                         * leading to data hazards and undefined behavior.
                         *
                         * Debugging revealed that inserting a delay between loop iterations resolved the issue,
                         * suggesting a race condition between packing and unpacking. However, since there is no
                         * semaphore or similar synchronization primitive available in the compute kernel, it is not
                         * possible to enforce proper synchronization programmatically.
                         *
                         * As a temporary workaround, swapping the order of read and write operations helped mitigate
                         * the issue, but this is not a robust or permanent solution. Proper synchronization between
                         * packer and unpacker threads is required to ensure data integrity and correct results.
                         *
                         * See also: https://github.com/tenstorrent/tt-metal/pull/22340
                         */
                        tile_regs_acquire();
                        copy_tile_to_dst_init_short_with_dt(
                            input_tensor_transposed_cb_index, index_tensor_transposed_cb_index);
                        copy_tile(index_tensor_transposed_cb_index, left_tile_id, index_dest_start);
                        copy_tile(index_tensor_transposed_cb_index, right_tile_id, index_dest_end);

                        copy_tile_to_dst_init_short_with_dt(
                            index_tensor_transposed_cb_index, input_tensor_transposed_cb_index);
                        copy_tile(input_tensor_transposed_cb_index, left_tile_id, input_dest_start);
                        copy_tile(input_tensor_transposed_cb_index, right_tile_id, input_dest_end);

                        ckernel::topk_local_sort(0, (int)dir, 5);

                        tile_regs_commit();
                        tile_regs_wait();

                        pack_reconfig_data_format(input_tensor_transposed_cb_index);
                        pack_tile<true>(input_dest_start, input_tensor_transposed_cb_index, left_tile_id);
                        pack_tile<true>(input_dest_end, input_tensor_transposed_cb_index, right_tile_id);

                        pack_reconfig_data_format(index_tensor_transposed_cb_index);
                        pack_tile<true>(index_dest_start, index_tensor_transposed_cb_index, left_tile_id);
                        pack_tile<true>(index_dest_end, index_tensor_transposed_cb_index, right_tile_id);

                        tile_regs_release();
                        DPRINT << TERM_COMPUTE << "[Compute] i = " << i << TERM_RESET << ENDL();
                    }
                }  // Wt_per_core loop
            }  // sub loop
        }  // stage loop
        DPRINT << TERM_COMPUTE << "[Compute] completed local sort, synchronizing" << TERM_RESET << ENDL();

        // BUG: Deadlock with Writer

        // transposed_cb only have Wt tiles, which are already in use therefore
        // PACKER blocks until UNPACKER completes cb_pop_front()
        cb_reserve_back(input_tensor_transposed_cb_index, Wt_per_core);
        cb_reserve_back(index_tensor_transposed_cb_index, Wt_per_core);

        // DPRINT << TERM_COMPUTE << "[Compute] cb_reserve_back()" << TERM_RESET << ENDL();

        // UNPACKER pop Wt tiles => transposed_cb becomes empty
        // => PACKER resumes
        cb_pop_front(input_tensor_transposed_cb_index, Wt_per_core);
        cb_pop_front(index_tensor_transposed_cb_index, Wt_per_core);

        // DPRINT << TERM_COMPUTE << "[Compute] cb_pop_front()" << TERM_RESET << ENDL();

        // UNPACK will consecutively call wait_front in transpose_and_pack
        // Therefore, PACKER must liberate buffer
        cb_push_back(input_tensor_transposed_cb_index, Wt_per_core);
        cb_push_back(index_tensor_transposed_cb_index, Wt_per_core);

        // DPRINT << TERM_COMPUTE << "[Compute] cb_push_back()" << TERM_RESET << ENDL();

        // Syncrhonize with Writer: Writer can start reading input_tensor_transposed_cb_index
        cb_reserve_back(sync_cb_index, one_tile);
        cb_push_back(sync_cb_index, one_tile);

        // Second phase: use second circular buffer
        // 1. read 2 tiles from input_tensor_cb / input_other_cb
        // 2. run minmax on tensors

        const uint32_t TILE_INPUT0 = 0;
        const uint32_t TILE_INPUT1 = 1;

        // TODO: exchange indices
        // TODO: Since minmax does not guarantee a stable sort, we will have to figure something else

        // Current prototype:
        // - 2 cores, 1D tensor
        // - Sort + min

        binary_op_init_common(TILE_INPUT0, TILE_INPUT1, TILE_INPUT0);

        constexpr uint32_t FIRST_TILE = 0;

        DPRINT << TERM_COMPUTE << "[Compute] finished local sorting" << TERM_RESET << ENDL();

        constexpr bool select_min = true;
        // TODO: Replace value tensor with transposed_tensor
        // TODO: value tensor should only be used at the very end
        // Second phase:
        // 1) re-iterate through value_tensor_cb
        // 2) apply min/max on it with value_other_cb
        // 3) Re-built bitonic sequence
        for (uint32_t i = 0; i < Wt_per_core; i++) {
            DPRINT_UNPACK(
                DPRINT << TERM_COMPUTE << "[Compute] waiting for tiles on other_cb = " << value_tensor_other_cb_index
                       << TERM_RESET << ENDL());
            cb_wait_front(value_tensor_other_cb_index, one_tile);
            cb_wait_front(input_tensor_transposed_cb_index, one_tile);

            DPRINT_UNPACK(DPRINT << TERM_COMPUTE << "[Compute] copying tiles" << TERM_RESET << ENDL());
            copy_tile_to_dst_init_short(value_tensor_other_cb_index, one_tile);
            copy_tile(value_tensor_other_cb_index, FIRST_TILE, TILE_INPUT1);

            copy_tile_to_dst_init_short(input_tensor_transposed_cb_index);
            copy_tile(input_tensor_transposed_cb_index, FIRST_TILE, TILE_INPUT0);

            tile_regs_acquire();
            DPRINT_MATH(DPRINT << TERM_COMPUTE << "[Compute] math running min/max" << TERM_RESET << ENDL());

            if (select_min) {
                binary_min_tile_init();
                binary_min_tile(TILE_INPUT0, TILE_INPUT1);
            } else {
                binary_max_tile_init();
                binary_max_tile(TILE_INPUT0, TILE_INPUT1);
            }

            tile_regs_commit();

            tile_regs_wait();
            pack_tile<true>(TILE_INPUT0, input_tensor_transposed_cb_index, i);
            tile_regs_release();

            cb_pop_front(value_tensor_other_cb_index, one_tile);
            cb_pop_front(input_tensor_transposed_cb_index, one_tile);
        }
        DPRINT << TERM_COMPUTE << "[Compute] finished min/max" << TERM_RESET << ENDL();

        // Repeat local sort (bitonic sequence + merge)
        // This is a naive approach, we could probably do something better

        // Right now, input_tensor_transposed_cb_index is empty
        cb_reserve_back(input_tensor_transposed_cb_index, Wt_per_core);
        cb_push_back(input_tensor_transposed_cb_index, Wt_per_core);

        // cb_wait_front(input_tensor_transposed_cb_index, Wt_per_core);

        // for (uint32_t stage = 2; stage <= stages; stage++) {
        //     for (uint32_t sub = stage; sub > 0; sub--) {
        //         uint32_t sub_dist = 1 << (sub - 1);
        //         for (uint32_t i = 0; i < Wt_per_core; i++) {
        //             uint32_t j = i ^ sub_dist;
        //             if (j > i) {
        //                 // Determine direction for this comparison block
        //                 const bool ascending_block = ((i >> stage) & 1) == 0;
        //                 const bool dir = ascending_block == ascending;

        //                 // Get indexes of tiles to compare
        //                 const uint32_t left_tile_id = i;
        //                 const uint32_t right_tile_id = j;
        //                 /**
        //                  * Compute kernel for performing bitonic sort on tiles with synchronization caveats.
        //                  *
        //                  * Potential Bug: Unpacker and Packer Threads Synchronization Issue
        //                  *
        //                  * After migrating to the blackhole architecture, undefined behavior was observed, resulting
        //                  in
        //                  * incorrect results. The core of the issue lies in the synchronization between the unpacker
        //                  * (reading tiles from CB to registers) and packer (writing tiles from registers back to CB)
        //                  * threads.
        //                  *
        //                  * In the this loop, two tiles are read from a circular buffer (CB) into LLK registers, an
        //                  LLK
        //                  * operation is performed, and then the results are written back to the same CB. If there is
        //                  * insufficient synchronization between the packer and unpacker threads, it is possible that
        //                  the
        //                  * packer thread does not have enough time to fully pack the tiles from the registers back to
        //                  * the CB before the next loop iteration begins. As a result, the unpacker thread in the next
        //                  * iteration may read tiles into registers before the previous packing operation is complete,
        //                  * leading to data hazards and undefined behavior.
        //                  *
        //                  * Debugging revealed that inserting a delay between loop iterations resolved the issue,
        //                  * suggesting a race condition between packing and unpacking. However, since there is no
        //                  * semaphore or similar synchronization primitive available in the compute kernel, it is not
        //                  * possible to enforce proper synchronization programmatically.
        //                  *
        //                  * As a temporary workaround, swapping the order of read and write operations helped mitigate
        //                  * the issue, but this is not a robust or permanent solution. Proper synchronization between
        //                  * packer and unpacker threads is required to ensure data integrity and correct results.
        //                  *
        //                  * See also: https://github.com/tenstorrent/tt-metal/pull/22340
        //                  */
        //                 tile_regs_acquire();
        //                 copy_tile_to_dst_init_short_with_dt(
        //                     input_tensor_transposed_cb_index, index_tensor_transposed_cb_index);
        //                 copy_tile(index_tensor_transposed_cb_index, left_tile_id, index_dest_start);
        //                 copy_tile(index_tensor_transposed_cb_index, right_tile_id, index_dest_end);

        //                 copy_tile_to_dst_init_short_with_dt(
        //                     index_tensor_transposed_cb_index, input_tensor_transposed_cb_index);
        //                 copy_tile(input_tensor_transposed_cb_index, left_tile_id, input_dest_start);
        //                 copy_tile(input_tensor_transposed_cb_index, right_tile_id, input_dest_end);

        //                 ckernel::topk_local_sort(0, (int)dir, 5);

        //                 tile_regs_commit();
        //                 tile_regs_wait();

        //                 pack_reconfig_data_format(input_tensor_transposed_cb_index);
        //                 pack_tile<true>(input_dest_start, input_tensor_transposed_cb_index, left_tile_id);
        //                 pack_tile<true>(input_dest_end, input_tensor_transposed_cb_index, right_tile_id);

        //                 pack_reconfig_data_format(index_tensor_transposed_cb_index);
        //                 pack_tile<true>(index_dest_start, index_tensor_transposed_cb_index, left_tile_id);
        //                 pack_tile<true>(index_dest_end, index_tensor_transposed_cb_index, right_tile_id);

        //                 tile_regs_release();
        //                 DPRINT << TERM_COMPUTE << "[Compute] i = " << i << TERM_RESET << ENDL();
        //             }
        //         } // Wt_per_core loop
        //     } // sub loop
        // } // stage loop

        DPRINT << TERM_COMPUTE << "[Compute] finished second local sort" << TERM_RESET << ENDL();
        cb_reserve_back(input_tensor_transposed_cb_index, Wt_per_core);

        cb_pop_front(input_tensor_transposed_cb_index, Wt_per_core);

        cb_push_back(input_tensor_transposed_cb_index, Wt_per_core);

        // TODO: Same with index_tensor_transposed_cb_index (we did not send it for now)

        // Write everything into value tensor
        // Values tensor
        transpose_and_pack(input_tensor_transposed_cb_index, value_tensor_cb_index, Wt_per_core);

        // Indexes tensor
        transpose_and_pack(index_tensor_transposed_cb_index, index_tensor_output_cb_index, Wt_per_core);

    }  // Ht loop
    DPRINT << TERM_COMPUTE << "[Compute] completed" << TERM_RESET << ENDL();
}

}  // namespace NAMESPACE
