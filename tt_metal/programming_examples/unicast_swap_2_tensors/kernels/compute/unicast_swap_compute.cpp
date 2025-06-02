// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "compute_kernel_api.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/reconfig_data_format.h"
#include "compute_kernel_api/pack.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/binary_max_min.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"

#include "debug/dprint.h"
#include "tt_metal/hw/inc/debug/dprint_tensix.h"

namespace NAMESPACE {
void MAIN {
    constexpr uint32_t input0_cb_index = get_compile_time_arg_val(0);
    constexpr uint32_t input1_cb_index = get_compile_time_arg_val(1);
    constexpr uint32_t output_cb_index = get_compile_time_arg_val(2);
    constexpr uint32_t num_tiles = get_compile_time_arg_val(3);
    constexpr bool select_min = (get_compile_time_arg_val(4) == 1);

    constexpr uint32_t ONE_TILE = 1;
    uint32_t first_tile = 0;

    constexpr uint32_t TILE_INPUT0 = 0;
    constexpr uint32_t TILE_INPUT1 = 1;

    binary_op_init_common(TILE_INPUT0, TILE_INPUT1, TILE_INPUT0);

    // Read two tiles and write one
    // Performs: oupput[i] = max(input0[i], input1[i])

    for (uint32_t i = 0; i < num_tiles; i++) {
        cb_wait_front(input0_cb_index, ONE_TILE);
        cb_wait_front(input1_cb_index, ONE_TILE);

        copy_tile_to_dst_init_short(input0_cb_index);
        copy_tile(input0_cb_index, first_tile, TILE_INPUT0);

        copy_tile_to_dst_init_short(input1_cb_index);
        copy_tile(input1_cb_index, first_tile, TILE_INPUT1);

        tile_regs_acquire();

        if (select_min) {
            binary_min_tile_init();
            binary_min_tile(TILE_INPUT0, TILE_INPUT1);
        } else {
            binary_max_tile_init();
            binary_max_tile(TILE_INPUT0, TILE_INPUT1);
        }

        tile_regs_commit();

        cb_pop_front(input1_cb_index, ONE_TILE);
        cb_pop_front(input0_cb_index, ONE_TILE);

        // Write TILE_INPUT0 to output buffer
        tile_regs_wait();
        cb_reserve_back(output_cb_index, ONE_TILE);
        pack_tile(TILE_INPUT0, output_cb_index);
        cb_push_back(output_cb_index, ONE_TILE);

        tile_regs_release();
    }
}
}  // namespace NAMESPACE
