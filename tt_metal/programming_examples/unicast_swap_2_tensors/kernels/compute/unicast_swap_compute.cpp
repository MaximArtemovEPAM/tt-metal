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

    DPRINT_MATH(DPRINT << "[Math] starting" << ENDL());
    DPRINT_PACK(DPRINT << "[Pack] starting" << ENDL());
    DPRINT_UNPACK(DPRINT << "[Unpack] starting" << ENDL());

    constexpr uint32_t ONE_TILE = 1;
    uint32_t first_tile = 0;

    constexpr uint32_t TILE_INPUT0 = 0;
    constexpr uint32_t TILE_INPUT1 = 1;

    binary_op_init_common(TILE_INPUT0, TILE_INPUT1, TILE_INPUT0);

    cb_wait_front(input0_cb_index, ONE_TILE);
    cb_wait_front(input1_cb_index, ONE_TILE);

    copy_tile_to_dst_init_short(input0_cb_index);
    copy_tile(input0_cb_index, first_tile, TILE_INPUT0);

    copy_tile_to_dst_init_short(input1_cb_index);
    copy_tile(input1_cb_index, first_tile, TILE_INPUT1);

    DPRINT << "[Math], tile input0 = \n";
    dprint_tensix_dest_reg(TILE_INPUT0);
    DPRINT << "[Math], tile input1 = \n";
    dprint_tensix_dest_reg(TILE_INPUT1);

    tile_regs_acquire();
    // MATH(dprint_tensix_dest_reg(TILE_INPUT0););

    DPRINT_MATH(DPRINT << "[Math] waiting before binary_max_tile" << ENDL());

    binary_max_tile_init();
    binary_max_tile(TILE_INPUT0, TILE_INPUT1);
    DPRINT_MATH(DPRINT << "[Math] binary_max_tile done" << ENDL());

    tile_regs_commit();

    cb_pop_front(input1_cb_index, ONE_TILE);
    cb_pop_front(input0_cb_index, ONE_TILE);

    // Write TILE_INPUT0 to output buffer
    tile_regs_wait();
    cb_reserve_back(output_cb_index, ONE_TILE);
    pack_tile(TILE_INPUT1, output_cb_index);
    cb_push_back(output_cb_index, ONE_TILE);

    tile_regs_release();

    DPRINT_MATH(DPRINT << "[Math] finished" << ENDL());
    DPRINT_PACK(DPRINT << "[Pack] finished" << ENDL());
    DPRINT_UNPACK(DPRINT << "[Unpack] finished" << ENDL());
}
}  // namespace NAMESPACE
