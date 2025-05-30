// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"

#include "debug/dprint.h"

void kernel_main() {
    uint32_t output_addr = get_arg_val<uint32_t>(0);

    const uint32_t input_cb_index = get_compile_time_arg_val(0);

    constexpr uint32_t ONE_TILE = 1;

    const uint32_t output_tile_bytes = get_tile_size(input_cb_index);
    const auto output_dataformat = get_dataformat(input_cb_index);

    const InterleavedAddrGenFast<false> l1_output_addrg = {
        .bank_base_address = output_addr, .page_size = output_tile_bytes, .data_format = output_dataformat};

    constexpr uint32_t first_tile = 0;

    DPRINT << "[Writer] waiting for tile" << ENDL();
    cb_wait_front(input_cb_index, ONE_TILE);

    uint32_t l1_read_addr = get_read_ptr(input_cb_index);

    // Display tile
    // DPRINT << "[Writer] TILE = \n";
    // float* ptr = (float*)l1_read_addr;
    // for (uint32_t i = 0; i < output_tile_bytes / sizeof(float); i++) {
    //     DPRINT << " " << ptr[i];
    // }
    // DPRINT << ENDL();

    DPRINT << "[Writer] writing tile" << ENDL();
    noc_async_write_tile(first_tile, l1_output_addrg, l1_read_addr);
    noc_async_write_barrier();

    cb_pop_front(input_cb_index, ONE_TILE);

    DPRINT << "[Writer] finished" << ENDL();
}
