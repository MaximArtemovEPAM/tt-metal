// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"

#include "debug/dprint.h"

void print_tile_row0(uint32_t tile_addr) {
    DPRINT << "tile = [";
    const float* ptr = (float*)tile_addr;
    for (int i = 0; i < 32; i++) {
        DPRINT << ptr[i] << " ";
    }
    DPRINT << "]";
}

void kernel_main() {
    uint32_t output_addr = get_arg_val<uint32_t>(0);

    const uint32_t input_cb_index = get_compile_time_arg_val(0);
    const uint32_t num_tiles = get_compile_time_arg_val(1);

    constexpr uint32_t ONE_TILE = 1;

    const uint32_t output_tile_bytes = get_tile_size(input_cb_index);
    const auto output_dataformat = get_dataformat(input_cb_index);

    const InterleavedAddrGenFast<false> l1_output_addrg = {
        .bank_base_address = output_addr, .page_size = output_tile_bytes, .data_format = output_dataformat};

    const InterleavedAddrGenFast<true> dram_output_addrg = {
        .bank_base_address = output_addr, .page_size = output_tile_bytes, .data_format = output_dataformat};

    DPRINT << "[Writer] starting" << ENDL();

    for (uint32_t i = 0; i < num_tiles; i++) {
        cb_wait_front(input_cb_index, ONE_TILE);

        uint32_t l1_read_addr = get_read_ptr(input_cb_index);

        DPRINT << "[Writer]" << ENDL();
        print_tile_row0(l1_read_addr);
        DPRINT << ENDL();

        noc_async_write_tile(i, dram_output_addrg, l1_read_addr);
        noc_async_write_barrier();

        cb_pop_front(input_cb_index, ONE_TILE);
    }

    DPRINT << "[Writer] finshed" << ENDL();
}
