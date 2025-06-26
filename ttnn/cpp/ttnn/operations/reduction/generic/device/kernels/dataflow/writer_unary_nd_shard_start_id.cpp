// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include "accessor/sharded_accessor.h"

void kernel_main() {
    uint32_t bank_base_address = get_arg_val<uint32_t>(0);
    uint32_t num_tiles = get_arg_val<uint32_t>(1);
    uint32_t start_id = get_arg_val<uint32_t>(2);

    constexpr tt::CBIndex cb_id_out = static_cast<tt::CBIndex>(get_compile_time_arg_val(0));
    constexpr uint32_t tile_elements = get_compile_time_arg_val(1);
    constexpr uint32_t page_size = get_compile_time_arg_val(2);
    constexpr uint32_t arg_index = 3;
    auto sharding_args = nd_sharding::make_args<arg_index, 3>();
    auto sharded_accessor = nd_sharding::make_sharded_accessor_from_args(sharding_args, bank_base_address, page_size);
    // single-tile ublocks
    constexpr uint32_t onetile = 1;
    const uint32_t tile_bytes = get_tile_size(cb_id_out);

    // auto start_addr = sharded_accessor.get_noc_addr(start_id);
    uint32_t end_id = start_id + num_tiles;
    for (uint32_t i = start_id; i < end_id; ++i) {
        cb_wait_front(cb_id_out, onetile);
        uint32_t l1_read_addr = get_read_ptr(cb_id_out);
        uint64_t curr_noc_addr = sharded_accessor.get_noc_addr(i);
        // uint64_t curr_noc_addr = start_addr + (i - start_id) * page_size;
        noc_async_write(l1_read_addr, curr_noc_addr, tile_bytes);
        noc_async_write_barrier();
        cb_pop_front(cb_id_out, onetile);
    }
}
