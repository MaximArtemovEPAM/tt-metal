// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "dataflow_api.h"
#include "debug/dprint.h"

// Zero out all tiles for a given circular buffer.
template <uint32_t cb_id>
FORCE_INLINE void zero_out_tiles() {
    constexpr uint32_t tile_size = get_tile_size(cb_id);
    static_assert(
        tile_size % MEM_ZEROS_SIZE == 0, "Tile size must be a multiple of MEM_ZEROS_BASE for zeroing out tiles");
    const uint32_t num_tiles = get_local_cb_interface(cb_id).fifo_num_pages;
    const uint32_t num_zeros_reads = (tile_size / MEM_ZEROS_SIZE) * num_tiles;
    uint64_t zeros_noc_addr = get_noc_addr(MEM_ZEROS_BASE);
    uint32_t write_addr = get_write_ptr(cb_id);

    noc_async_read_one_packet_set_state(zeros_noc_addr, MEM_ZEROS_SIZE);
    for (uint32_t i = 0; i < num_zeros_reads; ++i) {
        noc_async_read_one_packet_with_state(zeros_noc_addr, write_addr);
        write_addr += MEM_ZEROS_SIZE;
    }
    noc_async_write_barrier();
}

template <
    uint32_t dilation_w,
    uint32_t coalesced_read_bytes,
    uint32_t conv_act_c_read_bytes,
    uint32_t act_block_w_extra_align_bytes,
    uint32_t stride_w_bytes,
    uint32_t stride_w,
    uint32_t stride_h_bytes,
    uint32_t weight_size_w,
    uint32_t window_outer,
    uint32_t w_tiles>
FORCE_INLINE void read_sticks(
    volatile tt_l1_ptr uint32_t* packed_reader_indices_ptr,
    uint32_t reader_offset,
    uint32_t& l1_write_addr_act,
    uint32_t& reader_idx,
    bool first_write,
    uint32_t cb_start_addr,
    uint32_t first_in_block_h,
    uint32_t out_cb_id) {
    // uint16_t num_elems = packed_reader_indices_ptr[reader_idx] & 0xffff;

    // while (num_elems--) {
    if (first_in_block_h) {
        reader_idx++;
    }

        uint16_t start_ind = packed_reader_indices_ptr[reader_idx] & 0xffff;
        uint16_t end_ind = packed_reader_indices_ptr[reader_idx] >> 16;

        if constexpr (dilation_w == 1) {
            uint32_t counter = 0;
            for (uint16_t ind = start_ind; ind <= end_ind; ind += stride_w) {
                uint32_t act_l1_offset = reader_offset + (ind * conv_act_c_read_bytes);
                uint32_t outer = first_write ? 0 : window_outer - 1;
                l1_write_addr_act += outer * coalesced_read_bytes;
                act_l1_offset += outer * stride_h_bytes;

                for (uint32_t i = outer; i < window_outer; i++) {
                    noc_async_read_one_packet_with_state<true>(act_l1_offset, l1_write_addr_act);
                    l1_write_addr_act += coalesced_read_bytes;
                    act_l1_offset += stride_h_bytes;
                }

                l1_write_addr_act += act_block_w_extra_align_bytes;

                counter++;
                // TODO(sjovic): use constant for 32
                if (counter % 32 == 0) {
                    cb_push_back(out_cb_id, w_tiles);
                }
            }
        } else {
            for (uint16_t ind = start_ind; ind <= end_ind; ind += stride_w) {
                uint32_t act_l1_offset = reader_offset + (ind * conv_act_c_read_bytes);
                for (uint32_t inner = 0; inner < weight_size_w; inner++) {
                    noc_async_read_one_packet_with_state<true>(act_l1_offset, l1_write_addr_act);
                    l1_write_addr_act += conv_act_c_read_bytes;
                    act_l1_offset += stride_w_bytes;
                }
                l1_write_addr_act += act_block_w_extra_align_bytes;
            }
        }
    // }
    reader_idx++;
}
