// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include <tt-metalium/buffer_types.hpp>
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_manager.hpp"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "cpp/ttnn/operations/ccl/kernel_common/worker_sync_utils.hpp"
#include "cpp/ttnn/operations/ccl/ccl_host_types.hpp"
#include "cpp/ttnn/operations/ccl/kernel_common/sharding_addrgen.hpp"
#include "minimal_ccl_common.hpp"
#include <cstdint>
#include <utility>

using address_t = uint32_t;
using tt::tt_metal::BufferType;
using ttnn::ccl::Topology;

///////////////////////////////////////////////////
// COMPILE TIME ARGS
///////////////////////////////////////////////////

constexpr uint32_t my_chip_id = get_compile_time_arg_val(0);
constexpr uint32_t reserved_packet_header_cb_id = get_compile_time_arg_val(1);
constexpr uint32_t num_packet_headers_storable = get_compile_time_arg_val(2);
constexpr BufferType output_type = static_cast<BufferType>(get_compile_time_arg_val(3));
constexpr uint32_t cb_output_id = get_compile_time_arg_val(4);
constexpr uint32_t num_tiles_to_write_per_packet = get_compile_time_arg_val(5);
constexpr uint32_t output_page_size = get_compile_time_arg_val(6);
constexpr uint32_t num_targets_forward_direction = get_compile_time_arg_val(7);
constexpr uint32_t num_targets_backward_direction = get_compile_time_arg_val(8);
constexpr bool dynamic_alternate = get_compile_time_arg_val(9);
constexpr bool fuse_op = get_compile_time_arg_val(10);
constexpr Topology topology = static_cast<Topology>(get_compile_time_arg_val(11));
constexpr bool direction = get_compile_time_arg_val(12);  // 1 is forward, 0 is backward

void kernel_main() {
    ///////////////////////////////////////////////////
    // ARGS
    ///////////////////////////////////////////////////
    uint32_t arg_idx = 0;
    address_t output_address = get_arg_val<address_t>(arg_idx++);
    uint32_t input_tensor_Wt = get_arg_val<uint32_t>(arg_idx++);
    uint32_t input_tensor_Ht = get_arg_val<uint32_t>(arg_idx++);
    uint32_t output_tensor_Wt = get_arg_val<uint32_t>(arg_idx++);
    uint32_t output_tensor_Ht = get_arg_val<uint32_t>(arg_idx++);
    uint32_t gather_dim = get_arg_val<uint32_t>(arg_idx++);
    uint32_t input_batch_head_count = get_arg_val<uint32_t>(arg_idx++);
    uint32_t input_tile_id_start = get_arg_val<uint32_t>(arg_idx++);
    uint32_t input_tile_id_end = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t out_ready_sem_noc0_x = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t out_ready_sem_noc0_y = get_arg_val<uint32_t>(arg_idx++);
    uint32_t ring_size = get_arg_val<uint32_t>(arg_idx++);
    size_t out_ready_sem = get_arg_val<uint32_t>(arg_idx++);
    uint32_t start_pages_read_in_row = get_arg_val<uint32_t>(arg_idx++);
    uint32_t start_row_offset = get_arg_val<uint32_t>(arg_idx++);
    bool use_barrier_sem = get_arg_val<uint32_t>(arg_idx++);
    size_t barrier_sem = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t barrier_sem_noc0_x = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t barrier_sem_noc0_y = get_arg_val<uint32_t>(arg_idx++);

#ifdef OUTPUT_IS_SHARDED
    using tensor_shard_info = ShardedInfo<
        get_compile_time_arg_val(13),   // Memory layout
        get_compile_time_arg_val(14),   // The number of sharding cores
        get_compile_time_arg_val(15),   // The page size we offset each write to
        get_compile_time_arg_val(16),   // The number of pages in each sharding row not including padding pages
        get_compile_time_arg_val(17),   // This defines times when contiguous pages can't be calculated
        get_compile_time_arg_val(18),   // pages_per_shard_x
        get_compile_time_arg_val(19)>;  // pages_per_shard_y

    const auto [mapping_table, rt_increment] =
        experimental::shard_addr_gen_utils::get_shard_map<tensor_shard_info>(get_arg_addr(arg_idx));
    experimental::ShardedAddrGen<tensor_shard_info> output_addrgen = {
        .bank_base_address = output_address, .shard_array = mapping_table};

    arg_idx += rt_increment;
#else
    constexpr bool output_is_dram = output_type == tt::tt_metal::BufferType::DRAM;
    const InterleavedAddrGenFast<output_is_dram> output_addrgen = {
        .bank_base_address = output_address,
        .page_size = output_page_size,
        .data_format = get_dataformat(cb_output_id)};
#endif

    size_t arg_for_fab = arg_idx;
    auto fabric_connection = FabricConnectionManager::build_from_args(arg_for_fab);
    /* Args for overlapped all gather */
    OpSignaler op_signaler_sender;

    if constexpr (fuse_op) {
        arg_idx = arg_for_fab;
        op_signaler_sender = OpSignaler(arg_idx);
    }
    // packet header cb
    cb_reserve_back(reserved_packet_header_cb_id, 1);
    auto packet_header_buffer_addr = get_write_ptr(reserved_packet_header_cb_id);
    cb_push_back(reserved_packet_header_cb_id, 1);
    cb_reserve_back(reserved_packet_header_cb_id, 1);
    auto packet_header_buffer_seminc = get_write_ptr(reserved_packet_header_cb_id);
    cb_push_back(reserved_packet_header_cb_id, 1);
    // pre-populate packet headers
    volatile PACKET_HEADER_TYPE* pkt_hdr = reinterpret_cast<volatile PACKET_HEADER_TYPE*>(packet_header_buffer_addr);
    pkt_hdr->to_chip_unicast(1);

    volatile PACKET_HEADER_TYPE* pkt_hdr_sem_inc = reinterpret_cast<PACKET_HEADER_TYPE*>(packet_header_buffer_seminc);

    fabric_connection.open();

    // Due to the existing direction of fabric connections, forward writers will signal to backward writers
    // and backward writers will signal to forward writers
    bool signal_on_barrier_semaphore = use_barrier_sem && ((direction == 1 && num_targets_backward_direction) ||
                                                           (direction == 0 && num_targets_forward_direction));
    bool wait_on_barrier_semaphore = use_barrier_sem && ((direction == 1 && num_targets_backward_direction) ||
                                                         (direction == 0 && num_targets_forward_direction));
    if (signal_on_barrier_semaphore) {
        uint64_t sync_sem_noc_addr_in_pkt = safe_get_noc_addr(barrier_sem_noc0_x, barrier_sem_noc0_y, barrier_sem, 0);
        pkt_hdr_sem_inc->to_noc_unicast_atomic_inc(
            tt::tt_fabric::NocUnicastAtomicIncCommandHeader{sync_sem_noc_addr_in_pkt, static_cast<uint16_t>(1), 32});
        if (direction == 1) {
            fabric_connection.get_backward_connection().wait_for_empty_write_slot();
            pkt_hdr_sem_inc->to_chip_unicast(1);
            fabric_connection.get_backward_connection().send_payload_flush_blocking_from_address(
                packet_header_buffer_seminc, sizeof(PACKET_HEADER_TYPE));
        } else {
            fabric_connection.get_forward_connection().wait_for_empty_write_slot();
            pkt_hdr_sem_inc->to_chip_unicast(1);
            fabric_connection.get_forward_connection().send_payload_flush_blocking_from_address(
                packet_header_buffer_seminc, sizeof(PACKET_HEADER_TYPE));
        }
    }
    if (wait_on_barrier_semaphore) {
        noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(barrier_sem), 1);
        noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(barrier_sem), 0);
    }

    uint32_t slice_writes = 0;

    // Write out the local slice to both DRAM and forward and backward
    uint32_t pages_read_in_row = start_pages_read_in_row;
    uint32_t row_offset = start_row_offset;
    uint32_t tiles_read = input_tile_id_start;
    uint32_t tiles_to_read = input_tile_id_end;
    uint32_t tile_id_start = my_chip_id * input_tensor_Wt;

    if (gather_dim == 3) {
        tile_id_start = my_chip_id * input_tensor_Wt;
    } else {
        tile_id_start = my_chip_id * input_tensor_Ht * input_tensor_Wt;
    }
    for (uint32_t bh_idx = 0; bh_idx < input_batch_head_count; bh_idx++) {
        while (tiles_read < tiles_to_read) {
            uint32_t tiles_remaining_to_read = tiles_to_read - tiles_read;
            uint32_t tiles_to_put_in_current_packet = std::min(tiles_remaining_to_read, num_tiles_to_write_per_packet);

            cb_wait_front(cb_output_id, num_tiles_to_write_per_packet);
            size_t l1_read_addr = get_read_ptr(cb_output_id);

            // Will have more cases once scatter-write supports more than 2 distinct addresses
            switch (tiles_to_put_in_current_packet) {
#ifdef ARCH_WORMHOLE
                case 2: {
                    uint32_t tile_one_id = tile_id_start + row_offset + pages_read_in_row;
                    pages_read_in_row++;
                    if (pages_read_in_row >= input_tensor_Wt) {
                        row_offset += output_tensor_Wt;
                        pages_read_in_row = 0;
                    }

                    uint32_t tile_two_id = tile_id_start + row_offset + pages_read_in_row;
                    pages_read_in_row++;
                    if (pages_read_in_row >= input_tensor_Wt) {
                        row_offset += output_tensor_Wt;
                        pages_read_in_row = 0;
                    }

                    uint64_t remote_noc0_dest_noc_addr_tile_one =
                        get_noc_addr(tile_one_id, output_addrgen, 0 /*offset*/, 0 /*noc_id*/);
                    uint64_t remote_noc0_dest_noc_addr_tile_two =
                        get_noc_addr(tile_two_id, output_addrgen, 0 /*offset*/, 0 /*noc_id*/);

                    if (direction == 1) {
                        if (num_targets_backward_direction) {
                            scatter_write_for_fabric_write_backward(
                                remote_noc0_dest_noc_addr_tile_one,
                                remote_noc0_dest_noc_addr_tile_two,
                                pkt_hdr,
                                fabric_connection,
                                l1_read_addr,
                                output_page_size,
                                output_page_size);
                        }
                        uint64_t local_noc0_dest_noc_addr_tile_one = get_noc_addr(tile_one_id, output_addrgen);
                        uint64_t local_noc0_dest_noc_addr_tile_two = get_noc_addr(tile_two_id, output_addrgen);

                        noc_async_write(l1_read_addr, local_noc0_dest_noc_addr_tile_one, output_page_size);
                        noc_async_write(
                            l1_read_addr + output_page_size, local_noc0_dest_noc_addr_tile_two, output_page_size);
                        noc_async_write_barrier();
                    } else {
                        if (num_targets_forward_direction) {
                            scatter_write_for_fabric_write_forward(
                                remote_noc0_dest_noc_addr_tile_one,
                                remote_noc0_dest_noc_addr_tile_two,
                                pkt_hdr,
                                fabric_connection,
                                l1_read_addr,
                                output_page_size,
                                output_page_size);
                        }
                    }
                    tiles_read += 2;
                    break;
                }
#endif
                case 1:
                default: {
                    uint32_t tile_id = tile_id_start + row_offset + pages_read_in_row;
                    pages_read_in_row++;
                    if (pages_read_in_row >= input_tensor_Wt) {
                        row_offset += output_tensor_Wt;
                        pages_read_in_row = 0;
                    }

                    uint64_t remote_noc0_dest_noc_addr =
                        get_noc_addr(tile_id, output_addrgen, 0 /*offset*/, 0 /*noc_id*/);
                    if (direction == 1) {
                        if (num_targets_backward_direction) {
                            write_for_fabric_write_backward(
                                remote_noc0_dest_noc_addr, pkt_hdr, fabric_connection, l1_read_addr, output_page_size);
                        }
                        uint64_t local_noc0_dest_noc_addr = get_noc_addr(tile_id, output_addrgen);
                        noc_async_write(l1_read_addr, local_noc0_dest_noc_addr, output_page_size);
                        noc_async_write_barrier();
                    } else {
                        if (num_targets_forward_direction) {
                            write_for_fabric_write_forward(
                                remote_noc0_dest_noc_addr, pkt_hdr, fabric_connection, l1_read_addr, output_page_size);
                        }
                    }
                    tiles_read++;
                    break;
                }
            }

            cb_pop_front(cb_output_id, num_tiles_to_write_per_packet);
        }

        tile_id_start += output_tensor_Wt * output_tensor_Ht;
        tiles_read = input_tile_id_start;
        tiles_to_read = input_tile_id_end;
        pages_read_in_row = start_pages_read_in_row;
        row_offset = start_row_offset;
    }

    // 2. unicast output ready semaphore
    uint64_t out_ready_sem_noc_addr_in_pkt =
        safe_get_noc_addr(out_ready_sem_noc0_x, out_ready_sem_noc0_y, out_ready_sem, 0);
    pkt_hdr_sem_inc->to_noc_unicast_atomic_inc(tt::tt_fabric::NocUnicastAtomicIncCommandHeader{
        out_ready_sem_noc_addr_in_pkt,
        static_cast<uint16_t>(1),  // increment 1
        32});
    // Write the unicast packet
    if (direction == 1) {
        if (num_targets_backward_direction) {
            fabric_connection.get_backward_connection().wait_for_empty_write_slot();
            pkt_hdr_sem_inc->to_chip_unicast(1);
            fabric_connection.get_backward_connection().send_payload_flush_blocking_from_address(
                packet_header_buffer_seminc, sizeof(PACKET_HEADER_TYPE));
        }
    } else {
        if (num_targets_forward_direction) {
            fabric_connection.get_forward_connection().wait_for_empty_write_slot();
            pkt_hdr_sem_inc->to_chip_unicast(1);
            fabric_connection.get_forward_connection().send_payload_flush_blocking_from_address(
                packet_header_buffer_seminc, sizeof(PACKET_HEADER_TYPE));
        }
    }

    // increment locally
    if (fuse_op && direction == 1) {
        // Synchronize and signal that the local tensor slice is available
        op_signaler_sender.synchronize_workers_and_signal_op(my_chip_id);
    }

    uint32_t writes_expected = 0;
    if (topology == Topology::Linear) {
        if (direction == 1 && num_targets_backward_direction) {
            writes_expected = num_targets_forward_direction;
        } else if (direction == 0 && num_targets_forward_direction) {
            writes_expected = num_targets_backward_direction;
        }
    } else if (topology == Topology::Ring) {
        if (direction == 1) {
            writes_expected = num_targets_backward_direction - 1;
        } else {
            writes_expected = num_targets_forward_direction - 1;
        }
    }

    while (slice_writes < writes_expected) {
        // Direction == backward
        // Did I get something from my left to send to my right?
        // In the linear case, I expect num_targets_backward_direction slices from the left, and check if I have a
        // neighbor to the right
        // In the ring case, I expect to write to the right num_forward_target times
        // Direction == forward
        // Did I get something from my right to send to my left?
        // In the linear case, I expect num_targets_forward_direction slices from the right, and check if I have a
        // neighbor to the left
        // In the ring case, I expect to write to the left num_backward_target times
        int slice_chip_id;
        uint32_t actual_slice_chip_id;
        if (direction == 1) {
            slice_chip_id = my_chip_id + slice_writes + 1;
            actual_slice_chip_id = (slice_chip_id >= (int)ring_size) ? slice_chip_id - ring_size : slice_chip_id;
        } else {
            slice_chip_id = my_chip_id - slice_writes - 1;
            actual_slice_chip_id = (slice_chip_id < 0) ? ring_size + slice_chip_id : slice_chip_id;
        }
        uint32_t tiles_read = input_tile_id_start;
        uint32_t tiles_to_read = input_tile_id_end;
        uint32_t tile_id_start = actual_slice_chip_id * input_tensor_Wt;
        uint32_t row_offset = start_row_offset;
        uint32_t pages_read_in_row = start_pages_read_in_row;
        uint32_t slice_Wt = input_tensor_Wt;
        uint32_t stride_Wt = output_tensor_Wt;

        if (gather_dim == 3) {
            tile_id_start = actual_slice_chip_id * input_tensor_Wt;
        } else {
            tile_id_start = actual_slice_chip_id * input_tensor_Ht * input_tensor_Wt;
        }
        for (uint32_t bh_idx = 0; bh_idx < input_batch_head_count; bh_idx++) {
            while (tiles_read < tiles_to_read) {
                uint32_t tiles_remaining_to_read = tiles_to_read - tiles_read;
                uint32_t tiles_to_put_in_current_packet =
                    std::min(tiles_remaining_to_read, num_tiles_to_write_per_packet);

                cb_wait_front(cb_output_id, num_tiles_to_write_per_packet);
                size_t l1_read_addr = get_read_ptr(cb_output_id);

                // Will have more cases once scatter-write supports more than 2 distinct addresses
                switch (tiles_to_put_in_current_packet) {
#ifdef ARCH_WORMHOLE
                    case 2: {
                        uint32_t tile_one_id = tile_id_start + row_offset + pages_read_in_row;
                        pages_read_in_row++;
                        if (pages_read_in_row >= slice_Wt) {
                            row_offset += stride_Wt;
                            pages_read_in_row = 0;
                        }

                        uint32_t tile_two_id = tile_id_start + row_offset + pages_read_in_row;
                        pages_read_in_row++;
                        if (pages_read_in_row >= slice_Wt) {
                            row_offset += stride_Wt;
                            pages_read_in_row = 0;
                        }

                        uint64_t remote_noc0_dest_noc_addr_tile_one =
                            get_noc_addr(tile_one_id, output_addrgen, 0 /*offset*/, 0 /*noc_id*/);
                        uint64_t remote_noc0_dest_noc_addr_tile_two =
                            get_noc_addr(tile_two_id, output_addrgen, 0 /*offset*/, 0 /*noc_id*/);

                        if (direction == 1) {
                            scatter_write_for_fabric_write_backward(
                                remote_noc0_dest_noc_addr_tile_one,
                                remote_noc0_dest_noc_addr_tile_two,
                                pkt_hdr,
                                fabric_connection,
                                l1_read_addr,
                                output_page_size,
                                output_page_size);
                        } else {
                            scatter_write_for_fabric_write_forward(
                                remote_noc0_dest_noc_addr_tile_one,
                                remote_noc0_dest_noc_addr_tile_two,
                                pkt_hdr,
                                fabric_connection,
                                l1_read_addr,
                                output_page_size,
                                output_page_size);
                        }
                        tiles_read += 2;
                        break;
                    }
#endif
                    case 1:
                    default: {
                        uint32_t tile_id = tile_id_start + row_offset + pages_read_in_row;
                        pages_read_in_row++;
                        if (pages_read_in_row >= input_tensor_Wt) {
                            row_offset += output_tensor_Wt;
                            pages_read_in_row = 0;
                        }

                        uint64_t remote_noc0_dest_noc_addr =
                            get_noc_addr(tile_id, output_addrgen, 0 /*offset*/, 0 /*noc_id*/);

                        if (direction == 1) {
                            write_for_fabric_write_backward(
                                remote_noc0_dest_noc_addr, pkt_hdr, fabric_connection, l1_read_addr, output_page_size);
                        } else {
                            write_for_fabric_write_forward(
                                remote_noc0_dest_noc_addr, pkt_hdr, fabric_connection, l1_read_addr, output_page_size);
                        }
                        tiles_read++;
                        break;
                    }
                }

                cb_pop_front(cb_output_id, num_tiles_to_write_per_packet);
            }

            tile_id_start += output_tensor_Wt * output_tensor_Ht;
            tiles_read = input_tile_id_start;
            tiles_to_read = input_tile_id_end;
            row_offset = start_row_offset;
            pages_read_in_row = start_pages_read_in_row;
        }
        // 2. unicast output ready semaphore
        if (direction == 1) {
            fabric_connection.get_backward_connection().wait_for_empty_write_slot();
            pkt_hdr_sem_inc->to_chip_unicast(1);
            fabric_connection.get_backward_connection().send_payload_flush_blocking_from_address(
                packet_header_buffer_seminc, sizeof(PACKET_HEADER_TYPE));
        } else {
            fabric_connection.get_forward_connection().wait_for_empty_write_slot();
            pkt_hdr_sem_inc->to_chip_unicast(1);
            fabric_connection.get_forward_connection().send_payload_flush_blocking_from_address(
                packet_header_buffer_seminc, sizeof(PACKET_HEADER_TYPE));
        }
        slice_writes++;
    }

    fabric_connection.close();

    noc_async_write_barrier();
}
