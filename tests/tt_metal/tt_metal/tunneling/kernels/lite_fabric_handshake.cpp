// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "dataflow_api.h"
#include "debug/dprint.h"
#include "eth_chan_noc_mapping.h"

// enum SocketState : uint8_t {
//     IDLE = 0,
//     OPENING = 1,
//     ACTIVE = 2,
//     CLOSING = 3,
// };

template <uint32_t stream_id>
FORCE_INLINE void set_state(uint32_t state) {
    NOC_STREAM_WRITE_REG(stream_id, STREAM_REMOTE_DEST_BUF_SIZE_REG_INDEX, state);
}

template <uint32_t stream_id>
FORCE_INLINE int32_t get_state() {
    return (
        NOC_STREAM_READ_REG(stream_id, STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_REG_INDEX) &
        ((1 << REMOTE_DEST_WORDS_FREE_WIDTH) - 1));
}

template <uint32_t stream_id>
FORCE_INLINE void send_handshake_signal(uint32_t l1_handshake_addr) {
    constexpr uint32_t addr = STREAM_REG_ADDR(stream_id, STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_UPDATE_REG_INDEX);
    internal_::eth_write_remote_reg(0, addr, 1 << REMOTE_DEST_BUF_WORDS_FREE_INC);
    // volatile tt_l1_ptr uint32_t* handshake_addr_ptr = reinterpret_cast<volatile tt_l1_ptr
    // uint32_t*>(l1_handshake_addr); handshake_addr_ptr[0] = 0x39; internal_::eth_send_packet<false>(0,
    // l1_handshake_addr >> 4, l1_handshake_addr >> 4, 16 >> 4); while (eth_txq_is_busy()); handshake_addr_ptr[0] = 0x0;
}

void kernel_main() {
    uint32_t initial_state = get_arg_val<uint32_t>(0);
    uint32_t binary_address = get_arg_val<uint32_t>(1);     // only used by eths that set up kernels
    uint32_t binary_size_bytes = get_arg_val<uint32_t>(2);  // only used by eths that set up kernels
    bool multi_eth_cores_setup = get_arg_val<uint32_t>(3) == 1;
    uint32_t primary_eth_core_x = get_arg_val<uint32_t>(4);
    uint32_t primary_eth_core_y = get_arg_val<uint32_t>(5);
    uint32_t num_local_eths = get_arg_val<uint32_t>(6);
    uint32_t eth_chans_mask = get_arg_val<uint32_t>(6);

    constexpr uint32_t local_handshake_stream_id = 0;
    constexpr uint32_t remote_handshake_stream_id = 1;  // neighbour eth core will write here
    /*
        OLD
        I-am-mmio-eth-setting-up-remote: 0
        I-am-remote-eth-setting-up-local-eths: 1
        I-am-waiting-on-local-handshake: 3
        I-am-waiting-on-remote-handshake: 4
        I-am-done-with-handshake: 5 // THIS NEEDS TO BE IN A SEPARATE REGISTER BECAUSE LOCAL HANDSHAKE AND REMOTE
       HANDSHAKE CAN HAPPEN IN PARALLEL


        NEW - simple case where we only init one eth device
        I-am-mmio-eth-setting-up-remote: 0
        I-am-doing-local-handshake: 1
        I-am-remote-eth-setting-up-local-eths: 2
        I-am-wating-on-neighbour-handshake: 3
        I-am-done-with-handshake: 4
    */

    uint32_t launch_msg_addr = (uint32_t)&(((mailboxes_t*)MEM_AERISC_MAILBOX_BASE)->launch);
    uint32_t launch_and_go_msg_size_bytes = (sizeof(launch_msg_t) * launch_msg_buffer_num_entries) + sizeof(go_msg_t);
    uint32_t go_msg_addr = launch_msg_addr + (sizeof(launch_msg_t) * launch_msg_buffer_num_entries);

    constexpr uint32_t total_num_eths = sizeof(eth_chan_to_noc_xy[0]) / sizeof(eth_chan_to_noc_xy[0][0]);

    set_state<local_handshake_stream_id>(0);
    set_state<remote_handshake_stream_id>(0);  // maybe rename set_state

    int i = 0;

    uint32_t state = initial_state;
    while (state != 4) {
        invalidate_l1_cache();
        // if (i < 5) {
        //     // DPRINT << "My current state is: " << state << ENDL();
        //     i++;
        // }
        switch (state) {
            case 0: {
                // clobber the initial_state arg with the state that remote eth core should come up in
                // first send the rt args to the remote core
                // set additional rt args in the kernel that are the x-y of this core
                rta_l1_base[0] = 2;
                uint32_t rt_arg_base_addr = get_arg_addr(0);
                internal_::eth_send_packet<false>(0, rt_arg_base_addr >> 4, rt_arg_base_addr >> 4, 32 >> 4);

                DPRINT << "Sent runtime args to " << HEX() << rt_arg_base_addr << DEC() << ENDL();

                // send the kernel binary
                internal_::eth_send_packet<false>(0, binary_address >> 4, binary_address >> 4, binary_size_bytes >> 4);

                DPRINT << "Sent binary to " << HEX() << binary_address << DEC() << ENDL();

                // send launch, don't send go msg here because + go msg ... same as what was written on this core
                internal_::eth_send_packet<false>(
                    0, launch_msg_addr >> 4, launch_msg_addr >> 4, launch_and_go_msg_size_bytes >> 4);

                DPRINT << "Sent launch/go msg to 0x" << HEX() << launch_msg_addr << DEC() << " of size "
                       << launch_and_go_msg_size_bytes << " go msg addr " << HEX()
                       << (launch_msg_addr + (sizeof(launch_msg_t) * launch_msg_buffer_num_entries)) << DEC() << ENDL();

                // now we should go into local handshake state (skip for now) and then go into remote handshake state
                uint32_t next_state = multi_eth_cores_setup ? 1 : 3;
                state = next_state;
                break;
            }
            case 1: {
                // go over the eth chan header and do case 0 for all cores using noc writes
                // set additional rt args in the kernel that are the x-y of this core
                uint32_t local_handshake_addr =
                    STREAM_REG_ADDR(local_handshake_stream_id, STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_REG_INDEX);
                volatile uint32_t* local_handshake_ptr = reinterpret_cast<volatile uint32_t*>(local_handshake_addr);

                if (initial_state == 0 or initial_state == 2) {
                    uint32_t remaining_cores = eth_chans_mask;
                    for (uint32_t i = 0; i < total_num_eths; i++) {
                        if (remaining_cores == 0) {
                            break;
                        }
                        if ((remaining_cores & (0x1 << i)) && (exclude_eth_chan != i)) {  // exclude_eth_chan is self
                            uint64_t dest_handshake_addr =
                                get_noc_addr_helper(eth_chan_to_noc_xy[noc_index][i], local_handshake_addr);
                            noc_inline_dw_write<true>(dest_handshake_addr, 1 << REMOTE_DEST_BUF_WORDS_FREE_INC);
                            if (initial_state == 2) {
                                uint64_t dest_go_msg_addr =
                                    get_noc_addr_helper(eth_chan_to_noc_xy[noc_index][i], go_msg_addr);
                                noc_async_write(go_msg_addr, dest_go_msg_addr, 16);
                            }
                            remaining_cores &= ~(0x1 << i);
                        }
                    }

                    while (*local_handshake_ptr != num_local_eths - 1) {
                        if (initial_state == 0) {
                            // continue writing go msg now (can't send just 4 bytes with this api)
                            internal_::eth_send_packet<false>(0, go_msg_addr >> 4, go_msg_addr >> 4, 16 >> 4);
                        } else if (initial_state == 2) {
                            // continue wriitng go msg  ????
                        }
                        // wait for the subordinate eth cores to send us a handshake signal
                        invalidate_l1_cache();
                    }

                } else {
                    noc_inline_dw_write<true>(
                        get_noc_addr(primary_eth_core_x, primary_eth_core_y, local_handshake_addr),
                        1 << REMOTE_DEST_BUF_WORDS_FREE_INC);
                    while (*local_handshake_ptr != 1) {
                        // wait for the primary eth core
                        invalidate_l1_cache();
                    }
                }

                state = 3;
                break;
            }
            case 2: {
                rta_l1_base[0] = 1;
                uint32_t remaining_cores = eth_chans_mask;
                for (uint32_t i = 0; i < total_num_eths; i++) {
                    if (remaining_cores == 0) {
                        break;
                    }
                    if ((remaining_cores & (0x1 << i)) && (exclude_eth_chan != i)) {  // exclude_eth_chan is self
                        uint64_t dest_rt_args_addr =
                            get_noc_addr_helper(eth_chan_to_noc_xy[noc_index][i], rt_arg_base_addr);
                        uint64_t dest_binary_addr =
                            get_noc_addr_helper(eth_chan_to_noc_xy[noc_index][i], binary_address);
                        uint64_t dest_launch_and_go_addr =
                            get_noc_addr_helper(eth_chan_to_noc_xy[noc_index][i], launch_msg_addr);
                        noc_async_write(rt_arg_base_addr, dest_rt_args_addr, 32);
                        noc_async_write(binary_address, dest_binary_addr, binary_size_bytes);
                        noc_async_write(launch_msg_addr, dest_launch_and_go_addr, launch_and_go_msg_size_bytes);
                        remaining_cores &= ~(0x1 << i);
                    }
                }
                noc_async_write_barrier();

                state = 1;
                break;
            }
            case 3: {
                // we can only come into this state if our local handshakes are done or we are initializing one core

                // continue writing go msg now (can't send just 4 bytes with this api)
                internal_::eth_send_packet<false>(0, go_msg_addr >> 4, go_msg_addr >> 4, 16 >> 4);

                // if we just write it once then we will technically continue writing until we change state..?
                send_handshake_signal<remote_handshake_stream_id>(0);
                // volatile tt_l1_ptr uint32_t* handshake_addr_ptr = reinterpret_cast<volatile tt_l1_ptr
                // uint32_t*>(neighbour_handshake_addr); if (handshake_addr_ptr[0] == 0x39) {
                //     // this means that the remote eth core has sent us a handshake signal
                //     // we can now set our state to 5
                //     state = 5;
                // }

                if (get_state<remote_handshake_stream_id>() == 1) {
                    state = 4;
                }

                break;
            }
            default: ASSERT(false);
        }
        i++;
    }
}
