// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "dataflow_api.h"
#include "debug/dprint.h"

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
FORCE_INLINE void send_handshake_signal(uint32_t debug_addr) {
    constexpr uint32_t addr = STREAM_REG_ADDR(stream_id, STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_UPDATE_REG_INDEX);
    internal_::eth_write_remote_reg(0, addr, 1 << REMOTE_DEST_BUF_WORDS_FREE_INC);
    // volatile tt_l1_ptr uint32_t* debug_addr_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(debug_addr);
    // debug_addr_ptr[0] = 0x39;
    // internal_::eth_send_packet<false>(0, debug_addr >> 4, debug_addr >> 4, 16 >> 4);
    // while (eth_txq_is_busy());
    // debug_addr_ptr[0] = 0x0;
}

void kernel_main() {
    uint32_t initial_state = get_arg_val<uint32_t>(0);
    uint32_t src_binary_address = get_arg_val<uint32_t>(1);  // only used by eths that set up kernels
    uint32_t dst_binary_address = get_arg_val<uint32_t>(2);  // only used by eths that set up kernels
    uint32_t binary_size_bytes = get_arg_val<uint32_t>(3);   // only used by eths that set up kernels

    constexpr uint32_t local_state_stream_id = 0;
    constexpr uint32_t remote_handshake_stream_id = 1;  // neighbour eth core will write here
    /*
        I-am-mmio-eth-setting-up-remote: 0
        I-am-remote-eth-setting-up-local-eths: 1
        I-am-remote-eth-that-got-setup-by-a-local: 2
        I-am-waiting-on-doing-local-handshake: 3
        I-am-waiting-on-doing-remote-handshake: 4
        I-am-done-with-handshake: 5 // THIS NEEDS TO BE IN A SEPARATE REGISTER BECAUSE LOCAL HANDSHAKE AND REMOTE
       HANDSHAKE CAN HAPPEN IN PARALLEL
    */

    uint32_t launch_msg_addr = (uint32_t)&(((mailboxes_t*)MEM_AERISC_MAILBOX_BASE)->launch);
    uint32_t launch_and_go_msg_size_bytes = (sizeof(launch_msg_t) * launch_msg_buffer_num_entries) + sizeof(go_msg_t);
    uint32_t go_msg_addr = launch_msg_addr + (sizeof(launch_msg_t) * launch_msg_buffer_num_entries);

    // use a stream register to set the state (to allow other cores to change its state)- maybe it comes in as a runtime
    // arg?
    set_state<local_state_stream_id>(initial_state);
    set_state<remote_handshake_stream_id>(0);  // maybe rename set_state

    int i = 0;
    uint32_t state = get_state<local_state_stream_id>();
    while ((state = get_state<local_state_stream_id>()) != 5) {
        invalidate_l1_cache();
        // if (i < 5) {
        //     // DPRINT << "My current state is: " << state << ENDL();
        //     i++;
        // }
        switch (state) {
            case 0: {
                // clobber the initial_state arg with the state that remote eth core should come up in
                // first send the rt args to the remote core
                rta_l1_base[0] = 4;  // update this to be 1 later
                uint32_t rt_arg_base_addr = get_arg_addr(0);
                internal_::eth_send_packet<false>(0, rt_arg_base_addr >> 4, rt_arg_base_addr >> 4, 16 >> 4);

                DPRINT << "Sent runtime args to " << HEX() << rt_arg_base_addr << DEC() << ENDL();

                // send the kernel binary
                internal_::eth_send_packet<false>(
                    0, dst_binary_address >> 4, dst_binary_address >> 4, binary_size_bytes >> 4);

                DPRINT << "Sent binary to " << HEX() << dst_binary_address << DEC() << ENDL();

                // send launch, don't send go msg here because + go msg ... same as what was written on this core
                internal_::eth_send_packet<false>(
                    0, launch_msg_addr >> 4, launch_msg_addr >> 4, launch_and_go_msg_size_bytes >> 4);

                DPRINT << "Sent launch/go msg to 0x" << HEX() << launch_msg_addr << DEC() << " of size "
                       << launch_and_go_msg_size_bytes << " go msg addr " << HEX()
                       << (launch_msg_addr + (sizeof(launch_msg_t) * launch_msg_buffer_num_entries)) << DEC() << ENDL();

                // now we should go into local handshake state (skip for now) and then go into remote handshake state
                set_state<local_state_stream_id>(4);
                break;
            }
            case 4: {
                // we can only come into this state if our local handshakes are done

                // continue writing go msg now (can't send just 4 bytes)
                internal_::eth_send_packet<false>(0, go_msg_addr >> 4, go_msg_addr >> 4, 16 >> 4);

                // if we just write it once then we will technically continue writing until we change state..?
                send_handshake_signal<remote_handshake_stream_id>(src_binary_address + binary_size_bytes);
                // volatile tt_l1_ptr uint32_t* debug_addr_ptr = reinterpret_cast<volatile tt_l1_ptr
                // uint32_t*>(src_binary_address + binary_size_bytes);

                if (get_state<remote_handshake_stream_id>() == 1) {
                    set_state<local_state_stream_id>(5);
                }

                break;
            }
            default: ASSERT(false);
        }
        i++;
    }
}
