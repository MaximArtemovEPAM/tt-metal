// SPDX-FileCopyrightText: © 2025 Tenstorrent AI LLC
//
// SPDX-License-Identifier: Apache-2.0

using sem_ptr_t = volatile tt_l1_ptr uint32_t*;

constexpr uint32_t compute_core_id(uint32_t core_x, uint32_t core_y, uint32_t grid_size_x, uint32_t grid_size_y) {
    return core_x + core_y * grid_size_x;
}

/**
 * @brief Exchange 2 tiles between two cores using NOC
 *
 * This function implements a handshakes for exchanging tiles between two cores.
 * It uses semaphores to coordinate the exchange: data will be received at
 * `input_cb_write_addr` and will be sent to `input_other_noc_addr`
 *
 * @param this_core_id ID of the current core
 * @param other_core_id ID of the peer core to exchange with
 * @param sem_self_ptr Pointer to local semaphore in L1 memory
 * @param sem_input_other_noc_addr NOC address of peer's semaphore
 * @param input_cb_write_addr Local L1 address to write received data
 * @param input_other_noc_addr where to send data on the peer (NOC address)
 * @param input_tile_size Size of tile in bytes
 *
 * The exchange protocol:
 * 1. Uses total ordering (this_core_id < other_core_id) to avoid deadlocks
 * 2. For core with lower ID:
 *    - Waits for peer to be ready
 *    - Sends data
 *    - Signals peer that data is sent
 *    - Waits for peer to complete writing
 * 3. For core with higher ID:
 *    - Signals peer that it's ready
 *    - Waits for peer to send data
 *    - Receives and writes data
 *    - Signals peer that write is complete
 */
void sort_noc_exchange_tiles(
    uint32_t this_core_id,
    uint32_t other_core_id,
    sem_ptr_t sem_self_ptr,
    uint64_t sem_input_other_noc_addr,
    uint32_t input_cb_write_addr,
    uint64_t input_other_noc_addr,
    uint32_t input_tile_size) {
    if (this_core_id < other_core_id) {  // total order => avoid deadlocks
        // Write remote first / read then

        // Do not write until peer is ready => wait for him
        // We can't proceed before this because we don't know if it is safe to write into buffer
        noc_semaphore_wait(sem_self_ptr, 1);
        noc_semaphore_set(sem_self_ptr, 0);  // reset semaphore

        // Peer is ready => send data
        noc_async_write(input_cb_write_addr, input_other_noc_addr, input_tile_size);
        noc_async_write_barrier();

        // Data has been sent and written => we are ready to receive
        // Before this, we wake up other thread
        noc_semaphore_inc(sem_input_other_noc_addr, 1);

        // We need another semaphore to know when said data has been written
        // It will copy data into desired circular buffer => no need to to more
        noc_semaphore_wait(sem_self_ptr, 1);
        noc_semaphore_set(sem_self_ptr, 0);  // reset semaphore

    } else {
        // This section is similar to previous one, except that
        // 1) waiting and writing are reversed
        // 2)

        noc_semaphore_inc(sem_input_other_noc_addr, 1);

        // Note: other core does not need to notify this core that data has been written.
        // Because if next `noc_semaphore_wait()` passes, then that means other is ready to read
        // i.e. has written its data (write barrier ensures this behavior)

        // Wait for 1 reader
        noc_semaphore_wait(sem_self_ptr, 1);
        noc_semaphore_set(sem_self_ptr, 0);  // reset semaphore

        // Send data
        noc_async_write(input_cb_write_addr, input_other_noc_addr, input_tile_size);
        noc_async_write_barrier();

        // notify other that data has been written
        noc_semaphore_inc(sem_input_other_noc_addr, 1);
    }
}
