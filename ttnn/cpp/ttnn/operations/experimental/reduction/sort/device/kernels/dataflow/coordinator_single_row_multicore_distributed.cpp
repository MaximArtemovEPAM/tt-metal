// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <cstdint>
#include "dataflow_api.h"

#include "sort_distributed_common.hpp"
#include "../sort_debug_common.hpp"

void kernel_main() {
    const uint32_t core_loop_count = get_arg_val<uint32_t>(0);
    const uint32_t start_core_x = get_arg_val<uint32_t>(1);
    const uint32_t start_core_y = get_arg_val<uint32_t>(2);
    const uint32_t end_core_x = get_arg_val<uint32_t>(3);
    const uint32_t end_core_y = get_arg_val<uint32_t>(4);

    const uint32_t this_core_x = get_arg_val<uint32_t>(5);
    const uint32_t this_core_y = get_arg_val<uint32_t>(6);

    constexpr uint32_t intercore_stages = get_compile_time_arg_val(0);
    constexpr uint32_t num_cores_x = get_compile_time_arg_val(1);
    const uint32_t sem_barrier_addr = get_semaphore(get_compile_time_arg_val(2));
    const uint32_t sem_barrier_worker_addr = get_semaphore(get_compile_time_arg_val(3));

    sem_ptr_t sem_self_barrier_ptr = reinterpret_cast<sem_ptr_t>(sem_barrier_addr);
    sem_ptr_t sem_self_barrier_worker_ptr = reinterpret_cast<sem_ptr_t>(sem_barrier_worker_addr);

    const uint64_t sem_barrier_mcast_addr =
        get_noc_multicast_addr(start_core_x, start_core_y, end_core_x, end_core_y, sem_barrier_worker_addr);

    const uint64_t sem_noc_barrier_addr = get_noc_addr(this_core_x, this_core_y, sem_barrier_addr);

    DPRINT << TERM_BARRIER << "[Barrier] starting, with num_cores_x = " << num_cores_x << TERM_RESET << ENDL();
    DPRINT << TERM_BARRIER << "[Barrier] sem_barrier_addr = " << sem_barrier_addr
           << ", self global = " << sem_noc_barrier_addr << TERM_RESET << ENDL();
    DPRINT << TERM_BARRIER << "[Barrier] start core = {" << start_core_x << ", " << start_core_y << "}, end core = {"
           << end_core_x << ", " << end_core_y << "}, this core = {" << this_core_x << ", " << this_core_y << "}"
           << TERM_RESET << ENDL();

    for (uint32_t core_loop = 0; core_loop < core_loop_count; core_loop++) {
        for (uint32_t core_stage = 1; core_stage <= intercore_stages; core_stage++) {
            for (uint32_t sub = core_stage; sub > 0; sub--) {
                DPRINT << TERM_BARRIER << "[Barrier] waiting for " << num_cores_x << " cores, stage = " << core_stage
                       << ", sub = " << sub << TERM_RESET << ENDL();
                noc_semaphore_wait(sem_self_barrier_ptr, num_cores_x);
                noc_semaphore_set(sem_self_barrier_ptr, 0);

                DPRINT << TERM_BARRIER << "[Barrier] notifying cores, at " << HEX() << sem_barrier_worker_addr << " ("
                       << sem_barrier_mcast_addr << "), stage = " << DEC() << core_stage << ", sub = " << sub
                       << TERM_RESET << ENDL();

                for (uint32_t peer = start_core_x; peer < end_core_x; peer++) {
                    uint64_t sem_noc_worker_barrier_addr = get_noc_addr(peer, this_core_y, sem_barrier_worker_addr);
                    DPRINT << TERM_BARRIER << "[Barrier] notify core {" << peer << ", " << this_core_y << "} at "
                           << HEX() << sem_noc_worker_barrier_addr << DEC() << TERM_RESET << ENDL();
                    noc_semaphore_inc(sem_noc_worker_barrier_addr, 1);
                }

                // noc_semaphore_set_multicast(sem_barrier_worker_addr, sem_barrier_mcast_addr, num_cores_x);
                DPRINT << TERM_BARRIER << "[Barrier] notified cores, stage = " << core_stage << ", sub = " << sub
                       << TERM_RESET << ENDL();
            }
        }
    }

    DPRINT << TERM_BARRIER << "[Barrier] completed..." << TERM_RESET << ENDL();
}
