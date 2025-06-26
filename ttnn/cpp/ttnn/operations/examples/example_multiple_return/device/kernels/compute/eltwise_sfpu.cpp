// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "hostdevcommon/kernel_structs.h"
#include "compute_kernel_api.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "debug/dprint.h"
#include "debug/dprint_pages.h"
#include "debug/dprint_tensix.h"

uint32_t A = 12;
uint32_t B = 12;
uint32_t CX = 12;
uint32_t DX = 12;

constexpr std::uint32_t onetile = 1;

inline void add_nops_r(int n) {
    for (int i = 0; i < n; i++) {
        TTI_NOP;
    }
}

template <const int n>
inline void add_nops() {
    for (int i = 0; i < n; i++) {
        TTI_NOP;
    }
}

template <const int U, const int M, const int P>
inline void add_trisc_nops() {
    if constexpr (U) {
        UNPACK(add_nops<U>());
    }

    if constexpr (M) {
        MATH(add_nops<M>());
    }

    if constexpr (P) {
        PACK(add_nops<P>());
    }
}

namespace NAMESPACE {
void MAIN {
    // Circular Buffers
    constexpr auto cb_in = tt::CBIndex::c_0;
    constexpr auto cb_other = tt::CBIndex::c_1;
    constexpr auto cb_out = tt::CBIndex::c_2;

    // Variables
    constexpr uint32_t input_dst_reg = 0;

    binary_op_init_common(cb_in, cb_other, cb_out);

    tile_regs_acquire();
    cb_wait_front(cb_in, onetile);
    cb_wait_front(cb_other, onetile);

    UNPACK(tt::compute::common::print_full_tile(cb_out, 0, false);)

    {
        if (CX == 256 || DX == 256) {
            add_nops_r(A);
        } else {
            add_nops_r(B);
        }
    }

    add_trisc_nops<UNOPS, MNOPS, PNOPS>();
    dprint_tensix_dest_reg(0);

    mul_tiles_init(cb_in, cb_other);
    mul_tiles(cb_in, cb_other, 0, 0, 0);

    cb_pop_front(cb_in, onetile);
    cb_pop_front(cb_other, onetile);

    dprint_tensix_dest_reg(0);

    tile_regs_commit();

    tile_regs_wait();
    cb_reserve_back(cb_out, onetile);
    pack_tile(0, cb_out);
    cb_push_back(cb_out, onetile);
    tile_regs_release();

    cb_wait_front(cb_out, 1);

    UNPACK(tt::compute::common::print_full_tile(cb_out, 0, false);)
}
}  // namespace NAMESPACE
