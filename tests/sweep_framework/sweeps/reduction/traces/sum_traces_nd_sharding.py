# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple

import pytest
import torch
import ttnn
from loguru import logger
import torch.nn as nn

from tests.sweep_framework.sweep_utils.roofline_utils import get_run_return
from tests.ttnn.utils_for_testing import start_measuring_time, stop_measuring_time
from models.utility_functions import torch_random

TIMEOUT = 15

parameters = {
    "default": {
        "params": [
            # ((1, 1, 128, 4096), (256), 2, 4), # 4096/256=16 shards,  0..2 x 0..4 = 15 cores
            # ((1, 1, 128, 8192), (512), 2, 4), # 8192/512=16 shards,  0..2 x 0..4 = 15 cores
            # ((1, 1, 256, 4096), (256), 2, 4), # 4096/256=16 shards,  0..2 x 0..4 = 15 cores
            # ((1, 1, 256, 8192), (512), 2, 4), # 8192/512=16 shards,  0..2 x 0..4 = 15 cores
            # ((1, 1, 256, 14848), (512), 2, 4), # 14848/512=29 shards,  0..2 x 0..4 = 15 cores
            # ((1, 1, 128, 8192), (256), 5, 4), # 8192/256=32 shards,  0..5 x 0..4 = 30 cores
            # ((1, 1, 128, 16384), (512), 5, 4), # 16384/512=32 shards,  0..5 x 0..4 = 30 cores
            # ((1, 1, 256, 8192), (256), 5, 4), # 8192/256=32 shards,  0..5 x 0..4 = 30 cores
            # ((1, 1, 256, 16384), (512), 5, 4), # 16384/512=32 shards,  0..5 x 0..4 = 30 cores
            # ((1, 1, 256, 30208), (512), 5, 4), # 30208/512=59 shards,  0..5 x 0..4 = 30 cores
            # ((1, 1, 128, 16384), (256), 6, 8), # 16384/256=64 shards,  0..6 x 0..8 = 63 cores
            # ((1, 1, 128, 32768), (512), 6, 8), # 32768/512=64 shards,  0..6 x 0..8 = 63 cores
            # ((1, 1, 256, 16384), (256), 6, 8), # 16384/256=64 shards,  0..6 x 0..8 = 63 cores
            # ((1, 1, 256, 32768), (512), 6, 8), # 32768/512=64 shards,  0..6 x 0..8 = 63 cores
            # ((1, 1, 256, 64000), (512), 6, 8), # 64000/512=125 shards,  0..6 x 0..8 = 63 cores
            ((1, 1, 128, 4096), (256), 3, 3),  # 4096/256=16 shards,  0..2 x 0..4 = 16 cores
            ((1, 1, 128, 8192), (512), 3, 3),  # 8192/512=16 shards,  0..2 x 0..4 = 16 cores
            ((1, 1, 256, 4096), (256), 3, 3),  # 4096/256=16 shards,  0..2 x 0..4 = 16 cores
            ((1, 1, 256, 8192), (512), 3, 3),  # 8192/512=16 shards,  0..2 x 0..4 = 16 cores
            ((1, 1, 256, 14848), (512), 3, 3),  # 14848/512=29 shards,  0..2 x 0..4 = 16 cores
            ((1, 1, 128, 8192), (256), 7, 3),  # 8192/256=32 shards,  0..5 x 0..4 = 32 cores
            ((1, 1, 128, 16384), (512), 7, 3),  # 16384/512=32 shards,  0..5 x 0..4 = 32 cores
            ((1, 1, 256, 8192), (256), 7, 3),  # 8192/256=32 shards,  0..5 x 0..4 = 32 cores
            ((1, 1, 256, 16384), (512), 7, 3),  # 16384/512=32 shards,  0..5 x 0..4 = 32 cores
            ((1, 1, 256, 30208), (512), 7, 3),  # 30208/512=59 shards,  0..5 x 0..4 = 32 cores
            ((1, 1, 128, 16384), (256), 7, 7),  # 16384/256=64 shards,  0..6 x 0..8 = 64 cores
            ((1, 1, 128, 32768), (512), 7, 7),  # 32768/512=64 shards,  0..6 x 0..8 = 64 cores
            ((1, 1, 256, 16384), (256), 7, 7),  # 16384/256=64 shards,  0..6 x 0..8 = 64 cores
            ((1, 1, 256, 32768), (512), 7, 7),  # 32768/512=64 shards,  0..6 x 0..8 = 64 cores
            ((1, 1, 256, 64000), (512), 7, 7),  # 64000/512=125 shards,  0..6 x 0..8 = 64 cores
        ],
    },
}


def run_sum(device, params):
    [input_shape, shard_w, end_x, end_y] = params
    torch_input_tensor = torch.rand(input_shape, dtype=torch.float32)
    n, c, h, w = input_shape
    dim = -2
    keepdim = True
    torch_output_tensor = torch.sum(torch_input_tensor, dim, keepdim)

    memory_config = ttnn.MemoryConfig(
        buffer_type=ttnn.BufferType.L1,
        nd_shard_spec=ttnn.NdShardSpec(
            (1, 1, h, shard_w),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(end_x, end_y))}),
            ttnn.ShardOrientation.COL_MAJOR,
        ),
    )
    # Interleaved configs for potential comparsion
    # memory_config = None # DRAM Interleaved
    # memory_config = ttnn.L1_MEMORY_CONFIG # L1 Interleaved
    input_tensor = ttnn.from_torch(
        torch_input_tensor, dtype=ttnn.float32, device=device, layout=ttnn.TILE_LAYOUT, memory_config=memory_config
    )

    # ttnn.set_printoptions(profile="Full")
    start_time = start_measuring_time()
    op_output_tensor = ttnn.sum(input_tensor, dim=dim, keepdim=keepdim)
    output_tensor = ttnn.to_torch(op_output_tensor)
    e2e_perf = stop_measuring_time(start_time)
    expected_pcc = 0.999
    tensors = [input_tensor, op_output_tensor]
    print(f"oot {op_output_tensor}\not{output_tensor}\ntot {torch_output_tensor}")
    lst = []
    for i in range(w):
        x = output_tensor[0, 0, 0, i]
        if x < (h - 0.1) or x > (h + 0.1):
            lst.append(i)
    prev = -2
    for i in lst:
        if (prev + 1) != i:
            if prev != -2:
                print(prev)
            print(f"incorrect range {i}-", end="")
        prev = i

    if prev != -2:
        print(prev)
    mse = nn.MSELoss()(output_tensor, torch_output_tensor)
    print(f"MSE: {mse}")
    return get_run_return(torch_output_tensor, output_tensor, expected_pcc, tensors, e2e_perf)


@pytest.mark.parametrize("params", parameters["default"]["params"])
def test_default(device, params):
    (result, msg), e2e_perf = run_sum(device, params)
    assert result, msg
    logger.info(msg)
    if e2e_perf:
        logger.info(f"E2E Performance: {e2e_perf}")


def run(
    params,
    *,
    device,
) -> list:
    return run_sum(device, params)
