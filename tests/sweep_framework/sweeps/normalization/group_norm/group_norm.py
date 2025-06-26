# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple
from functools import partial

import torch
import random
import itertools
import ttnn
import pytest
from tests.sweep_framework.sweep_utils.utils import gen_shapes
from tests.sweep_framework.sweep_utils.roofline_utils import get_run_return
from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_func_with_cast_tt

from tests.ttnn.utils_for_testing import check_with_pcc, start_measuring_time, stop_measuring_time
from models.utility_functions import torch_random

from tests.ttnn.unit_tests.operations.eltwise.backward.utility_funcs import data_gen_with_range_batch_norm

TIMEOUT = 30

random.seed(0)

parameters = {
    "GN_SDXL_1": {
        "N": [1],
        "C": [640],
        "H": [128],
        "W": [128],
        "num_groups": [32],
        "num_out_blocks": [3],
        "cores_y": [4],
        "cores_x": [4],
    },
    "GN_SDXL_2": {
        "N": [1],
        "C": [960],
        "H": [128],
        "W": [128],
        "num_groups": [32],
        "num_out_blocks": [12],
        "cores_y": [2],
        "cores_x": [2],
    },
}


def run_group_norm(
    device,
    N,
    C,
    H,
    W,
    num_groups,
    num_out_blocks,
    cores_y,
    cores_x,
):
    data_seed = random.randint(0, 20000000)
    torch.manual_seed(data_seed)

    grid_size = ttnn.CoreGrid(y=cores_y, x=cores_x)

    # torch input tensor
    torch_input_tensor = torch.rand((N, C, H, W), dtype=torch.bfloat16)
    torch_weight = torch.rand((C,), dtype=torch.bfloat16)
    torch_bias = torch.rand((C,), dtype=torch.bfloat16)
    torch_output_tensor = torch.nn.functional.group_norm(
        torch_input_tensor, num_groups, weight=torch_weight, bias=torch_bias
    )
    torch_output_tensor = torch_output_tensor.permute(0, 2, 3, 1).view(N, 1, W * H, C)

    # input tensor
    input_tensor = torch_input_tensor.permute(0, 2, 3, 1).view(N, 1, W * H, C)
    input_tensor_row_major = ttnn.from_torch(
        input_tensor,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    input_tensor_tilized = ttnn.tilize_with_zero_padding(input_tensor_row_major, use_multicore=True)

    # input mask
    input_mask_tensor = ttnn.create_group_norm_input_mask(C, num_groups, grid_size.y)
    input_mask_tensor = ttnn.from_torch(
        input_mask_tensor,
        dtype=ttnn.DataType.BFLOAT8_B,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # gamma/beta
    gamma = ttnn.create_group_norm_weight_bias_rm(torch_weight, C, grid_size.y)
    beta = ttnn.create_group_norm_weight_bias_rm(torch_bias, C, grid_size.y)

    gamma_t = ttnn.from_torch(
        gamma,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    beta_t = ttnn.from_torch(
        beta,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # groupnorm
    start_time = start_measuring_time()
    output_tensor = ttnn.group_norm(
        input_tensor_tilized,
        num_groups=num_groups,
        input_mask=input_mask_tensor,
        weight=gamma_t,
        bias=beta_t,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        output_layout=ttnn.TILE_LAYOUT,
        core_grid=grid_size,
        inplace=False,
        num_out_blocks=num_out_blocks,
    )

    ttnn.synchronize_device(device)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor_torch = ttnn.to_torch(output_tensor)
    e2e_perf = stop_measuring_time(start_time)

    expected_pcc = 0.9996
    tensors = [input_tensor_tilized, input_mask_tensor, gamma_t, beta_t, output_tensor]
    return get_run_return(torch_output_tensor, output_tensor_torch, expected_pcc, tensors, e2e_perf)


def run(
    N,
    C,
    H,
    W,
    num_groups,
    num_out_blocks,
    cores_y,
    cores_x,
    *,
    device,
) -> list:
    return run_group_norm(
        device,
        N,
        C,
        H,
        W,
        num_groups,
        num_out_blocks,
        cores_y,
        cores_x,
    )


param_keys = parameters["GN_SDXL_1"].keys()
param_values = itertools.product(*parameters["GN_SDXL_1"].values())


@pytest.mark.parametrize(",".join(param_keys), list(param_values))
def test_group_norm(
    N,
    C,
    H,
    W,
    num_groups,
    num_out_blocks,
    cores_y,
    cores_x,
    device,
):
    run_group_norm(
        N,
        C,
        H,
        W,
        num_groups,
        num_out_blocks,
        cores_y,
        cores_x,
        device,
    )
