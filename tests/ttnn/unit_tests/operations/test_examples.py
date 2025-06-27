# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os

import pytest

import torch

import ttnn


def test_example(device):
    # shape = [1 * 32, 32]
    # magic_num = 0  # Fails with random  numbers
    # magic_num = 10  # Fails with random  numbers

    shape = [2 * 32, 32]
    magic_num = 23  # (23,24,26)

    torch_input = torch.ones(shape, dtype=torch.bfloat16) * -1.7014118e38
    torch_other = torch.zeros(shape, dtype=torch.bfloat16) * 0

    expected_output = torch_input * torch_other

    for x in range(1000):
        print("iter", x)
        print("nops", x + magic_num)
        os.environ["UNOPS"] = str(x + magic_num)
        tt_input = ttnn.from_torch(torch_input, layout=ttnn.TILE_LAYOUT, device=device)
        tt_other = ttnn.from_torch(torch_other, layout=ttnn.TILE_LAYOUT, device=device)

        tt_output1 = ttnn.empty(shape, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        tt_output = ttnn.empty(shape, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

        ttnn.prim.example(tt_other, tt_output1)
        ttnn.prim.example_multiple_return(tt_input, tt_output1, output=tt_output)

        actual_output = ttnn.to_torch(tt_output)

        ret = torch.allclose(actual_output, expected_output, atol=0, rtol=0)
        if not ret:
            print("actual_output", actual_output.to(torch.float32).flatten().tolist())
            print("FAILED")
            break
