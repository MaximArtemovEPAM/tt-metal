# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os
import math
import torch
import numpy as np
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import (
    comp_allclose,
    comp_pcc,
)
import ttnn
from loguru import logger
import pytest


def nearest_n(x, n):
    return ((x + n - 1) // n) * n


def nearest_pow_2(x):
    if x < 1:
        raise ValueError("x must be >= 1")
    import math

    power = math.ceil(math.log2(x))
    return 1 << power


def scaled_dot_product_attention_reference(Q, K, V, scale, is_causal=True):
    """
    Full-sequence causal SDPA reference.
    Q: (B, nh, S, d_qk), K/V: (B, nkv, S, d)
    """
    b, nh, S, d_qk = Q.shape
    _, nkv, _, d_v = V.shape
    # Expand KV to match Q heads
    head_rep = nh // nkv
    K_exp = K.repeat_interleave(head_rep, dim=1)  # (B, nh, S, d_qk)
    V_exp = V.repeat_interleave(head_rep, dim=1)  # (B, nh, S, d_v)
    # Use PyTorch’s builtin causal attention
    return torch.nn.functional.scaled_dot_product_attention(
        Q, K_exp, V_exp, attn_mask=None, scale=scale, is_causal=is_causal
    )


def flash_mla_prefill_tt(
    query,
    key,
    value,
    nh,
    nkv,
    is_causal=True,
):
    pass


def run_flash_mla_prefill_impl(
    device,
    batch,
    seq_len,
    nh,
    nkv,
    kv_lora_rank,
    d_rope,
    q_dtype,
    dtype,
):
    ######################
    ### Torch Setup
    ######################
    q = torch.randn(batch, nh, seq_len, kv_lora_rank + d_rope).float()  # (B, H, S (1 for decode), D)
    k = torch.randn(batch, nkv, seq_len, kv_lora_rank + d_rope).float()  # (B, H, S, D)
    v = torch.randn(batch, nkv, seq_len, kv_lora_rank + d_rope).float()  # (B, H, S, D)

    ######################
    ### TT Setup
    #######################

    padded_num_heads = nearest_pow_2(nearest_n(nh, n=32))
    q_chunk_size = padded_num_heads
    k_chunk_size = 128

    scale = (kv_lora_rank + d_rope) ** -0.5

    max_start_idx = seq_len // 2

    padded_layer_len = nearest_n(max_start_idx + 1, n=k_chunk_size)

    sdpa_program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
        q_chunk_size=q_chunk_size,
        k_chunk_size=k_chunk_size,
        exp_approx_mode=False,
    )

    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )

    tt_q = ttnn.from_torch(
        q,  # (B, H, S, D)
        device=device,
        dtype=q_dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tt_k = ttnn.from_torch(
        k,  # (B, H, S, D)
        device=device,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tt_v = ttnn.from_torch(
        v,  # (B, H, S, D)
        device=device,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    logger.info(f"TT Q shape: {tt_q.shape}, TT K shape: {tt_k.shape}, TT V shape: {tt_v.shape}")
    tt_out = ttnn.transformer.flash_mla_prefill(
        tt_q,
        tt_k,
        tt_v,
        scale=scale,
        program_config=sdpa_program_config,
        compute_kernel_config=compute_kernel_config,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        attn_mask=None,
        is_causal=True,
    )
    tt_back = ttnn.to_torch(tt_out)  # now (B, H_padded, S_padded, D)
    print("raw to_torch shape:", tt_back.shape)
    # slice out the padded heads and sequence length; no permute needed
    tt_out_torch = tt_back[:, :nh, :seq_len, :]  # (B, nh, S, D)

    ########################
    ### Validation
    ########################
    out_t = scaled_dot_product_attention_reference(
        q,
        k,
        v,
        scale,
        is_causal=True,
    )

    out_pass, out_pcc = comp_pcc(tt_out_torch, out_t)
    logger.info(f"Output PCC: {out_pcc}")

    assert out_pass, f"Output mismatch: PCC {out_pcc} < 0.99"


@pytest.mark.parametrize(
    "batch, seq_len, nh, nkv, kv_lora_rank, d_rope",
    # batch, seq_len, num heads q, num heads kv, kv lora rank, dim rope
    [
        # (2, 1024, 16, 16, 512, 64), # DeepSeek V3 TG TP=8, DP=4 (OOM)
        (2, 1024, 8, 8, 128, 64),
    ],
)
@pytest.mark.parametrize(
    "q_dtype, dtype",
    [
        (ttnn.bfloat16, ttnn.bfloat8_b),
    ],
)
def test_flash_mla_prefill(
    device,
    batch,
    seq_len,
    nh,
    nkv,
    kv_lora_rank,
    d_rope,
    q_dtype,
    dtype,
    use_program_cache,
    function_level_defaults,
    reset_seeds,
):
    run_flash_mla_prefill_impl(
        device,
        batch,
        seq_len,
        nh,
        nkv,
        kv_lora_rank,
        d_rope,
        q_dtype,
        dtype,
    )
