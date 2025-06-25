# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os
import math
import json
import torch
import numpy as np
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import (
    comp_allclose,
    comp_pcc,
)
import ttnn
from loguru import logger
import pytest
from dataclasses import dataclass

from models.demos.deepseek_v3.tt.rope import RotarySetup
from models.demos.t3000.llama2_70b.reference.llama.llama31_8b.model import RMSNorm as ReferenceRMSNorm
from models.common.rmsnorm import RMSNorm as RMSNorm
from models.demos.deepseek_v3_impl.model import (
    ModelArgs,
    precompute_freqs_cis,
    apply_rotary_emb,
)
from models.utility_functions import nearest_y


TP = 8
DP = 4


class ModelConfig:
    def __init__(self, model_args: ModelArgs):
        self.args = model_args
        self.args.qk_head_dim = self.args.qk_nope_head_dim + self.args.qk_rope_head_dim

        self.grid_size = (8, 8)
        self.bsz = 64 * 2  # Use padded shapes
        self.configs = {}

        #################
        ### MLA Configs
        #################

        # wq_a
        self.configs["WQA_IN0_SHAPE"] = (1, 1, self.bsz // DP, self.args.dim // TP)
        self.configs["WQA_IN1_SHAPE"] = (1, 1, self.args.dim // TP, self.args.q_lora_rank)
        self.configs["WQA_IN0_DTYPE"] = ttnn.bfloat8_b
        self.configs["WQA_IN1_DTYPE"] = ttnn.bfloat4_b
        self.configs["WQA_PROGRAM_CFG"] = None
        self.configs["WQA_IN0_MEM_CFG"] = ttnn.DRAM_MEMORY_CONFIG
        self.configs["WQA_IN1_MEM_CFG"] = ttnn.DRAM_MEMORY_CONFIG
        self.configs["WQA_OUT_MEM_CFG"] = ttnn.DRAM_MEMORY_CONFIG

        # wq_b
        self.configs["WQB_IN0_SHAPE"] = (1, 1, self.bsz // DP, self.args.q_lora_rank)
        self.configs["WQB_IN1_SHAPE"] = (1, 1, self.args.q_lora_rank, (self.args.n_heads * self.args.qk_head_dim) // TP)
        self.configs["WQB_IN0_DTYPE"] = ttnn.bfloat8_b
        self.configs["WQB_IN1_DTYPE"] = ttnn.bfloat4_b
        self.configs["WQB_PROGRAM_CFG"] = None
        self.configs["WQB_IN0_MEM_CFG"] = ttnn.DRAM_MEMORY_CONFIG
        self.configs["WQB_IN1_MEM_CFG"] = ttnn.DRAM_MEMORY_CONFIG
        self.configs["WQB_OUT_MEM_CFG"] = ttnn.DRAM_MEMORY_CONFIG

        # wkv_a
        self.configs["WKV_A_IN0_SHAPE"] = (1, 1, self.bsz // DP, self.args.dim // TP)
        self.configs["WKV_A_IN1_SHAPE"] = (
            1,
            1,
            self.args.dim // TP,
            self.args.kv_lora_rank + self.args.qk_rope_head_dim,
        )
        self.configs["WKV_A_IN0_DTYPE"] = ttnn.bfloat8_b
        self.configs["WKV_A_IN1_DTYPE"] = ttnn.bfloat4_b
        self.configs["WKV_A_PROGRAM_CFG"] = None
        self.configs["WKV_A_IN0_MEM_CFG"] = ttnn.DRAM_MEMORY_CONFIG
        self.configs["WKV_A_IN1_MEM_CFG"] = ttnn.DRAM_MEMORY_CONFIG
        self.configs["WKV_A_OUT_MEM_CFG"] = ttnn.DRAM_MEMORY_CONFIG

        # wkv_b1
        self.configs["WKV_B1_IN0_SHAPE"] = (self.bsz // DP, self.args.n_heads // TP, 1, self.args.qk_nope_head_dim)
        self.configs["WKV_B1_IN1_SHAPE"] = (
            self.bsz // DP,
            self.args.n_heads // TP,
            self.args.qk_nope_head_dim,
            self.args.kv_lora_rank,
        )
        self.configs["WKV_B1_IN0_DTYPE"] = ttnn.bfloat8_b
        self.configs["WKV_B1_IN1_DTYPE"] = ttnn.bfloat4_b
        self.configs["WKV_B1_PROGRAM_CFG"] = None
        self.configs["WKV_B1_IN0_MEM_CFG"] = ttnn.DRAM_MEMORY_CONFIG
        self.configs["WKV_B1_IN1_MEM_CFG"] = ttnn.DRAM_MEMORY_CONFIG
        self.configs["WKV_B1_OUT_MEM_CFG"] = ttnn.DRAM_MEMORY_CONFIG

        # wkv_b2
        self.configs["WKV_B2_IN0_SHAPE"] = (self.bsz // DP, self.args.n_heads // TP, 1, self.args.kv_lora_rank)
        self.configs["WKV_B2_IN1_SHAPE"] = (
            self.bsz // DP,
            self.args.n_heads // TP,
            self.args.kv_lora_rank,
            self.args.v_head_dim,
        )
        self.configs["WKV_B2_IN0_DTYPE"] = ttnn.bfloat8_b
        self.configs["WKV_B2_IN1_DTYPE"] = ttnn.bfloat4_b
        self.configs["WKV_B2_PROGRAM_CFG"] = None
        self.configs["WKV_B2_IN0_MEM_CFG"] = ttnn.DRAM_MEMORY_CONFIG
        self.configs["WKV_B2_IN1_MEM_CFG"] = ttnn.DRAM_MEMORY_CONFIG
        self.configs["WKV_B2_OUT_MEM_CFG"] = ttnn.DRAM_MEMORY_CONFIG

        # wo
        self.configs["WO_IN0_SHAPE"] = (1, self.bsz // DP, self.args.n_heads * self.args.v_head_dim)
        self.configs["WO_IN1_SHAPE"] = (1, 1, self.args.n_heads * self.args.v_head_dim, self.args.dim // TP)
        self.configs["WO_IN0_DTYPE"] = ttnn.bfloat8_b
        self.configs["WO_IN1_DTYPE"] = ttnn.bfloat4_b
        self.configs["WO_PROGRAM_CFG"] = None
        self.configs["WO_IN0_MEM_CFG"] = ttnn.DRAM_MEMORY_CONFIG
        self.configs["WO_IN1_MEM_CFG"] = ttnn.DRAM_MEMORY_CONFIG
        self.configs["WO_OUT_MEM_CFG"] = ttnn.DRAM_MEMORY_CONFIG

        # q_rope
        self.configs["QROPE_SHAPE"] = (1, self.bsz // DP, self.args.n_heads // TP, self.args.qk_rope_head_dim)
        self.configs["QROPE_DTYPE"] = ttnn.bfloat16

        q_rope_shard_height = nearest_y(self.configs["QROPE_SHAPE"][2], ttnn.TILE_SIZE)
        q_rope_shard_width = self.configs["QROPE_SHAPE"][3]
        q_rope_num_cores = self.configs["QROPE_SHAPE"][1]
        q_rope_core_grid = ttnn.num_cores_to_corerangeset(q_rope_num_cores, self.grid_size, row_wise=True)
        self.configs["QROPE_MEM_CFG"] = ttnn.create_sharded_memory_config(
            shape=(q_rope_shard_height, q_rope_shard_width),
            core_grid=q_rope_core_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            use_height_and_width_as_shard_shape=True,
        )

        # k_rope
        self.configs["KROPE_SHAPE"] = (1, self.bsz // DP // TP, 1, self.args.qk_rope_head_dim)
        self.configs["KROPE_DTYPE"] = ttnn.bfloat16
        k_rope_shard_height = nearest_y(self.configs["KROPE_SHAPE"][2], ttnn.TILE_SIZE)
        k_rope_shard_width = self.configs["KROPE_SHAPE"][3]
        k_rope_num_cores = self.configs["KROPE_SHAPE"][1]
        k_rope_core_grid = ttnn.num_cores_to_corerangeset(k_rope_num_cores, self.grid_size, row_wise=True)
        self.configs["KROPE_MEM_CFG"] = ttnn.create_sharded_memory_config(
            shape=(k_rope_shard_height, k_rope_shard_width),
            core_grid=k_rope_core_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            use_height_and_width_as_shard_shape=True,
        )

        # KVPE Cache
        self.configs["KVPE_SHAPE"] = (1, self.bsz // DP // TP, 1, self.args.kv_lora_rank + self.args.qk_rope_head_dim)
        self.configs["KVPE_DTYPE"] = ttnn.bfloat16
        kvpe_shard_height = nearest_y(self.configs["KVPE_SHAPE"][2], ttnn.TILE_SIZE)
        kvpe_shard_width = self.configs["KVPE_SHAPE"][3]
        kvpe_num_cores = self.configs["KVPE_SHAPE"][1]
        kvpe_core_grid = ttnn.num_cores_to_corerangeset(kvpe_num_cores, self.grid_size, row_wise=True)
        self.configs["KVPE_MEM_CFG"] = ttnn.create_sharded_memory_config(
            shape=(kvpe_shard_height, kvpe_shard_width),
            core_grid=kvpe_core_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            use_height_and_width_as_shard_shape=True,
        )

        self.configs["KVPE_CACHE_DTYPE"] = ttnn.bfloat4_b

        # q_norm
        self.configs["QNORM_SHAPE"] = (1, 1, self.bsz // DP, self.args.q_lora_rank)
        self.configs["QNORM_DTYPE"] = ttnn.bfloat16
        self.configs["QNORM_MEM_CFG"] = ttnn.DRAM_MEMORY_CONFIG

        # k_norm
        self.configs["KNORM_SHAPE"] = (1, 1, self.bsz // DP // TP, self.args.kv_lora_rank + self.args.qk_rope_head_dim)
        self.configs["KNORM_DTYPE"] = ttnn.bfloat16
        self.configs["KNORM_MEM_CFG"] = ttnn.DRAM_MEMORY_CONFIG
        # # TODO: Debug, gives bad PCC
        # knorm_num_cores = min(np.prod(self.grid_size), math.ceil(self.configs["KNORM_SHAPE"][3] / ttnn.TILE_SIZE))
        # knorm_core_grid = ttnn.num_cores_to_corerangeset(knorm_num_cores, self.grid_size, row_wise=True)
        # knorm_shard_height = nearest_y(self.configs["KNORM_SHAPE"][2], ttnn.TILE_SIZE) * np.prod(self.configs["KNORM_SHAPE"][:2])
        # knorm_shard_width = nearest_y(self.configs["KNORM_SHAPE"][3] // knorm_num_cores, ttnn.TILE_SIZE)
        # self.configs["KNORM_MEM_CFG"] = ttnn.create_sharded_memory_config(
        #     shape=(knorm_shard_height, knorm_shard_width),
        #     core_grid=knorm_core_grid,
        #     strategy=ttnn.ShardStrategy.WIDTH,
        #     use_height_and_width_as_shard_shape=True,
        # )


config_path = "models/demos/deepseek_v3_impl/configs/config_671B.json"
with open(config_path) as f:
    model_args = ModelArgs(**json.load(f))
cfg = ModelConfig(model_args)


#################
### Helper Funcs
#################


def run_matmul_impl(
    device,
    shapes,
    dtypes,
    program_config,
    memory_configs,
):
    layout = ttnn.TILE_LAYOUT

    in0_shape, in1_shape = shapes
    in0_dtype, in1_dtype = dtypes
    in0_mem_config, in1_mem_config, out_mem_config = memory_configs

    # Log configs
    logger.info("Running matmul with the following configurations:")
    logger.info(f"Input 0 Shape: {in0_shape}, Dtype: {in0_dtype}, Memory Config: {in0_mem_config}")
    logger.info(f"Input 1 Shape: {in1_shape}, Dtype: {in1_dtype}, Memory Config: {in1_mem_config}")
    logger.info(f"Output Memory Config: {out_mem_config}")
    logger.info(f"Program Config: {program_config}")

    #################
    ### Torch
    #################
    in0 = torch.randn(in0_shape).float()
    in1 = torch.randn(in1_shape).float()
    out_torch = in0 @ in1

    #################
    ### TT-NN
    #################
    tt_in0 = ttnn.from_torch(
        in0,
        device=device,
        dtype=in0_dtype,
        memory_config=in0_mem_config,
        layout=layout,
    )

    tt_in1 = ttnn.from_torch(
        in1,
        device=device,
        dtype=in1_dtype,
        memory_config=in1_mem_config,
        layout=layout,
    )

    tt_out = ttnn.matmul(
        tt_in0,
        tt_in1,
        memory_config=out_mem_config,
        program_config=program_config,
    )

    tt_out_torch = ttnn.to_torch(tt_out)

    #################
    ### Validation
    #################
    pcc_threshold = 0.99

    out_pass, out_pcc = comp_pcc(tt_out_torch, out_torch, pcc_threshold)
    logger.info(f"Output PCC: {out_pcc}")

    assert out_pass, f"Output mismatch: PCC {out_pcc} < 0.99"


def run_rope_impl(
    device,
    shape,
    dtype,
    mem_config,
):
    # TODO: Only testing without scaling for now!

    layout = ttnn.TILE_LAYOUT

    _, bsz, nh, head_dim = shape

    logger.info("Running rope with the following configurations:")
    logger.info(f"Shape: {shape}, Dtype: {dtype}, Memory Config: {mem_config}")
    logger.info(f"Max Seq Len: {cfg.args.max_seq_len}, Rope Theta: {cfg.args.rope_theta}")

    #################
    ### Torch
    #################
    position_ids = torch.randint(0, cfg.args.max_seq_len, (bsz,))
    input_torch = torch.randn(shape).float()
    freqs_cis = precompute_freqs_cis(cfg.args)[position_ids, :]
    out_torch = apply_rotary_emb(input_torch, freqs_cis)

    #################
    ### TT-NN
    #################
    rope_setup = RotarySetup(
        device=device,
        batch_size=bsz,
        reference_args=cfg.args,
    )

    tt_cos, tt_sin = rope_setup.get_rot_mats(position_ids)
    tt_trans_mat = rope_setup.get_both_trans_mats()["decode"]

    tt_input = ttnn.from_torch(
        input_torch,
        device=device,
        dtype=dtype,
        memory_config=mem_config,
        layout=layout,
    )

    tt_out = ttnn.experimental.rotary_embedding_llama(
        tt_input,
        tt_cos,
        tt_sin,
        tt_trans_mat,
        is_decode_mode=True,
    )

    tt_out_torch = ttnn.to_torch(tt_out)

    #################
    ### Validation
    #################
    pcc_threshold = 0.99

    out_pass, out_pcc = comp_pcc(tt_out_torch, out_torch, pcc_threshold)
    logger.info(f"Output PCC: {out_pcc}")

    assert out_pass, f"Output mismatch: PCC {out_pcc} < 0.99"


def run_update_cache_impl(
    device,
    shape,
    dtype,
    mem_config,
    cache_dtype,
):
    layout = ttnn.TILE_LAYOUT
    max_seq_len = cfg.args.max_seq_len

    logger.info("Running update cache with the following configurations:")
    logger.info(f"Shape: {shape}, Dtype: {dtype}, Memory Config: {mem_config}")
    logger.info(f"Max Seq Len: {max_seq_len}")

    _, bsz, nkv, head_dim = shape

    #################
    ### Torch
    #################
    cache_torch = torch.randn((bsz, nkv, max_seq_len, head_dim)).float()
    input_torch = torch.randn(shape).float()
    current_pos = torch.randint(0, max_seq_len, (bsz,))

    #################
    ### TT-NN
    #################
    tt_cache = ttnn.from_torch(
        cache_torch,
        device=device,
        dtype=cache_dtype,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=layout,
    )

    tt_input = ttnn.from_torch(
        input_torch,
        device=device,
        dtype=dtype,
        memory_config=mem_config,
        layout=layout,
    )

    tt_current_pos = ttnn.from_torch(
        current_pos,
        device=device,
        dtype=ttnn.int32,
    )

    ttnn.experimental.paged_update_cache(
        tt_cache,
        tt_input,
        update_idxs_tensor=tt_current_pos,
    )
    tt_cache_torch = ttnn.to_torch(tt_cache)

    #################
    ### Validation
    #################
    for b in range(bsz):
        inp = input_torch[:, b, ...].unsqueeze(1)  # [seq_len, b, nkv, head_dim]
        inp = inp.permute(1, 2, 0, 3)  # [b, nkv, seq_len, head_dim]

        pos = current_pos[b].item()

        cache_torch[b, :, pos : pos + 1, :] = inp

    pcc_threshold = 0.9999
    if cache_dtype == ttnn.bfloat4_b:
        pcc_threshold = 0.99

    out_pass, out_pcc = comp_pcc(tt_cache_torch, cache_torch, pcc_threshold)
    logger.info(f"Output PCC: {out_pcc}")

    assert out_pass, f"Output mismatch: PCC {out_pcc} < 0.99"


def run_rmsnorm_impl(
    device,
    shape,
    dtype,
    mem_config,
):
    layout = ttnn.TILE_LAYOUT

    logger.info("Running RMSNorm with the following configurations:")
    logger.info(f"Shape: {shape}, Dtype: {dtype}, Memory Config: {mem_config}")

    _, bsz, nh, head_dim = shape

    #################
    ### Torch
    #################
    input_torch = torch.randn(shape).float()
    rms_norm = ReferenceRMSNorm(head_dim, eps=1e-5)
    out_torch = rms_norm(input_torch)

    #################
    ### TT-NN
    #################
    tt_input = ttnn.from_torch(
        input_torch,
        device=device,
        dtype=dtype,
        memory_config=mem_config,
        layout=layout,
    )

    state_dict = {
        "rms_norm_weight.weight": rms_norm.weight.unsqueeze(0),
    }
    tt_rms_norm = RMSNorm(
        device=device,
        dim=head_dim,
        eps=1e-5,
        weight_key="rms_norm_weight",
        state_dict=state_dict,
    )

    tt_out = tt_rms_norm(tt_input, mode="decode")
    tt_out_torch = ttnn.to_torch(tt_out)

    #################
    ### Validation
    #################
    pcc_threshold = 0.9999

    out_pass, out_pcc = comp_pcc(tt_out_torch, out_torch, pcc_threshold)
    logger.info(f"Output PCC: {out_pcc}")

    assert out_pass, f"Output mismatch: PCC {out_pcc} < 0.99"


#################
### Tests
#################
@pytest.mark.parametrize(
    "shapes, dtypes, program_config, memory_configs",
    [
        (  # wq_a
            [cfg.configs["WQA_IN0_SHAPE"], cfg.configs["WQA_IN1_SHAPE"]],
            [cfg.configs["WQA_IN0_DTYPE"], cfg.configs["WQA_IN1_DTYPE"]],
            cfg.configs["WQA_PROGRAM_CFG"],
            [cfg.configs["WQA_IN0_MEM_CFG"], cfg.configs["WQA_IN1_MEM_CFG"], cfg.configs["WQA_OUT_MEM_CFG"]],
        ),
        (  # wq_b
            [cfg.configs["WQB_IN0_SHAPE"], cfg.configs["WQB_IN1_SHAPE"]],
            [cfg.configs["WQB_IN0_DTYPE"], cfg.configs["WQB_IN1_DTYPE"]],
            cfg.configs["WQB_PROGRAM_CFG"],
            [cfg.configs["WQB_IN0_MEM_CFG"], cfg.configs["WQB_IN1_MEM_CFG"], cfg.configs["WQB_OUT_MEM_CFG"]],
        ),
        (  # wkv_a
            [cfg.configs["WKV_A_IN0_SHAPE"], cfg.configs["WKV_A_IN1_SHAPE"]],
            [cfg.configs["WKV_A_IN0_DTYPE"], cfg.configs["WKV_A_IN1_DTYPE"]],
            cfg.configs["WKV_A_PROGRAM_CFG"],
            [cfg.configs["WKV_A_IN0_MEM_CFG"], cfg.configs["WKV_A_IN1_MEM_CFG"], cfg.configs["WKV_A_OUT_MEM_CFG"]],
        ),
        (  # wkv_b1
            [cfg.configs["WKV_B1_IN0_SHAPE"], cfg.configs["WKV_B1_IN1_SHAPE"]],
            [cfg.configs["WKV_B1_IN0_DTYPE"], cfg.configs["WKV_B1_IN1_DTYPE"]],
            cfg.configs["WKV_B1_PROGRAM_CFG"],
            [cfg.configs["WKV_B1_IN0_MEM_CFG"], cfg.configs["WKV_B1_IN1_MEM_CFG"], cfg.configs["WKV_B1_OUT_MEM_CFG"]],
        ),
        (  # wkv_b2
            [cfg.configs["WKV_B2_IN0_SHAPE"], cfg.configs["WKV_B2_IN1_SHAPE"]],
            [cfg.configs["WKV_B2_IN0_DTYPE"], cfg.configs["WKV_B2_IN1_DTYPE"]],
            cfg.configs["WKV_B2_PROGRAM_CFG"],
            [cfg.configs["WKV_B2_IN0_MEM_CFG"], cfg.configs["WKV_B2_IN1_MEM_CFG"], cfg.configs["WKV_B2_OUT_MEM_CFG"]],
        ),
        (  # wo
            [cfg.configs["WO_IN0_SHAPE"], cfg.configs["WO_IN1_SHAPE"]],
            [cfg.configs["WO_IN0_DTYPE"], cfg.configs["WO_IN1_DTYPE"]],
            cfg.configs["WO_PROGRAM_CFG"],
            [cfg.configs["WO_IN0_MEM_CFG"], cfg.configs["WO_IN1_MEM_CFG"], cfg.configs["WO_OUT_MEM_CFG"]],
        ),
    ],
    ids=[
        "wq_a",
        "wq_b",
        "wkv_a",
        "wkv_b1",
        "wkv_b2",
        "wo",
    ],
)
def test_matmuls(
    device,
    shapes,
    dtypes,
    program_config,
    memory_configs,
    use_program_cache,
    function_level_defaults,
    reset_seeds,
):
    run_matmul_impl(
        device,
        shapes=shapes,
        dtypes=dtypes,
        program_config=program_config,
        memory_configs=memory_configs,
    )


@pytest.mark.parametrize(
    "shape, dtype, mem_config",
    [
        (  # q_rope
            cfg.configs["QROPE_SHAPE"],
            cfg.configs["QROPE_DTYPE"],
            cfg.configs["QROPE_MEM_CFG"],
        ),
        (  # k_rope
            cfg.configs["KROPE_SHAPE"],
            cfg.configs["KROPE_DTYPE"],
            cfg.configs["KROPE_MEM_CFG"],
        ),
    ],
    ids=[
        "q_rope",
        "k_rope",
    ],
)
def test_ropes(
    device,
    shape,
    dtype,
    mem_config,
    use_program_cache,
    function_level_defaults,
    reset_seeds,
):
    run_rope_impl(
        device,
        shape=shape,
        dtype=dtype,
        mem_config=mem_config,
    )


@pytest.mark.parametrize(
    "shape, dtype, mem_config, cache_dtype",
    [
        (
            cfg.configs["KVPE_SHAPE"],
            cfg.configs["KVPE_DTYPE"],
            cfg.configs["KVPE_MEM_CFG"],
            cfg.configs["KVPE_CACHE_DTYPE"],
        ),
    ],
    ids=["kvpe"],
)
def test_update_caches(
    device,
    shape,
    dtype,
    mem_config,
    cache_dtype,
    use_program_cache,
    function_level_defaults,
    reset_seeds,
):
    run_update_cache_impl(
        device,
        shape=shape,
        dtype=dtype,
        mem_config=mem_config,
        cache_dtype=cache_dtype,
    )


@pytest.mark.parametrize(
    "shape, dtype, mem_config",
    [
        (
            cfg.configs["QNORM_SHAPE"],
            cfg.configs["QNORM_DTYPE"],
            cfg.configs["QNORM_MEM_CFG"],
        ),
        (
            cfg.configs["KNORM_SHAPE"],
            cfg.configs["KNORM_DTYPE"],
            cfg.configs["KNORM_MEM_CFG"],
        ),
    ],
    ids=["q_norm", "k_norm"],
)
def test_rmsnorms(
    device,
    shape,
    dtype,
    mem_config,
    use_program_cache,
    function_level_defaults,
    reset_seeds,
):
    run_rmsnorm_impl(
        device,
        shape=shape,
        dtype=dtype,
        mem_config=mem_config,
    )
