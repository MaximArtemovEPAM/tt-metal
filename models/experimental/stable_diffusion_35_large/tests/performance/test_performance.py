import torch
import torch.nn.functional as F
import pytest

import ttnn

from models.utility_functions import nearest_32
from models.experimental.stable_diffusion_35_large.tt.fun_linear import TtLinearParameters, sd_linear
from models.experimental.stable_diffusion_35_large.tt.fun_normalization import TtRmsNormParameters, sd_rms_norm
from models.experimental.stable_diffusion_35_large.tt.utils import assert_quality


def rms_norm(x, w, eps=1e-6):
    rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + eps)
    return x / rms * w


@pytest.mark.parametrize(
    "M, K, N",
    [
        [1, 2560, 3840],
    ],
    ids=["tp4"],
)
@pytest.mark.parametrize("shard_num_cores", [i for i in range(1, 64)])
def test_time_embedding(device, use_program_cache, M, K, N, shard_num_cores):
    """
    Test the graph of spatial and prompt time embedding
    """
    x = torch.randn(M, K)
    w = torch.randn(K, N)
    b = torch.randn(1, N)
    # breakpoint()

    x_tt = ttnn.from_torch(x, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    w_tt = ttnn.from_torch(w, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    b_tt = ttnn.from_torch(b, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    shard_grid = ttnn.num_cores_to_corerangeset(shard_num_cores, device.compute_with_storage_grid_size(), True)
    memory_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            shard_grid,
            [
                x_tt.padded_shape[0],
                nearest_32(x_tt.padded_shape[1] / shard_num_cores),
            ],
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )
    x_tt = ttnn.to_memory_config(x_tt, memory_config)
    parameters = TtLinearParameters(weight=w_tt, bias=b_tt)

    x_tt = sd_linear(x_tt, parameters, memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG)

    x_back = ttnn.to_torch(x_tt)

    assert_quality(F.linear(x, w.T, b), x_back)


def test_layernorm_scale_shift():
    """
    Test the graph of layernorm with scale and shift

    normed = layernorm(x)
    x = x * (1 + scale) + shift
    """


@pytest.mark.parametrize(
    "M, K, N, n_local_heads",
    [
        [1024, 2560, 1920, 10],
        [352, 2560, 1920, 10],
    ],
    ids=["spatial_tp4", "prompt_tp4"],
)
@pytest.mark.parametrize("shard_grid", [(None, None)] + [(i, j) for i in range(1, 9) for j in range(1, 9)])
def test_attention_proj(device, use_program_cache, M, K, N, n_local_heads, shard_grid):
    """
    Test the graph of
    qkv = sd_linear(x, proj_qkv)
    q, k, v = create_heads(qkv)
    q = norm(q)
    k = norm(k)
    """
    head_dim = N // 3 // n_local_heads
    x = torch.randn(M, K)
    wqkv = torch.randn(K, N)
    bqkv = torch.randn(1, N)
    norm_q_weight = torch.randn(head_dim)
    norm_k_weight = torch.randn(head_dim)
    x_qkv = F.linear(x, wqkv.T, bqkv)
    q, k, v = [t.reshape(M, n_local_heads, -1).permute(1, 0, 2) for t in x_qkv.chunk(3, dim=-1)]
    q = rms_norm(q, norm_q_weight)
    k = rms_norm(k, norm_k_weight)

    x_tt = ttnn.from_torch(x.unsqueeze(0), device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    wqkv_tt = ttnn.from_torch(wqkv, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    bqkv_tt = ttnn.from_torch(bqkv, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    norm_q_weight_tt = ttnn.from_torch(
        norm_q_weight.unsqueeze(0), device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16
    )
    norm_k_weight_tt = ttnn.from_torch(
        norm_k_weight.unsqueeze(0), device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16
    )

    linear_params = TtLinearParameters(weight=wqkv_tt, bias=bqkv_tt)

    norm_q_params = TtRmsNormParameters(weight=norm_q_weight_tt, eps=1e-6)
    norm_k_params = TtRmsNormParameters(weight=norm_k_weight_tt, eps=1e-6)

    core_grid = device.core_grid
    output_memory_config = ttnn.DRAM_MEMORY_CONFIG
    program_config = None
    if shard_grid[0] is not None:
        core_grid = ttnn.CoreGrid(x=shard_grid[0], y=shard_grid[1])
        memory_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(shard_grid[0] - 1, shard_grid[1] - 1))}
                ),
                [
                    nearest_32(x_tt.padded_shape[1] / shard_grid[1]),
                    nearest_32(x_tt.padded_shape[2] / shard_grid[0]),
                ],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        )
        output_memory_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(shard_grid[0] - 1, shard_grid[1] - 1))}
                ),
                [nearest_32(x_tt.padded_shape[1] / shard_grid[1]), nearest_32(wqkv_tt.padded_shape[1] / shard_grid[0])],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        )
        program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(x=shard_grid[0], y=shard_grid[1]),
            in0_block_w=4,
            out_subblock_h=1,
            out_subblock_w=1,
            per_core_M=M // shard_grid[1] // 32,
            per_core_N=K // shard_grid[0] // 32,
            transpose_mcast=False,
            fused_activation=None,
            fuse_batch=False,
        )
        x_tt = ttnn.to_memory_config(x_tt, memory_config)

    x_qkv_tt = sd_linear(
        x_tt, linear_params, core_grid=core_grid, memory_config=output_memory_config, program_config=program_config
    )

    q_tt, k_tt, v_tt = ttnn.transformer.split_query_key_value_and_split_heads(
        x_qkv_tt, num_heads=n_local_heads, transpose_key=False, memory_config=output_memory_config
    )

    q_tt = sd_rms_norm(q_tt, norm_q_params, deallocate=True)
    k_tt = sd_rms_norm(k_tt, norm_k_params, deallocate=True)

    q_back = ttnn.to_torch(q_tt)
    k_back = ttnn.to_torch(k_tt)
    v_back = ttnn.to_torch(v_tt)

    assert_quality(q, q_back)
    assert_quality(k, k_back)
    assert_quality(v, v_back)


def test_ring_joint_attention():
    pass


def test_ff():
    pass
