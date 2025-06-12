import torch
import torch.nn.functional as F
import pytest
import math
import ttnn

from models.utility_functions import nearest_32
from models.experimental.stable_diffusion_35_large.tt.fun_linear import TtLinearParameters, sd_linear
from models.experimental.stable_diffusion_35_large.tt.fun_normalization import TtRmsNormParameters, sd_rms_norm
from models.experimental.stable_diffusion_35_large.tt.utils import assert_quality


def rms_norm(x, w, eps=1e-6):
    rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + eps)
    return x / rms * w


def largest_factor(n: int, max_factor: int = None) -> int:
    """Return the largest proper factor of |n| (i.e. the greatest divisor < |n|).
    For primes the result is 1.  Raises ValueError for n = 0."""
    if n == 0:
        raise ValueError("Zero has infinitely many divisors.")
    n = abs(n)
    if n <= 1:
        return n  # 1 and –1 have no proper factor

    # The largest factor equals n // smallest_prime_factor.
    limit = math.isqrt(n)
    for d in range(2, limit + 1):
        if n % d == 0:
            if max_factor is not None and n // d < max_factor:
                return n // d
    return 1  # n is prime


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
    "M, K, N",
    [
        [1024, 2560, 1920],
        [352, 2560, 1920],
        [1024, 2560, 640],
        [352, 2560, 640],
        [1024, 2560, 2560],
        [352, 2560, 2560],
    ],
    ids=[
        "attn_proj_spatial_tp4",
        "attn_proj_prompt_tp4",
        "attn_out_proj_spatial_tp4",
        "attn_out_proj_prompt_tp4",
        "mlp_spatial_tp4",
        "mlp_prompt_tp4",
    ],
)
def test_sweep_dram_matmul(device, use_program_cache, M, K, N):
    """
    Sweep a reasonable set of program configs for 2d dram matmul
    """
    x = torch.randn(M, K)
    wqkv = torch.randn(K, N)
    bqkv = torch.randn(1, N)
    x_qkv = F.linear(x, wqkv.T, bqkv)

    x_tt = ttnn.from_torch(x.unsqueeze(0), device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    wqkv_tt = ttnn.from_torch(wqkv, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    bqkv_tt = ttnn.from_torch(bqkv, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)

    linear_params = TtLinearParameters(weight=wqkv_tt, bias=bqkv_tt)

    core_grid = device.core_grid
    output_memory_config = ttnn.DRAM_MEMORY_CONFIG
    TILE_SIZE = 32
    M_tiles = x_tt.padded_shape[1] // TILE_SIZE
    K_tiles = x_tt.padded_shape[2] // TILE_SIZE
    N_tiles = wqkv_tt.padded_shape[1] // TILE_SIZE
    per_core_M = math.ceil(M_tiles / core_grid.y)
    per_core_N = math.ceil(N_tiles / core_grid.x)
    for in0_block_w in [k for k in range(1, K_tiles + 1) if K_tiles % k == 0]:
        for out_subblock_h in [1, 2, 4]:
            for out_subblock_w in [1, 2, 4]:
                if out_subblock_h * out_subblock_w > 8:
                    continue
                if out_subblock_h > per_core_M or out_subblock_w > per_core_N:
                    continue
                program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                    compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
                    in0_block_w=in0_block_w,
                    out_subblock_h=out_subblock_h,
                    out_subblock_w=out_subblock_w,
                    per_core_M=per_core_M,
                    per_core_N=per_core_N,
                    transpose_mcast=False,
                    fused_activation=None,
                )
                print(f"in0_block_w: {in0_block_w}, out_subblock_h: {out_subblock_h}, out_subblock_w: {out_subblock_w}")
                try:
                    x_qkv_tt = sd_linear(
                        x_tt, linear_params, memory_config=output_memory_config, program_config=program_config
                    )

                    qkv_back = ttnn.to_torch(x_qkv_tt)

                    assert_quality(x_qkv, qkv_back)
                except Exception as e:
                    print(f"exception")


@pytest.mark.parametrize(
    "M, K, N",
    [
        [1024, 2560, 1920],
        [352, 2560, 1920],
        [1024, 2560, 640],
        [352, 2560, 640],
        [1024, 2560, 2560],
        [352, 2560, 2560],
    ],
    ids=[
        "attn_proj_spatial_tp4",
        "attn_proj_prompt_tp4",
        "attn_out_proj_spatial_tp4",
        "attn_out_proj_prompt_tp4",
        "mlp_spatial_tp4",
        "mlp_prompt_tp4",
    ],
)
@pytest.mark.parametrize("shard_grid", [(i, j) for i in range(1, 9) for j in range(1, 9)])
def test_sweep_block_sharded_matmul(device, use_program_cache, M, K, N, shard_grid):
    """
    For each shard grid, sweep a reasonable set of program configs for 2d block sharded matmul
    """
    x = torch.randn(M, K)
    wqkv = torch.randn(K, N)
    bqkv = torch.randn(1, N)
    x_qkv = F.linear(x, wqkv.T, bqkv)

    x_tt = ttnn.from_torch(x.unsqueeze(0), device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    wqkv_tt = ttnn.from_torch(wqkv, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    bqkv_tt = ttnn.from_torch(bqkv, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)

    linear_params = TtLinearParameters(weight=wqkv_tt, bias=bqkv_tt)

    in_shard_height = nearest_32(x_tt.padded_shape[1] / shard_grid[1])
    in_shard_width = nearest_32(x_tt.padded_shape[2] / shard_grid[0])
    grid_core_set = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(shard_grid[0] - 1, shard_grid[1] - 1))}
    )
    memory_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            grid_core_set,
            [
                in_shard_height,
                in_shard_width,
            ],
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )
    out_shard_width = nearest_32(wqkv_tt.padded_shape[1] / shard_grid[0])
    output_memory_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            grid_core_set,
            [in_shard_height, out_shard_width],
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )

    x_tt = ttnn.to_memory_config(x_tt, memory_config)

    per_core_K = in_shard_width // 32
    per_core_M = in_shard_height // 32
    per_core_N = out_shard_width // 32
    for in0_block_w in [k for k in range(1, per_core_K + 1) if per_core_K % k == 0]:
        for out_subblock_h in [1, 2, 4]:
            for out_subblock_w in [1, 2, 4]:
                if out_subblock_h * out_subblock_w > 8:
                    continue
                if out_subblock_h > per_core_M or out_subblock_w > per_core_N:
                    continue
                program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                    compute_with_storage_grid_size=ttnn.CoreCoord(shard_grid[0], shard_grid[1]),
                    in0_block_w=in0_block_w,
                    out_subblock_h=out_subblock_h,
                    out_subblock_w=out_subblock_w,
                    per_core_M=per_core_M,
                    per_core_N=per_core_N,
                    transpose_mcast=False,
                    fused_activation=None,
                )

                print(f"in0_block_w: {in0_block_w}, out_subblock_h: {out_subblock_h}, out_subblock_w: {out_subblock_w}")
                try:
                    x_qkv_tt = sd_linear(
                        x_tt, linear_params, memory_config=output_memory_config, program_config=program_config
                    )

                    qkv_back = ttnn.to_torch(x_qkv_tt)

                    assert_quality(x_qkv, qkv_back)
                except Exception as e:
                    print(f"exception")


@pytest.mark.parametrize(
    "M, K, N, n_local_heads",
    [
        [1024, 2560, 1920, 10],
        [352, 2560, 1920, 10],
    ],
    ids=["spatial_tp4", "prompt_tp4"],
)
@pytest.mark.parametrize("shard_grid", [(None, None)] + [(i, j) for i in range(1, 9) for j in range(1, 9)])
def test_attention_proj_graph(device, use_program_cache, M, K, N, n_local_heads, shard_grid):
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
        core_grid = None
        in_shard_height = nearest_32(x_tt.padded_shape[1] / shard_grid[1])
        in_shard_width = nearest_32(x_tt.padded_shape[2] / shard_grid[0])
        grid_core_set = ttnn.CoreRangeSet(
            {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(shard_grid[0] - 1, shard_grid[1] - 1))}
        )
        memory_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                grid_core_set,
                [
                    in_shard_height,
                    in_shard_width,
                ],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        )
        out_shard_width = nearest_32(wqkv_tt.padded_shape[1] / shard_grid[0])
        output_memory_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                grid_core_set,
                [in_shard_height, out_shard_width],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        )
        block_w_tiles_per_shard = in_shard_width // 32

        program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(shard_grid[0], shard_grid[1]),
            in0_block_w=largest_factor(block_w_tiles_per_shard, max_factor=8),
            out_subblock_h=1,
            out_subblock_w=4,
            per_core_M=in_shard_height // 32,
            per_core_N=out_shard_width // 32,
            transpose_mcast=False,
            fused_activation=None,
        )
        x_tt = ttnn.to_memory_config(x_tt, memory_config)

        # # num_heads x seq_len x head_dim
        # heads_memory_config = ttnn.MemoryConfig(
        #     ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        #     ttnn.BufferType.L1,
        #     ttnn.ShardSpec(

        #     )
        # )

    print(f"shard_grid: {shard_grid}")
    print(f"input_memory_config: {memory_config}")
    print(f"output_memory_config: {output_memory_config}")
    print(f"program_config: {program_config}")
    x_qkv_tt = sd_linear(
        x_tt, linear_params, core_grid=core_grid, memory_config=output_memory_config, program_config=program_config
    )

    x_qkv_tt = ttnn.to_memory_config(x_qkv_tt, ttnn.DRAM_MEMORY_CONFIG)

    q_tt, k_tt, v_tt = ttnn.transformer.split_query_key_value_and_split_heads(
        x_qkv_tt, num_heads=n_local_heads, transpose_key=False
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
