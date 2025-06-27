# models/demos/deepseek_v3/tests/test_embedding_1d.py
# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import os
import tempfile
from pathlib import Path

import pytest
import torch
from loguru import logger
from transformers import AutoConfig

import ttnn

# Import from local reference files instead of HuggingFace
from models.demos.deepseek_v3.reference.rmsnorm import DeepseekV3RMSNorm
from models.demos.deepseek_v3.tt.rms_norm import RMSNorm
from models.demos.deepseek_v3.utils.run_config import create_run_config
from models.utility_functions import comp_pcc


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def hf_config():
    """Load DeepSeek config for testing"""
    config = AutoConfig.from_pretrained("deepseek-ai/DeepSeek-R1-0528", trust_remote_code=True)
    return config


@pytest.fixture
def reference_model(hf_config):
    """Get the actual DeepSeek RMSNorm model using local implementation."""
    return DeepseekV3RMSNorm(
        hidden_size=hf_config.hidden_size,
        eps=hf_config.rms_norm_eps,
    )


@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (4, 8)}.get(
            os.environ.get("MESH_DEVICE"), (1, ttnn.get_num_devices())
        )
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "mode, batch, seq_len, is_distributed, is_in_sharded",
    [
        ("decode", 128, 1, True, True),  # Batch decode with distributed and sharded inputs
        ("decode", 32, 1, False, True),  # Batch decode with non-dtributed but sharded inputs
    ],
)
def test_rmsnorm_forward_pass(
    mode,
    batch,
    seq_len,
    is_distributed,
    is_in_sharded,
    reference_model,
    hf_config,
    temp_dir,
    mesh_device,
):
    """Test embedding forward pass against reference model."""
    # Setup: Convert weights and get weight_config
    hf_state_dict = reference_model.state_dict()
    weight_config = RMSNorm.convert_weights(
        hf_config, hf_state_dict, temp_dir, mesh_device, is_distributed=is_distributed
    )

    # Generate appropriate config
    if mode == "prefill":
        model_config = RMSNorm.prefill_model_config(hf_config, mesh_device)
    else:
        model_config = RMSNorm.decode_model_config(hf_config, mesh_device)

    # Create RunConfig using both weight_config and model_config
    run_config = create_run_config(model_config, weight_config, mesh_device)

    # Instantiate the model
    tt_rmsnorm = RMSNorm(hf_config, mesh_device, norm_category="decoder", is_distributed=is_distributed)

    # Prepare input - in decode mode batch is placed into seq_len dimension anyway
    if mode == "decode":
        torch_input = torch.randn(1, 1, batch, hf_config.hidden_size)
    else:
        torch_input = torch.randn(1, batch, seq_len, hf_config.hidden_size)

    tt_input = ttnn.from_torch(
        torch_input,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(-2, -1), mesh_shape=list(mesh_device.shape))
        if is_distributed
        else ttnn.ReplicateTensorToMesh(mesh_device),
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
    )
    if is_in_sharded:
        shard_core_grid = ttnn.CoreGrid(x=4, y=7)
        sharded_memory_config = ttnn.create_sharded_memory_config(
            shape=(
                tt_input.shape[0] * tt_input.shape[1] * tt_input.shape[2],
                tt_input.shape[3] // shard_core_grid.num_cores,
            ),
            core_grid=shard_core_grid,
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        tt_input = ttnn.to_memory_config(tt_input, memory_config=sharded_memory_config)

    # TTNN forward pass
    tt_output = tt_rmsnorm.forward(tt_input, run_config, mesh_device)
    if is_distributed:
        tt_output_torch = ttnn.to_torch(
            tt_output,
            mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(-2, -1), mesh_shape=list(mesh_device.shape)),
        )
    else:
        tt_output_torch = ttnn.to_torch(ttnn.get_device_tensors(tt_output)[0])

    tt_output_torch = tt_output_torch[..., :batch, :]
    # Reference forward pass
    reference_output = reference_model(torch_input)

    # Compare outputs
    pcc_required = 0.99  # Embedding should be exact match (just lookup)
    passing, pcc_message = comp_pcc(reference_output, tt_output_torch, pcc_required)

    logger.info(f"Mode: {mode}, Seq len: {seq_len}")
    logger.info(f"Reference shape: {reference_output.shape}")
    logger.info(f"TTNN output shape: {tt_output_torch.shape}")
    logger.info(f"PCC: {pcc_message}")

    assert passing, f"Embedding output does not meet PCC requirement {pcc_required}: {pcc_message}"

    # Cleanup
    ttnn.deallocate(tt_output)
    ttnn.deallocate(tt_input)


if __name__ == "__main__":
    pytest.main([__file__])
