# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import tempfile
from pathlib import Path

import pytest
import torch
from loguru import logger
from transformers import AutoConfig

import ttnn
from models.demos.deepseek_v3.tt.mla_1d import MLA_1D
from models.demos.deepseek_v3.utils.run_config import create_run_config

# Import from local reference files instead of HuggingFace
from models.demos.deepseek_v3_impl.model import MLA, ModelArgs, precompute_freqs_cis
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
    config.num_hidden_layers = 1  # Reduce layers for testing
    return config


@pytest.fixture
def reference(hf_config):
    """Get the actual DeepSeek MLP model using local implementation."""

    config_path = "models/demos/deepseek_v3_impl/configs/config_671B.json"
    with open(config_path) as f:
        model_args = ModelArgs(**json.load(f))
    return model_args, MLA(model_args)


def get_mesh_device():
    """Fixture to provide mesh device configuration."""
    mesh_device = os.environ.get("MESH_DEVICE", "N150")
    mesh_config = {
        "N150": (1, 1),
        "N300": (1, 2),
        "T3K": (1, 8),
        "TG": (8, 4),
    }.get(mesh_device, (1, ttnn.get_num_devices()))
    return mesh_config


# Unit Tests
@pytest.mark.parametrize(
    "mesh_device",
    [
        get_mesh_device(),
    ],
    indirect=True,
)
def test_convert_weights(reference, hf_config, temp_dir, mesh_device):
    """Test that weights are correctly converted to TTNN format."""

    _, reference_model = reference

    output_path = temp_dir

    logger.info(f"Converting weights for MLA_1D to {output_path}")

    # Convert weights - now returns weight_config
    weight_config = MLA_1D.convert_weights(hf_config, reference_model.state_dict(), output_path, mesh_device)

    assert "wq_a" in weight_config
    assert "wq_b" in weight_config
    assert "wkv_a" in weight_config
    assert "wkv_b1" in weight_config
    assert "wkv_b2" in weight_config
    assert "wo" in weight_config


@pytest.mark.parametrize(
    "mesh_device",
    [get_mesh_device()],
    indirect=True,
)
def test_decode_config_generation(hf_config, mesh_device):
    """Test decode config generation."""

    logger.info(f"Generating decode config for MLA_1D")
    model_config = MLA_1D.decode_model_config(hf_config, mesh_device)

    assert model_config["mode"] == "decode"


@pytest.mark.parametrize(
    "mesh_device",
    [get_mesh_device()],
    indirect=True,
)
def test_run_config_creation(reference, hf_config, temp_dir, mesh_device):
    """Test creating runtime config from ModelConfig and weights."""
    logger.info(f"Creating run config for MLA_1D")

    # Get state dict from actual model - pass directly to convert_weights
    _, reference_model = reference
    hf_state_dict = reference_model.state_dict()

    # First convert weights and get weight_config
    weights_path = temp_dir
    logger.info(f"Converting weights for MLA_1D to {weights_path}")
    weight_config = MLA_1D.convert_weights(hf_config, hf_state_dict, weights_path, mesh_device)

    # Generate model config
    model_config = MLA_1D.decode_model_config(hf_config, mesh_device)

    # Create RunConfig using both weight_config and model_config
    run_config = create_run_config(model_config, weight_config, mesh_device)

    # Verify mode is accessible
    assert run_config["mode"] == "decode"


# Integration Tests
@pytest.mark.parametrize(
    "mesh_device",
    [get_mesh_device()],
    indirect=True,
)
@pytest.mark.parametrize(
    "mode, seq_len, batch_size",
    [
        ("decode", 1024, 2),
    ],
)
def test_forward_pass(
    mode,
    seq_len,
    batch_size,
    reference,
    hf_config,
    temp_dir,
    mesh_device,
):
    reference_args, reference_model = reference

    ############################
    ### Set up configs
    ############################
    num_devices = mesh_device.get_num_devices()
    TG_GRID = (8, 4)  # TP, DP

    weights_path = temp_dir
    logger.info(f"Converting weights for MLA_1D to {weights_path}")
    weight_config = MLA_1D.convert_weights(hf_config, reference_model.state_dict(), weights_path, mesh_device)

    if mode == "prefill":
        model_config = MLA_1D.prefill_model_config(hf_config, mesh_device)
    else:
        model_config = MLA_1D.decode_model_config(hf_config, mesh_device)

    run_config = create_run_config(model_config, weight_config, mesh_device)

    ############################
    ### Torch inputs
    ############################
    start_pos = seq_len
    freqs_cis = precompute_freqs_cis(reference_args)[start_pos, :]

    if mode == "prefill":
        torch_input = torch.randn(batch_size, seq_len, hf_config.hidden_size)
    else:
        torch_input = torch.randn(batch_size, 1, hf_config.hidden_size)
    torch_input = torch_input.to(dtype=torch.bfloat16)

    ############################
    ### Torch reference
    ############################
    reference_output = reference_model(
        torch_input,
        start_pos=start_pos,
        freqs_cis=freqs_cis,
        mask=None,
    )

    ############################
    ### TTNN inputs
    ############################
    torch_input = torch_input.permute(1, 0, 2).unsqueeze(0)

    # if num_devices == 1:
    #     torch_input = torch_input[..., :torch_input.shape[-1] // TG_GRID[0]]

    tt_input = ttnn.from_torch(
        torch_input,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(-1, None), mesh_shape=list(mesh_device.shape)),
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
    )

    ############################
    ### TTNN forward pass
    ############################
    tt_mla = MLA_1D(hf_config, mesh_device)

    tt_output = tt_mla.forward(tt_input, run_config, mesh_device)
    tt_output_torch = ttnn.to_torch(tt_output)

    if mode == "prefill":
        pass
    else:
        tt_output_torch = tt_output_torch.squeeze(1).permute(1, 0, 2)

    ############################
    ### Validation
    ############################
    pcc_required = 0.98  # Slightly lower due to bfloat conversions
    passing, pcc_message = comp_pcc(reference_output, tt_output_torch, pcc_required)

    logger.info(f"Mode: {mode}, Seq len: {seq_len}")
    logger.info(f"PCC: {pcc_message}")

    assert passing, f"MLA output does not meet PCC requirement {pcc_required}: {pcc_message}"


if __name__ == "__main__":
    pytest.main([__file__])
