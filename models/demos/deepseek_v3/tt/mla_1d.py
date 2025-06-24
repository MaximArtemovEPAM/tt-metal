# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0


import torch

import ttnn
from models.demos.deepseek_v3.tt.abstract_module import AbstractModule
from models.demos.deepseek_v3.utils.config_helpers import TILE_SIZE


class MLA_1D(AbstractModule):
    """ """

    MAX_BATCH_SIZE = TILE_SIZE

    @staticmethod
    def convert_weights(hf_config, state_dict, output_path, mesh_device):
        """Convert PyTorch weights to TTNN format for 1D tensor parallelism.

        Args:
            hf_config: HuggingFace model configuration object
            state_dict: PyTorch state dict for this layer
            output_path: Path to save converted weights
            mesh_device: TTNN mesh device

        Returns:
            Dict mapping operation names to their TTNN weight file paths
        """

        weight_config = {}

        dim = hf_config.hidden_size
        hidden_dim = hf_config.intermediate_size
        num_heads = hf_config.num_attention_heads
        kv_lora_rank = hf_config.kv_lora_rank
        qk_nope_head_dim = hf_config.qk_nope_head_dim
        v_head_dim = hf_config.v_head_dim

        num_devices = mesh_device.get_num_devices()

        TG_GRID = (8, 4)  # TP, DP

        def add_weight_config(
            torch_weight,
            our_name,
            kwarg_name,
            dtype,
            mem_config,
            layout,
            mesh_mapper,
        ):
            """Helper function to convert and save weights, updating weight_config."""
            ttnn_weight = ttnn.as_tensor(
                torch_weight,
                dtype=dtype,
                device=mesh_device,
                mesh_mapper=mesh_mapper,
                layout=layout,
                memory_config=mem_config,
            )
            ttnn_weight = ttnn.unsqueeze_to_4D(ttnn_weight)

            weight_file_path = output_path / f"{our_name}.{kwarg_name}.weight"
            ttnn.dump_tensor(weight_file_path, ttnn_weight)
            ttnn.deallocate(ttnn_weight)

            # Add to weight config
            weight_config[our_name] = {kwarg_name: str(weight_file_path)}

        hf_ttnn_name_mapping = {
            "q_a_proj": "wq_a",
            "q_b_proj": "wq_b",
            "kv_a_proj_with_mqa": "wkv_a",
            "kv_b_proj": "wkv_b",
            "o_proj": "wo",
        }

        # wq_a
        hf_name = "q_a_proj"
        our_name = hf_ttnn_name_mapping[hf_name]
        torch_weight = state_dict[f"{our_name}.weight"]
        torch_weight = torch.transpose(torch_weight, -2, -1)

        if num_devices == 1:
            torch_weight = torch_weight[: torch_weight.shape[0] // TG_GRID[0], :]

        add_weight_config(
            torch_weight,
            our_name,
            "input_tensor_b",
            dtype=ttnn.bfloat8_b,
            mem_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                mesh_device,
                dims=[-2, None],
                mesh_shape=list(mesh_device.shape),
            ),
        )

        # wq_b
        hf_name = "q_b_proj"
        our_name = hf_ttnn_name_mapping[hf_name]
        torch_weight = state_dict[f"{our_name}.weight"]
        torch_weight = torch.transpose(torch_weight, -2, -1)

        if num_devices == 1:
            torch_weight = torch_weight[:, : torch_weight.shape[1] // TG_GRID[0]]

        add_weight_config(
            torch_weight,
            our_name,
            "input_tensor_b",
            dtype=ttnn.bfloat8_b,
            mem_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                mesh_device,
                dims=[-1, None],
                mesh_shape=list(mesh_device.shape),
            ),
        )

        # wkv_a
        hf_name = "kv_a_proj_with_mqa"
        our_name = hf_ttnn_name_mapping[hf_name]
        torch_weight = state_dict[f"{our_name}.weight"]
        torch_weight = torch.transpose(torch_weight, -2, -1)

        if num_devices == 1:
            torch_weight = torch_weight[: torch_weight.shape[0] // TG_GRID[0], :]

        add_weight_config(
            torch_weight,
            our_name,
            "input_tensor_b",
            dtype=ttnn.bfloat8_b,
            mem_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                mesh_device,
                dims=[-2, None],
                mesh_shape=list(mesh_device.shape),
            ),
        )

        # wkv_b1
        hf_name = "kv_b_proj"
        our_name = hf_ttnn_name_mapping[hf_name]
        torch_weight = state_dict[f"{our_name}.weight"]

        # This weight needs to be split
        torch_weight = torch_weight.view(kv_lora_rank, num_heads * (qk_nope_head_dim + v_head_dim))
        torch_weight = torch_weight.reshape(num_heads, -1, kv_lora_rank)

        torch_weight_k = torch_weight[:, :qk_nope_head_dim, :]  # [num_heads, qk_nope_head_dim, kv_lora_rank]
        torch_weight_v = torch_weight[:, qk_nope_head_dim:, :].transpose(
            -2, -1
        )  # [num_heads, kv_lora_rank, v_head_dim]

        if num_devices == 1:
            torch_weight_k = torch_weight_k[: torch_weight_k.shape[0] // TG_GRID[0], ...]
            torch_weight_v = torch_weight_v[: torch_weight_v.shape[0] // TG_GRID[0], ...]

        add_weight_config(
            torch_weight_k,
            our_name + "1",
            "input_tensor_b",
            dtype=ttnn.bfloat8_b,
            mem_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                mesh_device,
                dims=[-3, None],
                mesh_shape=list(mesh_device.shape),
            ),
        )

        add_weight_config(
            torch_weight_v,
            our_name + "2",
            "input_tensor_b",
            dtype=ttnn.bfloat8_b,
            mem_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                mesh_device,
                dims=[-3, None],
                mesh_shape=list(mesh_device.shape),
            ),
        )

        # wo
        hf_name = "o_proj"
        our_name = hf_ttnn_name_mapping[hf_name]
        torch_weight = state_dict[f"{our_name}.weight"]
        torch_weight = torch.transpose(torch_weight, -2, -1)

        if num_devices == 1:
            torch_weight = torch_weight[:, : torch_weight.shape[1] // TG_GRID[0]]

        add_weight_config(
            torch_weight,
            our_name,
            "input_tensor_b",
            dtype=ttnn.bfloat8_b,
            mem_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                mesh_device,
                dims=[-1, None],
                mesh_shape=list(mesh_device.shape),
            ),
        )

        return weight_config

    @staticmethod
    def prefill_model_config(hf_config, mesh_device):
        """Prefill model config for an MLP with 1D tensor parallelism.

        Args:
            hf_config: HuggingFace model configuration object
            mesh_device: TTNN mesh device

        Returns:
            Dict containing operator configurations for prefill mode
        """
        # Extract dimensions from HF config
        dim = hf_config.hidden_size
        num_devices = mesh_device.get_num_devices()

        config = {"mode": "prefill"}

        # TODO: Need to implement

        return config

    @staticmethod
    def decode_model_config(hf_config, mesh_device):
        """Generate decode operator configuration for this MLP layer.

        Args:
            hf_config: HuggingFace model configuration object
            mesh_device: TTNN mesh device

        Returns:
            Dict containing operator configurations for decode mode
        """
        # Extract dimensions from HF config
        dim = hf_config.hidden_size
        hidden_dim = hf_config.intermediate_size
        num_devices = mesh_device.get_num_devices()

        config = {"mode": "decode"}

        config["wq_a"] = {
            "memory_config": ttnn.DRAM_MEMORY_CONFIG,
            "program_config": None,
        }

        config["wq_b"] = {
            "memory_config": ttnn.DRAM_MEMORY_CONFIG,
            "program_config": None,
        }

        config["wkv_a"] = {
            "memory_config": ttnn.DRAM_MEMORY_CONFIG,
            "program_config": None,
        }

        config["wkv_b1"] = {
            "memory_config": ttnn.DRAM_MEMORY_CONFIG,
            "program_config": None,
        }

        config["wkv_b2"] = {
            "memory_config": ttnn.DRAM_MEMORY_CONFIG,
            "program_config": None,
        }

        config["wo"] = {
            "memory_config": ttnn.DRAM_MEMORY_CONFIG,
            "program_config": None,
        }

        return config

    def __init__(self, mesh_device, hf_config):
        """Initialize the MLP with the given mesh device and HuggingFace config

        We use this to define lambdas for dynamic prefill program configs that
        will be used in the forward pass; putting dynamic functions in the
        model config returns is discouraged as we will not be able to convert
        them to JSON in the future.

        Also keeping them here reminds us that this ugliness is real and perhaps
        we should find a way to make it beautiful and fast instead.

        Args:
            mesh_device: TTNN mesh device
            hf_config: HuggingFace model configuration object

        """
        super().__init__(mesh_device, hf_config)

        dim = hf_config.hidden_size
        hidden_dim = hf_config.intermediate_size
        num_devices = mesh_device.get_num_devices()

        # TODO: Set up RoPE module

        # TODO: Set up KVPE Cache

    def forward(self, x, cfg, mesh_device):
        """Decode is very straightforward but prefill reshapes and has dynamic program configs
        so we implement forward as two functions for clarity.
        """
        if cfg["mode"] == "decode":
            return self._forward_decode(x, cfg, mesh_device)
        else:
            assert cfg["mode"] == "prefill"
            return self._forward_prefill(x, cfg, mesh_device)

    def _forward_decode(self, x, cfg, mesh_device):
        """Straightforward forward pass for decode mode"""
        # Gate and up projections
        q = ttnn.linear(x, **cfg["w1"])

        return q

    def _forward_prefill(self, x, cfg, mesh_device):
        """Forward pass of the MLP.

        Prefill mode we reshape to respect cfg["max_rows"] and generate program configs from the seq-len lambda.

        Args:
            x: Input tensor
            cfg: RunConfig containing weights and op configurations
            mesh_device: TTNN mesh device for multi-device operations

        Returns:
            Output tensor after MLP computation
        """

        return x
