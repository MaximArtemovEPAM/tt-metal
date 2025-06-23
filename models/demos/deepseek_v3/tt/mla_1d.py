# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import math

import torch

import ttnn
from models.demos.deepseek_v3.tt.abstract_module import AbstractModule
from models.demos.deepseek_v3.utils.config_helpers import (
    COMPUTE_KERNEL_CONFIG_HIFI2_FP16,
    COMPUTE_KERNEL_CONFIG_LOFI,
    TILE_SIZE,
    dram_shard_core_grid_for_k_and_n,
    dram_sharded_matmul_config,
    find_prefill_grid,
    matmul_config,
)
from models.utility_functions import is_blackhole, is_wormhole_b0


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
                torch_weight.unsqueeze(0).unsqueeze(0),
                dtype=dtype,
                device=mesh_device,
                mesh_mapper=mesh_mapper,
                layout=layout,
                memory_config=mem_config,
            )

            weight_file_path = output_path / f"{our_name}.{kwarg_name}.weight"
            ttnn.dump_tensor(weight_file_path, ttnn_weight)
            ttnn.deallocate(ttnn_weight)

            # Add to weight config
            weight_config[our_name] = {kwarg_name: str(weight_file_path)}

        hf_ttnn_name_mapping = {
            "q_a_proj": "wq_a",
            "q_a_layernorm": "wq_a_ln",
            "q_b_proj": "wq_b",
            "kv_a_proj_with_mqa": "wkv_a",
            "kv_a_layernorm": "wkv_a_ln",
            "kv_b_proj": "wkv_b",
            "o_proj": "wo",
        }

        # wq_a
        hf_name = "q_a_proj"
        our_name = hf_ttnn_name_mapping[hf_name]
        torch_weight = state_dict[f"{hf_name}.weight"]
        torch_weight = torch.transpose(torch_weight, -2, -1)

        # FIXME: Single device only for now
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
        torch_weight = state_dict[f"{hf_name}.weight"]
        torch_weight = torch.transpose(torch_weight, -2, -1)

        # FIXME: Single device only for now
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

        # # TODO: This should be handled by RMSNorm module??
        # # wq_a_ln
        # hf_name = "q_a_layernorm"
        # our_name = hf_ttnn_name_mapping[hf_name]
        # torch_weight = state_dict[f"{hf_name}.weight"]

        # add_weight_config(
        #     torch_weight,
        #     our_name,
        #     "input_tensor_b", # TODO: Check
        #     dtype=ttnn.bfloat16,
        #     mem_config=ttnn.DRAM_MEMORY_CONFIG,
        #     layout=ttnn.TILE_LAYOUT,
        #     mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        # )

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

        # Maximum rows to process at once in prefill mode
        config["max_rows"] = 512 if is_blackhole() else 1024

        # Program configs are dynamically generated in __init__ based on sequence length
        # See __init__ function for implementation details
        # Note: avoid the temptation to combine these - create_run_config will look for these names
        # and will fill them with the weights, so each op needs its own config entry.
        config["w1"] = {
            "memory_config": ttnn.DRAM_MEMORY_CONFIG,
            "compute_kernel_config": COMPUTE_KERNEL_CONFIG_HIFI2_FP16,
        }

        config["w3"] = {
            "memory_config": ttnn.DRAM_MEMORY_CONFIG,
            "compute_kernel_config": COMPUTE_KERNEL_CONFIG_HIFI2_FP16,
        }

        config["w2"] = {
            "memory_config": ttnn.DRAM_MEMORY_CONFIG,
            "compute_kernel_config": COMPUTE_KERNEL_CONFIG_HIFI2_FP16,
        }

        # Activation configurations
        config["mul"] = {
            "memory_config": ttnn.DRAM_MEMORY_CONFIG,
            "input_tensor_a_activations": [ttnn.UnaryOpType.SILU],
        }

        # All-reduce configuration for multi-device synchronization
        config["all_reduce"] = {
            "cluster_axis": 0,
            "dim": 3,
            "num_reduce_scatter_links": 1,
            "num_all_gather_links": 1,
            "topology": ttnn.Topology.Ring if num_devices == 8 else ttnn.Topology.Linear,
            "dtype": ttnn.bfloat8_b,
            "use_composite": dim >= 8192,  # Use composite for larger models
        }

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

        # Decode mode configurations
        config["w1"] = {
            "memory_config": ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            "program_config": dram_sharded_matmul_config(dim, hidden_dim, mesh_device),
            "compute_kernel_config": COMPUTE_KERNEL_CONFIG_LOFI,  # FP16 accumulation saves L1
        }

        config["w3"] = {
            "memory_config": ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            "program_config": dram_sharded_matmul_config(dim, hidden_dim, mesh_device),  # Same as w1
            "compute_kernel_config": COMPUTE_KERNEL_CONFIG_LOFI,
        }

        tile_padded_batch_rows = TILE_SIZE * int(math.ceil(MLA_1D.MAX_BATCH_SIZE / TILE_SIZE))
        mlp2_core_grid = dram_shard_core_grid_for_k_and_n(hidden_dim // num_devices, dim)
        config["w2_reshard"] = {
            "memory_config": ttnn.create_sharded_memory_config(
                (
                    tile_padded_batch_rows,
                    hidden_dim // num_devices // mlp2_core_grid.num_cores,
                ),
                mlp2_core_grid,
                ttnn.ShardStrategy.WIDTH,
                ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
        }
        config["w2"] = {
            "memory_config": ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            "program_config": dram_sharded_matmul_config(hidden_dim // num_devices, dim, mesh_device),
            "compute_kernel_config": COMPUTE_KERNEL_CONFIG_LOFI,
        }

        # Activation configurations
        config["mul"] = {
            "memory_config": ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            "input_tensor_a_activations": [ttnn.UnaryOpType.SILU],
        }

        # All-reduce configuration for multi-device synchronization
        config["all_reduce"] = {
            "cluster_axis": 0,
            "dim": 3,
            "num_reduce_scatter_links": 1,
            "num_all_gather_links": 1,
            "topology": ttnn.Topology.Ring if num_devices == 8 else ttnn.Topology.Linear,
            "dtype": ttnn.bfloat8_b,
            # FIXME: From tt_transformers/tt/mlp.py, surely >= and not ==?
            # FIXME: Why this value and not e.g. 7*1024?
            "use_composite": dim == 8192,  # Use composite for larger models
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
        super().__init__()

        prefill_rows = 8  # TODO if BH = 10, if wh = 8
        dim = hf_config.hidden_size
        hidden_dim = hf_config.intermediate_size
        num_devices = mesh_device.get_num_devices()

        mlp1_3_grid = find_prefill_grid(prefill_rows, dim // TILE_SIZE)
        mlp2_grid = find_prefill_grid(prefill_rows, hidden_dim // TILE_SIZE)

        # Using dram_shard_grid_width to ensure per_core_N matches DRAM shard width for P100, otherwise matmuls silently give bad PCC
        dram_shard_grid_width = 8 if is_wormhole_b0() else mesh_device.dram_grid_size().x  # 7 for P100, 8 for P150

        n_w1_w3 = hidden_dim // num_devices  # weights are 1d sharded across devices
        self.w1_w3_pc = lambda seq_len: matmul_config(
            m=seq_len,
            k=dim // num_devices,
            n=n_w1_w3,
            grid_size=mlp1_3_grid,
            per_core_N=math.ceil(n_w1_w3 / (TILE_SIZE * dram_shard_grid_width)),
        )

        n_w2 = dim
        self.w2_pc = lambda seq_len: matmul_config(
            m=seq_len,
            k=hidden_dim // num_devices,
            n=n_w2,
            grid_size=mlp2_grid,
            per_core_N=math.ceil(n_w2 / (TILE_SIZE * dram_shard_grid_width)),
        )

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
        w1_out = ttnn.linear(x, **cfg["w1"])
        w3_out = ttnn.linear(x, **cfg["w3"])
        ttnn.deallocate(x)

        # Apply activation and multiply
        activated = ttnn.mul(w1_out, w3_out, **cfg["mul"])
        ttnn.deallocate(w1_out)
        ttnn.deallocate(w3_out)

        # w2 may use a different core grid, this is a no-op if they already match
        activated = ttnn.to_memory_config(activated, **cfg["w2_reshard"])

        # Down projection
        output = ttnn.linear(activated, **cfg["w2"])
        ttnn.deallocate(activated)

        # All-reduce across devices to sum partial results
        return ttnn.all_reduce(output, mesh_device=mesh_device, **cfg["all_reduce"])

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
        seq_len = x.shape[-2]

        # Handle large sequence lengths
        if seq_len > cfg["max_rows"]:
            # Reshape input to process in chunks
            original_shape = x.shape
            num_chunks = seq_len // cfg["max_rows"]
            x = ttnn.reshape(x, [1, num_chunks, cfg["max_rows"], -1])

            # Get current sequence length for program config
            current_seq_len = x.shape[-2]
        else:
            current_seq_len = seq_len
            original_shape = None

        # Gate and up projections with dynamic program configs
        w1_out = ttnn.linear(x, program_config=self.w1_w3_pc(current_seq_len), **cfg["w1"])
        w3_out = ttnn.linear(x, program_config=self.w1_w3_pc(current_seq_len), **cfg["w3"])
        ttnn.deallocate(x)

        # Apply activation and multiply
        activated = ttnn.mul(w1_out, w3_out, **cfg["mul"])
        ttnn.deallocate(w1_out)
        ttnn.deallocate(w3_out)

        # Down projection with dynamic program configs
        output = ttnn.linear(activated, program_config=self.w2_pc(current_seq_len), **cfg["w2"])
        ttnn.deallocate(activated)

        # All-reduce across devices to sum partial results
        output = ttnn.all_reduce(output, mesh_device=mesh_device, **cfg["all_reduce"])

        # Reshape output to expected format if we reshaped the input
        if original_shape is not None:
            output = ttnn.reshape(output, original_shape)

        return output
