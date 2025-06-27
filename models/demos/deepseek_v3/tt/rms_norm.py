# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.demos.deepseek_v3.utils.abstract_module import AbstractModule
from models.demos.deepseek_v3.utils.config_dataclass import RMSNormConfig
from models.demos.deepseek_v3.utils.config_helpers import COMPUTE_KERNEL_CONFIG_HIFI2, TILE_SIZE, save_and_get_path


class RMSNorm(AbstractModule):
    """Distributed RMSNorm module with 1D tensor parallelism from TTT code.
    Uses DRAM-sharded weights split 1D across 8 wormholes of 8x4 mesh device"""

    @staticmethod
    def convert_weights(hf_config, state_dict, output_path, mesh_device, is_distributed=False):
        """DRAM-sharded weights split 1D across all wormholes

        Args:
            hf_config: HuggingFace model configuration object
            state_dict: PyTorch state dict for this layer
            output_path: Path to save converted weights
            mesh_device: TTNN mesh device

        Returns:
            Dict mapping operation names to their TTNN weight file paths
        """
        weight_config = {}

        # Get the embedding weight from the state dict (in the full model: model.embed_tokens.weight)
        torch_weight = state_dict["weight"]

        # Convert to TTNN tensor with 1D sharded across columns of mesh device
        # Reshape to tile width sticks for optimal performance
        torch_weight = torch_weight.reshape([1, 1, torch_weight.shape[-1] // TILE_SIZE, TILE_SIZE])
        ttnn_weight = ttnn.as_tensor(
            torch_weight,
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(None, 2), mesh_shape=list(mesh_device.shape))
            if is_distributed
            else ttnn.ReplicateTensorToMesh(mesh_device),
        )

        # Save to disk with standard naming - "rmsnorm" must match the op name used in the model config
        # so that RunConfig can populate it with the actual weight tensors at runtime
        weight_config["rmsnorm"] = {
            "weight": save_and_get_path(output_path / "rmsnorm.weight", ttnn_weight),
        }

        return weight_config

    @staticmethod
    def prefill_model_config(hf_config, mesh_device):
        """Prefill model config for an RMSNorm with 1D tensor parallelism.

        Args:
            hf_config: HuggingFace model configuration object
            mesh_device: TTNN mesh device

        Returns:
            Dict containing operator configurations for prefill mode
        """
        config = {"mode": "prefill"}

        # Embedding configuration for prefill mode
        config["rmsnorm"] = RMSNormConfig(
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.TILE_LAYOUT,
        )

        return config

    @staticmethod
    def decode_model_config(hf_config, mesh_device):
        """Generate decode operator configuration for this embedding layer.
        Same as prefill mode for this module.

        Args:
            hf_config: HuggingFace model configuration object
            mesh_device: TTNN mesh device

        Returns:
            Dict containing operator configurations for decode mode
        """
        config = {"mode": "decode"}
        # RMSNorm configuration for decode mode
        config["rmsnorm"] = RMSNormConfig(
            epsilon=hf_config.rms_norm_eps,
            compute_kernel_config=COMPUTE_KERNEL_CONFIG_HIFI2,
        )
        return config

    def __init__(self, hf_config, mesh_device, norm_category="decoder", is_distributed=False):
        """Initialize the rms norm with the given HuggingFace config and mesh device."""
        super().__init__(hf_config, mesh_device)
        self.norm_category = norm_category
        self.is_distributed = is_distributed
        if list(mesh_device.shape)[-1] == 1:
            # If mesh device is 1D, we cannot shard across it
            self.is_distributed = False
        if norm_category == "decoder":
            self.is_sharded_out = True
            self.output_memory_config = None
        else:
            self.is_sharded_out = False
            self.output_memory_config = ttnn.DRAM_MEMORY_CONFIG

        self.stats_memcfg = ttnn.DRAM_MEMORY_CONFIG
        if is_distributed and self.is_sharded_out:
            self.stats_memcfg = ttnn.create_sharded_memory_config(
                shape=[1, 1, TILE_SIZE, TILE_SIZE * list(mesh_device.shape)[-1]],
                core_grid=ttnn.CoreGrid(y=1, x=1),
                strategy=ttnn.ShardStrategy.WIDTH,
            )

    def forward(self, x, cfg, mesh_device):
        """Forward pass of the embedding.

        Args:
            x: Input tensor (token indices)
            cfg: RunConfig containing weights and op configurations
            mesh_device: TTNN mesh device for multi-device operations

        Returns:
            Output tensor after embedding lookup
        """
        program_config = None
        if self.is_sharded_out:
            grid_size_x = x.memory_config().shard_spec.grid.bounding_box().grid_size().x
            grid_size_y = x.memory_config().shard_spec.grid.bounding_box().grid_size().y
            shard_shape = x.memory_config().shard_spec.shape
            program_config = ttnn.LayerNormShardedMultiCoreProgramConfig(
                compute_with_storage_grid_size=(grid_size_x, grid_size_y),
                subblock_w=1,
                block_h=shard_shape[0] // TILE_SIZE,
                block_w=shard_shape[1] // TILE_SIZE,
                inplace=False,
            )
        if self.is_distributed:
            return self._distributed_rmsnorm(
                x,
                mesh_device=mesh_device,
                epsilon=cfg["rmsnorm"].epsilon,
                weight=cfg["rmsnorm"].weight,
                compute_kernel_config=cfg["rmsnorm"].compute_kernel_config,
                program_config=program_config,
                memory_config=self.stats_memcfg,
                output_dtype=ttnn.bfloat16,
            )
        else:
            return ttnn.rms_norm(
                x, **cfg["rmsnorm"], program_config=program_config, memory_config=self.output_memory_config
            )

    def _distributed_rmsnorm(
        self,
        inp,
        mesh_device,
        epsilon=None,
        weight=None,
        compute_kernel_config=None,
        program_config=None,
        memory_config=None,
        output_dtype=None,
    ):
        # Run distributed rmsnorm part 1
        tt_stats = ttnn.rms_norm_pre_all_gather(inp, program_config=program_config, dtype=output_dtype)
        # AllGather stats
        tt_stats = ttnn.all_gather(
            tt_stats,
            dim=3,
            num_links=1,
            cluster_axis=1,
            mesh_device=mesh_device,
            memory_config=self.stats_memcfg,
            topology=ttnn.Topology.Linear,
        )
        # Run distributed rmsnorm part 2
        tt_out = ttnn.rms_norm_post_all_gather(
            inp,
            epsilon=epsilon,
            weight=weight,
            program_config=program_config,
            stats=tt_stats,
            memory_config=self.output_memory_config,
            dtype=output_dtype,
        )
        tt_stats.deallocate(True)

        return tt_out
