# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.deepseek_v3_impl.model import ModelArgs, precompute_freqs_cis
from models.tt_transformers.tt.common import gather_cos_sin
from models.utility_functions import nearest_32


def get_rot_transformation_mat():
    # ROPE op uses a single tile
    dhead = 32
    rot_emb_matrix = torch.zeros(1, 1, dhead, dhead)
    rot_emb_matrix[..., torch.arange(0, dhead, 2), torch.arange(1, dhead, 2)] = 1
    rot_emb_matrix[..., torch.arange(1, dhead, 2), torch.arange(0, dhead, 2)] = -1
    return rot_emb_matrix


def get_cos_sin_matrix(reference_args: ModelArgs):
    freqs_cis = precompute_freqs_cis(reference_args)
    freqs = freqs_cis.angle()
    cos, sin = torch.cos(freqs), torch.sin(freqs)
    cos, sin = gather_cos_sin(torch.arange(reference_args.max_seq_len), cos, sin)

    return cos, sin


class RotarySetup(LightweightModule):
    """
    Class to set up rotary positional embeddings for DeepSeekV3 models.

    Duplicate + changes from models/tt_transformers/tt/rope.py
    """

    def __init__(
        self,
        device,
        batch_size: int,
        reference_args,
        datatype=ttnn.bfloat16,
    ):
        super().__init__()

        self.batch_size = batch_size
        self.dim = reference_args.qk_rope_head_dim
        self.device = device
        self.num_devices = device.get_num_devices()

        if self.num_devices == 32:
            assert False, "This is untested!"
            self.batch_size_per_device_group = max(self.batch_size // list(device.shape)[1], 1)
        else:
            self.batch_size_per_device_group = self.batch_size
        self.core_grid = device.compute_with_storage_grid_size()

        # Generate the cos/sin matrices needed for ttnn.embedding op
        cos_matrix, sin_matrix = get_cos_sin_matrix(reference_args)

        self.cos_matrix = ttnn.from_torch(
            cos_matrix,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            dtype=datatype,
            mesh_mapper=ttnn.ReplicateTensorToMesh(device),
        )
        self.sin_matrix = ttnn.from_torch(
            sin_matrix,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            dtype=datatype,
            mesh_mapper=ttnn.ReplicateTensorToMesh(device),
        )

        self.batch_grid = (
            ttnn.CoreGrid(y=4, x=8)
            if ttnn.get_arch_name() == "blackhole"
            else ttnn.num_cores_to_corerangeset(batch_size, self.core_grid, row_wise=True)
        )
        # Generate the transformation matrix
        trans_mat = get_rot_transformation_mat().repeat(
            1,
            1,
            batch_size,
            1,
            # 1, 1, num_cores, 1
        )  # Repeat across all cores on device
        trans_mat_mem_config = ttnn.create_sharded_memory_config(
            shape=(ttnn.TILE_SIZE, ttnn.TILE_SIZE),
            core_grid=self.batch_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        self.transformation_mat = ttnn.from_torch(
            trans_mat,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            dtype=datatype,
            memory_config=trans_mat_mem_config,
            mesh_mapper=ttnn.ReplicateTensorToMesh(device),
        )

        prefill_trans_mat_torch = get_rot_transformation_mat()
        self.transformation_mat_prefill = ttnn.from_torch(
            prefill_trans_mat_torch,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            dtype=datatype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(device),
        )

    def get_both_trans_mats(self):
        assert self.transformation_mat is not None, "Transformation matrix not initialized"
        assert self.transformation_mat_prefill is not None, "Prefill Transformation matrix not initialized"
        return {"decode": self.transformation_mat, "prefill": self.transformation_mat_prefill}

    def get_rot_idxs(self, position_idxs, on_host=False):
        assert isinstance(position_idxs, torch.Tensor), "Position ids must be a torch tensor"
        assert len(position_idxs.shape) == 1, "position idxs must be a [batch] tensor"

        batch = position_idxs.shape[0]
        position_idxs = position_idxs.reshape(1, batch)  # [1, 1, 1, batch]
        assert position_idxs.shape == (1, batch), "position idxs must be a [1, batch] tensor"
        assert torch.min(position_idxs) >= 0, "position idxs must be non-negative"

        # Add padding if needed
        pad_size = nearest_32(batch) - batch
        position_idxs = torch.nn.functional.pad(position_idxs, (0, pad_size), "constant", 0)

        rot_idxs = ttnn.as_tensor(
            position_idxs,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.device),
            device=None if on_host else self.device,
            memory_config=None if on_host else ttnn.DRAM_MEMORY_CONFIG,
        )

        return rot_idxs

    def get_rot_mats(self, position_idxs, return_rot_idxs=False):
        device = self.device

        # If position_idxs is a torch tensor, get the TTNN version of it
        if isinstance(position_idxs, torch.Tensor):
            rot_idxs = self.get_rot_idxs(position_idxs)
        else:
            rot_idxs = position_idxs
            assert len(rot_idxs.shape) == 2 and rot_idxs.shape[0] == 1, "rot_idxs must be a [1, batch] tensor"

        # Send the idxs to device
        if rot_idxs.device != device:
            rot_idxs = ttnn.to_device(rot_idxs, device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        embedding_layout = ttnn.TILE_LAYOUT
        cos = ttnn.embedding(rot_idxs, self.cos_matrix, layout=embedding_layout)  # [1, batch, dim]
        sin = ttnn.embedding(rot_idxs, self.sin_matrix, layout=embedding_layout)  # [1, batch, dim]

        cos = ttnn.unsqueeze_to_4D(cos)  # [1, 1, batch, dim]
        sin = ttnn.unsqueeze_to_4D(sin)  # [1, 1, batch, dim]

        cos = ttnn.transpose(cos, 1, 2)  # [1, batch, 1[32], dim]
        sin = ttnn.transpose(sin, 1, 2)  # [1, batch, 1[32], dim]

        if self.batch_size_per_device_group % ttnn.TILE_SIZE != 0:
            cos = cos[:, : self.batch_size_per_device_group, :, :]
            sin = sin[:, : self.batch_size_per_device_group, :, :]

        mem_config = ttnn.create_sharded_memory_config(
            shape=(ttnn.TILE_SIZE, self.dim),
            core_grid=self.batch_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

        cos = ttnn.interleaved_to_sharded(cos, mem_config)  # [1, 1 (= batch / shard_num_cores), 1[32], self.dim]
        sin = ttnn.interleaved_to_sharded(sin, mem_config)  # [1, 1 (= batch / shard_num_cores), 1[32], self.dim]

        if return_rot_idxs:
            return [cos, sin], rot_idxs
        return [cos, sin]
