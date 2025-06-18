# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.tt_transformers.tt.ccl import tt_distributed_rmsnorm, tt_sharded_distributed_rmsnorm


class DistributedNorm(LightweightModule):
    def __init__(
        self,
        norm,
        args,
        TG=False,
        ccl_sub_device_crs=None,
        worker_sub_device_id=None,
    ):
        self.norm = norm
        self.args = args

        self.ccl_sub_device_crs = ccl_sub_device_crs
        self.worker_sub_device_id = worker_sub_device_id

        if TG:
            core_grid_ln = (
                min(4, args.dim // 4 // 32 // 8),
                8,
            )  # dividing by 4 and 8 for num_cols and num_rows of mesh, and 32 for tile size
            num_cores_ln = core_grid_ln[0] * core_grid_ln[1]
            hidden_size_per_device_distributed_ln = args.dim // 4
            self.gather_in_mem_cfg = ttnn.create_sharded_memory_config(
                shape=(1, 1, 32, hidden_size_per_device_distributed_ln),
                core_grid=ttnn.CoreGrid(y=core_grid_ln[0], x=core_grid_ln[1]),
                strategy=ttnn.ShardStrategy.WIDTH,
            )
            self.ln_prg_cfg = ttnn.LayerNormShardedMultiCoreProgramConfig(
                compute_with_storage_grid_size=(core_grid_ln[1], core_grid_ln[0]),
                subblock_w=(hidden_size_per_device_distributed_ln // num_cores_ln) // 32,
                block_h=1,
                block_w=(hidden_size_per_device_distributed_ln // num_cores_ln) // 32,
                inplace=False,
            )
            self.ln_sharded_stats_memcfg = ttnn.create_sharded_memory_config(
                shape=[1, 1, 32, 32 * 4],
                core_grid=ttnn.CoreGrid(y=1, x=1),
                strategy=ttnn.ShardStrategy.WIDTH,
            )
            self.ln_cfg = ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi2,
                math_approx_mode=False,
                fp32_dest_acc_en=False,
                packer_l1_acc=False,
            )
        self.TG = TG

    def forward(self, x, mode):
        """Apply a norm, possibly gathering inputs if required."""
        if self.TG:
            if mode == "decode":
                return tt_sharded_distributed_rmsnorm(
                    x,
                    epsilon=self.norm.eps,
                    gamma=self.norm.weight_distributed,
                    mesh_device=self.args.mesh_device,
                    ln_sharded_input_memcfg=self.gather_in_mem_cfg,
                    ln_sharded_progcfg=self.ln_prg_cfg,
                    ln_sharded_stats_memcfg=self.ln_sharded_stats_memcfg,
                )
            else:
                return tt_distributed_rmsnorm(
                    x,
                    epsilon=self.norm.eps,
                    gamma=self.norm.weight_distributed,
                    mesh_device=self.args.mesh_device,
                    compute_kernel_config=self.ln_cfg,
                )

        input_mem_cfg = self.norm.sharded_output_config if mode == "decode" else ttnn.DRAM_MEMORY_CONFIG

        # Distributed norm already performs a gather
        if self.args.is_multichip and not self.args.is_distributed_norm(mode):
            x = ttnn.all_gather(x, dim=3, num_links=1, topology=self.args.ccl_topology(), memory_config=input_mem_cfg)
        else:
            x = ttnn.to_memory_config(x, input_mem_cfg)

        x = self.norm(x, mode=mode, in_sharded=(mode == "decode"), out_sharded=(mode == "decode"))

        # Distributed norm requires a gather
        if self.args.is_distributed_norm(mode):
            use_all_gather_async_minimal_interleaved_any = not x.is_sharded() and x.layout == ttnn.TILE_LAYOUT
            if use_all_gather_async_minimal_interleaved_any:
                ag_input_dtype = x.dtype
                ag_output_shape = list(x.shape)
                ag_output_shape[3] *= self.args.mesh_device.get_num_devices()

                persistent_intermediate_buffer = ttnn.from_torch(
                    torch.zeros(ag_output_shape),
                    device=self.args.mesh_device,
                    layout=ttnn.TILE_LAYOUT,
                    dtype=ag_input_dtype,
                    memory_config=input_mem_cfg,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(self.args.mesh_device),
                )

                persistent_output_buffer = ttnn.from_torch(
                    torch.zeros(ag_output_shape),
                    device=self.args.mesh_device,
                    layout=ttnn.TILE_LAYOUT,
                    dtype=ag_input_dtype,
                    memory_config=input_mem_cfg,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(self.args.mesh_device),
                )

                multi_device_global_semaphore = [
                    ttnn.create_global_semaphore(self.args.mesh_device, self.ccl_sub_device_crs, 0) for _ in range(2)
                ]

                x = ttnn.experimental.all_gather_async(
                    x,
                    persistent_intermediate_buffer=persistent_intermediate_buffer,
                    persistent_output_buffer=persistent_output_buffer,
                    dim=3,
                    multi_device_global_semaphore=multi_device_global_semaphore,
                    num_links=1,
                    memory_config=input_mem_cfg,
                    topology=self.args.ccl_topology(),
                    subdevice_id=self.worker_sub_device_id,
                )
            else:
                multi_device_global_semaphore = ttnn.create_global_semaphore(
                    self.args.mesh_device, self.ccl_sub_device_crs, 0
                )

                x = ttnn.experimental.all_gather_async(
                    x,
                    dim=3,
                    multi_device_global_semaphore=multi_device_global_semaphore,
                    num_links=1,
                    memory_config=input_mem_cfg,
                    topology=self.args.ccl_topology(),
                    subdevice_id=self.worker_sub_device_id,
                )

            ttnn.synchronize_device(self.args.mesh_device, sub_device_ids=[self.worker_sub_device_id])

            print("22222222")
            print(x.shape[0])
            print(x.shape[1])
            print(x.shape[2])
            print(x.shape[3])
            print(x.dtype)
            print("---------")

        return x
