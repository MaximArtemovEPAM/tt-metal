from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Union

import ttnn

# Union type for all possible program configs used with ttnn.linear
ProgramConfig = Union[
    ttnn.MatmulMultiCoreReuseMultiCastProgramConfig,
    ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig,
    ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig,
    None,
]


@dataclass
class ConfigBase:
    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def keys(self):
        return tuple(f.name for f in fields(self))


@dataclass
class LinearConfig(ConfigBase):
    """Common parameters for a ttnn.linear layer, weights are in input_tensor_b"""

    input_tensor_b: ttnn.Tensor | Path | None = None
    memory_config: ttnn.MemoryConfig | None = None
    compute_kernel_config: ttnn.DeviceComputeKernelConfig | None = None
    program_config: ProgramConfig = None


@dataclass
class EmbeddingConfig(ConfigBase):
    """Common parameters for a ttnn.embedding layer"""

    weight: ttnn.Tensor | Path | None = None
    memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG
    layout: ttnn.Layout = ttnn.TILE_LAYOUT


@dataclass
class MulConfig(ConfigBase):
    memory_config: ttnn.MemoryConfig | None = None
    input_tensor_a_activations: list[ttnn.UnaryOpType] | None = None


@dataclass
class AllReduceConfig(ConfigBase):
    cluster_axis: int
    dim: int
    num_reduce_scatter_links: int
    num_all_gather_links: int
    topology: ttnn.Topology
    dtype: ttnn.DataType
    use_composite: bool


@dataclass
class ReshardConfig(ConfigBase):
    """Simple config for operations that only need memory configuration"""

    memory_config: ttnn.MemoryConfig


@dataclass
class AllGatherConfig(ConfigBase):
    cluster_axis: int
    dim: int
    num_links: int
    topology: ttnn.Topology


@dataclass
class RMSNormConfig(ConfigBase):
    """RMSNorm config"""

    epsilon: float
    weight: ttnn.Tensor | Path | None = None
    compute_kernel_config: ttnn.DeviceComputeKernelConfig | None = None
