# Reduce h op nd sharding performance
[**Performance table**](https://docs.google.com/spreadsheets/d/1JHHGMV8-sA5JIItnPeju_-9AFRQFWQJ3CF3IcNruqJc/edit?usp=sharing)

## Algorithm overview
A simple sum-reduction over H dimensions (-2).

- Input: 4 dim tensor with shape (B, C, H, W)
- Output: 3 dim tensor with shape (B, C, 1, W)

Python api:
```python
dim = -2
keepdim = True
out = ttnn.sum(input_tensor, dim=dim, keepdim=keepdim)
# Torch analog:
out = torch.sum(torch_input_tensor, dim, keepdim)
```

Work is distributed along W dimension (i.e. each core iterates over full height)

```
tiles are read in the N W_skip H W_chunk order
W_skip(chunk size) represents the number of tile columns whose reading will be intertwined
H W_chunk represent tiles of the chunk read in row major order
exmpl. Ht = 3; Wt = 4; row_chunk = 2;
       read order (H, W):
       1. chunk:  1:(0, 0)  2:(0, 1)  3:(1, 0)   4:(1, 1)   5:(2, 0)   6:(2, 1)
       2. chunk:  7:(0, 2)  8:(0, 3)  9:(1, 2)  10:(1, 3)  11:(2, 2)  12:(2, 3)
```

Vizualization of tiles read order:

![](./images/reduce_h_tile_iteration_viz.png)

- [Host setup code](https://github.com/tenstorrent/tt-metal/blob/c6713ac633149d238a946741e61660c4da0196ba/ttnn/cpp/ttnn/operations/reduction/generic/device/multi_core_h/reduce_op_multi_core_h.cpp#L18)
- [Compute kernel](https://github.com/tenstorrent/tt-metal/blob/philei/nd-sharding-reduce-h-perf-improvements/ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/compute/reduce_h.cpp) is identical between all implementations
- [Writer kernel](https://github.com/tenstorrent/tt-metal/blob/philei/nd-sharding-reduce-h-perf-improvements/ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/dataflow/writer_unary_nd_shard_start_id.cpp) is identical between all implementations

### DRAM/L1 Interleaved
- Data is evenly distributed between all banks
- row_chunk = number of dest registers
- [Reader kernel](https://github.com/tenstorrent/tt-metal/blob/philei/nd-sharding-reduce-h-perf-improvements/ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/dataflow/reader_unary_transpose_wh_interleaved_input_cols_partitioned.cpp)

### 2d sharding
- Assumes that number of shards = number of cores
- row_chunk = 1
- Based on assumption above, indexing logic is much simpler
- [Reader kernel](https://github.com/tenstorrent/tt-metal/blob/philei/nd-sharding-reduce-h-perf-improvements/ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/dataflow/reader_unary_transpose_wh_interleaved_input_cols_partitioned_sharded.cpp)

### ND sharding
- Both host and device code is identical to interleaved
- The only difference is using ShardedAccessor instead odf InterleavedAddrGenFast


## Initial bringup
[**Performance table**](https://docs.google.com/spreadsheets/d/1JHHGMV8-sA5JIItnPeju_-9AFRQFWQJ3CF3IcNruqJc/edit?usp=sharing)

To avoid copying whe whole table in this doc, we'l track performance of `256x512 64 shards on 64 cores` case (tensor shape: (1, 1, 256, 32768), shard shape: (1, 1, 256, 512))


When tensor is created with the following nd sharding memory config:
```python
memory_config = ttnn.MemoryConfig(
    buffer_type=ttnn.BufferType.L1,
    nd_shard_spec=ttnn.NdShardSpec(
        (1, 1, h, shard_w),
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(end_x, end_y))}),
    ),
)
```

Performance is not good (2.3x-4.1x slower than 2d sharding):

| 256x512 64 shards on 64 cores | DRAM interleaved | L1 interleaved | 2d width sharding | nd sharding baseline |
| ----------------------------- | ---------------- | -------------- | ----------------- | -------------------- |
| time (ns)                     | 194891           | 179025         | 31490             | 129263               |


## Proper distribution of work cores
This op doesn't require any inter-core communication, and we have to make sure that core works with local shard if possible.

Adding the following snippet inside of get_noc_addr reveled that even for cases when (n work core == n shard), core (x, y) was working with shard (y, x):

```bash
export TT_METAL_DPRINT_CORES="(0,0)-(7,7)"
```

```c++
uint32_t x = (packed_xy_coords[bank_id] >> 8) & 0xFF;
uint32_t y = packed_xy_coords[bank_id] & 0xFF;
DPRINT_DATA0(DPRINT << "bank_id: " << (uint32_t)bank_id << ", x: " << x << ", y: " << y << ", my_x: " << (uint32_t) my_x[0] << ", my_y: " << (uint32_t) my_y[0] << ENDL());
```

This can be easily fixed with a col-major core orientation or if a proper host setup code is used
```python
memory_config = ttnn.MemoryConfig(
    buffer_type=ttnn.BufferType.L1,
    nd_shard_spec=ttnn.NdShardSpec(
        (1, 1, h, shard_w),
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(end_x, end_y))}),
        ttnn.ShardOrientation.COL_MAJOR,
    ),
)
```


Before:
```
...
0:(x=0,y=0):NC: bank_id: 0, x: 18, y: 18, my_x: 18, my_y: 18
0:(x=0,y=1):NC: bank_id: 1, x: 19, y: 18, my_x: 18, my_y: 19
0:(x=0,y=2):NC: bank_id: 2, x: 20, y: 18, my_x: 18, my_y: 20
0:(x=0,y=3):NC: bank_id: 3, x: 21, y: 18, my_x: 18, my_y: 21
0:(x=0,y=4):NC: bank_id: 4, x: 22, y: 18, my_x: 18, my_y: 22
0:(x=0,y=5):NC: bank_id: 5, x: 23, y: 18, my_x: 18, my_y: 23
0:(x=0,y=6):NC: bank_id: 6, x: 24, y: 18, my_x: 18, my_y: 24
0:(x=0,y=7):NC: bank_id: 7, x: 25, y: 18, my_x: 18, my_y: 25
...
```

After:
```
...
0:(x=0,y=0):NC: bank_id: 0, x: 18, y: 18, my_x: 18, my_y: 18
0:(x=0,y=1):NC: bank_id: 1, x: 18, y: 19, my_x: 18, my_y: 19
0:(x=0,y=2):NC: bank_id: 2, x: 18, y: 20, my_x: 18, my_y: 20
0:(x=0,y=3):NC: bank_id: 3, x: 18, y: 21, my_x: 18, my_y: 21
0:(x=0,y=4):NC: bank_id: 4, x: 18, y: 22, my_x: 18, my_y: 22
0:(x=0,y=5):NC: bank_id: 5, x: 18, y: 23, my_x: 18, my_y: 23
0:(x=0,y=6):NC: bank_id: 6, x: 18, y: 24, my_x: 18, my_y: 24
0:(x=0,y=7):NC: bank_id: 7, x: 18, y: 25, my_x: 18, my_y: 25
...
```

| 256x512 64 shards on 64 cores | DRAM interleaved | L1 interleaved | 2d width sharding | nd sharding baseline | nd sharding col-major |
| ----------------------------- | ---------------- | -------------- | ----------------- | -------------------- | --------------------- |
| time (ns)                     | 194891           | 179025         | 31490             | 129263               | 65393                 |

## Squashing 4d into 2d
As shown in [this PR](https://github.com/tenstorrent/tt-metal/pull/22929), `get_noc_addr` performance scales linearly with rank.

[Recent PR](https://github.com/tenstorrent/tt-metal/commit/76ab7b266d918fb99df108c2428ec89255b42fb3) introduced implicit dimension squashing, so that (1, 1, H, W) is visible as (H, W) for accessor. That change gives a massive performance boost

| 256x512 64 shards on 64 cores | DRAM interleaved | L1 interleaved | 2d width sharding | nd sharding baseline | nd sharding col-major | nd sharding col-major squeeze dims |
| ----------------------------- | ---------------- | -------------- | ----------------- | -------------------- | --------------------- | ---------------------------------- |
| time (ns)                     | 194891           | 179025         | 31490             | 129263               | 65393                 | 36961                              |


## Removing get_noc_addr
- Removing `get_noc_addr` with getting shard base address, and calculating offset directly in the kernel wins a few more percentages.
- [Here](https://github.com/tenstorrent/tt-metal/commit/88f6e2b0aca1e9c39e2415153143bbcad0b3f33a) how this can be done

| 256x512 64 shards on 64 cores | DRAM interleaved | L1 interleaved | 2d width sharding | nd sharding baseline | nd sharding col-major | nd sharding col-major squeeze dims | nd sharding col-major no get_noc_addr |
| ----------------------------- | ---------------- | -------------- | ----------------- | -------------------- | --------------------- | ---------------------------------- | ------------------------------------- |
| time (ns)                     | 194891           | 179025         | 31490             | 129263               | 65393                 | 36961                              | 34074                                 |

## Reusing 2d-sharded reader kernel
Single in case of n shards over n cores, tensor data is organized in the same exact way as in case of 2d sharding, we can reuse sharded reader kernel. This pretty much aligns performance with 2d sharding
| 256x512 64 shards on 64 cores | DRAM interleaved | L1 interleaved | 2d width sharding | nd sharding baseline | nd sharding col-major | nd sharding col-major squeeze dims | nd sharding col-major no get_noc_addr | nd sharding reuse 2d sharding kernel |
| ----------------------------- | ---------------- | -------------- | ----------------- | -------------------- | --------------------- | ---------------------------------- | ------------------------------------- | ------------------------------------ |
| time (ns)                     | 194891           | 179025         | 31490             | 129263               | 65393                 | 36961                              | 34074                                 | 31519                                |



### Reproduce benchmark results
```bash
tt_metal/tools/profiler/profile_this.py -c 'pytest tests/sweep_framework/sweeps/reduction/traces/sum_traces_nd_sharding.py::test_default'
```

- Should create a .csv file under generated/profiler/reports/time_stamp/
- Column "DEVICE KERNEL DURATION [ns]" should contain kernel duration
