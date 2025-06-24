# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import time

import pytest
from loguru import logger

import ttnn
from models.demos.mobilenetv2.reference.mobilenetv2 import Mobilenetv2
from models.demos.mobilenetv2.tests.mobilenetv2_common import MOBILENETV2_BATCH_SIZE, MOBILENETV2_L1_SMALL_SIZE
from models.demos.mobilenetv2.tt import ttnn_mobilenetv2
from models.demos.mobilenetv2.tt.model_preprocessing import (
    create_mobilenetv2_input_tensors,
    create_mobilenetv2_model_parameters,
)
from models.utility_functions import run_for_wormhole_b0


@run_for_wormhole_b0()
@pytest.mark.models_performance_bare_metal
@pytest.mark.models_performance_virtual_machine
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": MOBILENETV2_L1_SMALL_SIZE, "trace_region_size": 6434816, "num_command_queues": 2}],
    indirect=True,
)
@pytest.mark.parametrize(
    "batch_size",
    ((MOBILENETV2_BATCH_SIZE),),
)
def test_run_mobilenetv2_trace_2cqs_inference(
    device,
    use_program_cache,
    batch_size,
    model_location_generator,
):
    torch_model = Mobilenetv2()
    torch_model.eval()

    torch_input_tensor, _ = create_mobilenetv2_input_tensors(batch=batch_size, input_height=224, input_width=224)
    torch_output_tensor = torch_model(torch_input_tensor)

    model_parameters = create_mobilenetv2_model_parameters(torch_model, device=device)

    ttnn_model = ttnn_mobilenetv2.TtMobileNetV2(model_parameters, device, batchsize=batch_size)

    _, host_input_tensor = create_mobilenetv2_input_tensors(
        batch=batch_size, input_height=224, input_width=224, input_channels=16
    )

    dram_cores = 10
    dram_shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dram_cores - 1, 0))}),
        [host_input_tensor.shape[-2] // dram_cores, host_input_tensor.shape[-1]],
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    input_dram_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.DRAM, dram_shard_spec
    )

    input_l1_mem_config = ttnn.create_sharded_memory_config(
        shape=(7840, host_input_tensor.shape[-1]),
        core_grid=ttnn.CoreGrid(y=8, x=8),  # (0,0) to (7,7)
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    input_dram_tensor = ttnn.allocate_tensor_on_device(
        host_input_tensor.shape, host_input_tensor.dtype, host_input_tensor.layout, device, input_dram_mem_config
    )

    # Dummy record an op event on CQ 0 since we wait on this first in the loop
    op_event = ttnn.record_event(device, 0)

    # First run to compile the model
    # Stall CQ 1 for the input tensor consumer (CQ 0) to signal it has finished
    ttnn.wait_for_event(1, op_event)
    # Write the input tensor on CQ 1
    ttnn.copy_host_to_device_tensor(host_input_tensor, input_dram_tensor, cq_id=1)
    # Signal that the write has finished on CQ 1
    write_event = ttnn.record_event(device, 1)
    # Make CQ 0 stall until CQ 1 has signalled that the write has finished
    ttnn.wait_for_event(0, write_event)
    # Run the first operation of the model on CQ 0 - move from DRAM to L1
    input_l1_tensor = ttnn.reshard(input_dram_tensor, input_l1_mem_config)
    # Signal to CQ 1 that CQ 0 is finished with the input and it can be overwritten
    op_event = ttnn.record_event(device, 0)
    # Run the rest of the model on CQ 0
    output_tensor = ttnn_model(input_l1_tensor)

    # Capture the trace of the model
    ttnn.wait_for_event(1, op_event)
    ttnn.copy_host_to_device_tensor(host_input_tensor, input_dram_tensor, cq_id=1)
    write_event = ttnn.record_event(device, 1)
    ttnn.wait_for_event(0, write_event)
    input_l1_tensor = ttnn.reshard(input_dram_tensor, input_l1_mem_config)
    op_event = ttnn.record_event(device, 0)

    # Record the address of the input tensor to trace
    input_trace_addr = input_l1_tensor.buffer_address()
    spec = input_l1_tensor.spec

    # Deallocate the previous output tensor so we can allocate at the right address
    output_tensor.deallocate(force=True)

    tid = ttnn.begin_trace_capture(device, cq_id=0)
    # Keep the output tensor on device for reading after trace execution
    output_tensor = ttnn_model(input_l1_tensor)

    # Allocate persistent input tensor and verify it matches the trace address
    input_l1_tensor = ttnn.allocate_tensor_on_device(spec, device)
    assert input_trace_addr == input_l1_tensor.buffer_address()

    ttnn.end_trace_capture(device, tid, cq_id=0)

    outputs = []
    iterations = 32

    start = time.time()
    for iteration in range(iterations):
        # Stall CQ 1 for the input tensor consumer (CQ 0) to signal it has finished
        ttnn.wait_for_event(1, op_event)
        # Write the next input tensor on CQ 1
        ttnn.copy_host_to_device_tensor(host_input_tensor, input_dram_tensor, cq_id=1)
        # Signal that the write has finished on CQ 1
        write_event = ttnn.record_event(device, 1)
        # Make CQ 0 stall until CQ 1 has signalled that the write has finished
        ttnn.wait_for_event(0, write_event)
        # Move from DRAM to L1 using in-place reshard to reuse the address
        input_l1_tensor = ttnn.reshard(input_dram_tensor, input_l1_mem_config, input_l1_tensor)
        # Signal to CQ 1 that CQ 0 is finished with the input and it can be overwritten
        op_event = ttnn.record_event(device, 0)
        # Execute the traced model on CQ 0
        ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
        outputs.append(output_tensor.cpu(blocking=False))

    # Final synchronize to wait for all outputs to be read to host
    ttnn.synchronize_device(device)

    end = time.time()
    inference_time = (end - start) / iterations
    logger.info(f"Average model time={1000.0 * inference_time : .2f} ms")
    logger.info(f"Average model performance={iterations * batch_size / (end-start) : .2f} fps")

    # Verify outputs match PyTorch
    # TODO: DO THIS!

    logger.info(f"MobileNetV2 trace 2CQ inference test passed for batch_size={batch_size}")
