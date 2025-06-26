# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
from loguru import logger
import os
from pathlib import Path
import csv
import ttnn
import numpy as np

from tt_metal.tools.profiler.process_device_log import import_log_run_stats
import tt_metal.tools.profiler.device_post_proc_config as device_post_proc_config
from tt_metal.tools.profiler.common import PROFILER_LOGS_DIR, PROFILER_DEVICE_SIDE_LOG, rm

profiler_log_path = PROFILER_LOGS_DIR / PROFILER_DEVICE_SIDE_LOG


def get_op_duration(op_name):
    # Import profiler log file and run perf related statistic calculation
    setup = device_post_proc_config.perf_analysis()
    setup.deviceInputLog = profiler_log_path
    device_data = import_log_run_stats(setup)

    # TODO provide more bullet proof solution, currently assumes there is only 1 device and 1 core
    core_name = list(device_data["devices"][0]["cores"].keys())[0]

    # Get event timeseries for Unpacker and Packer cores (TRISC_0 and TRISC_2)
    trisc0_data = device_data["devices"][0]["cores"][core_name]["riscs"]["TRISC_0"]["timeseries"]
    trisc2_data = device_data["devices"][0]["cores"][core_name]["riscs"]["TRISC_2"]["timeseries"]

    # TODO provide safe guards, OP existis, there is same number of OP occurences etc... Add error handling
    # Get Unpacker op_name zone start time series
    trisc0_op_start = [i for i in trisc0_data if (i[0]["zone_name"] == op_name and i[0]["type"] == "ZONE_START")]

    # Get Packer op_name zone end time series
    trisc2_op_end = [i for i in trisc2_data if (i[0]["zone_name"] == op_name and i[0]["type"] == "ZONE_END")]

    # Get op duration for each OP occurence in timeseries, assuming everything is in place
    op_duration = []
    for i in range(len(trisc0_op_start)):
        op_duration.append(trisc2_op_end[i][1] - trisc0_op_start[i][1])

    return op_duration


def get_profiler_data(perf_scope, op_name, op_duration=False):
    # Import profiler log file and run perf related statistic calculation
    setup = device_post_proc_config.perf_analysis()
    setup.deviceInputLog = profiler_log_path
    deviceData = import_log_run_stats(setup)
    data = []

    # Add UNTILIZE-BLOCK/OP zone average duration per trisc core
    data.append(
        deviceData["devices"][0]["cores"]["DEVICE"]["analysis"][f"trisc0_{op_name}_{perf_scope}_duration"]["stats"][
            "Average"
        ]
    )
    data.append(
        deviceData["devices"][0]["cores"]["DEVICE"]["analysis"][f"trisc1_{op_name}_{perf_scope}_duration"]["stats"][
            "Average"
        ]
    )
    data.append(
        deviceData["devices"][0]["cores"]["DEVICE"]["analysis"][f"trisc2_{op_name}_{perf_scope}_duration"]["stats"][
            "Average"
        ]
    )

    op_name_profiler = op_name + "-op"
    op_name_profiler = op_name_profiler.upper()
    if op_duration:
        data.append(np.mean(get_op_duration(deviceData, op_name_profiler)))

    return data


def test_tilize():
    # Set log csv file name, file will be used to store perf data
    ENVS = dict(os.environ)
    TT_METAL_HOME = Path(ENVS["TT_METAL_HOME"])
    log_file = TT_METAL_HOME / "generated" / f"eltwise_add_sweep_cpp.csv"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    num_iterations = 8
    with open(log_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        csv_header = [
            "num_iterations",
            "num_tiles",
            "op_duration(mean)",
            "op_duration(min)",
            "op_duration(max)",
        ]
        writer.writerow(csv_header)

        for num_tiles in [1, 2, 3, 4, 5, 8, 16, 32, 64, 128, 256, 512]:
            # Clean profiler log file
            # ttnn.DumpDeviceProfiler(device)
            rm(profiler_log_path)

            # Put os system call to execute cpp test
            os.system(
                f"./build_Release_tracy/test/tt_metal/test_eltwise_binary --num_tiles {num_tiles} --iter {num_iterations}"
            )

            # Process profiler log file and extract tilize data
            # ttnn.DumpDeviceProfiler(device)
            op_duration = get_op_duration("ELTWISE-BINARY-OP")
            csv_data = [
                num_iterations,
                num_tiles,
                f"{np.mean(op_duration) / num_tiles:.2f}",
                f"{np.min(op_duration) / num_tiles:.2f}",
                f"{np.max(op_duration) / num_tiles:.2f}",
            ]
            writer.writerow(csv_data)
            file.flush()
