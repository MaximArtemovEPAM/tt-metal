#!/bin/bash
# /opt/test_env.sh

echo "=== Environment Variables on $(hostname) ==="
echo "ARCH_NAME: $ARCH_NAME"
echo "TT_METAL_HOME: $TT_METAL_HOME" 
echo "PYTHONPATH: $PYTHONPATH"
echo "TT_METAL_ENV: $TT_METAL_ENV"
echo "TT_METAL_MESH_ID: $TT_METAL_MESH_ID"
echo "TT_METAL_HOST_RANK_ID: $TT_METAL_HOST_RANK_ID"
echo "Current working directory: $(pwd)"
echo "Hostname: $(hostname)"
echo "Date: $(date)"
echo "============================================="
tt-smi -r
/home/asaigal/tt-metal-2/build_Release/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="*TestCustomMultiMeshMcast*" |& tee /home/asaigal/tt-metal-2/log.txt
tt-smi -r
/home/asaigal/tt-metal-2/build_Release/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="*TestCustomUnicastRaw*" |& tee /home/asaigal/tt-metal-2/log.txt
#TT_METAL_SLOW_DISPATCH_MODE=1 /home/asaigal/tt-metal/build_Release/test/tt_metal/test_dram_loopback_single_core
#TT_METAL_SLOW_DISPATCH=1 /home/asaigal/build_Release/test/tt-metal/test_dram_loopback_single_core
# Verify tt-smi is available
#if command -v tt-smi >/dev/null 2>&1; then
#	echo "tt-smi is available $(hostname)"
#    tt-smi -r
#else
#	echo "tt-smi not found in PATH $(hostname)"
#fi
