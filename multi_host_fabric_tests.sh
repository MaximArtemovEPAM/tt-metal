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
$TT_METAL_HOME/build/test/tt_metal/multi_host_fabric_tests |& tee $TT_METAL_HOME/log.txt
