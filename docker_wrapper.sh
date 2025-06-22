#!/bin/bash
# /home/asaigal/tt-metal/docker_wrapper.sh

HOST=$1
shift

export ARCH_NAME=wormhole_b0
export TT_METAL_HOME=/home/asaigal/tt-metal
export PYTHONPATH=/home/asaigal/tt-metal
export TT_METAL_ENV=dev
export TT_METAL_MESH_ID=0
export TT_METAL_HOST_RANK=0

echo "$(date): Called with HOST=$HOST, COMMAND=$*" >> /home/asaigal/tt-metal/mpi_wrapper_calls.log

ssh -l asaigal "$HOST" sudo docker exec \
  -e ARCH_NAME=wormhole_b0 \
  -e TT_METAL_HOME=/home/asaigal/tt-metal-2 \
  -e PYTHONPATH=/home/asaigal/tt-metal-2 \
  -e TT_METAL_ENV=dev \
  -e TT_METAL_MESH_ID=1 \
  -e TT_METAL_HOST_RANK_ID=0 \
  asaigal-host-mapped "$@"
