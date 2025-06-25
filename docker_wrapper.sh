#!/bin/bash
# /home/asaigal/tt-metal/docker_wrapper.sh

HOST=$1
shift

ssh -l asaigal "$HOST" sudo docker exec \
  -e ARCH_NAME=wormhole_b0 \
  -e TT_METAL_HOME=/home/asaigal/tt-metal-2 \
  -e PYTHONPATH=/home/asaigal/tt-metal-2 \
  -e TT_METAL_ENV=dev \
  -e TT_MESH_ID=1 \
  -e TT_HOST_RANK=0 \
  asaigal-host-mapped "$@"
