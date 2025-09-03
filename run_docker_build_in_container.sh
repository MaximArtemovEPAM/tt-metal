#!/usr/bin/env bash

set -o xtrace

cd /workspace
ls
bash +x ./create_venv.sh
bash +x ./install_dependencies.sh
bash +x ./build_metal.sh
