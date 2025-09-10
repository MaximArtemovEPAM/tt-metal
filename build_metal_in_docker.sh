#!/usr/bin/env bash

set -o nounset
set -x

DISTRO="$1"
COMPILER="$2"

case "$DISTRO" in
    "debian"|"fedora"|"rhel"|"ubuntu")
        ;;
    *)
        echo "Expected DISTRO as debian/fedora/rhel/ubuntu, got '${DISTRO}'"
        exit 1
        ;;
esac

case "$COMPILER" in
    "clang_17")
        TOOLCHAIN_PATH="cmake/x86_64-linux-clang-17-libstdcpp-toolchain.cmake"
        ;;
    "gcc_12")
        TOOLCHAIN_PATH="cmake/x86_64-linux-gcc-12-toolchain.cmake"
        ;;
    *)
        echo "Error: COMPILER must be 'clang_17' or 'gcc_12', got: '$COMPILER'"
        exit 1
        ;;
esac

IMAGE_NAME="docker-conan-test-${DISTRO}"

# Check if image exists and remove it
if docker images -q "${IMAGE_NAME}" > /dev/null 2>&1; then
    docker rmi "${IMAGE_NAME}"
fi

docker build --progress=plain \
    -f dockerfile/Dockerfile.conan-test.${DISTRO} -t "${IMAGE_NAME}" .

docker run -it --rm -v $(pwd):/workspace "${IMAGE_NAME}" \
    ./build_metal_with_conan.sh /tt-metal-conan/conan-build --toolchain-path "${TOOLCHAIN_PATH}"
