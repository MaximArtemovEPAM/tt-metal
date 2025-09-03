#!/bin/bash
libs=(
    "build_Release/tt_metal/libtt_metal.so"
    "build_Release/tt_metal/llrt/hal/tt-1xx/libhal_1xx.a"
    "build_Release/tt_stl/libtt_stl.so"
    "build_Release/tt_metal/third_party/umd/device/libdevice.so"
)

for lib in "${libs[@]}"; do
    echo "============================================"
    echo "Library: $lib"
    echo "============================================"

    if [[ ! -f "$lib" ]]; then
        echo "File not found!"
        continue
    fi

    # Method 1: nm with demangling
    if [[ $lib == *.so ]]; then
        nm -D -C "$lib" 2>/dev/null
    else
        nm -C "$lib" 2>/dev/null
    fi

    # Method 2: objdump with c++filt
    objdump -t "$lib" 2>/dev/null | c++filt

    echo
done
