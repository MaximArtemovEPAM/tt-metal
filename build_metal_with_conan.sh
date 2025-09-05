#!/usr/bin/env bash

conan profile detect --force
conan install tool_conanfile.txt --output-folder=conan_build --build-require --build=missing -c tools.cmake.cmakedeps:new=recipe_will_break --profile conan_profile.txt
cat conan_build/conanbuild.sh
source conan_build/conanbuild.sh
conan install conanfile.txt --output-folder=conan_build --build-require --build=missing -c tools.cmake.cmakedeps:new=recipe_will_break --profile conan_profile.txt
cmake --version
which cmake
ninja --version
which ninja
echo "$PATH"

./build_metal.sh
