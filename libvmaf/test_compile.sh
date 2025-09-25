#!/bin/bash
set -e

rm -r build_*
meson setup --reconfigure -Denable_cuda=true -Denable_avx512=true -Denable_nvtx=false -Dwerror=false -Denable_nvcc=true --buildtype release build_cuda .
ninja -C build_cuda -j20
meson setup --reconfigure -Denable_cuda=true -Denable_avx512=true -Denable_nvtx=true -Dwerror=false -Denable_nvcc=true --buildtype release build_cuda_nvtx .
ninja -C build_cuda_nvtx -j20
meson setup --reconfigure -Denable_cuda=false -Denable_avx512=true -Denable_nvtx=false -Dwerror=false -Denable_nvcc=true --buildtype release build .
ninja -C build -j20
meson setup --reconfigure -Denable_cuda=false -Denable_avx512=true -Denable_nvtx=true -Dwerror=false -Denable_nvcc=true --buildtype release build_nvtx .
ninja -C build_nvtx -j20
