#!/usr/bin/env bash
# /build-vmaf implementation. See SKILL.md for the invocation contract.
set -euo pipefail

backend=cpu
config=release
sanitizers=none
reconfigure=0
clean=0

for arg in "$@"; do
    case "$arg" in
        --backend=*)    backend="${arg#*=}" ;;
        --config=*)     config="${arg#*=}" ;;
        --sanitizers=*) sanitizers="${arg#*=}" ;;
        --reconfigure)  reconfigure=1 ;;
        --clean)        clean=1 ;;
        -h|--help)
            sed -n '/^## Invocation/,/^## /p' "$(dirname "$0")/SKILL.md"
            exit 0 ;;
        *) echo "unknown arg: $arg" >&2; exit 2 ;;
    esac
done

repo_root=$(git rev-parse --show-toplevel)
cd "$repo_root/libvmaf"

opts=()
case "$backend" in
    cpu)  opts+=(-Denable_cuda=false -Denable_sycl=false) ;;
    cuda) command -v nvcc >/dev/null || { echo "nvcc not in PATH" >&2; exit 3; }
          opts+=(-Denable_cuda=true  -Denable_sycl=false) ;;
    sycl) command -v icpx >/dev/null || { echo "icpx not in PATH" >&2; exit 3; }
          opts+=(-Denable_cuda=false -Denable_sycl=true ) ;;
    hip)  opts+=(-Denable_cuda=false -Denable_sycl=false -Denable_hip=true) ;;
    all)  command -v nvcc >/dev/null || { echo "nvcc missing for --backend=all" >&2; exit 3; }
          command -v icpx >/dev/null || { echo "icpx missing for --backend=all" >&2; exit 3; }
          opts+=(-Denable_cuda=true -Denable_sycl=true) ;;
    *) echo "unknown backend: $backend" >&2; exit 2 ;;
esac

case "$config" in
    debug)          opts+=(--buildtype=debug) ;;
    release)        opts+=(--buildtype=release) ;;
    relwithdebinfo) opts+=(--buildtype=release -Db_ndebug=true) ;;
    *) echo "unknown config: $config" >&2; exit 2 ;;
esac

san="$sanitizers"
if [[ "$san" != none ]]; then
    if [[ "$san" == *tsan* ]]; then
        if [[ "$san" == *asan* || "$san" == *ubsan* ]]; then
            echo "tsan is mutually exclusive with asan/ubsan" >&2; exit 2
        fi
        opts+=(-Db_sanitize=thread --buildtype=debug)
    else
        # asan / ubsan / both
        case "$san" in
            asan)        opts+=(-Db_sanitize=address --buildtype=debug) ;;
            ubsan)       opts+=(-Db_sanitize=undefined --buildtype=debug) ;;
            asan,ubsan|ubsan,asan|address,undefined)
                         opts+=(-Db_sanitize=address,undefined --buildtype=debug) ;;
            *) echo "unknown sanitizers: $san" >&2; exit 2 ;;
        esac
    fi
fi

(( clean ))       && rm -rf build
(( reconfigure )) && opts=(--reconfigure "${opts[@]}")

t0=$(date +%s)
if [[ ! -d build ]]; then
    meson setup build "${opts[@]}"
else
    meson configure build "${opts[@]}" 2>/dev/null || meson setup --reconfigure build "${opts[@]}"
fi
ninja -C build "-j$(nproc)"
t1=$(date +%s)

echo
echo "built in $((t1 - t0))s"
echo "cli : $repo_root/libvmaf/build/tools/vmaf"
echo "lib : $repo_root/libvmaf/build/src/libvmaf.so.3.0.0"
