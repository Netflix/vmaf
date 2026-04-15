#!/usr/bin/env bash
# scaffold.sh — create a new SIMD path for an existing feature extractor.
# Usage: bash .claude/skills/add-simd-path/scaffold.sh <isa> <feature>
#   <isa>     one of: avx2, avx512, avx10, neon, sve, rvv
#   <feature> an existing feature name under libvmaf/src/feature/ (adm, vif, motion, ...)

set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "usage: $0 <isa> <feature>" >&2
    exit 2
fi

isa="$1"
feature="$2"
repo_root=$(git rev-parse --show-toplevel)
tpl="$repo_root/.claude/skills/add-simd-path/templates"

case "$isa" in
    avx2|avx512|avx10)
        dir="$repo_root/libvmaf/src/feature/x86"
        header='#include <immintrin.h>'
        align=64
        ;;
    neon|sve)
        dir="$repo_root/libvmaf/src/feature/arm64"
        header='#include <arm_neon.h>'
        align=16
        ;;
    rvv)
        dir="$repo_root/libvmaf/src/feature/riscv"
        header='#include <riscv_vector.h>'
        align=16
        ;;
    *)
        echo "unknown isa: $isa" >&2
        exit 2
        ;;
esac

mkdir -p "$dir"
out="$dir/${feature}_${isa}.c"
if [[ -e "$out" ]]; then
    echo "refuse: $out already exists" >&2
    exit 1
fi

sed -e "s/{{FEATURE}}/$feature/g" \
    -e "s/{{ISA}}/$isa/g" \
    -e "s|{{ISA_HEADER}}|$header|g" \
    -e "s/{{ALIGN}}/$align/g" \
    "$tpl/simd_feature.c.template" > "$out"

echo "wrote $out"
echo "next: register runtime dispatch in libvmaf/src/feature/${feature}.c and add a bit-exact test."
