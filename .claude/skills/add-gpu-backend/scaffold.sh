#!/usr/bin/env bash
#
# scaffold.sh — materialize the add-gpu-backend templates for a named backend.
#
# Usage: bash .claude/skills/add-gpu-backend/scaffold.sh <backend>
#   where <backend> is a short lowercase token (hip, vulkan, metal, rocm, opencl).

set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "usage: $0 <backend>" >&2
    exit 2
fi

backend="$1"
backend_upper="${backend^^}"
# Capitalize first letter for pascal-case (hip → Hip).
backend_pascal="${backend^}"

repo_root=$(git rev-parse --show-toplevel)
tpl="$repo_root/.claude/skills/add-gpu-backend/templates"
src="$repo_root/libvmaf/src"

subst() {
    sed -e "s/{{BACKEND}}/$backend/g" \
        -e "s/{{BACKEND_UPPER}}/$backend_upper/g" \
        -e "s/{{BACKEND_PASCAL}}/$backend_pascal/g" \
        "$@"
}

mkdir -p "$src/$backend" "$src/feature/$backend"

subst "$tpl/common.h.template"           > "$src/$backend/common.h"
subst "$tpl/common.c.template"           > "$src/$backend/common.c"
subst "$tpl/meson.build.template"        > "$src/$backend/meson.build"

for feat in adm vif motion; do
    FEATURE="$feat" \
    sed -e "s/{{BACKEND}}/$backend/g" \
        -e "s/{{BACKEND_UPPER}}/$backend_upper/g" \
        -e "s/{{BACKEND_PASCAL}}/$backend_pascal/g" \
        -e "s/{{FEATURE}}/$feat/g" \
        "$tpl/feature_stub.c.template" \
        > "$src/feature/$backend/${feat}_${backend}.c"
done

echo "scaffolded $backend backend under libvmaf/src/$backend/ + libvmaf/src/feature/$backend/"
echo "next steps:"
echo "  1. add 'option('enable_$backend', type: 'feature', value: 'auto')' to libvmaf/meson_options.txt"
echo "  2. wire libvmaf/src/$backend/meson.build into libvmaf/meson.build"
echo "  3. fill in the TODOs in common.c / *_${backend}.c"
echo "  4. add a matrix row for $backend in .github/workflows/ci.yml"
