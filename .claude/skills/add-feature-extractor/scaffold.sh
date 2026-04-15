#!/usr/bin/env bash
# scaffold.sh — create a new perceptual feature extractor skeleton.
# Usage: bash .claude/skills/add-feature-extractor/scaffold.sh <name>

set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "usage: $0 <name>" >&2
    exit 2
fi

name="$1"
name_upper="${name^^}"
name_pascal=$(echo "$name" | awk -F_ '{for(i=1;i<=NF;i++)$i=toupper(substr($i,1,1)) substr($i,2);}1' OFS='')

repo_root=$(git rev-parse --show-toplevel)
tpl="$repo_root/.claude/skills/add-feature-extractor/templates"
dst="$repo_root/libvmaf/src/feature"

for suffix in h c; do
    out="$dst/${name}.${suffix}"
    if [[ -e "$out" ]]; then
        echo "refuse: $out already exists" >&2
        exit 1
    fi
    sed -e "s/{{NAME}}/$name/g" \
        -e "s/{{NAME_UPPER}}/$name_upper/g" \
        -e "s/{{NAME_PASCAL}}/$name_pascal/g" \
        "$tpl/feature.${suffix}.template" > "$out"
    echo "wrote $out"
done

echo "next: register $name in libvmaf/src/feature/feature_extractor.c and add a model-json entry."
