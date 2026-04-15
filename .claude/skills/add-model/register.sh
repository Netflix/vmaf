#!/usr/bin/env bash
# register.sh — validate a VMAF model JSON/pkl/onnx and wire it into the build.
# Usage: bash .claude/skills/add-model/register.sh <path-to-model>

set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "usage: $0 <path-to-model>" >&2
    exit 2
fi

model="$1"
if [[ ! -f "$model" ]]; then
    echo "not a file: $model" >&2
    exit 1
fi

ext="${model##*.}"
repo_root=$(git rev-parse --show-toplevel)

case "$ext" in
    json)
        python3 -c "import json, sys; json.load(open(sys.argv[1]))" "$model"
        ;;
    pkl)
        python3 -c "import pickle, sys; pickle.load(open(sys.argv[1], 'rb'))" "$model" \
            || { echo "invalid pickle (or unsafe — beware of trusted sources only)"; exit 1; }
        ;;
    onnx)
        if command -v python3 >/dev/null 2>&1 && python3 -c "import onnx" 2>/dev/null; then
            python3 -c "import onnx; onnx.checker.check_model(onnx.load('$model'))"
        else
            echo "warn: onnx package not installed, skipping schema check"
        fi
        ;;
    *)
        echo "unknown extension: .$ext (expected .json / .pkl / .onnx)" >&2
        exit 2
        ;;
esac

dest="$repo_root/model/$(basename "$model")"
if [[ "$(realpath "$model")" != "$(realpath "$dest" 2>/dev/null || true)" ]]; then
    cp "$model" "$dest"
    echo "copied to $dest"
fi

echo "registered $(basename "$dest"). Next: reference it via \`-m version=$(basename "${dest%.*}")\` or \`-m path=$dest\`."
