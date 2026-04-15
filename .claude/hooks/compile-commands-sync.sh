#!/usr/bin/env bash
# PostToolUse hook: keep the repo-root compile_commands.json symlink fresh for clangd
# (D14 — clangd is the chosen C/C++ LSP). Fires when meson.build or meson_options.txt
# changes; no-op otherwise.
set -euo pipefail

file="${CLAUDE_TOOL_INPUT_file_path:-}"
[[ -z "$file" ]] && exit 0

case "$(basename "$file")" in
    meson.build|meson_options.txt) ;;
    *) exit 0 ;;
esac

repo_root=$(git rev-parse --show-toplevel 2>/dev/null || pwd)
build_dir="$repo_root/libvmaf/build"
cc_json="$build_dir/compile_commands.json"
link="$repo_root/compile_commands.json"

[[ ! -f "$cc_json" ]] && exit 0

if [[ -L "$link" && "$(readlink "$link")" == "$cc_json" ]]; then
    exit 0
fi

ln -sfn "$cc_json" "$link"
echo "compile_commands.json symlink refreshed -> $cc_json" >&2
exit 0
