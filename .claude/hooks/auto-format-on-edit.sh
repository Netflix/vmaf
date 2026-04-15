#!/usr/bin/env bash
# PostToolUse hook (matcher: Edit|Write): auto-format files after the agent edits them.
# Uses repo-local tool versions when available; silently skips if a formatter is not installed.
set -euo pipefail

file="${CLAUDE_TOOL_INPUT_file_path:-}"
[[ -z "$file" || ! -f "$file" ]] && exit 0

# Never reformat files that are explicitly upstream-touched or generated
case "$file" in
    */resource/doc/*|*/subprojects/*|*/build/*|*/testdata/*.yuv|*.json|*.onnx|*.pkl)
        exit 0 ;;
esac

case "$file" in
    *.c|*.h|*.cpp|*.hpp|*.cu|*.cuh)
        command -v clang-format >/dev/null 2>&1 && clang-format -i --style=file "$file" || true
        ;;
    *.py)
        command -v black >/dev/null 2>&1 && black -q "$file" || true
        command -v isort >/dev/null 2>&1 && isort -q "$file" || true
        ;;
    *.sh)
        command -v shfmt >/dev/null 2>&1 && shfmt -w -i 4 -ci "$file" || true
        ;;
esac

exit 0
