#!/usr/bin/env bash
# SessionStart hook: print a brief orientation line when a Claude session starts.
# Quiet on happy path; warns only when something actionable is off.
set -euo pipefail

repo_root=$(git rev-parse --show-toplevel 2>/dev/null || exit 0)
cd "$repo_root"

branch=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "?")
echo "VMAF fork session — branch: $branch" >&2

# Upstream delta (if upstream remote is configured)
if git remote get-url upstream >/dev/null 2>&1; then
    git fetch --quiet upstream master 2>/dev/null || true
    behind=$(git rev-list --count HEAD..upstream/master 2>/dev/null || echo 0)
    if [[ "$behind" -gt 0 ]]; then
        echo "  upstream/master has $behind commits we don't have — consider /sync-upstream" >&2
    fi
fi

# compile_commands freshness for clangd
cc="$repo_root/compile_commands.json"
if [[ -L "$cc" ]]; then
    target=$(readlink "$cc")
    if [[ -f "$target" ]]; then
        for mf in libvmaf/meson.build libvmaf/meson_options.txt; do
            [[ -f "$mf" && "$mf" -nt "$target" ]] && {
                echo "  WARN: $mf is newer than compile_commands.json — rebuild to refresh clangd" >&2
                break
            }
        done
    fi
else
    echo "  NOTE: no compile_commands.json symlink at repo root; clangd will have no DB" >&2
fi

# Pre-commit install check
if [[ -f .pre-commit-config.yaml && ! -x .git/hooks/pre-commit ]]; then
    echo "  WARN: pre-commit not installed. Run 'pre-commit install --hook-type pre-commit --hook-type commit-msg --hook-type pre-push'" >&2
fi

exit 0
