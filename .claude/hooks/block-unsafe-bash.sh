#!/usr/bin/env bash
# PreToolUse hook: block dangerous Bash commands before they execute.
# Exit 0  -> allow. Exit non-zero -> deny and surface stderr to the agent.
set -euo pipefail

cmd="${CLAUDE_TOOL_INPUT_command:-}"

# Patterns that are always blocked. Kept narrow so legitimate work is not impeded.
blocked_patterns=(
    # Force-push to protected branches
    'git[[:space:]]+push[[:space:]]+(--force|--force-with-lease|-f)[[:space:]]+origin[[:space:]]+(master|main|sycl)\b'
    'git[[:space:]]+push[[:space:]]+origin[[:space:]]+(master|main|sycl)[[:space:]]+(--force|-f)\b'
    # Hard reset of tracked branches to remote (destructive)
    'git[[:space:]]+reset[[:space:]]+--hard[[:space:]]+(origin/|upstream/)'
    # Clean / rm on the repo root or above
    'rm[[:space:]]+-rf?[[:space:]]+(/|~|\.\.|\.git(\b|/))'
    # Curl/wget piped into a shell (supply-chain risk)
    '(curl|wget)[[:space:]]+[^|]*\|[[:space:]]*(sh|bash|zsh|fish)\b'
    # Disabling git hooks or signing
    '--no-verify\b'
    '--no-gpg-sign\b'
    # Modifying Netflix golden assertions (paths per D24)
    "sed[[:space:]]+-i[[:space:]]+.*python/test/(quality_runner_test|vmafexec_test|vmafexec_feature_extractor_test|feature_extractor_test|result_test)\\.py"
)

for pat in "${blocked_patterns[@]}"; do
    if [[ "$cmd" =~ $pat ]]; then
        echo "BLOCKED: matched unsafe pattern: $pat" >&2
        echo "Command: $cmd" >&2
        echo "If this is intentional, run it manually outside the agent session." >&2
        exit 2
    fi
done

exit 0
