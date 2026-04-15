#!/usr/bin/env bash
# Stop hook: brief session exit summary. Only prints if something is actionable.
set -euo pipefail

repo_root=$(git rev-parse --show-toplevel 2>/dev/null || exit 0)
cd "$repo_root"

dirty=$(git status --porcelain | wc -l | tr -d ' ')
unpushed=$(git log --oneline '@{u}..HEAD' 2>/dev/null | wc -l | tr -d ' ')

if [[ "$dirty" -eq 0 && "$unpushed" -eq 0 ]]; then
    exit 0
fi

echo "Session exit summary:" >&2
[[ "$dirty" -gt 0 ]]   && echo "  $dirty modified file(s) not committed" >&2
[[ "$unpushed" -gt 0 ]] && echo "  $unpushed commit(s) not pushed to origin" >&2

exit 0
