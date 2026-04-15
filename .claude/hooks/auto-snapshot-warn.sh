#!/usr/bin/env bash
# PostToolUse hook: warn when a numerical-code file was edited, reminding the agent
# that snapshot regeneration may be required via /regen-snapshots.
# Does not block; informational only.
set -euo pipefail

file="${CLAUDE_TOOL_INPUT_file_path:-}"
[[ -z "$file" ]] && exit 0

case "$file" in
    */libvmaf/src/feature/*.c|*/libvmaf/src/feature/*.h \
    |*/libvmaf/src/feature/*/*.c|*/libvmaf/src/feature/*/*.h \
    |*/libvmaf/src/feature/*/*.cu|*/libvmaf/src/feature/*/*.cuh \
    |*/libvmaf/src/feature/*/*.cpp|*/libvmaf/src/feature/*/*.hpp)
        cat >&2 <<EOF
NOTICE: edited feature extractor code: $file

If this changes numerical output (even slightly), test snapshots under
'testdata/scores_cpu_*.json' and 'testdata/netflix_benchmark_results.json' may need
regeneration. Run '/regen-snapshots' and include a justification in the commit message.

If this change is expected to be bit-exact (pure perf), run '/cross-backend-diff' to
confirm ULP == 0 before proceeding.

Netflix golden-data tests (python/test/*.py assertAlmostEqual) must NEVER be modified.
EOF
        ;;
    */python/test/quality_runner_test.py \
    |*/python/test/vmafexec_test.py \
    |*/python/test/vmafexec_feature_extractor_test.py \
    |*/python/test/feature_extractor_test.py \
    |*/python/test/result_test.py)
        cat >&2 <<EOF
ERROR: edit detected to a Netflix golden-data test file: $file

These tests contain the canonical ground-truth assertions for VMAF numerical correctness
(see docs/principles.md §3.1 and CLAUDE.md §8). They are preserved verbatim. If your
change is strictly additive (new test cases in separate functions), that is allowed, but
existing 'assertAlmostEqual' values must not be modified. Review your diff before commit.
EOF
        ;;
esac

exit 0
